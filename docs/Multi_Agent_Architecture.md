# Multi-Agent Architecture
### Redwood AI Insurance — Agentic AI Platform

---

> **Core Framing:**
> Agent orchestration in this platform is not a chatbot loop. It is a
> **stateful claim and underwriting lifecycle manager** — where each graph
> node is a bounded unit of work, edges encode business routing logic, and
> HITL interrupts are first-class citizens, not afterthoughts.

---

## Table of Contents
1. [Layer Separation](#1-layer-separation)
2. [FNOL Multi-Agent System](#2-fnol-multi-agent-system)
3. [Underwriting Quote Agent](#3-underwriting-quote-agent)
4. [State Schema Design](#4-state-schema-design)
5. [HITL Interrupt Pattern](#5-hitl-interrupt-pattern)
6. [Checkpointing Strategy](#6-checkpointing-strategy)
7. [Sync vs Async Inference Mapping](#7-sync-vs-async-inference-mapping)
8. [LangChain Integration Points](#8-langchain-integration-points)
9. [What NOT to Use from LangChain](#9-what-not-to-use-from-langchain)
10. [Deployment Mapping](#10-deployment-mapping)

---

## 1. Layer Separation

Three distinct layers with hard module boundaries. No layer imports from a lower layer.

```
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER — LangGraph                                │
│  StateGraph topology · node routing · HITL interrupts           │
│  checkpointing · conditional edges · graph compilation          │
├─────────────────────────────────────────────────────────────────┤
│  PRIMITIVES LAYER — LangChain                                   │
│  BaseChatModel (Bedrock / AzureOpenAI / vLLM)                   │
│  ChatPromptTemplate · PydanticOutputParser · BaseRetriever      │
│  LCEL chains inside individual nodes                            │
├─────────────────────────────────────────────────────────────────┤
│  BUSINESS LOGIC / ML LAYER — Pure Python + Pydantic             │
│  XGBoost scoring · feature store · Neo4j queries                │
│  entity resolution · regulatory mask · SHAP explainability      │
│  NO LangChain imports in this layer                             │
└─────────────────────────────────────────────────────────────────┘
```

**Module boundary rule:** `features/`, `models/`, `graph/`, `entities/` have no
LangChain imports. Enforced by import linting in CI. LangChain lives only in
`agents/nodes/` and `agents/chains/`.

---

## 2. FNOL Multi-Agent System

### 2a. Full Graph Topology

```
FNOL Submission
      │
      ▼
┌─────────────┐
│  intake     │  Parse + validate FNOL payload. Write claim_id,
│  node       │  policy lookup, claimant identity to state.
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  feature    │  Redis online feature lookup (<5ms).
│  enrichment │  Assemble sync feature vector from resolved entities.
│  node       │  Writes feature_vector to state.
└──────┬──────┘
       │
       ├─────────────────────────────────────────────────────────┐
       ▼                                                         ▼
┌─────────────┐                                        ┌────────────────┐
│  device     │  IP reputation, browser/app            │  graph query   │
│  check node │  metadata, geolocation delta.          │  node          │
│             │  Writes device_flags to state.         │  1–2 hop Neo4j │
└──────┬──────┘                                        │  neighborhood. │
       │                                               │  Writes        │
       └──────────────────┬────────────────────────────┤  graph_context │
                          ▼                            └────────────────┘
                 ┌─────────────────┐
                 │  tabular        │  XGBoost ensemble score.
                 │  scoring node   │  Writes sync_fraud_score [0.0–1.0].
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  decision       │  Applies risk tier thresholds +
                 │  routing node   │  regulatory rules + coverage check.
                 └────────┬────────┘
                          │
          ┌───────────────┼──────────────────┬──────────────────┐
          ▼               ▼                  ▼                  ▼
   score < 0.25     0.25 – 0.60        0.60 – 0.85          > 0.85
          │               │                  │                  │
          ▼               ▼                  ▼                  ▼
   ┌───────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  stp node │  │  evidence    │  │  siu referral│  │  payment     │
   │  (auto)   │  │  request     │  │  node        │  │  hold node   │
   │           │  │  node        │  │              │  │              │
   └─────┬─────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │               │                  │                  │
         │        ┌──────┴──────────────────┴──────────────────┘
         │        │      HITL INTERRUPT (suspend graph here)
         │        ▼
         │  ┌───────────────────┐
         │  │  investigator     │  SHAP explanation, graph path,
         │  │  copilot node     │  evidence checklist, suggested actions.
         │  │                   │  Awaits investigator decision.
         │  └────────┬──────────┘
         │           │
         │           ▼
         │  ┌───────────────────┐
         │  │  feedback node    │  Write decision to label store.
         │  │                   │  Trigger retraining eligibility check.
         │  └────────┬──────────┘
         │           │
         └─────┬─────┘
               │
               ▼
       ┌───────────────┐
       │  audit node   │  Write full decision record: score, tier, node path,
       │               │  investigator decision, SHAP values, timestamp.
       └───────┬───────┘
               │
               ▼
     [ASYNC PATH — post-submission, non-blocking]
               │
       ┌───────┴────────────────────────────────────────┐
       ▼               ▼               ▼                ▼
┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐
│  image    │  │  deep graph  │  │  narrative│  │  third party  │
│  analysis │  │  node        │  │  llm node │  │  enrichment   │
│  node     │  │  3+ hop      │  │  (LangCh.)│  │  node         │
│  (ViT)    │  │  traversal   │  │           │  │  MVR/credit   │
└─────┬─────┘  └──────┬───────┘  └─────┬────┘  └──────┬────────┘
      └────────────────┴────────────────┴───────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  async rescore  │  Re-evaluate ensemble with
                       │  node           │  enriched signals.
                       └────────┬────────┘
                                │
                    async_score > sync_score + threshold?
                                │
                      YES ──────┴────── NO
                       │                │
                       ▼                ▼
               escalate tier       no action
               (re-enter graph     (audit log
               at routing node)     updated)
```

### 2b. Conditional Edge Logic

```python
def route_by_score(state: FNOLState) -> str:
    score = state["sync_fraud_score"]
    if score < 0.25:
        return "stp_node"
    elif score < 0.60:
        return "evidence_request_node"
    elif score < 0.85:
        return "siu_referral_node"
    else:
        return "payment_hold_node"
```

Thresholds are configured in `config.py` — not hardcoded in the graph.
They are calibrated to minimize expected dollar loss, not maximize accuracy.

---

## 3. Underwriting Quote Agent

```
Quote Request
      │
      ▼
┌─────────────┐
│  intake     │  Parse quote request. Validate required fields.
│  node       │  Write quote_id, applicant identity to state.
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  entity     │  Confirm resolved entities exist for vehicle, person,
│  check node │  address. Fail fast if entity resolution not completed.
└──────┬──────┘
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
┌─────────────┐                        ┌──────────────────┐
│  MVR pull   │  Pull driving record   │  credit pull     │  Pull credit score.
│  node       │  from DMV/LexisNexis.  │  node            │  Skip if state in
│             │  Writes mvr_result.    │                  │  CA / MA / MI / HI.
└──────┬──────┘                        │                  │  Writes credit_score
       │                               │                  │  (null if excluded).
       └──────────────┬────────────────┘                  │
                      │                └──────────────────┘
                      ▼
             ┌─────────────┐
             │  feature    │  Assemble feature vector from feature store
             │  assembly   │  using MVR result + credit_score (if present).
             │  node       │  Apply state regulatory mask. Writes feature_vector.
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │  risk       │  Hurdle model: XGBoost frequency × severity.
             │  scoring    │  Calibrate output. Writes risk_score [0.0–1.0].
             │  node       │
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │  premium    │  rating_tables[risk_tier] × base_rate × modifiers.
             │  calc node  │  Pure Python — no LLM. Writes premium_usd.
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │  compliance │  Verify adverse action documentation triggers.
             │  check node │  Flag restricted-state credit exclusions.
             │             │  Writes regulatory_flags to state.
             └──────┬──────┘
                    │
          ┌─────────┴──────────────────┐
     risk_tier ≠ HIGH             risk_tier = HIGH
          │                            │
          │                            ▼
          │                  ┌──────────────────────┐
          │                  │  underwriter review  │  HITL INTERRUPT.
          │                  │  node                │  Presents risk_score,
          │                  │                      │  premium_usd, mvr_result,
          │                  │                      │  regulatory_flags.
          │                  │                      │  Awaits UW decision.
          │                  └──────────┬───────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
                ┌─────────────┐
                │  quote gen  │  LangChain LCEL chain: RAG retrieval from
                │  node       │  enterprise-rag-platform + LLM generates
                │  (LangCh.)  │  customer-facing quote explanation.
                │             │  Writes quote_explanation to state.
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │  issuance   │  Collect payment (sync external call).
                │  node       │  Write policy_id to state.
                │             │  Persist risk_score_at_issuance to feature
                │             │  store (shared data spine — feeds fraud
                │             │  scoring at FNOL). Payment failure raises;
                │             │  graph faults before policy_id is written.
                └─────────────┘
```

### 3a. Conditional Edge Logic

```python
def route_by_risk_tier(state: QuoteState) -> str:
    # HIGH tier triggers manual underwriter review before quoting.
    # LOW and MEDIUM proceed directly to quote generation.
    if state["risk_tier"] == "HIGH":
        return "underwriter_review_node"
    return "quote_gen_node"
```

Risk tier thresholds are configured in `config.py` — not hardcoded in the graph.

### 3b. Third-Party Check Parallelism

MVR and credit pulls run in parallel fan-out from `entity_check_node`. Both
are blocking inputs to `feature_assembly_node` — unlike FNOL's async
`third_party_enrichment_node`, the underwriting path cannot score without them.

```python
graph.add_edge("entity_check_node", "mvr_pull_node")
graph.add_edge("entity_check_node", "credit_pull_node")
graph.add_edge(["mvr_pull_node", "credit_pull_node"], "feature_assembly_node")
```

Credit pull is skipped at the node level (returns `credit_score=None` immediately)
when `state["applicant_state"]` is in the restricted set — the node always
executes so the graph topology stays uniform; the skip logic is internal.

**Key invariant:** `risk_score_at_issuance` is written to the feature store at
issuance — not at quote time. A quote that does not convert to a policy does not
pollute the fraud signal. See DEC-010.

---

## 4. State Schema Design

### FNOL State

```python
from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class FNOLState(TypedDict):
    # Identity
    claim_id: str
    policy_id: str
    claimant_id: str

    # Inputs
    fnol_payload: dict[str, Any]          # raw FNOL submission
    feature_vector: dict[str, Any]        # assembled from feature store

    # Scoring
    sync_fraud_score: float               # XGBoost tabular score
    async_fraud_score: float | None       # post-enrichment rescore
    graph_context: dict[str, Any]         # Neo4j neighborhood result
    device_flags: dict[str, Any]

    # Routing
    risk_tier: str                        # LOW / MEDIUM / HIGH / EXTREME
    decision: str                         # STP / EVIDENCE / SIU / HOLD

    # HITL
    evidence_requests: list[str]
    investigator_decision: str | None     # CONFIRM / CLEAR / ESCALATE
    investigator_override_reason: str | None

    # Audit
    shap_values: dict[str, float]
    audit_trail: list[dict[str, Any]]
    node_path: list[str]                  # which nodes executed, in order

    # LLM message thread (LangChain)
    messages: Annotated[list[BaseMessage], add_messages]
```

### Quote State

```python
class QuoteState(TypedDict):
    # Identity
    quote_id: str
    applicant_id: str
    applicant_state: str                  # US state code — drives regulatory mask

    # Inputs
    quote_request: dict[str, Any]
    feature_vector: dict[str, Any]

    # Third-party checks (blocking pre-score inputs)
    mvr_result: dict[str, Any]            # DMV / LexisNexis driving record
    credit_score: float | None            # None if state-excluded (CA/MA/MI/HI)

    # Scoring
    risk_score: float
    risk_tier: str                        # LOW / MEDIUM / HIGH
    premium_usd: float

    # Compliance
    regulatory_flags: list[str]
    credit_eligible: bool

    # HITL (underwriter review — HIGH tier only)
    underwriter_decision: str | None      # APPROVE / DECLINE / REFER
    underwriter_override_reason: str | None

    # Output
    quote_explanation: str | None         # LLM-generated customer explanation
    policy_id: str | None                 # populated at issuance

    # LLM message thread (LangChain)
    messages: Annotated[list[BaseMessage], add_messages]
```

**Design rules:**
- All state fields are plain Python types or Pydantic models — no LangGraph internals leak into state definitions.
- `messages` is the only field that uses a LangGraph reducer (`add_messages`). All other fields are last-write-wins.
- `audit_trail` and `node_path` are append-only by convention. Use a custom reducer if parallel branches write simultaneously.
- State is serialized to JSON for checkpointing — no non-serializable objects in state.

---

## 5. HITL Interrupt Pattern

LangGraph suspends graph execution at a designated node boundary. State is
persisted to the checkpoint store. Execution resumes when the human submits a
decision — which may be hours or days later. The same pattern applies to both
the FNOL investigator review and the underwriting underwriter review.

### FNOL — Investigator Copilot

```python
graph = StateGraph(FNOLState)

# Suspend BEFORE investigator_copilot_node executes.
graph.add_edge("evidence_request_node", "investigator_copilot_node")
graph.add_edge("siu_referral_node", "investigator_copilot_node")

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["investigator_copilot_node"],
)
```

**Resume flow:**

```python
# 1. Investigator submits decision via API
investigator_decision = {
    "investigator_decision": "CONFIRM",
    "investigator_override_reason": "Shared address with 3 confirmed ring members"
}

# 2. Update state with decision
app.update_state(
    config={"configurable": {"thread_id": claim_id}},
    values=investigator_decision,
)

# 3. Resume graph from the interrupt point
result = app.invoke(None, config={"configurable": {"thread_id": claim_id}})
```

### Underwriting — Underwriter Review (HIGH tier only)

```python
graph = StateGraph(QuoteState)

# Suspend BEFORE underwriter_review_node executes.
# Only reached when risk_tier == "HIGH" (see route_by_risk_tier).

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["underwriter_review_node"],
)
```

**Resume flow:**

```python
# 1. Underwriter submits decision via API
underwriter_decision = {
    "underwriter_decision": "APPROVE",
    "underwriter_override_reason": "MVR violations are >5 years old; risk acceptable"
}

# 2. Update state with decision
app.update_state(
    config={"configurable": {"thread_id": quote_id}},
    values=underwriter_decision,
)

# 3. Resume — proceeds to quote_gen_node or terminates on DECLINE
result = app.invoke(None, config={"configurable": {"thread_id": quote_id}})
```

If `underwriter_decision == "DECLINE"`, `quote_gen_node` is skipped and the
graph routes to a rejection notice node (not shown — same pattern as audit_node).

**Shared HITL rules:**
- The graph does not poll. It suspends and waits for an external `update_state` + `invoke(None)` call.
- The decision field (`investigator_decision` / `underwriter_decision`) is `None` at interrupt time. The review node checks for `None` and blocks if not yet populated.
- FNOL: STP claims (`score < 0.25`) never hit the interrupt — they route directly to `audit_node`.
- UW: LOW and MEDIUM risk tiers never hit the interrupt — they route directly to `quote_gen_node`.
- Override rate is tracked as a model health signal for both graphs.

---

## 6. Checkpointing Strategy

```
Node completes
      │
      ▼
LangGraph writes state snapshot to checkpoint store
      │
      ▼
Next node starts
```

| Environment | Checkpointer | Rationale |
|---|---|---|
| Local / dev | `SqliteSaver` | Zero infrastructure. File-backed. |
| Production (AWS) | `PostgresSaver` (RDS) or custom Redis | Durable, query-able by `claim_id` |
| Production (Azure) | `PostgresSaver` (Azure Database for PostgreSQL) | Same API |
| Production (OpenShift) | `PostgresSaver` (CrunhyData PGO) | Operator-managed Postgres |

**Why not Redis as primary checkpoint store:**
Redis is used for online feature serving at sub-10ms latency. Claim state
checkpoints are written once per node (~10–20 writes per claim), are larger
payloads, and require durability guarantees Redis RDB/AOF alone does not
provide in all configurations. Postgres is the safer default for checkpoint
durability.

**Checkpoint key:** `thread_id = claim_id`. Every graph run for the same claim
resumes from the last committed checkpoint — including async path re-scores
and investigator-resumed HITL flows.

---

## 7. Sync vs Async Inference Mapping

**FNOL graph:**

| Graph Section | Path | Latency | Blocks Customer? |
|---|---|---|---|
| `intake_node` → `decision_routing_node` | Sync | <65ms total | Yes — FNOL submission |
| `stp_node` → `audit_node` | Sync | <10ms | Yes |
| `evidence_request_node` | Sync | <10ms | Yes — sends request |
| `siu_referral_node` | Sync | <10ms | Yes — notifies |
| `payment_hold_node` | Sync | <10ms | Yes — blocks payment |
| `investigator_copilot_node` | HITL | Hours–days | No — async wait |
| `image_analysis_node` | Async | Seconds | No |
| `deep_graph_node` | Async | Seconds | No |
| `narrative_llm_node` | Async | 2–10s | No |
| `third_party_enrichment_node` | Async | 5–30s | No |
| `async_rescore_node` | Async | <5ms after enrichment | No |

**Underwriting quote graph:**

| Graph Section | Path | Latency | Blocks Customer? |
|---|---|---|---|
| `intake_node` → `entity_check_node` | Sync | <20ms | Yes — quote request |
| `mvr_pull_node` + `credit_pull_node` (parallel) | Sync | 1–5s | Yes — blocking pre-score |
| `feature_assembly_node` → `premium_calc_node` | Sync | <30ms | Yes |
| `compliance_check_node` | Sync | <10ms | Yes |
| `underwriter_review_node` | HITL | Hours–days | No — async wait (HIGH tier only) |
| `quote_gen_node` | Sync | 2–8s | Yes — LLM call |
| `issuance_node` (incl. payment call) | Sync | <500ms | Yes — payment + write |

**Implementation pattern:**
The sync and async paths are separate graph compilations — not parallel branches
in one graph. The sync graph completes and returns within the FNOL HTTP request
lifecycle. The async graph is triggered as a background job (SQS / Azure Service
Bus / OCP Job) using the same `claim_id` as `thread_id`, resuming from the
checkpoint written by the sync graph.

---

## 8. LangChain Integration Points

LangChain is used only inside agent node implementations. It never appears in
routing logic, state definitions, or the ML scoring path.

| Node | LangChain Component | Purpose |
|---|---|---|
| `narrative_llm_node` | `ChatBedrock` / `AzureChatOpenAI` / `ChatOpenAI` (vLLM) | LLM call for narrative consistency analysis |
| `narrative_llm_node` | `ChatPromptTemplate` | Structured prompt with claim narrative + prior statements |
| `narrative_llm_node` | `PydanticOutputParser` | Parse LLM response into `NarrativeAnalysisResult` |
| `quote_gen_node` | `ChatBedrock` / `AzureChatOpenAI` / `ChatOpenAI` | Quote explanation generation |
| `quote_gen_node` | LCEL chain | `retriever \| prompt \| llm \| parser` |
| `quote_gen_node` | `enterprise-rag-platform` `BaseRetriever` | Policy document retrieval for quote context |
| `investigator_copilot_node` | `ChatPromptTemplate` | Case summary prompt assembly |
| `underwriter_review_node` | `ChatPromptTemplate` | Risk summary prompt: score, premium, MVR, flags |
| All LLM nodes | `add_messages` reducer | Accumulate LLM message history in state |

**Substrate switching:**
All LLM nodes instantiate the model via a factory function that reads
`LLM_PROVIDER` from environment (`bedrock` / `azure_openai` / `vllm`).
The node implementation does not change — only the `BaseChatModel` instance
returned by the factory changes. This satisfies the three-binding deployment
model (DEC-014).

```python
def get_llm() -> BaseChatModel:
    provider = os.environ["LLM_PROVIDER"]
    if provider == "bedrock":
        return ChatBedrock(model_id=os.environ["BEDROCK_MODEL_ID"])
    elif provider == "azure_openai":
        return AzureChatOpenAI(deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"])
    elif provider == "vllm":
        return ChatOpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key="none")
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
```

---

## 9. What NOT to Use from LangChain

| Component | Status | Replacement |
|---|---|---|
| `AgentExecutor` | Excluded | LangGraph `StateGraph` |
| `initialize_agent` | Excluded | LangGraph `StateGraph` |
| `ConversationalRetrievalChain` | Excluded | LCEL chain inside a LangGraph node |
| `RetrievalQAChain` | Excluded | LCEL chain inside a LangGraph node |
| `SequentialChain` | Excluded | LangGraph edges |
| `ConversationBufferMemory` | Excluded | LangGraph state (`messages` field) |
| `ConversationSummaryMemory` | Excluded | LangGraph state with summary node |
| `StructuredChatAgent` | Excluded | LangGraph tool-calling node |
| `OpenAIFunctionsAgent` | Excluded | LangGraph tool-calling node |

**Why this matters:** These are pre-LangGraph abstractions that hide control
flow and make HITL impossible to implement. `AgentExecutor` has no suspend
primitive — it runs to completion or errors. `ConversationBufferMemory` is
mutable shared state outside the graph — it breaks checkpointing.

---

## 10. Deployment Mapping

| Component | AWS | Azure | OpenShift |
|---|---|---|---|
| FNOL agent graph | EKS pod (LangGraph) | AKS pod (LangGraph) | OCP Deployment (LangGraph) |
| Quote agent graph | EKS pod (LangGraph) | AKS pod (LangGraph) | OCP Deployment (LangGraph) |
| Checkpoint store | RDS PostgreSQL | Azure Database for PostgreSQL | CrunchyData PGO |
| Async job trigger | SQS + Lambda | Azure Service Bus + Function | OCP Job / Tekton |
| LLM backend | Bedrock | Azure OpenAI | vLLM on GPU nodepool |
| Feature store (sync) | ElastiCache Redis | Azure Cache for Redis | Redis Operator |
| Graph queries | Amazon Neptune | Cosmos DB (Gremlin) | Neo4j Operator |

LangGraph itself has no cloud dependency — the same `StateGraph` code runs
on all three substrates. Only the checkpointer backend and LLM factory
function are substrate-specific. See DEC-014.

---

*Architecture version: 2026-Q2 | Applies to: `fnol-claims-multi-agent-system/`, `intelligent-underwriting-platform/`*
