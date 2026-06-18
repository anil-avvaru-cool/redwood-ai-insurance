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
   - [2c. DocumentVerification Agent](#2c-documentverification-agent)
   - [2d. Subrogation Agent](#2d-subrogation-agent)
3. [Underwriting Quote Agent](#3-underwriting-quote-agent)
4. [State Schema Design](#4-state-schema-design)
5. [HITL Interrupt Pattern](#5-hitl-interrupt-pattern)
6. [Checkpointing Strategy](#6-checkpointing-strategy)
7. [Sync vs Async Inference Mapping](#7-sync-vs-async-inference-mapping)
8. [LangChain Integration Points](#8-langchain-integration-points)
9. [What NOT to Use from LangChain](#9-what-not-to-use-from-langchain)
10. [Deployment Mapping](#10-deployment-mapping)
11. [MCP Tool Layer](#11-mcp-tool-layer)
12. [Agent-to-Agent Integration Patterns](#12-agent-to-agent-integration-patterns)

---

## 1. Layer Separation

Four distinct layers with hard module boundaries. No layer imports from a lower layer.

```
┌─────────────────────────────────────────────────────────────────┐
│  AGENT INTEGRATION LAYER                                        │
│  Event-bus triggers · RemoteGraph invocation                    │
│  MCP client calls · cross-agent state via claim store           │
├─────────────────────────────────────────────────────────────────┤
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

**MCP tool calls** are made from graph nodes via a thin async MCP client. MCP client
code lives in `agents/nodes/` — same boundary as LangChain. Business logic and ML
scoring layers have no MCP dependency.

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
     [EVENT emitted → DocumentVerificationAgent triggered (Pattern A)]
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

### 2c. DocumentVerification Agent

**Deployment:** Independent LangGraph server (`doc-verification-agent`). Not a subgraph
inside the FNOL service — a separately deployed FastAPI + LangGraph instance.

**Lifecycle:** Async. Triggered at `audit_node` completion via event-bus (SQS / Azure
Service Bus / OCP Job). See [DEC-017](./DECISION_LOG.md#dec-017----event-bus-trigger-vs-remotegraph-invocation-decision-boundary) for the trigger pattern decision.

**Thread ID:** `doc_verify_{claim_id}` — isolated from the FNOL graph's `claim_id`
checkpoint to prevent key collision.

**State source:** Reads claim context from the claim store directly, not from the FNOL
LangGraph checkpoint. Keeps this agent decoupled from FNOL graph internals.

**MCP tools used:** `exif_analyzer`, `perceptual_hash_checker` (via `doc-verification-mcp-server`);
`repair_estimate_validator`, `narrative_consistency_checker` (via `doc-verification-mcp-server`).
See [MCP_Tool_Contracts.md](./MCP_Tool_Contracts.md) for full schemas.

```
Document Submission Event
(claim_id + document_refs — emitted at audit_node completion)
      │
      ▼
┌──────────────────┐
│  doc_intake_node │  Load claim context from claim store.
│                  │  Validate document_refs is non-empty.
│                  │  Write claim_context, doc_refs to state.
└────────┬─────────┘
         │
         ├──────────────────────────────┬──────────────────────────────┐
         ▼                              ▼                              ▼
┌──────────────────┐     ┌──────────────────────────┐    ┌────────────────────────┐
│ photo_auth_node  │     │ estimate_check_node       │    │ narrative_check_node   │
│                  │     │                           │    │                        │
│ MCP tools:       │     │ MCP tool:                 │    │ MCP tool:              │
│ exif_analyzer    │     │ repair_estimate_validator │    │ narrative_consistency  │
│ perceptual_hash  │     │                           │    │ _checker               │
│ _checker         │     │ Repair estimate vs        │    │                        │
│                  │     │ regional historical       │    │ Rule-based:            │
│ EXIF metadata +  │     │ actuals for vehicle       │    │ loss_type ↔ damage     │
│ perceptual hash. │     │ make/model/year.          │    │ description            │
│ Flag forged or   │     │ Writes                    │    │ consistency.           │
│ reused images.   │     │ estimate_check_result.    │    │ Writes                 │
│ Writes           │     │                           │    │ narrative_check_result.│
│ photo_auth_      │     │                           │    │                        │
│ result.          │     │                           │    │                        │
└────────┬─────────┘     └──────────────┬────────────┘    └──────────────┬─────────┘
         └────────────────────────────────┴──────────────────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  doc_decision_node    │
                              │                       │
                              │  Aggregate all three  │
                              │  check results.       │
                              │  SERVICE_UNAVAILABLE  │
                              │  tool errors →        │
                              │  INCONCLUSIVE (not    │
                              │  PASS or FLAG).       │
                              │  PASS: all clear.     │
                              │  FLAG: ≥1 failed.     │
                              │  Writes doc_verdict,  │
                              │  doc_evidence_summary.│
                              └──────────┬────────────┘
                                         │
                              PASS ───────┴────── FLAG
                               │                    │
                               ▼                    ▼
                   ┌──────────────────┐  ┌────────────────────────┐
                   │  doc_audit_node  │  │  flag_escalation_node  │
                   │  Write PASS to   │  │                        │
                   │  claim store.    │  │  Active SIU case:      │
                   └──────────────────┘  │  append FLAG to        │
                                         │  InvestigatorCopilot   │
                                         │  evidence_checklist.   │
                                         │                        │
                                         │  STP claim (no active  │
                                         │  case): create manual  │
                                         │  review task.          │
                                         │  Writes flag_action.   │
                                         └──────────┬─────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────┐
                                         │  doc_audit_node  │
                                         │  Write FLAG +    │
                                         │  evidence to     │
                                         │  claim store.    │
                                         └──────────────────┘
```

**Design decisions:**
- No dedicated HITL interrupt. FLAG results are routed to the InvestigatorCopilot's
  existing evidence checklist via `flag_escalation_node`, avoiding a second review
  queue for investigators.
- `photo_auth_node`, `estimate_check_node`, and `narrative_check_node` run in parallel
  fan-out — no data dependency between them. `doc_decision_node` is the fan-in point.
- `thread_id = doc_verify_{claim_id}` prevents checkpoint key collision with the FNOL
  graph's `thread_id = claim_id`.
- MCP tool `SERVICE_UNAVAILABLE` errors produce `INCONCLUSIVE` verdict in
  `doc_decision_node` — not a false FLAG. Audit node records the partial result.

---

### 2d. Subrogation Agent

**Deployment:** Independent LangGraph server (`subrogation-agent`). Separately deployed
FastAPI + LangGraph instance.

**Lifecycle:** Post-settlement. Triggered when a claim status transitions to `SETTLED`
via event-bus (SQS / Azure Service Bus / OCP Job). This is a distinct event source —
claims settle days to weeks after the initial FNOL submission, so this agent cannot be
a node in the FNOL async path.

**Thread ID:** `subrogation_{claim_id}` — isolated from both the FNOL and
DocumentVerification checkpoints.

**HITL:** Applies only when `recovery_amount > SUBROGATION_HIGH_VALUE_THRESHOLD` (config).
Requires senior claims handler approval before a recovery demand is issued.

**MCP tools used:** `police_report_api`, `telematics_crash_query`, `evidence_collector`
(via `claims-data-mcp-server`); `recovery_demand_queue` (via `subrogation-mcp-server`).
See [MCP_Tool_Contracts.md](./MCP_Tool_Contracts.md) for full schemas.

```
Claim SETTLED Event
(claim_id + settlement_amount — emitted on claim status → SETTLED)
      │
      ▼
┌─────────────────────┐
│ subrogation_intake  │  Load settled claim from claim store.
│ _node               │  Validate settlement_amount > 0.
│                     │  Load police_report_ref, telematics_crash_ref.
│                     │  Write claim_context, settlement_amount to state.
└──────────┬──────────┘
           │
           ├──────────────────────────────────────────────┐
           ▼                                              ▼
┌──────────────────────┐                   ┌─────────────────────────────┐
│ liability_scoring    │                   │ evidence_assembly_node       │
│ _node                │                   │                             │
│ MCP tools:           │                   │ MCP tool:                   │
│ police_report_api    │                   │ evidence_collector          │
│ telematics_crash_    │                   │                             │
│ query                │                   │ Collect: police report,     │
│                      │                   │ photos, witness statements. │
│ Police report +      │                   │ Score evidence completeness.│
│ telematics_crash_    │                   │ Writes evidence_refs,       │
│ match. Computes      │                   │ evidence_score to state.    │
│ at_fault_pct,        │                   └─────────────────┬───────────┘
│ confidence.          │                                     │
│ Writes               │                                     │
│ liability_assessment.│                                     │
└──────────┬───────────┘                                     │
           └────────────────────────┬────────────────────────┘
                                    ▼
                     ┌──────────────────────────────────┐
                     │  recovery_estimation_node         │
                     │                                  │
                     │  recovery = settlement_amount     │
                     │    × at_fault_pct × confidence.  │
                     │  viable = recovery >              │
                     │    SUBROGATION_MIN_RECOVERY.      │
                     │  Writes recovery_opportunity.     │
                     └──────────────┬───────────────────┘
                                    │
                         viable ────┴──── not viable
                           │                   │
                           ▼                   ▼
              ┌─────────────────────┐  ┌──────────────────────┐
              │ recovery_routing    │  │ subrogation_close    │
              │ _node               │  │ _node                │
              │                     │  │                      │
              │ recovery >          │  │ Write not-viable     │
              │ HIGH_VALUE_         │  │ result to claim      │
              │ THRESHOLD?          │  │ store. Audit log.    │
              └──────────┬──────────┘  └──────────────────────┘
                         │
             YES ─────────┴──── NO
              │                   │
              ▼                   ▼
[HITL INTERRUPT]       ┌────────────────────────┐
(suspend graph)        │ subrogation_referral   │
              │        │ _node                  │
              ▼        │ MCP tool:              │
┌──────────────────┐   │ recovery_demand_queue  │
│ senior_handler   │   │                        │
│ _review_node     │   │ Queue recovery demand  │
│                  │   │ to recovery team.      │
│ Present:         │   │ Write referral_id.     │
│ liability,       │   └────────────────────────┘
│ recovery_amount, │
│ evidence_score.  │
│ Awaits:          │
│ APPROVE / DECLINE│
└────────┬─────────┘
         │
  APPROVE┴── DECLINE
     │              │
     ▼              ▼
┌──────────────┐  ┌──────────────────────┐
│ subrogation  │  │ subrogation_close    │
│ _referral    │  │ _node                │
│ _node        │  └──────────────────────┘
└──────────────┘
```

**Design decisions:**
- Separate compilation from the FNOL graph is required — the settlement trigger fires
  on a different event (claim status change) and potentially weeks after the original
  FNOL submission. Cannot be a late async node in the FNOL graph.
- `liability_scoring_node` and `evidence_assembly_node` run in parallel fan-out —
  evidence completeness scoring is independent of the liability calculation.
- HITL scoped to high-value recovery only. Low-value demands (≤ `SUBROGATION_HIGH_VALUE_THRESHOLD`)
  proceed automatically to avoid manual overhead that would exceed recovery value.
- `settlement_amount` is a required input from the SETTLED event payload. The Subrogation
  agent does not infer or read settlement amount from the FNOL graph checkpoint.
- `SUBROGATION_MIN_RECOVERY` and `SUBROGATION_HIGH_VALUE_THRESHOLD` are both config
  values — not hardcoded in the graph.

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

### DocVerification State

```python
class DocVerificationState(TypedDict):
    # Identity
    claim_id: str
    policy_id: str

    # Input
    doc_refs: list[str]                       # URIs to submitted documents
    claim_context: dict[str, Any]             # loss_type, incident_description snapshot

    # Sub-check results (written by parallel fan-out nodes)
    photo_auth_result: dict[str, Any] | None
    estimate_check_result: dict[str, Any] | None
    narrative_check_result: dict[str, Any] | None

    # Decision
    doc_verdict: str | None                   # PASS / FLAG / INCONCLUSIVE
    doc_evidence_summary: list[str]

    # Escalation
    flag_action: str | None                   # APPENDED_TO_CASE / REVIEW_TASK_CREATED

    # Audit
    node_path: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
```

### Subrogation State

```python
class SubrogationState(TypedDict):
    # Identity
    claim_id: str
    policy_id: str

    # Input (from SETTLED event — both required at graph entry)
    settlement_amount: float
    claim_context: dict[str, Any]

    # Evidence (loaded by intake and assembly nodes)
    police_report_ref: str | None
    telematics_crash_ref: str | None
    evidence_refs: list[str]
    evidence_score: float | None              # completeness score [0.0–1.0]

    # Scoring
    liability_assessment: dict[str, Any] | None   # at_fault_pct, confidence, evidence list
    recovery_opportunity: dict[str, Any] | None   # recovery_amount, viable

    # HITL (high-value recovery only)
    senior_handler_decision: str | None           # APPROVE / DECLINE
    senior_handler_notes: str | None

    # Output
    referral_id: str | None

    # Audit
    node_path: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
```

**Design rules:**
- All state fields are plain Python types or Pydantic models — no LangGraph internals leak into state definitions.
- `messages` is the only field that uses a LangGraph reducer (`add_messages`). All other fields are last-write-wins.
- `audit_trail` and `node_path` are append-only by convention. Use a custom reducer if parallel branches write simultaneously.
- State is serialized to JSON for checkpointing — no non-serializable objects in state.
- `doc_verdict` now has three values: `PASS`, `FLAG`, `INCONCLUSIVE` (MCP tool unavailable).

---

## 5. HITL Interrupt Pattern

LangGraph suspends graph execution at a designated node boundary. State is
persisted to the checkpoint store. Execution resumes when the human submits a
decision — which may be hours or days later. The same pattern applies to all
three HITL interrupt points: FNOL investigator review, underwriting underwriter
review, and subrogation senior handler review (high-value recovery only).
DocumentVerification has no dedicated HITL — FLAG results are delegated to the
InvestigatorCopilot workflow via `flag_escalation_node`.

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

### Subrogation — Senior Handler Review (high-value recovery only)

```python
graph = StateGraph(SubrogationState)

# Suspend BEFORE senior_handler_review_node executes.
# Only reached when recovery_amount > SUBROGATION_HIGH_VALUE_THRESHOLD (config).

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["senior_handler_review_node"],
)
```

**Resume flow:**

```python
# 1. Senior handler submits decision via API
handler_decision = {
    "senior_handler_decision": "APPROVE",
    "senior_handler_notes": "Police report confirms clear third-party liability"
}

# 2. Update state with decision
app.update_state(
    config={"configurable": {"thread_id": f"subrogation_{claim_id}"}},
    values=handler_decision,
)

# 3. Resume — proceeds to subrogation_referral_node on APPROVE
result = app.invoke(None, config={"configurable": {"thread_id": f"subrogation_{claim_id}"}})
```

If `senior_handler_decision == "DECLINE"`, the graph routes to `subrogation_close_node`.

**Shared HITL rules:**
- The graph does not poll. It suspends and waits for an external `update_state` + `invoke(None)` call.
- The decision field (`investigator_decision` / `underwriter_decision` / `senior_handler_decision`) is `None` at interrupt time. The review node checks for `None` and blocks if not yet populated.
- FNOL: STP claims (`score < 0.25`) never hit the interrupt — they route directly to `audit_node`.
- UW: LOW and MEDIUM risk tiers never hit the interrupt — they route directly to `quote_gen_node`.
- Subrogation: Only claims where `recovery_amount > SUBROGATION_HIGH_VALUE_THRESHOLD` hit the interrupt — all others route directly to `subrogation_referral_node`.
- Override rate is tracked as a model health signal for all three graphs.

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
| Production (OpenShift) | `PostgresSaver` (CrunchyData PGO) | Operator-managed Postgres |

**Why not Redis as primary checkpoint store:**
Redis is used for online feature serving at sub-10ms latency. Claim state
checkpoints are written once per node (~10–20 writes per claim), are larger
payloads, and require durability guarantees Redis RDB/AOF alone does not
provide in all configurations. Postgres is the safer default for checkpoint
durability.

**Checkpoint keys by graph:**

| Graph | `thread_id` | Rationale |
|---|---|---|
| FNOL sync + async | `claim_id` | Single thread for both compilations — async path resumes from sync checkpoint |
| DocumentVerification | `doc_verify_{claim_id}` | Avoids key collision with FNOL checkpoint |
| Subrogation | `subrogation_{claim_id}` | Different lifecycle (post-settlement); distinct namespace |
| Underwriting | `quote_id` | Separate claim lifecycle; never shares thread with FNOL |

All four graphs use the same checkpoint store (PostgreSQL). Thread ID prefix convention makes claim history queryable across graphs for audit purposes.

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

**DocumentVerification agent:**

| Graph Section | Path | Latency | Blocks Customer? |
|---|---|---|---|
| `doc_intake_node` | Async | <10ms | No — post-audit |
| `photo_auth_node` + `estimate_check_node` + `narrative_check_node` (parallel, MCP calls) | Async | 1–10s | No |
| `doc_decision_node` | Async | <5ms | No |
| `flag_escalation_node` | Async | <50ms | No |
| `doc_audit_node` | Async | <10ms | No |

**Subrogation agent:**

| Graph Section | Path | Latency | Blocks Customer? |
|---|---|---|---|
| `subrogation_intake_node` | Async | <10ms | No — post-settlement |
| `liability_scoring_node` + `evidence_assembly_node` (parallel, MCP calls) | Async | 500ms–5s | No |
| `recovery_estimation_node` | Async | <5ms | No |
| `recovery_routing_node` | Async | <5ms | No |
| `senior_handler_review_node` | HITL | Hours–days | No (high-value only) |
| `subrogation_referral_node` (MCP call) | Async | <50ms | No |
| `subrogation_close_node` | Async | <10ms | No |

**Implementation pattern:**
The sync and async paths are separate graph compilations — not parallel branches
in one graph. The sync graph completes and returns within the FNOL HTTP request
lifecycle. The async graph is triggered as a background job (SQS / Azure Service
Bus / OCP Job) using the same `claim_id` as `thread_id`, resuming from the
checkpoint written by the sync graph.

DocumentVerification and Subrogation agents are each separate service deployments
with their own thread ID prefix. DocumentVerification is triggered alongside the FNOL
async path at `audit_node` completion. Subrogation is triggered independently by
the claim SETTLED status event — a different event source with a different
operational timeline.

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
| FNOL agent (`fnol-claims-agent`) | EKS pod (LangGraph server) | AKS pod (LangGraph server) | OCP Deployment (LangGraph server) |
| Quote agent (`underwriting-quote-agent`) | EKS pod (LangGraph server) | AKS pod (LangGraph server) | OCP Deployment (LangGraph server) |
| DocVerification agent (`doc-verification-agent`) | EKS pod (LangGraph server) | AKS pod (LangGraph server) | OCP Deployment (LangGraph server) |
| Subrogation agent (`subrogation-agent`) | EKS pod (LangGraph server) | AKS pod (LangGraph server) | OCP Deployment (LangGraph server) |
| MCP servers (`doc-verification-mcp-server`, `claims-data-mcp-server`, `subrogation-mcp-server`) | EKS pods | AKS pods | OCP Deployments |
| Checkpoint store (all agents) | RDS PostgreSQL | Azure Database for PostgreSQL | CrunchyData PGO |
| Async job trigger | SQS + Lambda | Azure Service Bus + Function | OCP Job / Tekton |
| LLM backend | Bedrock | Azure OpenAI | vLLM on GPU nodepool |
| Feature store (sync) | ElastiCache Redis | Azure Cache for Redis | Redis Operator |
| Graph queries | Amazon Neptune | Cosmos DB (Gremlin) | Neo4j Operator |

LangGraph itself has no cloud dependency — the same `StateGraph` code runs
on all three substrates. Only the checkpointer backend and LLM factory
function are substrate-specific. See DEC-014.

See [DEPLOYMENT_TOPOLOGIES.md](./DEPLOYMENT_TOPOLOGIES.md) for full service inventory,
scaling configuration, health check patterns, and the per-substrate RemoteGraph URL map.

---

## 11. MCP Tool Layer

Graph nodes in `DocumentVerificationAgent` and `SubrogationAgent` call external
systems through MCP tools rather than direct Python function calls. This keeps node
logic as pure orchestration.

**Three MCP servers, grouped by data access domain:**

| MCP Server | Tools | Used By |
|---|---|---|
| `doc-verification-mcp-server` | `exif_analyzer`, `perceptual_hash_checker`, `repair_estimate_validator`, `narrative_consistency_checker` | `DocumentVerificationAgent` |
| `claims-data-mcp-server` | `police_report_api`, `telematics_crash_query`, `evidence_collector` | `SubrogationAgent`; shared candidates for `FNOLAgent` |
| `subrogation-mcp-server` | `recovery_demand_queue` | `SubrogationAgent` |

**FNOL sync path does not use MCP.** Nodes on the `intake_node` → `decision_routing_node`
path call Python functions directly — adding a network hop to an already
sub-65ms budget is not acceptable.

**Tool error handling in nodes:** `ToolDependencyError` (upstream unavailable) is caught
at the node boundary and written to state as a `SERVICE_UNAVAILABLE` flag — the exception
never propagates to the LangGraph runtime. `doc_decision_node` treats a `SERVICE_UNAVAILABLE`
result as `INCONCLUSIVE`, not `FLAG`.

See [MCP_Tool_Contracts.md](./MCP_Tool_Contracts.md) for full input/output schemas,
flag definitions, versioning policy, and error contracts for all eight tools.

---

## 12. Agent-to-Agent Integration Patterns

Two distinct patterns govern how agents trigger or communicate with each other.
The choice is determined by whether the caller requires a response.

### Pattern A — Event-Bus Trigger (fire-and-forget)

Used when the caller does not need a response and the triggered agent operates
on a different lifecycle event.

```
Caller graph completes a node
        │
        ▼
Emit event (SQS message / Service Bus message / OCP Job)
        │
        ▼
Target agent service consumes event and starts independently
```

**Current uses:**

| Trigger Point | Event | Target Agent |
|---|---|---|
| `audit_node` completion | `doc_verify_requested` + `{claim_id, doc_refs}` | `DocumentVerificationAgent` |
| Claim status → `SETTLED` | `claim_settled` + `{claim_id, settlement_amount}` | `SubrogationAgent` |

### Pattern B — RemoteGraph Invocation (synchronous, response required)

Used when the calling agent needs the result before it can proceed. Uses LangGraph's
`RemoteGraph` — the target agent runs as a subgraph call, retaining its own checkpoint
store and HITL support.

```python
from langgraph.pregel.remote import RemoteGraph

doc_verify_agent = RemoteGraph(
    "doc-verification-agent",
    url=os.environ["DOC_VERIFY_AGENT_URL"],
)

result = await doc_verify_agent.ainvoke({
    "claim_id": state["claim_id"],
    "doc_refs": state["doc_refs"],
    "claim_context": state["claim_context"],
})
```

**Current candidates (not yet implemented):**

| Calling Agent | Target Agent | Trigger Condition |
|---|---|---|
| `UnderwritingAgent / issuance_node` | `DocumentVerificationAgent` | If document verification is required to gate policy issuance |

### Pattern Selection Rule

| Condition | Pattern |
|---|---|
| Caller does not need result | A — Event-Bus |
| Caller needs result before continuing | B — RemoteGraph |
| Triggered by a lifecycle event on a different timeline (SETTLED, AUDIT_COMPLETE) | A — Event-Bus |
| Triggered within the same request lifecycle | B — RemoteGraph |

**Do not implement a custom A2A protocol.** LangGraph `RemoteGraph` is the standard
inter-agent call mechanism. It preserves checkpoint isolation, HITL support, and
substrate portability without additional infrastructure.

See [DEC-017](./DECISION_LOG.md#dec-017----event-bus-trigger-vs-remotegraph-invocation-decision-boundary) for full rationale.

---

*Architecture version: 2026-Q2 | Applies to: `fnol-claims-multi-agent-system/`, `intelligent-underwriting-platform/`, `doc-verification-agent/`, `subrogation-agent/`*
