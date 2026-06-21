# Multi-Agent Architecture
### Redwood AI Insurance — Agentic AI Platform

---

> **Core Framing:**
> Agent orchestration in this platform is not a chatbot loop. It is a
> **stateful claim and underwriting lifecycle manager** — where each graph
> node is a bounded unit of work, edges encode business routing logic, and
> HITL interrupts are first-class citizens, not afterthoughts.

---

> **Architecture Pattern: Choreographed Specialist Network**
> Four independent LangGraph services (FNOL, Underwriting, DocumentVerification, Subrogation),
> each owning a bounded claim lifecycle domain. There is no central coordinator and no LLM
> planner — routing is deterministic Python (score thresholds, config-driven). Agents are
> triggered by domain lifecycle events (SQS / Azure Service Bus / OCP Job) rather than by
> a central orchestrator telling them what to do. LLM capability is injected only at specific
> leaf nodes (`narrative_llm_node`, `quote_gen_node`, `investigator_copilot_node`); it never
> controls graph topology. This is **choreography over orchestration**: each agent reacts to
> an event on its own lifecycle boundary, with inter-agent calls handled by Pattern A
> (event-bus fire-and-forget) or Pattern B (RemoteGraph synchronous subgraph call). See §12.

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
13. [TigerGraph — Graph Scaling Path](#13-tigergraph--graph-scaling-path)
14. [Evaluation and Observability](#14-evaluation-and-observability)

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

---

## 13. TigerGraph — Graph Scaling Path

The current architecture commits to Neo4j (local/OCP), Amazon Neptune (AWS), and
Cosmos DB Gremlin (Azure) for all graph queries. This section documents the
architectural threshold at which TigerGraph becomes the correct substrate, and
exactly what changes when that threshold is crossed.

### The Specific Problem: Insurance Fraud Ring Structure

Insurance fraud graphs are not social graphs. They are heterogeneous multi-entity
graphs where a single fraud ring spans claimants, vehicles, accidents, repair shops,
medical providers, billing entities, and attorneys — each connected by different edge
types. A full ring detection traversal looks like this:

```
Claimant A ──── involved_in ───→ Accident X ←─── involved_in ──── Claimant B
     │                                                                    │
     ├── treated_by ──→ Medical Provider M ←── treated_by ───────────────┘
     │                         │
     │               bills_through ──→ Billing Entity B
     │                                       │
     │                         shares_TIN_with ──→ Provider N  ← [known fraud flag]
     │
     ├── represented_by ──→ Attorney L ←── represented_by ──── Claimant B
     │
     └── vehicle_repaired_by ──→ Repair Shop R
                                       │
                              estimated_by ──→ Adjuster J ← [flagged ring member]
```

This is a **6–7 hop traversal** across heterogeneous edge types. In the current
architecture, the sync path (`graph_query_node`) handles 1–2 hop neighborhood
lookups — fast enough for sub-65ms scoring. The `deep_graph_node` in the async
path currently targets 3+ hop ring expansion.

**The scaling pressure point:** as the cumulative entity graph grows to hundreds of
millions of nodes (policies, persons, vehicles, providers, shops, attorneys, billing
entities) under 100K+ daily claims, Neo4j's query planner — which optimizes for OLTP-style
lookups — hits traversal limits that the async enrichment SLA cannot absorb. A 6+ hop
pattern match against a 500M-node graph under concurrent load produces full-graph scans,
not index-bounded traversals.

### Why TigerGraph Fits This Workload

TigerGraph uses a **Native Parallel Graph (NPG) engine**: queries are compiled to
parallel execution plans distributed across the graph's partition map. The execution
model is fundamentally different from Neo4j's single-planner approach.

| Dimension | Neo4j (current) | TigerGraph (candidate) |
|---|---|---|
| Query language | Cypher | GSQL (procedural, compiled to parallel plan) |
| Traversal model | Iterative, single-planner | Distributed parallel execution across partitions |
| Optimal hop depth | 1–3 hops | 5–10+ hops across billions of edges |
| Graph size sweet spot | Tens of millions of nodes | Hundreds of millions to billions |
| Insurance fraud OOTB | Manual Cypher ring queries | Fraud detection starter kit (ring detection, money flow, shared identity) |
| Streaming ingest | Kafka connector | Native Kafka connector + CDC; supports concurrent ingest + query |
| Cloud-native maturity | Mature (OCP operator, Neptune, Cosmos) | Growing (TigerGraph Cloud on AWS/Azure; Helm chart, no dedicated OCP operator) |

TigerGraph's **fraud detection starter kit** is particularly relevant: pre-built GSQL
queries for shared identity detection (same address/phone/SSN across policies), money
flow ring tracing, and provider billing cluster analysis. These patterns map directly
to the staged accident, owner give-up, and medical billing fraud categories that
drive SIU referrals in this platform.

### What Changes — and What Doesn't

Only the graph substrate in the **Business Logic / ML layer** changes. The
LangGraph orchestration, node signatures, and state schema are unaffected.

```
┌─────────────────────────────────────────────────────────────────┐
│  BUSINESS LOGIC / ML LAYER — Pure Python + Pydantic             │
│  graph queries ← CHANGE: pyTigerGraph replaces neo4j-driver     │
│                           GSQL replaces Cypher                  │
│  XGBoost scoring · feature store · entity resolution ← unchanged│
│  regulatory mask · SHAP explainability              ← unchanged  │
│  NO LangChain imports in this layer                             │
└─────────────────────────────────────────────────────────────────┘
```

Concretely:
- `graph_query_node` and `deep_graph_node` swap `neo4j` driver for `pyTigerGraph` client
- GSQL ring detection queries replace Cypher traversals
- `FNOLState.graph_context: dict[str, Any]` is substrate-agnostic — the field and its
  downstream consumers (tabular scoring, SHAP, audit) do not change
- The layer boundary rule holds: the graph client lives in the Business Logic layer.
  No LangGraph node imports the graph driver directly

**Deployment mapping change:**

| Environment | Graph Substrate (current) | Graph Substrate (TigerGraph path) |
|---|---|---|
| AWS | Amazon Neptune (Gremlin) | TigerGraph Cloud (AWS-hosted) or self-managed EC2 cluster |
| Azure | Cosmos DB (Gremlin) | TigerGraph Cloud (Azure-hosted) |
| OpenShift | Neo4j Operator | TigerGraph Helm chart (no dedicated OCP operator — operational gap) |

### Migration Trigger Criteria

Do not migrate speculatively. Migrate when **all three** conditions are met:

1. **Entity graph scale:** cumulative graph exceeds ~200M nodes with active concurrent
   ring detection queries
2. **SLA breach:** `deep_graph_node` async p95 latency regularly exceeds the enrichment
   budget; `neo4j EXPLAIN` shows full-graph scans on ring detection patterns
3. **Ring complexity:** production SIU referrals require 6+ hop traversals to close
   false-negative gaps that 3-hop Neo4j queries miss

Until then, Neo4j covers the current workload and avoids the operational cost of
running a TigerGraph cluster alongside the existing substrates.

### What Stays the Same

- LangGraph `StateGraph` topology — no change
- HITL interrupt pattern — no change
- All MCP tool contracts — no change
- `FNOLState` schema — no change
- Sync path is not affected: 1–2 hop sync queries continue to run against the
  existing graph substrate; TigerGraph migration targets the async `deep_graph_node` first

---

---

## 14. Evaluation and Observability

> **Scope of this section:** The tabular ML layer (XGBoost, feature store, SHAP) already
> has a signal path: override rates flow back to `feedback_node`, and the label store
> triggers retraining eligibility checks. This section covers the **LLM node layer only**
> — `narrative_llm_node`, `quote_gen_node`, `investigator_copilot_node` — which produce
> unstructured outputs with no existing quality signal.

**Critical constraint:** No PII in any eval log, trace, or dataset. All claim identifiers
logged to external services must use `claim_id` / `quote_id` only. Raw narratives,
claimant names, SSNs, and policy numbers must be masked before export. See masking
rules in [DEC-020](./DECISION_LOG.md#dec-020) (to be added).

---

### 14a. Scope — What Gets Evaluated

| Node | Output Type | Existing Signal | Eval Gap |
|---|---|---|---|
| `narrative_llm_node` | Narrative consistency verdict + explanation | None | Is the LLM flagging real inconsistencies or hallucinating? |
| `quote_gen_node` | Customer-facing quote explanation | None | Does explanation accurately reflect risk score and policy terms? |
| `investigator_copilot_node` | Case summary + SHAP rendering + evidence checklist | Override rate (indirect) | Is the summary faithful to SHAP values, or does it introduce hallucinated risk reasoning? |
| `tabular_scoring_node` | XGBoost fraud score | Label store + SHAP | Covered — not in scope here |
| `risk_scoring_node` | XGBoost risk score | Override rate + SHAP | Covered — not in scope here |
| MCP tool nodes | Structured data results | `INCONCLUSIVE` error flag | Covered by MCP error contract — not in scope here |

**Three-tier evaluation stack:**

```
┌──────────────────────────────────────────────────────────────┐
│  PRODUCTION MONITORING — Arize Phoenix (self-hosted)         │
│  Embedding drift · output distribution · latency p95         │
│  Hallucination rate trend · token cost per node              │
├──────────────────────────────────────────────────────────────┤
│  OFFLINE EVALUATION — Braintrust                             │
│  Golden-set regression · prompt change gating                │
│  Ground truth from feedback_node label store                 │
├──────────────────────────────────────────────────────────────┤
│  TRACING — LangSmith                                         │
│  Per-run prompt + response + token count + latency           │
│  Node execution path · LCEL chain intermediate steps         │
└──────────────────────────────────────────────────────────────┘
```

Each tier answers a different question:
- **LangSmith:** What did the LLM do on this specific run?
- **Braintrust:** Does the current prompt/model pass the regression suite before deploy?
- **Arize Phoenix:** Is LLM behavior shifting in production over time?

---

### 14b. Tracing — LangSmith

LangGraph emits traces to LangSmith via the `LANGCHAIN_TRACING_V2` environment variable.
No code changes to graph nodes are required — all LLM calls, LCEL chain steps, and
node execution are captured automatically.

**Environment variables (added to `.env` / deployment config):**

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...                     # from LangSmith project settings
LANGCHAIN_PROJECT=fnol-claims-agent       # one project per agent service
```

**Per-agent LangSmith project mapping:**

| Agent Service | LangSmith Project |
|---|---|
| `fnol-claims-agent` | `fnol-claims-agent` |
| `underwriting-quote-agent` | `underwriting-quote-agent` |
| `doc-verification-agent` | `doc-verification-agent` |
| `subrogation-agent` | `subrogation-agent` |

**What LangSmith captures automatically:**

| Signal | Source | LangSmith Field |
|---|---|---|
| Full prompt sent to LLM | `ChatPromptTemplate` render | `inputs.messages` |
| Raw LLM response | `BaseChatModel` output | `outputs.generations` |
| Parsed output | `PydanticOutputParser` | `outputs.output` |
| Token count (prompt + completion) | LLM response metadata | `token_usage` |
| Latency per node | Node start/end timestamps | `latency_ms` |
| LCEL chain intermediate steps | `quote_gen_node` retriever → prompt → llm → parser | `child_runs` |
| Node execution order | LangGraph run metadata | `tags` / `metadata` |

**Tagging runs with claim context (no PII):**

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    tags=["fnol", "narrative_llm"],
    metadata={
        "claim_id": state["claim_id"],         # opaque ID — not claimant PII
        "risk_tier": state["risk_tier"],
        "sync_fraud_score": state["sync_fraud_score"],
    }
)
result = chain.invoke(inputs, config=config)
```

`claim_id` is the only identifier logged. Never log `claimant_id`, names, addresses,
or narrative text verbatim to LangSmith — pass the structured metadata only.

**Design decisions:**
- LangSmith is the first tracing integration because it requires zero instrumentation
  changes to graph nodes — the LangChain callback system handles it automatically.
- One LangSmith project per agent service. This keeps FNOL traces separate from
  underwriting traces, matching the checkpoint store namespace convention.
- LangSmith is used for debugging and run-level inspection. It is not the production
  monitoring layer — Arize Phoenix handles drift and distribution tracking (§14d).

---

### 14c. Offline Evaluation — Braintrust

Braintrust manages eval datasets and scoring runs. It is the gating layer for prompt
changes before deployment: a failing Braintrust suite blocks the deploy, the same way
a failing unit test suite does.

**Three eval datasets, one per LLM node:**

#### Dataset 1 — `narrative_consistency_evals`

Ground truth source: `feedback_node` label store. When an investigator confirms a
fraud ring (`investigator_decision = CONFIRM`), the associated `narrative_llm_node`
output that flagged the inconsistency is a true positive. When an investigator clears
(`investigator_decision = CLEAR`), a prior FLAG from the narrative node is a false positive.

```python
import braintrust
from braintrust import Eval

def narrative_verdict_match(output, expected) -> float:
    # 1.0 if LLM verdict matches investigator ground truth, 0.0 otherwise
    llm_verdict = output["narrative_verdict"]          # CONSISTENT / INCONSISTENT
    investigator_outcome = expected["investigator_decision"]  # CONFIRM / CLEAR
    if investigator_outcome == "CONFIRM" and llm_verdict == "INCONSISTENT":
        return 1.0
    if investigator_outcome == "CLEAR" and llm_verdict == "CONSISTENT":
        return 1.0
    return 0.0

Eval(
    "narrative-consistency-eval",
    data=lambda: load_labeled_narrative_cases(),    # from label store, PII-masked
    task=lambda input: run_narrative_llm_node(input),
    scores=[narrative_verdict_match],
)
```

#### Dataset 2 — `quote_explanation_evals`

Scoring function: RAG faithfulness — does the explanation cite terms that exist in
the retrieved policy documents? Uses an LLM-as-judge scorer with a rubric prompt.

```python
from braintrust.scorers import LLMClassifier

quote_faithfulness = LLMClassifier(
    name="QuoteExplanationFaithfulness",
    prompt_template="""
You are evaluating whether a quote explanation is faithful to the policy terms provided.

Policy terms retrieved: {{retrieved_docs}}
Quote explanation generated: {{output}}
Risk score: {{risk_score}}
Premium (USD): {{premium_usd}}

Does the explanation accurately reflect the retrieved policy terms without introducing
facts not present in the retrieval context? Answer YES or NO with a one-sentence reason.
""",
    choice_scores={"YES": 1.0, "NO": 0.0},
)
```

#### Dataset 3 — `copilot_summary_evals`

Scoring function: SHAP faithfulness — does the investigator copilot summary correctly
identify the top-3 SHAP features by rank? Rule-based, no LLM judge needed.

```python
def shap_rank_faithfulness(output, expected) -> float:
    # Check top-3 SHAP features appear in copilot summary (order-insensitive)
    top_shap_features = set(expected["shap_top3_features"])
    summary_text = output["case_summary"].lower()
    mentioned = sum(1 for f in top_shap_features if f.lower() in summary_text)
    return mentioned / len(top_shap_features)
```

**CI integration — block deploy on regression:**

```yaml
# .github/workflows/eval.yml (or equivalent CI config)
- name: Run Braintrust evals
  run: |
    braintrust eval src/agents/evals/narrative_eval.py \
                     src/agents/evals/quote_eval.py \
                     src/agents/evals/copilot_eval.py \
      --threshold 0.85        # fail CI if any suite scores below 0.85
  env:
    BRAINTRUST_API_KEY: ${{ secrets.BRAINTRUST_API_KEY }}
```

**Eval suite → retraining trigger connection:**

```
feedback_node label store
        │
        ▼ (nightly export job, PII-masked)
Braintrust dataset — narrative_consistency_evals
        │
        ▼ (eval run on prompt/model change)
narrative_verdict_match score
        │
  score drops below 0.85?
        │
   YES ─┴─ NO
    │          │
    ▼          ▼
flag for    no action
prompt
review
```

**Design decisions:**
- Braintrust eval datasets are built from the existing `feedback_node` label store —
  no additional human labeling pipeline is required to bootstrap. Investigator and
  underwriter decisions are the ground truth.
- PII masking runs as a transformation step in the nightly export job before any
  case data reaches Braintrust. The export job is the single PII boundary — claim
  narratives are replaced with tokenized placeholders keyed to `claim_id`.
- LLM-as-judge is used only for `quote_gen_node` faithfulness, where a rubric is
  required. The other two nodes use rule-based scorers to avoid eval-on-eval
  cost and latency.

---

### 14d. Production Monitoring — Arize Phoenix

Arize Phoenix (open-source, self-hosted) instruments the LLM nodes via OpenTelemetry.
It tracks output distribution, embedding drift, and hallucination rate trend in production.
Self-hosted Phoenix is required — claim narratives cannot leave the deployment boundary
even masked, as re-identification risk under CCPA/HIPAA adjacent insurance regulations
is non-trivial.

**Instrumentation (one call at service startup):**

```python
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Phoenix collector — runs as a sidecar container in the same pod
exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
LangChainInstrumentor().instrument(
    tracer_provider=...,         # configured with exporter above
    skip_dep_check=True,
)
```

Phoenix is deployed as a **sidecar container** in each agent pod — not a shared
service. This keeps telemetry within the pod boundary until aggregation is required.

**Deployment mapping:**

| Agent | Phoenix Sidecar | Aggregation |
|---|---|---|
| `fnol-claims-agent` | Phoenix container in same pod | Phoenix UI (internal ingress) |
| `underwriting-quote-agent` | Phoenix container in same pod | Phoenix UI (internal ingress) |
| `doc-verification-agent` | Phoenix container in same pod | Phoenix UI (internal ingress) |
| `subrogation-agent` | Phoenix container in same pod | Phoenix UI (internal ingress) |

**Metrics tracked per LLM node:**

| Metric | Node | Alert Threshold |
|---|---|---|
| `narrative_llm.flag_rate` | `narrative_llm_node` | >3σ shift over 7-day baseline |
| `narrative_llm.latency_p95_ms` | `narrative_llm_node` | >5000ms (budget: 2–10s) |
| `quote_gen.hallucination_score` | `quote_gen_node` | Mean > 0.15 (LLM-judge estimate) |
| `quote_gen.latency_p95_ms` | `quote_gen_node` | >10000ms (budget: 2–8s) |
| `copilot.token_cost_usd` | `investigator_copilot_node` | >$0.05 per invocation (cost guard) |
| `copilot.shap_feature_recall` | `investigator_copilot_node` | <0.80 (from Braintrust metric exported) |

**Embedding drift detection for `narrative_llm_node`:**

```python
# Emit claim narrative embedding alongside each LLM trace
# Embedding is computed from the anonymized narrative — not the raw text
# Phoenix tracks centroid drift over rolling 7-day window

px.log_embedding(
    embedding=narrative_embedding,      # computed from masked narrative
    raw_data=None,                      # raw text never logged
    link_to_data=state["claim_id"],
)
```

If the incoming narrative embedding distribution shifts significantly (e.g., a new
fraud pattern emerges with distinct linguistic structure), Phoenix raises a drift alert
before the Braintrust eval suite would catch it — giving lead time to build new
eval cases.

**Design decisions:**
- Phoenix over Arize AI cloud: Insurance data with PII-adjacent content cannot
  leave the deployment boundary even masked. Self-hosted Phoenix eliminates the
  data-egress compliance question entirely.
- Sidecar deployment over a shared Phoenix service: Avoids cross-agent trace
  commingling and makes pod-level decommission straightforward.
- Embedding drift monitoring on `narrative_llm_node` specifically because claim
  narratives are the highest-variance free-text input in the system. Quote
  explanations and copilot summaries are generated from structured inputs and
  are less susceptible to distribution shift.

---

### 14e. Feedback Signal → Eval Pipeline

The full pipeline connecting HITL decisions back to the eval and monitoring layers:

```
FNOL graph — feedback_node
        │
        │  investigator_decision written to label store
        │  (CONFIRM / CLEAR / ESCALATE + claim_id)
        ▼
┌───────────────────────────────────┐
│  label store (PostgreSQL)         │
│  claim_id, investigator_decision  │
│  narrative_llm_output (masked)    │
│  shap_values, sync_fraud_score    │
└────────────┬──────────────────────┘
             │
    nightly export job (PII masking runs here)
             │
      ┌──────┴──────────────────────────┐
      ▼                                 ▼
Braintrust dataset               Arize Phoenix
(offline eval — regression        (online monitor —
 gating on deploy)                 drift + distribution)
      │                                 │
      ▼                                 ▼
CI eval gate                     Alert → Slack / PagerDuty
(blocks deploy if                (triggers prompt review
 score < 0.85)                    or model health check)
```

**Retraining eligibility check — extended:**

The `feedback_node` currently triggers a retraining eligibility check for the
XGBoost tabular model. Extend this to include LLM eval signal:

```python
def feedback_node(state: FNOLState) -> FNOLState:
    # Existing: write decision to label store
    label_store.write(
        claim_id=state["claim_id"],
        investigator_decision=state["investigator_decision"],
        shap_values=state["shap_values"],
        sync_fraud_score=state["sync_fraud_score"],
    )

    # Existing: trigger tabular model retraining eligibility check
    check_retraining_eligibility(state["claim_id"])

    # Extended: tag narrative LLM output with ground truth for Braintrust export
    label_store.write_llm_label(
        claim_id=state["claim_id"],
        node="narrative_llm_node",
        llm_output=state.get("narrative_analysis"),        # stored in state by narrative node
        ground_truth=state["investigator_decision"],
    )

    return state
```

The `write_llm_label` call writes to the same PostgreSQL label store — the nightly
export job picks it up, applies PII masking, and pushes to Braintrust.

**Design decisions:**
- The nightly export job is the single PII boundary. Masking is not the responsibility
  of `feedback_node` or any graph node — it runs as a separate offline process on
  the label store records. This prevents masking logic from leaking into the graph.
- `narrative_analysis` is stored in `FNOLState` by `narrative_llm_node` today via
  the `messages` field. If the node is updated to write a structured field
  (e.g., `narrative_analysis: NarrativeAnalysisResult`), the `feedback_node` call
  above reads from that field directly. The label write is a no-op if the field is absent.
- Do not implement eval logging inside graph nodes. Nodes are pure orchestration units —
  telemetry is the responsibility of the instrumentation layer (LangSmith callbacks,
  Phoenix OpenTelemetry) and the export job, not the node implementation.

---

### 14f. Framework Selection Summary

| Layer | Primary | Alternative | Why Primary |
|---|---|---|---|
| Tracing | LangSmith | LangFuse (open-source) | Native LangGraph integration; zero code changes |
| Offline eval | Braintrust | W&B Weave | Dataset versioning + CI gate + LLM-judge scoring in one tool |
| Production monitoring | Arize Phoenix (self-hosted) | Arize AI cloud (if PII controls confirmed) | Self-hosted required for insurance data boundary compliance |
| LLM-as-judge scorer | Claude (via Bedrock / Azure OpenAI) | GPT-4o | Reuses the same LLM factory from §8; no additional provider dependency |

**What is intentionally excluded:**

| Tool | Exclusion Reason |
|---|---|
| LangSmith as production monitor | LangSmith is a run-level debugger, not a time-series drift monitor. Use Phoenix for distribution tracking. |
| Helicone / PromptLayer | Proxy-based interceptors break LangGraph's async streaming guarantees. |
| Weights & Biases (W&B) ML monitoring | W&B is the right tool for XGBoost model tracking (already in scope for the tabular layer). Do not conflate ML model monitoring with LLM eval — they are separate pipelines with separate toolchains. |

---

*Architecture version: 2026-Q2 | Applies to: `fnol-claims-multi-agent-system/`, `intelligent-underwriting-platform/`, `doc-verification-agent/`, `subrogation-agent/`*
