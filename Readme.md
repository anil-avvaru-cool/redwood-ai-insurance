# Redwood AI Insurance Platform

> End-to-end intelligent insurance operations platform spanning the full lifecycle —
> from quote generation through claims settlement — through risk scoring,
> real-time fraud detection, and autonomous claims automation.

**[Local setup →](docs/Local_setup.md)** | **Prerequisites:** Python 3.11+, Docker, UV

---

## Platform Overview

Modern insurance AI is not a collection of isolated models. It is a unified system
where underwriting context flows directly into fraud detection, entity resolution
underpins every scoring decision, and human investigators are augmented rather than
replaced.

Redwood AI Insurance is built on three architectural principles:

**One shared data spine.** The risk score assigned at underwriting re-enters the
claims pipeline as a fraud feature. A high-risk policy filed shortly after issuance
is one of the strongest fraud signals in personal auto. Without a shared spine,
that signal is invisible.

**Entity resolution as foundation.** Vehicle, Person, Address, and Phone are
resolved once as independent entities before any feature computation or graph
construction begins. Both platforms consume resolved entities — neither owns them.

**Adversarial by design.** Fraud detection is not a supervised classification
problem. It is an adversarial risk orchestration system combining anomaly detection,
graph intelligence, behavioral analytics, and human investigation workflows.

---

## Platform Architecture

```
UNDERWRITING PLATFORM                    CLAIMS PLATFORM
─────────────────────                    ───────────────
Customer requests quote                  Accident happens
          ↓                                     ↓
    Risk Scoring                           FNOL Agent
  (P(claim), E[cost])                   (intake, triage)
          ↓                                     ↓
 Quote Generation Agent                  Fraud Scoring
          ↓                                     ↓
   Policy Issuance                      Claims Automation
                                               ↓
                         ┌─────────────────────┤
                         ↓                     ↓
                    STP (auto)         Investigator Copilot
                                          (HITL)

          └──────────── shared feature store ────────────┘
                  risk_score_at_issuance flows into
                  fraud scoring as a feature input

          └────────── shared entity resolution ──────────┘
                  Vehicle, Person, Address, Policy
                  resolved once, consumed by both platforms
```

### Data Flow

```
Customer / Claimant
        │
        ▼
┌───────────────────┐
│  Entity Resolution│  ← VIN decode, address normalize, person dedup
│   (Layer 0)       │    Runs once offline. Both platforms consume.
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Feature Store   │  ← Versioned. Immutable snapshots. Sub-10ms online serving.
│  (Online + Offline│    State regulatory mask applied here (not at model level).
└────────┬──────────┘
         │
    ┌────┴─────┐
    ▼          ▼
Underwriting  Claims
 Platform    Platform
    │          │
    │   risk_score_at_issuance
    └────────► │  (shared data spine: underwriting → fraud detection)
               ▼
         Fraud Score
         [0.0 – 1.0]
               │
         ┌─────┴──────┐
         ▼            ▼
     STP (auto)   SIU Referral /
                  Investigator
                    Copilot
```

---

## Repositories

| Repo | Description | Focus Area |
|---|---|---|
| [insurance-data-platform](https://github.com/anil-avvaru-cool/insurance-data-platform) | Entity resolution, feature store, synthetic data generation | Production data architecture, Decision Log |
| [ai-fraud-detection-platform](https://github.com/anil-avvaru-cool/ai-fraud-detection-platform) | Ensemble fraud scoring, graph intelligence, sync/async inference | ML systems design, adversarial framing |
| [intelligent-underwriting-platform](https://github.com/anil-avvaru-cool/intelligent-underwriting-platform) | Hurdle model risk scoring, quote generation agent | Actuarial ML, risk modeling |
| [fnol-claims-multi-agent-system](https://github.com/anil-avvaru-cool/fnol-claims-multi-agent-system) | FNOL Agent, Document Verification, Subrogation, Investigator Copilot | Agentic AI, HITL workflows |
| [enterprise-rag-platform](https://github.com/anil-avvaru-cool/enterprise-rag-platform) | Reusable RAG library consumed by both underwriting and claims agents | RAG architecture, document AI |

---

## Technology Stack

Deployment topology is segment-driven. The same platform components run on three infrastructure substrates — AWS, Azure, and OpenShift. See [docs/DEPLOYMENT_TOPOLOGIES.md](./docs/DEPLOYMENT_TOPOLOGIES.md) for the full segment decision matrix and per-component portability map.

| Layer | AWS | Azure | OpenShift | Portable |
|---|---|---|---|---|
| Tabular scoring | XGBoost, LightGBM | XGBoost, LightGBM | XGBoost, LightGBM | Yes — pure Python |
| Graph intelligence | Amazon Neptune | Cosmos DB (Gremlin) | Neo4j Operator on OCP | Cypher — Neptune and Neo4j both support |
| Online feature serving | ElastiCache (Redis) | Azure Cache for Redis | Redis Operator on OCP | Redis — same API across all |
| Offline feature pipeline | S3 + AWS Glue | ADLS + Synapse | ODF + Spark on RHOAI | Parquet + Spark |
| Model serving | KServe on EKS | KServe on AKS | KServe on RHOAI | KServe InferenceService spec |
| LLM inference | Bedrock Titan | Azure OpenAI | vLLM on GPU nodepool | OpenAI-compatible API |
| Graph neural network | GraphSAGE | GraphSAGE | GraphSAGE | Yes |
| NLP narrative analysis | Transformer (fine-tuned) | Transformer (fine-tuned) | Transformer (fine-tuned) | Yes |
| Vision fraud detection | ViT / CLIP | ViT / CLIP | ViT / CLIP | Yes |
| Explainability | SHAP | SHAP | SHAP | Yes |
| API layer | FastAPI + AWS Lambda | FastAPI + Azure Functions | FastAPI on OCP | FastAPI |
| Agent orchestration | LangGraph on EKS | LangGraph on AKS | LangGraph on OCP | LangGraph — no cloud dependency |
| Agent primitives | LangChain (Bedrock) | LangChain (AzureOpenAI) | LangChain (vLLM) | LangChain `BaseChatModel` — substrate-agnostic |
| Drift monitoring | CloudWatch + PSI | Azure Monitor + PSI | Prometheus + Grafana | PSI — pure Python |
| Audit trail | S3 + Athena | ADLS + Synapse | ODF + Trino | Parquet schema identical |

---

## Key Design Decisions

This platform was built with explicit, documented reasoning at every architectural
junction. The full Decision Log is in [`docs/DECISION_LOG.md`](./docs/DECISION_LOG.md).
Selected decisions that shaped the design:

**[DEC-001](./docs/DECISION_LOG.md#dec-001----telematics-nulls-null-not-default) — Telematics nulls: null, not default**
XGBoost's native missing-value handling learns the actual risk distribution for
non-telematics users. Non-telematics users skew slightly higher risk on average
due to adverse selection — imputing a neutral value masks this signal for 60% of
the user base.

**[DEC-004](./docs/DECISION_LOG.md#dec-004----feature_definitionspy-as-layer-0) — `feature_definitions.py` as Layer 0**
Training-serving skew is the most common silent failure mode in production ML.
Making feature definitions the dependency root forces both pipelines to share one
implementation from day one. Archetypes import feature names from
`feature_definitions.py` — never the reverse.

**[DEC-010](./docs/DECISION_LOG.md#dec-010----risk_score_at_issuance-as-fraud-feature-shared-spine) — `risk_score_at_issuance` as fraud feature**
The underwriting risk score is stored in the feature store and re-used as a fraud
scoring input at FNOL time. This is the "shared data spine" — a high-risk policy
filed shortly after issuance is one of the strongest fraud signals in personal auto.

**[DEC-011](./docs/DECISION_LOG.md#dec-011----entity-resolution-as-independent-pre-graph-layer) — Entity resolution as independent pre-graph layer**
Vehicle, Person, Address, and Phone are resolved before the graph is built. A
`shares_address` edge between two unnormalized strings degrades Louvain community
detection silently — fraud rings appear disconnected.

**[DEC-013](./docs/DECISION_LOG.md#dec-013----source-datetimes-in-raw-parquet-derived-ints-unchanged-in-feature-store-option-a) — Source datetimes in raw parquet; derived ints unchanged in feature store**
Source datetime columns (`fnol_submitted_at`, `quote_requested_at`) live in raw
parquet only — consumed by `psi_drift.py` for drift monitoring. Derived int
features (`reporting_delay_days`, `policy_inception_days`) remain unchanged in
the feature vector and trained artifacts. Non-breaking. No model retraining required.

**[DEC-014](./docs/DECISION_LOG.md#dec-014----deployment-topology-as-a-first-class-decision) — Deployment topology as a first-class decision**
LangGraph is the default agent orchestration layer across all three substrate bindings (AWS / Azure / OpenShift). No cloud dependency — the same `StateGraph` runs everywhere.

**[DEC-015](./docs/DECISION_LOG.md#dec-015----langchain-as-primitives-layer-alongside-langgraph) — LangChain as primitives layer alongside LangGraph**
LangChain provides model wrappers, prompt templates, output parsers, and RAG retrieval inside agent nodes. Hard module boundary: ML scoring and feature store modules have no LangChain dependency. See [Multi_Agent_Architecture.md](./docs/Multi_Agent_Architecture.md) for full graph topology, state schemas, and HITL patterns.

---

## Platform Status

In active development — 2026 Q2.

| Component | Status |
|---|---|
| Entity resolution layer | ✅ Complete |
| Synthetic data generator | ✅ Complete |
| Feature store | ✅ Complete |
| Risk scoring (hurdle model) | ✅ Complete |
| Fraud scoring (ensemble) | ✅ Complete |
| FNOL Agent | ✅ Complete |
| Claims automation | ✅ Complete |
| RAG library | ✅ Complete |
| Cloud deployment (AWS / Azure / OpenShift) | 📋 Planned — 2026 Q3 |

---

## Related Writing

- [Your Fraud Labels Are Lying to You](https://anilavvaru.substack.com/p/your-fraud-labels-are-lying-to-you) — published
- "How we built a sub-100ms fraud scorer on AWS" — in progress
- "What C-suite gets wrong about claims AI" — in progress

---

*Platform version: 2026-Q2 | Stack: Python, XGBoost, Neo4j, Redis, FastAPI, AWS*