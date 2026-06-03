# Redwood AI Insurance Platform

> End-to-end intelligent insurance operations platform spanning the full lifecycle —
> from quote generation through claims settlement — through risk scoring,
> real-time fraud detection, and autonomous claims automation.

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

## Architecture

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

---

## Platform Repositories

| Repo | Description | Focus Area |
|---|---|---|
| [insurance-data-platform](https://github.com/anil-avvaru-cool/insurance-data-platform) | Entity resolution, feature store, synthetic data generation | Production data architecture, Decision Log |
| [ai-fraud-detection-platform](https://github.com/anil-avvaru-cool/ai-fraud-detection-platform) | Ensemble fraud scoring, graph intelligence, sync/async inference | ML systems design, adversarial framing |
| [intelligent-underwriting-platform](https://github.com/anil-avvaru-cool/intelligent-underwriting-platform) | Hurdle model risk scoring, quote generation agent | Actuarial ML, risk modeling |
| [fnol-claims-multi-agent-system](https://github.com/anil-avvaru-cool/fnol-claims-multi-agent-system) | FNOL Agent, Document Verification, Subrogation, Investigator Copilot | Agentic AI, HITL workflows |
| [enterprise-rag-platform](https://github.com/anil-avvaru-cool/enterprise-rag-platform) | Reusable RAG library consumed by both underwriting and claims agents | RAG architecture, document AI |

---

## Key Design Decisions

This platform was built with explicit, documented reasoning at every architectural
junction. Selected decisions that shaped the design:

**DEC-004 — feature_definitions.py as Layer 0**
Training-serving skew is the most common silent failure mode in production ML.
Making feature definitions the dependency root — not an output of data generation —
forces both pipelines to share one implementation from day one.

**DEC-010 — risk_score_at_issuance as fraud feature**
The underwriting risk score is stored in the feature store and re-used as a fraud
scoring input, not discarded after policy issuance. This is the architectural
"shared data spine" — risk score is not a claims pipeline stage, it re-enters
claims as a feature.

**DEC-011 — Entity resolution as independent pre-graph layer**
Vehicle, Person, Address, and Phone are resolved before the graph is built.
A shares_address edge between two unnormalized strings is unreliable and degrades
Louvain community detection silently.

Full decision log with all 13 entries: [insurance-data-platform/DECISION_LOG.md](https://github.com/anil-avvaru-cool/insurance-data-platform/blob/main/DECISION_LOG.md)

---

## Technology Stack

| Layer | Technology |
|---|---|
| Tabular scoring | XGBoost, LightGBM |
| Graph intelligence | Neo4j / Amazon Neptune |
| Online feature serving | Redis / ElastiCache |
| Offline feature pipeline | Spark / AWS Glue |
| Graph neural network | GraphSAGE |
| NLP narrative analysis | Transformer (fine-tuned) |
| Vision fraud detection | ViT / CLIP |
| Explainability | SHAP |
| API layer | FastAPI + AWS Lambda |
| Agents | LLM-powered with RAG |
| Drift monitoring | PSI / CSI |

---

## Platform Status

In active development — 2026 Q2.

| Component | Status |
|---|---|
| Entity resolution layer | 🔨 In progress |
| Synthetic data generator | 🔨 In progress |
| Feature store | 🔨 In progress |
| Risk scoring (hurdle model) | 📋 Planned — Week 3 |
| Fraud scoring (ensemble) | 📋 Planned — Week 4 |
| FNOL Agent | 📋 Planned — Week 5 |
| Claims automation | 📋 Planned — Week 6 |
| AWS deployment | 📋 Planned — Week 7 |

---

## Related Content

Article published
https://anilavvaru.substack.com/p/your-fraud-labels-are-lying-to-you

Articles in progress — publishing on Towards Data Science and LinkedIn:

- "How we built a under 100ms fraud scorer on AWS" — concrete, deployable
- "What C-suite gets wrong about claims AI" — executive reach

---

*Platform version: 2026-Q2 | Stack: Python, XGBoost, Neo4j, Redis, FastAPI, AWS*
