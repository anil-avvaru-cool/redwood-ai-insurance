# Platform Architecture Diagrams
### Redwood AI Insurance — AI Platform

The platform architecture is split into two focused diagrams. Each is a standalone SVG — embed directly in any markdown renderer that supports inline SVG, or reference as a separate `.svg` file.

---

## Diagram 1 — Platform data flow

Shows the full top-down data flow: entity resolution → feature store → underwriting and claims platforms in parallel → shared data spine → HITL feedback loop.

![Redwood platform data flow](https://raw.githubusercontent.com/anil-avvaru-cool/redwood-ai-insurance/refs/heads/main/docs/Architecture/redwood_platform_data_flow.svg)

---

## Diagram 2 — Decision tiers and inference paths

Shows what happens after the fraud score is produced: the four risk-tier routing outcomes, and the sync vs async inference split with component-level latency budgets.

![Redwood Decision Tiers](https://raw.githubusercontent.com/anil-avvaru-cool/redwood-ai-insurance/refs/heads/main/docs/Architecture/redwood_decision_tiers.svg)

---

## Reading the diagrams

### Diagram 1 — data flow tiers

| Row | What it represents |
|---|---|
| Entity resolution | Runs once offline before any pipeline. Outputs resolved vehicles, persons, addresses, and policies to `data/entities/`. |
| Feature store | Single source of truth for all model inputs. State regulatory mask applied here — credit score set to null for CA/MA/MI/HI before the vector is assembled. Source datetimes live in raw parquet only (DEC-013). |
| Underwriting column (left) | Risk scoring → quote agent → policy issuance. `risk_score_at_issuance` written to feature store at quote completion. |
| Claims column (right) | FNOL agent → ensemble fraud scoring → decision orchestration. Reads `risk_score_at_issuance` from feature store at FNOL time. |
| Shared data spine (dashed amber) | The architectural link from DEC-010. A high-risk policy filed shortly after issuance is one of the strongest fraud signals in personal auto — without this link it is invisible to the fraud model. |
| Feedback loop (dashed gray, left margin) | Investigator decisions flow into the label store. Confirmed fraud labels (keyed on `fraud_confirmed_at`) trigger retraining eligibility. Override patterns surface model blind spots. |

### Diagram 2 — decision tiers and inference paths

| Section | What it represents |
|---|---|
| Four tier boxes | Score-based routing after the ensemble meta-learner produces its output. Thresholds are calibrated to minimize expected dollar loss, not accuracy. |
| Sync path (<100ms) | Blocks FNOL submission. Redis lookup + device check + 1–2 hop graph query + XGBoost score + routing ≈ 65ms total. Entity resolution is a pre-computation step — it does not run inside this latency budget. |
| Async path (seconds–minutes) | Post-submission enrichment. Deep ViT analysis, full 3+ hop graph traversal, LLM narrative reconciliation, third-party data (MVR, credit, LexisNexis). |
| Async re-score note | If the async path materially raises the score above the sync score, the claim is automatically re-evaluated and can escalate to a higher tier. |

---

---

## Related Architecture Documents

| Document | Covers |
|---|---|
| [Multi_Agent_Architecture.md](./Multi_Agent_Architecture.md) | LangGraph StateGraph topology, state schemas, HITL interrupt pattern, checkpointing, LangChain integration |
| [Fraud_Detection_Architecture.md](./Fraud_Detection_Architecture.md) | Ensemble fraud scoring, sync/async inference, investigator copilot |
| [Risk_Scoring_Architecture.md](./Risk_Scoring_Architecture.md) | Hurdle model, feature store, calibration, drift monitoring |

---

*Diagram version: 2026-Q2*