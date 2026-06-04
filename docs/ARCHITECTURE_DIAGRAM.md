# Platform Architecture Diagram

---

> **Full interactive SVG available in the platform repository.**
> See `docs/architecture/redwood_platform_data_flow.svg` `docs/architecture/redwood_decision_tiers.svg`.

---

## Reading the Diagram

The architecture has four horizontal tiers:

1. **Entity Resolution (Stage 0)** — runs once offline before any model or pipeline.
   Outputs resolved vehicles, persons, addresses, and policies to `data/entities/`.

2. **Feature Store** — the single source of truth for all model inputs. State
   regulatory mask applied here. Source datetimes in raw parquet only; derived ints
   in feature vector (DEC-013).

3. **Platform columns** — Underwriting (left) and Claims (right) run in parallel,
   both consuming from the shared feature store.

4. **Decision outcomes** — STP auto-approve, evidence request, SIU referral, and
   Investigator Copilot HITL. A feedback loop from the label store flows back into
   retraining.

### Shared Data Spine

The dashed amber arrow from **Policy Issuance → Fraud Scoring** carries
`risk_score_at_issuance`. This is the architectural "shared spine" described in
DEC-010: the underwriting risk score re-enters the claims pipeline as a fraud
feature. Without it, the fraud model has no knowledge of the underwriting context
at the time the policy was issued.

### Inference Paths

**Synchronous (<100ms):** Online feature retrieval (Redis) → device lookup → graph
neighborhood query → tabular XGBoost → decision orchestration.

**Asynchronous (seconds–minutes):** Deep ViT image analysis → full graph traversal
→ LLM narrative reconciliation → third-party enrichment → score update.

See `Fraud_Detection_Architecture.md` §7 for per-component latency budgets.