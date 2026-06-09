# Decision Log
### Redwood AI Insurance — AI Platform
> Living document. Update immediately when a decision is made. Decisions without rationale become mysteries.

---

## How to Use This Document

Each entry captures:
- **Decision** — what was decided
- **Rationale** — why, including the alternative considered
- **Trade-off** — what was accepted or rejected
- **Date + context** — when and what triggered it

---

## DEC-001 — Telematics nulls: null not default

**Date:** 2026-Q2
**Decision:** Telematics features are set to `null` for the ~60% of users without devices. No default or neutral value is imputed.

**Rationale:** XGBoost's native missing-value handling learns the actual risk distribution for non-telematics users from training data. Non-telematics users skew slightly higher risk on average due to adverse selection — safer drivers are more likely to opt in. Imputing a neutral value masks this learned signal and produces a systematically miscalibrated model for a majority of the user base.

**Alternative considered:** Mean imputation or a neutral midpoint value.

**Trade-off accepted:** Model behavior for the 60% null cohort is less interpretable to non-technical stakeholders. Mitigated by SHAP explanation: `telematics_available=False` surfaces as a positive fraud/risk contributor in investigator copilot output.

**Applies to:** `feature_definitions.py` — `build_telematics_features()`, both platforms.

---

## DEC-002 — Credit score nulls: null not default (regulatory)

**Date:** 2026-Q2
**Decision:** `credit_score` is set to `null` for policies in CA, MA, MI, HI. It is never imputed with any value.

**Rationale:** Imputing any value — even a neutral one — risks acting as a credit proxy in states where credit-based insurance scoring is prohibited by statute. XGBoost's null path is the only safe route.

**Alternative considered:** Setting a fixed neutral value (e.g. national median score of 703).

**Trade-off accepted:** Slightly reduced model accuracy in restricted states. Accepted — regulatory compliance is non-negotiable.

**Applies to:** `feature_definitions.py` — `apply_state_regulatory_mask()`, underwriting platform.

**States affected:** CA, MA, MI, HI (see `CREDIT_RESTRICTED_STATES` in `config.py`).

---

## DEC-003 — Telematics trio convention

**Date:** 2026-Q2
**Decision:** Every nullable telematics signal is accompanied by two derived features: an availability flag (`telematics_available: bool`) and, for the claims platform, an enrolled-but-missing fraud signal (`telematics_enrolled_but_missing: bool`).

**Rationale:** A single nullable float cannot express three distinct states: (1) device present and signal populated, (2) no device, (3) enrolled but feed absent at claim time. State 3 is a meaningful fraud signal — organized rings frequently suppress telematics data.

**Alternative considered:** A single categorical feature. Rejected — XGBoost handles boolean flags better than categorical encoding for this pattern.

**Trade-off accepted:** Adds 2 derived features per telematics signal.

**Applies to:** `entity_vehicle.py`, `feature_definitions.py` — `build_telematics_features()`, both platforms.

**Convention:** Apply this trio pattern to any future nullable signal group.

---

## DEC-004 — feature_definitions.py as Layer 0

**Date:** 2026-Q2
**Decision:** `features/feature_definitions.py` is built before archetypes. Archetypes import feature names from `feature_definitions.py`. It has no imports from archetypes, generator, or `entity_*.py` modules.

**Rationale:** Training-serving skew is the most common silent failure mode in production ML. Making `feature_definitions.py` the dependency root forces both pipelines to share one implementation from day one.

**Alternative considered:** Define feature names in archetypes and import into `feature_definitions.py`. Rejected — inverts the dependency.

**Trade-off accepted:** `feature_definitions.py` must be written before any data generation work begins.

---

## DEC-005 — Graph features as second-pass enrichment

**Date:** 2026-Q2
**Decision:** Graph features (`graph_hop_distance`, `shared_attribute_count`, `attorney_centrality_score`) are computed as a second pass after the offline pipeline runs — not baked into the generator.

**Rationale:** At inference time, graph features are queried live from Neo4j. Pre-joining them in the generator breaks online/offline parity — a form of training-serving skew.

**Alternative considered:** Pre-computing inside `generator.py`. Rejected.

**Trade-off accepted:** Graph enrichment requires Neo4j to be loaded before `offline_pipeline.py` can produce a complete feature set.

---

## DEC-006 — 20K records, 20 features per platform

**Date:** 2026-Q2
**Decision:** Experiment dataset is 20K quotes + 20K claims with 20 features each.

**Rationale:** 10 features was too few — several signals come in natural trios (telematics trio). 10K records was borderline for the Gamma severity model (~800 severity training records at ~8% claim rate). 20K produces ~1,600.

**Trade-off accepted:** Slightly more upfront work in archetype definitions. Total dataset is ~10MB Parquet — trivially fits local Docker.

---

## DEC-007 — Fraud rate 33% in synthetic data (vs ~15% production)

**Date:** 2026-Q2
**Decision:** Synthetic claims dataset targets ~33% fraud rate across 10 archetypes.

**Rationale:** At 15% fraud with 20K records, some archetypes would have fewer than 100 examples — too few to learn archetype-specific feature patterns. 33% provides ~660 fraud records per archetype.

**Production class imbalance** is corrected via `scale_pos_weight` in XGBoost, not by adjusting the training dataset ratio.

**Trade-off accepted:** Thresholds calibrated on synthetic data will not match production thresholds. Accepted — threshold calibration is done on a held-out production sample.

---

## DEC-008 — No PostgreSQL

**Date:** 2026-Q2
**Decision:** PostgreSQL is not included in the platform stack.

**Rationale:** Every data workload maps to a more appropriate technology: online feature serving → Redis, offline training data → S3 + Parquet, graph relationships → Neo4j, audit trail → JSON on S3, model metadata → MLflow.

**Trade-off accepted:** If a legacy policy system is Postgres-backed, a read-only connection will be needed for FNOL Agent policy lookup. This is a source system integration, not a platform database.

---

## DEC-009 — Pydantic models over standalone JSON Schema files

**Date:** 2026-Q2
**Decision:** Input validation schemas are defined as Pydantic models in `api/schemas.py`, not as standalone `.json` schema files.

**Rationale:** Pydantic provides runtime validation, serialization, and OpenAPI documentation. Maintaining both creates two sources of truth that drift apart.

**Trade-off accepted:** External partners who need JSON Schema receive a generated file (`model.schema_json()`), not hand-maintained.

---

## DEC-010 — risk_score_at_issuance as fraud feature (shared spine)

**Date:** 2026-Q2
**Decision:** The underwriting risk score at quote time is stored in the feature store and re-used as a fraud scoring input feature, not discarded after policy issuance.

**Rationale:** High-risk policies filed shortly after issuance is one of the strongest fraud signals in personal auto. Without `risk_score_at_issuance`, the fraud model has no knowledge of the underwriting context. This is the architectural "shared data spine."

**Dependency created:** Quotes must be scored by the risk model before claims can be generated.

**Trade-off accepted:** Tight coupling between underwriting and claims pipelines via the feature store. Accepted — this coupling is the intended architecture.

---

## DEC-011 — Entity resolution as independent pre-graph layer

**Date:** 2026-Q2
**Decision:** Vehicle, Person, Address, and Phone are resolved as independent entities in `entities/` before `graph_builder.py` runs. `graph_builder.py` loads resolved entities as nodes and edges only.

**Rationale:** Vehicle base attributes have no graph dependency. Address and phone normalization must happen before graph edges are created — a `shares_address` edge between two unnormalized strings is unreliable and degrades Louvain community detection silently.

**Trade-off accepted:** Adds an explicit entity resolution step to the build order and a new `entities/` directory. Accepted — prevents a class of silent failures.

**Applies to:** `entities/` layer (new), `graph_builder.py` (scope reduced), `offline_pipeline.py`.

---

## DEC-012 — Correct other units to US standard units like miles

**Date:** 2026-Q2
**Decision:** `ip_geolocation_delta_km` → `ip_geolocation_delta_miles`. All units corrected to US standard.

**Rationale:** Agent spec says "USA standard units in data and outputs" — non-negotiable. Mixing km and miles creates confusion for investigators and business stakeholders reading raw feature values.

---

## DEC-013 — Source datetimes in raw parquet; derived ints unchanged in feature store (Option A)

**Date:** 2026-Q2
**Decision:** Explicit source datetime columns are added to `quotes.parquet` and `claims.parquet` as raw pipeline columns only. The derived int features `policy_inception_days` and `reporting_delay_days` are unchanged in the feature store, feature vector, and all trained artifacts.

**Columns added to `quotes.parquet`:** `quote_requested_at`, `quote_completed_at`, `policy_inception_date`

**Columns added to `claims.parquet`:** `loss_event_datetime`, `fnol_submitted_at`, `claim_closed_at` (nullable), `fraud_confirmed_at` (nullable)

**Rationale:** Without datetime columns, PSI monitoring has no time axis. Option A is non-breaking — both pipelines continue to consume the same derived ints. Source datetimes serve three distinct consumers the ints cannot: (1) PSI current-period window, (2) concept drift / confirmed label window keyed on `fraud_confirmed_at`, (3) temporal integrity validation.

**Alternative considered (Option B):** Replace derived ints with source datetimes in the feature store. Rejected — adds compute to the <100ms online serving path, risks training-serving skew.

**Trade-off accepted:** `policy_inception_days` and `reporting_delay_days` have two representations. The derivation relationship must be kept in sync in `feature_definitions.py`.

**Applies to:** `generator.py`, `archetypes_claims.py`, `archetypes_underwriting.py`, `entity_policy.py`, `hurdle_model.py`, `validator.py`, `feature_definitions.py`, `monitoring/psi_drift.py`.

---

## DEC-014 — Deployment Topology as a First-Class Decision

**Date:** 2026-Q2

**Decision:** Deployment topology is segment-driven, not stack-driven. The platform exposes three substrate bindings (AWS, Azure, OpenShift) via Helm values files. Component implementations are substrate-agnostic. LLM substrate is abstracted behind an OpenAI-compatible interface in `enterprise-rag-platform`.

**Rationale:** A single deployment target would disqualify Redwood from regulated carrier and SI delivery engagements where infrastructure is either dictated by compliance posture or by an existing enterprise contract. The three-binding approach covers the full commercial carrier market without platform fragmentation.

**Agent framework selection by segment:**
- **LangGraph** — default for all segments. No cloud dependency, state machine model maps directly to claim lifecycle states, portable across all three substrate bindings.
- **CrewAI** — available for insurtech and mid-market segments where rapid prototyping speed outweighs operational control. Not recommended for regulated deployments.
- **vLLM** — LLM serving layer for any deployment where inference traffic cannot leave the customer network. Exposes OpenAI-compatible API — no application code changes when switching from managed API to vLLM.

**Alternative considered:** A single AWS-only deployment target. Rejected — excludes regulated carriers under DOI consent orders and SI engagements with existing OpenShift contracts.

**Trade-off accepted:** Three values files require maintenance when platform components change. Mitigated by the portability map in `DEPLOYMENT_TOPOLOGIES.md` — substrate bindings are isolated to `values-*.yaml` and operator manifests. Core platform logic has no substrate imports.

**Applies to:** `redwood-ai-insurance/helm/`, `redwood-ai-insurance/openshift/`, `enterprise-rag-platform/config.py`, `fnol-claims-multi-agent-system/`, `DEPLOYMENT_TOPOLOGIES.md`.