# Platform Strategy
### Redwood AI Insurance — AI Platform
> Version: 2026-Q2

---

## The Core Problem

Insurance fraud detection is not a supervised classification problem.

The framing matters. A classification mindset produces a model that learns to
replicate past detection — replicating past *blind spots* along with it. Fraud
evolves. Tactics that worked last year leave traces in labeled data; tactics
emerging now don't. The model has no way to detect what it has never seen.

The right framing is **adversarial risk orchestration**: a system that combines
anomaly detection (what's unusual relative to this population?), graph intelligence
(who is this person connected to?), behavioral analytics (does this pattern match
known fraud tactics?), and human investigation workflows (what does an experienced
investigator need to close this case?).

The models are components of that system — not the system itself.

---

## The Shared Data Spine

The single most impactful architectural decision in this platform is one line in the
feature store:

```
risk_score_at_issuance: float  # stored at quote time, re-read at FNOL time
```

A high-risk policy filed shortly after issuance is one of the strongest fraud signals
in personal auto. Without a shared data spine connecting underwriting to claims, that
signal is invisible to the fraud model. It sees a claim — it does not see the
underwriting context under which the policy was issued.

This is not a data engineering convenience. It is a fraud signal that cannot be
recovered after the fact without architectural intent.

The shared spine enables a second-order capability: over time, the fraud model's
feedback (which policies led to confirmed fraud) flows back to improve underwriting
risk scores, which in turn improves fraud detection. The platforms learn together.

**Implementation:** `risk_score_at_issuance` is written to the feature store at
quote completion and re-read as a feature input at FNOL time. See DEC-010.

---

## Entity Resolution as Foundation

Both platforms depend on the same underlying entities: vehicles (VINs), people
(policyholders, drivers, claimants), addresses, phone numbers, and policies.

If those entities are not resolved consistently, every downstream system degrades
silently:

- A `shares_address` edge between "123 Main St" and "123 Main Street" does not
  exist. A fraud ring spanning both appears disconnected in the graph.
- A VIN that fails to decode produces a vehicle record with null MSRP and null ADAS
  efficacy. The risk model scores it as though it is an average vehicle.
- A person who appears as both a policyholder and a claimant — under slightly
  different name spellings — is treated as two distinct people. Cross-role fraud
  signals disappear.

Entity resolution runs once, offline, before any feature computation or graph
construction begins. Both platforms consume resolved entities — neither owns them.
This is Stage 0 in the build order, not an afterthought.

**Implementation:** `entities/` layer (`entity_vehicle.py`, `entity_person.py`,
`entity_address.py`, `entity_phone.py`, `entity_policy.py`). See DEC-011.

---

## Training-Serving Parity

The most common silent failure mode in production ML is training-serving skew:
the offline training pipeline and the online serving pipeline compute features
differently.

This platform prevents it structurally:

- `feature_definitions.py` is Layer 0 — the dependency root. Archetypes import
  feature names from it; it imports nothing from them.
- Entity resolution outputs are passed as arguments to feature functions — not
  re-computed at serving time.
- Graph features are computed as a second-pass enrichment querying Neo4j, mirroring
  exactly what happens at inference time.
- Source datetime columns live in raw parquet only — never in the feature vector —
  so there is no path for them to contaminate model training. See DEC-013.

**Implementation:** `feature_definitions.py` as Layer 0 (DEC-004), graph features
as second-pass enrichment (DEC-005).

---

## Label Quality

Standard binary cross-entropy trains the fraud model to replicate past detection
patterns — including past *misses*. If 30% of your "legitimate" training records are
actually undetected fraud, the model learns that those patterns are safe.

This platform treats label quality as a first-class concern:

- **PU Learning** — treats negative labels as "unlabeled," not confirmed clean.
- **Confident Learning** (cleanlab) — identifies likely mislabeled examples via
  out-of-fold probability calibration.
- **Delayed label pipeline** — retrains on confirmed fraud labels post-SIU closure,
  keyed on `fraud_confirmed_at` in `claims.parquet`. Without this column, confirmed
  outcomes cannot be joined to training cohorts without reconstructing dates from
  unstructured SIU notes.
- **Cost-sensitive learning** — asymmetric loss: missed fraud >> false positive.
  Threshold set to minimize expected dollar loss, not accuracy.

**Implementation:** See `Fraud_Detection_Architecture.md` §5, DEC-007.

---

## Drift is the Operational Risk

Models do not fail suddenly. They drift, leading to adverse selection where
mispriced risks accumulate undetected.

PSI monitors input feature distributions over time. A state-mix shift (more CA
quotes) causes `credit_score` null rate to spike — that is real drift, not noise.
A telematics opt-in rate shift signals a change in population behavior affecting
the ~60% non-telematics cohort.

PSI alone is not sufficient. Concept drift — where the *relationship* between
features and outcomes changes — requires monitoring model performance metrics
(Gini, loss ratio by tier) alongside feature distributions.

The Champion-Challenger loop closes the cycle: PSI alert → challenger retrain →
shadow deployment → Gini check → phased rollout. No model goes to 100% traffic
without evidence of lift.

**Implementation:** `monitoring/psi_drift.py`, `champion_challenger.py`. PSI
current-period window keyed on `fnol_submitted_at` / `quote_requested_at` from
raw parquet (DEC-013).

---

## Human-in-the-Loop by Design

The Investigator Copilot is not a fallback for when the model is uncertain. It is
a designed component of the system.

Some claims require judgment that a model cannot provide: a claimant whose
telematics data is absent but who has a plausible explanation, a repair estimate
that is inflated but not fraudulent, a fraud ring where the evidence is
circumstantial. Investigators make these calls.

The Copilot makes investigators faster and better-calibrated — not redundant:

- Auto-generated case summary with top SHAP contributors surfaced in plain language.
- Interactive graph explorer showing fraud ring connections with risk path highlighting.
- Evidence checklist dynamically generated based on fraud type and score drivers.
- Suggested next actions based on similar historical cases that led to confirmed fraud.

Investigator override rates are tracked as a model health signal. Systematic
disagreement patterns surface model blind spots.

---

## Synthetic Data Strategy

The experiment dataset targets 33% fraud rate (vs ~15% production) across 10
archetypes with 20 features each. This is intentional — not a mistake.

At 15% fraud with 20K records, rare archetypes (coordinated fraud ring, VIN cloning)
have fewer than 100 examples — too few for XGBoost to learn archetype-specific
feature patterns. 33% produces ~660 fraud records per archetype with sufficient
support for each signal layer.

Production class imbalance is corrected via `scale_pos_weight` in XGBoost — not
by adjusting the training dataset ratio. Production thresholds are calibrated on a
held-out production sample, not on synthetic data. See DEC-007.

---

## Deployment Topology

The platform components are fixed. The deployment topology varies by customer constraint.

Every customer engagement starts with three questions before any technology is selected:

1. **Can LLM inference traffic leave the customer network?** Regulated carriers (DOI consent orders, NFIP, state-run funds) require air-gapped vLLM. Cloud-native carriers use Bedrock or Azure OpenAI. The answer to this question alone determines 80% of the infrastructure selection.

2. **What is the existing infrastructure substrate?** AWS-committed carriers get EKS + KServe + ElastiCache. Azure-committed carriers get AKS + Azure Cache. Carriers with an existing OpenShift contract get RHOAI + KServe built-in. SI engagements inherit the client's substrate. Greenfield defaults to EKS.

3. **What is the regulatory and audit posture?** DOI filing exposure requires full MLflow lineage, immutable feature snapshots, and KServe InferenceService versioning. Internal tooling deployments can use a simplified configuration.

The platform exposes three substrate bindings — AWS, Azure, OpenShift — via Helm values files. Core platform logic has no substrate imports. Substrate bindings are isolated to `values-*.yaml` and operator manifests.

**Non-negotiable portability constraint:** All LLM calls in `enterprise-rag-platform` and `fnol-claims-multi-agent-system` go through a single OpenAI-compatible endpoint configured in `enterprise-rag-platform/config.py`. No direct cloud-provider SDK calls anywhere in agent or RAG code. Swapping Bedrock for vLLM is a config change, not a code change.

**Implementation:** `enterprise-rag-platform/config.py` (LLM substrate config), `redwood-ai-insurance/helm/` (three values files). See [DEC-014](./DECISION_LOG.md#dec-014----deployment-topology-as-a-first-class-decision) and [DEPLOYMENT_TOPOLOGIES.md](./DEPLOYMENT_TOPOLOGIES.md).

---

## What This Platform Is Not

- Not a single model solving a single classification task.
- Not a batch-scoring system with a nightly run.
- Not a rules engine with ML bolted on.
- Not a black box that outputs a score with no explanation.

It is a real-time risk orchestration system where every decision is explainable,
every feature is auditable, every model version is reproducible, and human
judgment is a designed component — not a workaround.

---

*Strategy version: 2026-Q2*