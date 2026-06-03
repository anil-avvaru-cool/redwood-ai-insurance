# Redwood AI Insurance — Portfolio Task List
> Priority order: portfolio first, meetup later.
> Repo structure: `github.com/anil-avvaru-cool/` personal namespace.

---

## Repo Structure

```
github.com/<username>/
├── redwood-ai-insurance                  ← platform umbrella (parent, no code)
├── ai-fraud-detection-platform
├── intelligent-underwriting-platform
├── fnol-claims-multi-agent-system
├── enterprise-rag-platform
└── insurance-data-platform              ← entity resolution + feature store
```

---

## Phase 1 — Platform Foundation
> Start here. Umbrella repo must exist before any child repo can reference it.

- [ ] Create `redwood-ai-insurance` umbrella repo — empty, no code
- [ ] Write platform README with full Redwood AI Insurance story — overview, architecture summary, links to all 5 child repos
- [ ] Create platform architecture diagram — all 6 components connected through the shared data spine
- [ ] Add STRATEGY.md and key architecture docs to umbrella repo
- [ ] Create all 5 child repos with "Part of Redwood AI Insurance Platform" banner linking back to umbrella

---

## Phase 2 — Data Platform Repo
> Highest-signal repo. Decision Log is the portfolio differentiator.

**Repo:** `insurance-data-platform`

- [ ] Write README with entity resolution + feature store overview — explain why entity resolution is Layer 0
- [ ] Add architecture diagram: 7-layer build order (config → archetypes → generator → entities → graph → features → models)
- [ ] Port DECISION_LOG.md (DEC-001 to DEC-013) into repo — shows production reasoning, not just code
- [ ] Add roadmap with concrete week-by-week milestones — signals active development, not a graveyard

---

## Phase 3 — Fraud Detection Repo
> Most demo-able. Lead with the adversarial framing.

**Repo:** `ai-fraud-detection-platform`

- [ ] Write README — "not a classification problem" framing, ensemble architecture overview
- [ ] Add fraud scoring architecture diagram — XGBoost + GNN + NLP + ViT → stacking meta-learner → SHAP
- [ ] Add sync vs async inference diagram — concrete latency budgets per component for the <100ms path
- [ ] Add decision orchestration tier table — low / medium / high / extreme risk with thresholds and actions

---

## Phase 4 — Underwriting + RAG Repos
> Adds portfolio depth and demonstrates the shared spine across platforms.

**Repo:** `intelligent-underwriting-platform`

- [ ] Write README — risk score vs premium framing, two-stage hurdle model architecture
- [ ] Add risk scoring architecture diagram — Stage 1 frequency → Stage 2 severity → calibration → premium

**Repo:** `enterprise-rag-platform`

- [ ] Write README — standalone reusable library framing, explain how FNOL Agent and Quote Generation Agent both consume it
- [ ] Add RAG integration diagram — policy document RAG (FNOL) and underwriting guidelines RAG (quote) as consumers

---

## Phase 5 — Multi-Agent Claims Repo
> Ties the claims narrative together. Shows HITL workflow and cross-repo RAG dependency.

**Repo:** `fnol-claims-multi-agent-system`

- [ ] Write README — 4-agent system overview: FNOL Agent, Document Verification, Subrogation Agent, Investigator Copilot
- [ ] Add HITL workflow diagram — auto STP path vs evidence request vs SIU escalation vs investigator copilot
- [ ] Add agent interaction diagram showing RAG dependency — FNOL Agent consuming enterprise-rag-platform

---

## Phase 6 — Meetup Prep
> After portfolio is live. Needs a working demo, not just docs.

- [ ] Build live fraud scoring walkthrough — score a claim live using the sync path (5-min demo centerpiece)
- [ ] Create meetup slide deck — adversarial framing → platform overview → live demo → sync/async deep dive → Q&A (20–25 min)
- [ ] Update STRATEGY.md with meetup demo script — concrete talking points per section, not just the outline

---

## README Template (all child repos)

```markdown
> **Part of the [Redwood AI Insurance Platform](https://github.com/<username>/redwood-ai-insurance)**
> — an end-to-end intelligent insurance operations platform spanning underwriting through claims settlement.

## Overview
[what problem this repo solves — 1 paragraph]

## Architecture
[diagram]

## Status
In active development — see roadmap below.

## Roadmap
[concrete milestones with weeks]

## Design Decisions
[pull from DECISION_LOG.md — this section is the differentiator]
```

---

*Version: 2026-Q2 | 22 tasks across 6 phases*
