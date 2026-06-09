# Deployment Topologies
### Redwood AI Insurance — Infrastructure Substrate Guide
> See also: [DEC-014](./DECISION_LOG.md#dec-014----deployment-topology-as-a-first-class-decision)

---

## Core Principle

Platform components are fixed. The **deployment topology** varies by customer constraint. Same fraud ensemble, same feature store design, same entity resolution layer — different infrastructure substrate.

---

## Customer Segments

| Segment | Characteristics | Example Profile |
|---|---|---|
| **Tier-1 Cloud-Native Carrier** | High volume, AWS or Azure committed, modern data platform | Top-10 P&C carrier on cloud transformation program |
| **Regulated / Restricted Carrier** | Data sovereignty, PII cannot leave network, state or federal oversight | State-run fund, NFIP, workers comp carrier under DOI consent order |
| **Mid-Market Regional Carrier** | Limited platform team, cost-sensitive, wants managed services | Regional carrier, $500M–$2B DWP, no dedicated MLOps team |
| **Insurtech / Embedded Carrier** | Speed over control, API-first, small eng team | MGA, embedded insurance, usage-based carrier |
| **Large SI / Consulting Delivery** | Delivering Redwood into a carrier — infrastructure dictated by end client | System integrator implementation at a carrier with existing OpenShift footprint |

---

## Segment Decision Matrix

| Segment | LLM Strategy | Inference | Orchestration | Agent Framework |
|---|---|---|---|---|
| **Tier-1 Cloud-Native (AWS)** | Bedrock Titan | KServe on EKS | EKS + Helm | LangGraph on EKS |
| **Tier-1 Cloud-Native (Azure)** | Azure OpenAI | KServe on AKS | AKS + Helm | LangGraph on AKS |
| **Regulated / Restricted** | vLLM self-hosted, air-gapped | KServe on OpenShift AI | RHOAI + Operators | LangGraph, no external API calls |
| **Mid-Market Regional** | Bedrock or Anthropic API | SageMaker RT or FastAPI on EKS | Lightweight EKS or k3s | LangGraph or CrewAI |
| **Insurtech / Embedded** | Anthropic / OpenAI API | FastAPI + serverless (Modal / Lambda) | Minimal k8s or Render | CrewAI or LangGraph |
| **SI / Consulting Delivery** | Client-dictated | Client-dictated | OpenShift AI preferred | LangGraph (portable across all targets) |

---

## Three Driver Questions

Every customer engagement starts here before any technology is selected.

```
1. Can LLM inference traffic leave the customer network?
   ├── Yes → Bedrock / Azure OpenAI / Anthropic API
   └── No  → vLLM self-hosted
              ├── OpenShift AI footprint exists → RHOAI + KServe + vLLM
              └── No OpenShift → EKS GPU nodepool + KServe + vLLM

2. What is the existing infrastructure substrate?
   ├── AWS committed    → EKS + KServe + SageMaker + ElastiCache
   ├── Azure committed  → AKS + KServe + AzureML + Azure Cache
   ├── OpenShift        → RHOAI + KServe (built-in) + Redis Operator
   ├── On-prem only     → OpenShift AI or Rancher + self-managed Redis + Neo4j
   └── Greenfield       → EKS recommended (Redwood default stack)

3. What is the regulatory and audit posture?
   ├── DOI filing / adverse action exposure →
   │       Full MLflow lineage + immutable feature snapshots +
   │       KServe InferenceService versioning + audit log on S3/blob
   └── Internal tooling / no filing exposure →
           Simplified MLflow, relaxed snapshot retention
```

---

## Per-Component Portability Map

Platform components don't change. Only the substrate bindings change.

| Platform Component | AWS | Azure | OpenShift | Portable Layer |
|---|---|---|---|---|
| Online feature store | ElastiCache (Redis) | Azure Cache for Redis | Redis Operator on OCP | Redis — same API across all |
| Offline feature store | S3 + Glue | ADLS + Synapse | ODF + Spark on RHOAI | Parquet + Spark — same schema |
| Model serving (sync path) | KServe on EKS | KServe on AKS | KServe on RHOAI | KServe InferenceService spec — portable YAML |
| LLM inference | Bedrock Titan | Azure OpenAI | vLLM on GPU nodepool | OpenAI-compatible API — same client code |
| Graph database | Amazon Neptune | Cosmos DB (Gremlin API) | Neo4j Operator on OCP | Cypher — Neptune and Neo4j both support |
| Agent orchestration | LangGraph on EKS | LangGraph on AKS | LangGraph on OCP | LangGraph — no cloud dependency |
| Drift monitoring | CloudWatch + PSI | Azure Monitor + PSI | Prometheus + Grafana | PSI logic is pure Python — substrate-agnostic |
| Audit trail | S3 + Athena | ADLS + Synapse | ODF + Trino | Parquet schema — identical across all |

---

## Critical Portability Constraint

**Agent and RAG code must call LLM via OpenAI-compatible interface only — never via a cloud-provider SDK directly.**

This is enforced in `enterprise-rag-platform` as a single configurable endpoint URL:

```python
# enterprise-rag-platform/config.py — one change swaps the entire LLM substrate
LLM_BASE_URL = os.environ["LLM_BASE_URL"]   # set per deployment target
LLM_MODEL    = os.environ["LLM_MODEL"]
LLM_API_KEY  = os.environ["LLM_API_KEY"]
```

Target values by substrate:

| Substrate | LLM_BASE_URL | LLM_MODEL |
|---|---|---|
| Anthropic API | `https://api.anthropic.com` | `claude-sonnet-4-6` |
| Azure OpenAI | `https://<resource>.openai.azure.com` | deployment name |
| vLLM on-prem | `http://vllm-service:8000` | model path |
| AWS Bedrock (via proxy) | Bedrock-compatible proxy URL | `amazon.titan-text-express-v1` |

Swapping Bedrock for vLLM is a config change, not a code change. This constraint is non-negotiable for all contributors to `enterprise-rag-platform` and `fnol-claims-multi-agent-system`.

---

## Helm Values Files

Three values files cover the full substrate matrix. Core platform logic has no substrate imports — substrate bindings are isolated to these files and operator manifests.

```
redwood-ai-insurance/helm/
├── values-aws.yaml         ← EKS, ElastiCache, Neptune, S3, Bedrock
├── values-azure.yaml       ← AKS, Azure Cache, Cosmos DB, ADLS, Azure OpenAI
└── values-openshift.yaml   ← RHOAI, Redis Operator, Neo4j Operator, ODF, vLLM
```

When a platform component changes, all three values files must be updated. The portability map above identifies which rows require substrate-specific changes versus which are substrate-agnostic.
