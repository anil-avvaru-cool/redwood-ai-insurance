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

---

## Independent Agent Service Topology

`DocumentVerificationAgent` and `SubrogationAgent` are deployed as **separate LangGraph
server instances** — not as modules inside the FNOL agent service. This section defines
their service boundaries, scaling, and checkpoint isolation.

### Service Inventory

| Service Name | Triggered By | Trigger Pattern | HITL | Shares Checkpoint Store |
|---|---|---|---|---|
| `fnol-claims-agent` | FNOL HTTP submission | Sync request | Yes — investigator review | Yes (shared PostgreSQL, `thread_id=claim_id`) |
| `doc-verification-agent` | `audit_node` completion event | Event-bus (Pattern A) | No | Yes (shared PostgreSQL, `thread_id=doc_verify_{claim_id}`) |
| `subrogation-agent` | Claim `SETTLED` status event | Event-bus (Pattern A) | Yes — senior handler (high-value only) | Yes (shared PostgreSQL, `thread_id=subrogation_{claim_id}`) |
| `underwriting-quote-agent` | Quote HTTP request | Sync request | Yes — underwriter review (HIGH tier) | Yes (shared PostgreSQL, `thread_id=quote_id`) |

All four services share one PostgreSQL checkpoint store. Thread ID prefix convention
(`doc_verify_`, `subrogation_`) provides namespace isolation and makes claim history
queryable across all agents for audit purposes.

### Service Boundaries

Each service is a self-contained deployment unit with:
- Its own FastAPI application (wrapping the LangGraph `StateGraph`)
- Its own health check endpoint (`/health` → `{"status": "ok", "graph": "<name>"}`)
- Its own MCP client configuration (tool endpoint URLs injected via environment)
- Its own Helm values override block (scaling, resource limits, image tag)
- Its own service account with least-privilege IAM/RBAC policy

Services do not share in-process state. The only shared infrastructure is:
- PostgreSQL checkpoint store (all agents)
- Redis feature store (FNOL and Underwriting only — DocVerification and Subrogation do not hit Redis)
- Claim store (read-only for all agents; write access scoped per service)

### Per-Substrate Deployment Map (Independent Agents)

| Service | AWS | Azure | OpenShift |
|---|---|---|---|
| `doc-verification-agent` | EKS Deployment (LangGraph server) | AKS Deployment (LangGraph server) | OCP Deployment (LangGraph server) |
| `subrogation-agent` | EKS Deployment (LangGraph server) | AKS Deployment (LangGraph server) | OCP Deployment (LangGraph server) |
| Event trigger (doc-verify) | SQS → Lambda invokes agent HTTP | Service Bus → Function invokes agent HTTP | OCP Job / Tekton pipeline |
| Event trigger (subrogation) | SQS → Lambda invokes agent HTTP | Service Bus → Function invokes agent HTTP | OCP Job / Tekton pipeline |
| MCP servers (`doc-verification-mcp-server`, `claims-data-mcp-server`, `subrogation-mcp-server`) | EKS Deployment | AKS Deployment | OCP Deployment |

### Scaling Configuration

`DocumentVerificationAgent` and `SubrogationAgent` are async, non-customer-facing services.
Scaling is event-driven, not latency-driven.

| Service | Scaling Driver | Min Replicas | Max Replicas | Notes |
|---|---|---|---|---|
| `doc-verification-agent` | SQS/Service Bus queue depth | 1 | 10 | Scale on queue depth > `DOC_VERIFY_QUEUE_DEPTH_THRESHOLD` (config) |
| `subrogation-agent` | SQS/Service Bus queue depth | 1 | 5 | Lower volume — settlement events are less frequent than FNOL |
| `fnol-claims-agent` | HTTP request rate (p95 latency) | 2 | 20 | Sync path — scale on latency, not queue depth |
| `underwriting-quote-agent` | HTTP request rate | 2 | 15 | Sync path |

### Health Check Pattern

All LangGraph services expose a standard health endpoint used by Kubernetes liveness
and readiness probes:

```python
@app.get("/health")
async def health():
    # Verify checkpoint store connectivity
    await checkpointer.aget_tuple(config={"configurable": {"thread_id": "health-check"}})
    return {"status": "ok", "service": os.environ["SERVICE_NAME"]}
```

Readiness probe also verifies MCP server reachability before accepting traffic:

```python
@app.get("/ready")
async def ready():
    for tool_name, tool_url in mcp_client.tool_registry.items():
        await mcp_client.ping(tool_url)
    return {"status": "ready"}
```

### RemoteGraph Invocation (Pattern B — Future)

When a calling agent requires a synchronous response from `DocumentVerificationAgent`
(e.g., Underwriting gating issuance on document verification), use `RemoteGraph`:

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

`DOC_VERIFY_AGENT_URL` is substrate-specific and injected via Helm values:

| Substrate | `DOC_VERIFY_AGENT_URL` value |
|---|---|
| AWS (EKS) | `http://doc-verification-agent.claims.svc.cluster.local:8000` |
| Azure (AKS) | `http://doc-verification-agent.claims.svc.cluster.local:8000` |
| OpenShift | `http://doc-verification-agent.claims.svc.cluster.local:8000` |

In-cluster DNS is the same pattern across all three substrates — only the cluster
domain suffix differs and is transparent to the application.

See [DEC-017](./DECISION_LOG.md#dec-017----event-bus-trigger-vs-remotegraph-invocation-decision-boundary)
for the decision rule governing when to use event-bus vs `RemoteGraph`.
