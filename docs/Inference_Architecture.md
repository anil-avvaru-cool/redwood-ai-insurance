# Inference Architecture
### Redwood AI Insurance — Enterprise LLM Inference Layer

---

> **Core Framing:**
> LLM inference in this platform is not a cloud API call.
> For regulated carriers it is a **network-bound, auditable, versioned infrastructure layer** —
> deployed inside the enterprise perimeter, subject to the same change management
> and compliance controls as any other production ML system.
> The inference layer must be substrate-portable: the same agent code runs against
> Bedrock, Azure OpenAI, or a self-hosted vLLM cluster by changing three environment
> variables and nothing else.

---

## Table of Contents
1. [Layer Placement](#1-layer-placement)
2. [The Hosting Decision Tree](#2-the-hosting-decision-tree)
3. [vLLM as the Self-Hosted Inference Engine](#3-vllm-as-the-self-hosted-inference-engine)
4. [Model Registry and Air-Gapped Loading](#4-model-registry-and-air-gapped-loading)
5. [GPU Nodepool Sizing](#5-gpu-nodepool-sizing)
6. [Network Topology](#6-network-topology)
7. [KServe InferenceService Patterns](#7-kserve-inferenceservice-patterns)
8. [RAG and Embedding Inference](#8-rag-and-embedding-inference)
9. [Latency Budget](#9-latency-budget)
10. [Autoscaling](#10-autoscaling)
11. [Regulatory and Audit Controls](#11-regulatory-and-audit-controls)
12. [Champion-Challenger for LLMs](#12-champion-challenger-for-llms)
13. [Substrate Inference Map](#13-substrate-inference-map)
14. [SGLang as the Structured-Output Inference Engine](#14-sglang-as-the-structured-output-inference-engine)
15. [Customer-Based Use Cases — vLLM vs SGLang](#15-customer-based-use-cases--vllm-vs-sglang)

---

## 1. Layer Placement

The inference layer sits between the agent orchestration layer and the network
perimeter. It is not visible to business logic — agents call a single
OpenAI-compatible endpoint regardless of what is behind it.

```
┌──────────────────────────────────────────────────────────────┐
│  AGENT ORCHESTRATION LAYER — LangGraph                       │
│  FNOL multi-agent · Quote agent · Investigator copilot       │
│  Calls LLM via get_llm() factory — no substrate imports      │
├──────────────────────────────────────────────────────────────┤
│  RAG LAYER — enterprise-rag-platform                         │
│  Retrieval · Reranking · Context assembly                    │
│  Calls embedding model via same OpenAI-compatible interface  │
├──────────────────────────────────────────────────────────────┤
│  INFERENCE LAYER  ◄── this document                          │
│                                                              │
│  Cloud-hosted path         Self-hosted path                  │
│  ┌──────────────────┐      ┌───────────────────────────────┐ │
│  │ Bedrock          │      │ vLLM on GPU nodepool          │ │
│  │ Azure OpenAI     │      │ SGLang on GPU nodepool        │ │
│  │ Anthropic API    │      │ KServe InferenceService       │ │
│  └──────────────────┘      │ RHOAI / EKS / AKS             │ │
│                             └───────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE SUBSTRATE                                    │
│  EKS · AKS · OpenShift AI (RHOAI)                           │
│  See DEPLOYMENT_TOPOLOGIES.md for substrate decision matrix  │
└──────────────────────────────────────────────────────────────┘
```

**Non-negotiable constraint (DEC-014):** All LLM and embedding calls use the
OpenAI-compatible interface. No direct cloud-provider SDK calls anywhere in
agent or RAG code. Substrate is determined by three env vars:

```bash
LLM_BASE_URL=http://vllm-service.inference.svc.cluster.local:8000/v1
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
LLM_API_KEY=none   # vLLM accepts a placeholder; Bedrock proxy uses real key
```

---

## 2. The Hosting Decision Tree

```
Can LLM inference traffic leave the customer network?
│
├── YES
│    │
│    ├── AWS committed?    → Bedrock (Titan / Nova / Anthropic on Bedrock)
│    ├── Azure committed?  → Azure OpenAI Service
│    └── No cloud commitment / multi-cloud → Anthropic API (direct)
│
└── NO  (regulated carrier: data sovereignty, DOI consent order,
         state fund, federal program, air-gapped network)
     │
     ├── OpenShift AI (RHOAI) footprint exists?
     │    YES → RHOAI + KServe (built-in) + vLLM serving runtime
     │    NO  → EKS or AKS GPU nodepool + KServe + vLLM
     │
     ├── GPU available in cluster?
     │    YES → Full-precision or AWQ-quantized model on GPU
     │    NO  → CPU-only inference (latency-tolerant async path only;
     │          not acceptable for sync agent path < 8s SLA)
     │
     └── Internet egress to HuggingFace Hub allowed?
          YES → Pull model at deploy time via initContainer
          NO  → Model must be pre-staged in internal artifact registry
                (Artifactory / Nexus / S3 internal / ODF)

After selecting the self-hosted path, choose the inference engine:
     │
     ├── Primary workload is structured JSON extraction (FNOL detail,
     │   quote explanation schema, coverage determination)?
     │    AND running on EKS or AKS (not RHOAI)?
     │    YES → Deploy SGLang InferenceService alongside vLLM
     │           (see Section 14 for configuration and decision criteria)
     │
     └── Default → vLLM only
          (long-form generation, investigator summaries, embeddings,
           RHOAI deployments — vLLM covers all these workloads)
```

**Regulated carrier is the design-forcing constraint.** The self-hosted
vLLM path must work completely without public internet access. Cloud-native
carriers get the same codebase with different env vars.

---

## 3. vLLM as the Self-Hosted Inference Engine

vLLM is selected for self-hosted inference because:
- Exposes an OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`) —
  the same interface used for Bedrock and Azure OpenAI, so no agent code changes
- PagedAttention enables high-throughput concurrent requests on a single GPU
- Supports tensor parallelism across multiple GPUs without changing the serving API
- Quantization-aware loading (AWQ, GPTQ, INT8) reduces GPU memory without
  retraining — critical in enterprise GPU environments with strict quota allocations
- Active project with RHOAI / OpenShift AI native integration via `ServingRuntime`
  custom resource

### vLLM Configuration Reference

Key parameters that must be set per deployment. These are env vars on the
vLLM pod, not agent-side config.

| Parameter | Value | Rationale |
|---|---|---|
| `--model` | Internal registry path or HF model ID | What to serve |
| `--tensor-parallel-size` | 1 (single A100) / 2 (two A100 40GB for 13B) / 4 (70B) | Split model weight shards across GPUs |
| `--quantization` | `awq` for 7B–13B in memory-constrained nodes | Cuts VRAM ~50%; minimal quality loss for instruction-tuned models |
| `--max-model-len` | 8192 (default) / 32768 for long-context RAG prompts | Context window; larger = more VRAM for KV cache |
| `--gpu-memory-utilization` | `0.90` | Reserve 10% for CUDA overhead; increase OOM risk above 0.95 |
| `--enable-prefix-caching` | Enabled | Reuse KV cache across requests with shared system prompts — important for RAG where system prompt is large and identical across requests |
| `--served-model-name` | Canonical alias (e.g. `redwood-llm`) | Agents reference this alias, not the HF path — decouples model upgrade from agent config |

### Model Selection by Use Case

| Use Case | Recommended Model Class | Rationale |
|---|---|---|
| Narrative inconsistency (FNOL async) | 7B–13B instruction-tuned | Latency-tolerant async path; quality over speed |
| Quote explanation generation | 7B instruction-tuned | Sync path; must complete within 8s latency budget |
| Investigator copilot summary | 13B+ instruction-tuned | Long-context; reads full claim state including graph context |
| Embedding (RAG retrieval) | Dedicated embedding model (e.g. BGE-M3, E5-large) | Do not use the generative LLM for embeddings — served as a separate InferenceService |
| Cross-encoder reranking | Dedicated reranking model (e.g. BGE-Reranker-v2) | Separate endpoint; called only when `rerank=True` in RAGClient |

**Never serve embedding and generative models on the same vLLM process.**
They have different batching characteristics and memory profiles. Two
`InferenceService` resources with separate GPU allocations is the correct pattern.

---

## 4. Model Registry and Air-Gapped Loading

For carriers with no public internet egress, models must be pre-staged before
deployment. The pull-at-deploy pattern used in cloud-native environments
(initContainer `huggingface-cli download`) is replaced with a push-to-registry
pattern during the release pipeline.

### Artifact Flow (Air-Gapped)

```
Release Pipeline (runs in DMZ or with controlled egress)
      │
      ├── Pull model weights from HuggingFace Hub
      ├── Run quantization if required (AWQ/GPTQ with autoawq)
      ├── Push to internal artifact registry
      │    ├── AWS:        S3 internal bucket (VPC endpoint only)
      │    ├── Azure:      Azure Blob (private endpoint)
      │    └── OpenShift:  ODF (OpenShift Data Foundation) S3-compatible store
      │
      └── Tag release: model_name + version + quantization_config + SHA256

Deployment (no public internet required)
      │
      └── vLLM pod initContainer pulls from internal registry
           using internal service account credentials
```

### Model Version Pinning

Models must be version-pinned in the `InferenceService` manifest, not
referenced by a mutable tag like `latest`. The SHA256 of the model weights
is recorded in the audit log at deployment time. This is required for:

- Adverse action defensibility (regulators can ask "which model version
  generated this explanation?")
- Rollback — a pinned version can be re-deployed deterministically
- Champion-challenger — two versions running simultaneously must be
  unambiguously identified in the inference log

```yaml
# Pinned by digest, not by mutable tag
model: redwood-llm@sha256:a3f9...
```

---

## 5. GPU Nodepool Sizing

### Memory Sizing Formula

```
Required VRAM (GB) ≈ (model_params_B × precision_bytes) + kv_cache_overhead

AWQ 4-bit precision: 0.5 bytes/param
FP16 precision:      2 bytes/param
FP32 precision:      4 bytes/param (avoid on inference nodes)

KV cache overhead: ~20–30% of model weight VRAM at max_model_len=8192
```

| Model Size | Precision | VRAM Required | GPU Fit |
|---|---|---|---|
| 7B | FP16 | ~14GB + cache ≈ 18GB | 1× A100 40GB (comfortable) |
| 7B | AWQ 4-bit | ~4GB + cache ≈ 8GB | 1× A10G 24GB (tight) |
| 13B | FP16 | ~26GB + cache ≈ 34GB | 1× A100 80GB or 2× A100 40GB (TP=2) |
| 13B | AWQ 4-bit | ~7GB + cache ≈ 12GB | 1× A100 40GB |
| 70B | AWQ 4-bit | ~35GB + cache ≈ 45GB | 2× A100 40GB (TP=2) |
| 70B | FP16 | ~140GB | 4× A100 80GB (TP=4) — avoid unless required |

### Nodepool Configuration

```yaml
# EKS nodepool (managed node group)
instanceType: g5.2xlarge   # 1× A10G 24GB — adequate for 7B FP16 or 13B AWQ
instanceType: p3.2xlarge   # 1× V100 16GB — legacy; tight for 7B FP16
instanceType: p4d.24xlarge # 8× A100 40GB — for multi-model or 70B

# AKS nodepool
vmSize: Standard_NC24ads_A100_v4   # 1× A100 80GB
vmSize: Standard_NC48ads_A100_v4   # 2× A100 80GB

# OpenShift (via MachineSet)
instanceType: g5.2xlarge   # AWS-backed ROSA
instanceType: Standard_NC24ads_A100_v4  # Azure-backed ARO
```

Inference GPU nodepools use a taint (`inference=true:NoSchedule`) to prevent
non-inference workloads from consuming GPU quota. vLLM pods carry the matching
toleration. Agent pods (CPU-only) never land on GPU nodes.

---

## 6. Network Topology

### Traffic Flow — Self-Hosted (Regulated Carrier)

```
External (Internet)
      │  ← blocked at perimeter for LLM inference traffic
      ✗

Enterprise Perimeter
      │
      ▼
Agent Pod (LangGraph)                    RAG Pod (enterprise-rag-platform)
      │                                         │
      │ HTTP/2 to cluster-internal service      │ HTTP/2 to embedding service
      ▼                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  inference namespace                                            │
│                                                                 │
│  vllm-llm-service (ClusterIP)      vllm-embedding-service      │
│       │                                    │                    │
│       ▼                                    ▼                    │
│  InferenceService: redwood-llm     InferenceService: redwood-   │
│  (KServe + vLLM runtime)           embedding (KServe + vLLM)   │
│       │                                    │                    │
│       ▼                                    ▼                    │
│  GPU nodepool                       GPU nodepool                │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Internal model registry (S3/ODF/Blob — VPC/private endpoint only)
No traffic exits the cluster network boundary
```

### Service Mesh Policy

For carriers running Istio (EKS/AKS) or OpenShift Service Mesh (OCP):

- `AuthorizationPolicy` restricts inference service access to the `agents`
  and `rag` namespaces only — inference endpoints are not reachable from
  other namespaces or ingress
- mTLS enforced on all intra-cluster calls to inference services
- `PeerAuthentication` set to `STRICT` in the `inference` namespace
- No `ServiceEntry` for external LLM APIs exists in regulated deployments —
  an accidental SDK call to `api.anthropic.com` or `api.openai.com` will
  fail at the network layer, not silently succeed

### DNS Naming Convention

```bash
# Generative LLM
vllm-llm-service.inference.svc.cluster.local:8000

# Embedding model
vllm-embedding-service.inference.svc.cluster.local:8000

# Reranking model (if self-hosted; else Cohere API for cloud-native carriers)
vllm-rerank-service.inference.svc.cluster.local:8000
```

These are the values set in `LLM_BASE_URL`, `EMBEDDING_BASE_URL`, and
`RERANK_BASE_URL` in the self-hosted deployment env.

---

## 7. KServe InferenceService Patterns

KServe is used on all three substrates. On RHOAI, KServe is the built-in
model serving layer. On EKS and AKS, KServe is installed via Helm as part
of the Redwood platform stack.

### Generative LLM — InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: redwood-llm
  namespace: inference
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      storageUri: s3://redwood-models-internal/mistral-7b-instruct-awq-v0.3/
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "20Gi"
          cpu: "4"
        limits:
          nvidia.com/gpu: "1"
          memory: "24Gi"
      env:
        - name: SERVED_MODEL_NAME
          value: "redwood-llm"
        - name: TENSOR_PARALLEL_SIZE
          value: "1"
        - name: QUANTIZATION
          value: "awq"
        - name: MAX_MODEL_LEN
          value: "8192"
        - name: GPU_MEMORY_UTILIZATION
          value: "0.90"
        - name: ENABLE_PREFIX_CACHING
          value: "true"
      tolerations:
        - key: "inference"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
```

### Embedding Model — InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: redwood-embedding
  namespace: inference
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      storageUri: s3://redwood-models-internal/bge-m3/
      resources:
        requests:
          nvidia.com/gpu: "1"
          memory: "8Gi"
        limits:
          nvidia.com/gpu: "1"
          memory: "10Gi"
      env:
        - name: SERVED_MODEL_NAME
          value: "redwood-embedding"
        - name: MAX_MODEL_LEN
          value: "512"       # embedding models use short context windows
        - name: GPU_MEMORY_UTILIZATION
          value: "0.85"
```

### OpenShift-Specific: ServingRuntime

On RHOAI, vLLM is registered as a `ServingRuntime` custom resource.
The `InferenceService` references it by name rather than embedding runtime config.

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: vllm-runtime
  namespace: inference
spec:
  supportedModelFormats:
    - name: vLLM
      version: "1"
      autoSelect: true
  containers:
    - name: kserve-container
      image: quay.io/modh/vllm:rhoai-2.16   # Red Hat supported vLLM image
      command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
      args:
        - "--model=/mnt/models"
        - "--served-model-name={{.Name}}"
        - "--tensor-parallel-size=$(TENSOR_PARALLEL_SIZE)"
```

Use the Red Hat (`quay.io/modh/vllm`) image on RHOAI — it is validated
against the RHOAI operator version and receives security patches via the
Red Hat advisory cycle. The upstream `vllm/vllm-openai` image is used on
EKS and AKS where RHOAI operator is not present.

---

## 8. RAG and Embedding Inference

The RAG pipeline has two inference calls: embedding and optional reranking.
Both are on the hot path for the `quote_gen_node` (sync, user-facing) and
on the async path for the `narrative_llm_node`.

### Retrieval Inference Flow

```
RAGClient.retrieve(query, top_k=5, rerank=True)
      │
      ├── 1. Embed query
      │       POST /v1/embeddings → vllm-embedding-service
      │       Model: redwood-embedding (BGE-M3 or E5-large)
      │       Target latency: < 30ms
      │
      ├── 2. Vector search
      │       pgvector (AWS RDS / Azure PostgreSQL / CrunchyData PGO)
      │       ANN search: HNSW index, cosine similarity
      │       Returns top_k × 3 candidates before reranking
      │       Target latency: < 20ms
      │
      └── 3. Rerank (if rerank=True)
              POST /v1/rerank or /v1/completions (cross-encoder scoring)
              Model: BGE-Reranker-v2-m3 or Cohere Rerank (cloud-native only)
              Returns top_k final candidates
              Target latency: < 80ms (cross-encoder is slower than bi-encoder)

Total retrieval budget: < 130ms (fits within quote_gen_node 8s window)
```

### Embedding Caching

The `embedding_cache.py` module in `enterprise-rag-platform` caches
embeddings to avoid re-embedding unchanged documents and to short-circuit
repeated identical queries within a session.

| Cache Level | Backend | TTL | Scope |
|---|---|---|---|
| Document embedding cache | Redis (online feature store, separate key namespace) | 7 days | Ingestion pipeline — not real-time path |
| Query embedding cache | In-memory LRU (per pod) | Request lifetime | Real-time path — same query in same request |

Do not cache query embeddings to Redis across requests. Query context is
claim- or quote-specific — a cache hit from a different claim's query
is meaningless and risks surfacing mismatched context.

### Reranking Placement

| Carrier Type | Reranker | Location |
|---|---|---|
| Regulated / air-gapped | BGE-Reranker-v2 | Self-hosted vLLM in `inference` namespace |
| Cloud-native (AWS) | Cohere Rerank API | External API call (egress allowed) |
| Cloud-native (Azure) | Cohere Rerank API or Azure AI Search semantic ranker | External |

Self-hosted reranker adds ~60–80ms to retrieval latency. This is acceptable
for the sync RAG path because the total latency budget is dominated by the
generative LLM call (2–6s), not retrieval (130ms target).

---

## 9. Latency Budget

### Sync Agent Path — Quote Generation (p95 target: 8s)

```
quote_gen_node latency breakdown (sync, user-facing)

  RAG retrieval
  ├── embed query               ~25ms
  ├── vector search             ~15ms
  └── rerank (if self-hosted)   ~75ms
  Subtotal retrieval:          ~115ms

  Context assembly              ~5ms

  LLM generation (vLLM)
  ├── Time-to-first-token      ~300ms   (depends on model size, batch depth)
  └── Token generation         ~1.5s    (512 output tokens at ~350 tok/s on A100)
  Subtotal LLM:                ~1.8s

  Total quote_gen_node:        ~1.9s    (well within 8s window)

  Remaining quote agent budget: 6.1s
  ├── mvr_pull_node + credit_pull_node (parallel): 1–5s
  ├── feature_assembly_node:   ~20ms
  ├── risk_scoring_node:        ~5ms
  ├── premium_calc_node:        ~3ms
  └── compliance_check_node:   ~10ms
```

### Sync Agent Path — FNOL Intake (p95 target: 65ms)

The FNOL sync path (`intake` → `decision_routing`) does **not** call the LLM.
LLM calls are on the async path only. The 65ms budget is for tabular scoring
and feature enrichment. Do not add synchronous LLM calls to the FNOL sync path.

```
FNOL sync path — NO LLM calls

  feature_enrichment_node (Redis):  ~5ms
  device_check_node:                ~8ms
  graph_query_node (Neo4j, 1-hop):  ~12ms
  tabular_scoring_node (XGBoost):   ~3ms
  decision_routing_node:            ~2ms
  Total sync path:                  ~30ms  (p50)
                                    ~60ms  (p95, under graph query variance)
```

### LLM Latency Levers

If vLLM latency exceeds budget:

| Lever | Effect | Trade-off |
|---|---|---|
| Reduce `max_new_tokens` | Cuts generation time proportionally | Shorter output |
| Enable prefix caching | Eliminates prompt re-processing for shared system prompts | Memory overhead |
| Apply AWQ quantization | ~1.5–2× throughput increase | Minor quality degradation |
| Reduce `max_model_len` | Frees KV cache memory for more concurrent requests | Shorter context window |
| Increase `tensor_parallel_size` | Distributes compute across more GPUs | Requires more GPU quota |
| Add a second replica | Reduces queue depth under concurrent load | 2× GPU cost |

---

## 10. Autoscaling

### Metric-Based Scaling

LLM inference pods do not scale on CPU or memory — they scale on queue depth
and token throughput. Use KEDA (EKS/AKS) or OpenShift HPA extensions with
custom metrics.

```yaml
# KEDA ScaledObject — EKS / AKS
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: redwood-llm-scaler
  namespace: inference
spec:
  scaleTargetRef:
    name: redwood-llm-predictor
  minReplicaCount: 1
  maxReplicaCount: 4
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring.svc:9090
        metricName: vllm_request_queue_depth
        threshold: "5"          # scale up when > 5 requests queued per pod
        query: |
          sum(vllm_request_queue_depth{service="redwood-llm"})
          / count(up{job="redwood-llm"})
```

Scale-up is fast (new pod pre-loads model weights in ~60s from S3/ODF for
a 7B AWQ model). Scale-to-zero is disabled — cold start latency (60s+)
violates the sync agent SLA. Minimum 1 replica always running.

### GPU Quota Guardrails

Set hard `ResourceQuota` on the `inference` namespace to prevent unconstrained
scaling from exhausting cluster-wide GPU allocation:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: inference-gpu-quota
  namespace: inference
spec:
  hard:
    requests.nvidia.com/gpu: "8"    # max 8 GPUs across all pods in namespace
    limits.nvidia.com/gpu: "8"
```

---

## 11. Regulatory and Audit Controls

### Prompt and Completion Logging

**What must be logged:**

| Field | Purpose | Retention |
|---|---|---|
| `claim_id` / `quote_id` | Links inference call to the regulated decision | 7 years (DOI requirement) |
| `model_name` + `model_version` | Identifies which model generated the output | 7 years |
| `node_name` | Which agent node triggered the call | 7 years |
| `input_token_count` + `output_token_count` | Cost attribution and audit | 90 days |
| `latency_ms` | SLA monitoring | 30 days |

**What must NOT be logged:**

| Field | Reason |
|---|---|
| Raw prompt content containing PII | PII in logs violates data handling policy |
| Claimant name, address, SSN, DOB | PII — redact before any logging |
| Raw policy document chunks retrieved | May contain PII; log document ID + page range instead |
| LLM completion containing PII | Redact using PII detection before writing to log store |

The `narrative_llm_node` and `quote_gen_node` must pass their inputs through
the `pii_redactor` utility before writing to the inference audit log.
Completion text is stored in the claim/quote state (checkpointed to Postgres),
not in the inference log — the inference log records metadata only.

### Model Change Freeze

For carriers under DOI filing or active adverse action exposure:

- No LLM model version changes are permitted during the 30-day window before
  and after a rate filing effective date
- Model version changes require a regression test run against the regulatory
  test fixture (minimum 500 sample claims with known outcomes) before promotion
- The champion-challenger system (section 12) is used to validate a new model
  version before it becomes the serving version

### Adverse Action Defensibility

If an adverse action (declination, premium surcharge) is generated with LLM
involvement (e.g., the quote explanation node produces a reason code), the
following must be retrievable from the audit log:

1. `model_name` + `model_version` at time of decision
2. Prompt template version (stored in git, referenced by tag in the log)
3. Retrieved document IDs and relevance scores (not document text — use ID + page)
4. Structured output parsed from the LLM completion (`PydanticOutputParser` result)
5. Human override flag — whether `underwriter_decision` overrode the LLM output

---

## 12. Champion-Challenger for LLMs

LLM quality in production is evaluated differently from tabular models. There
is no single AUC metric. Evaluation uses:

- **Task-specific rubrics** (LLM-as-judge): a second LLM evaluates the output
  on correctness, grounding (did the response cite retrieved document content?),
  and tone (appropriate for insurance claimant communication)
- **Human spot-check rate**: investigators and underwriters flag low-quality
  LLM outputs via the copilot UI; flag rate is tracked per model version
- **Override rate**: if underwriters override the quote explanation at a higher
  rate for a new model version, the challenger loses

### Shadow Mode Deployment

```
Incoming request to quote_gen_node
      │
      ├── Champion (current serving model)
      │       Handles actual request
      │       Response returned to agent state
      │
      └── Challenger (candidate model version)
              Receives a copy of the same prompt
              Response written to shadow log only
              NOT returned to agent — no user impact
              Evaluated offline by LLM-as-judge rubric
```

Shadow mode runs for a minimum of 500 requests before a promotion decision.
The shadow log is stored in the same audit store as the main inference log,
keyed by `model_version=challenger`.

### Promotion Criteria

| Metric | Champion threshold | Challenger must meet |
|---|---|---|
| LLM-as-judge score (grounding) | Baseline | ≥ baseline − 2pp |
| LLM-as-judge score (correctness) | Baseline | ≥ baseline − 1pp |
| Human flag rate | Baseline | ≤ baseline + 0.5pp |
| p95 latency | Baseline | ≤ baseline + 500ms |
| Override rate (UW/investigator) | Baseline | ≤ baseline + 1pp |

If the challenger fails any criterion, it is rejected. Model promotions require
approval from the ML platform lead and, for regulated carriers, the compliance
officer.

---

## 13. Substrate Inference Map

| Inference Component | AWS (EKS) | Azure (AKS) | OpenShift (RHOAI) |
|---|---|---|---|
| LLM serving (generative) | KServe + vLLM on GPU nodepool | KServe + vLLM on GPU nodepool | RHOAI KServe + vLLM ServingRuntime |
| vLLM container image | `vllm/vllm-openai:latest` (pinned tag) | `vllm/vllm-openai:latest` (pinned) | `quay.io/modh/vllm:rhoai-2.16` |
| Structured extraction engine | KServe + SGLang (RawDeployment) | KServe + SGLang (RawDeployment) | vLLM guided_json (no SGLang custom runtime by default) |
| SGLang container image | `lmsysorg/sglang:<pinned>` | `lmsysorg/sglang:<pinned>` | Custom ServingRuntime — platform team approval required |
| Model storage | S3 (VPC endpoint) | Azure Blob (private endpoint) | ODF S3-compatible (internal) |
| Embedding model | KServe + vLLM (`/v1/embeddings`) | KServe + vLLM (`/v1/embeddings`) | RHOAI KServe + vLLM |
| Reranker | Self-hosted BGE or Cohere API | Self-hosted BGE or Cohere API | Self-hosted BGE (air-gapped) |
| Vector store (pgvector) | RDS PostgreSQL | Azure Database for PostgreSQL | CrunchyData PGO |
| Autoscaling | KEDA + Prometheus | KEDA + Prometheus | OpenShift HPA + user workload monitoring |
| Service mesh | Istio (AWS) | Istio (Azure) or OSM | OpenShift Service Mesh (Istio-based) |
| GPU nodepool taint | `inference=true:NoSchedule` | `inference=true:NoSchedule` | `inference=true:NoSchedule` |
| Inference audit log | S3 + Athena | ADLS + Synapse | ODF + Trino |

**LLM endpoint** (`LLM_BASE_URL`) resolves to a ClusterIP service on all
three substrates — agents never call a GPU pod directly. KServe manages the
routing and load-balancing across replicas.

---

## 14. SGLang as the Structured-Output Inference Engine

SGLang (Structured Generation Language) is an inference engine and serving framework optimized for workloads where structured output, shared-prefix caching, and multi-step inference programs are the primary requirements. It exposes an OpenAI-compatible API and is a first-class option in the Redwood inference stack for use cases where vLLM's general-purpose design is not optimal.

### Why SGLang Exists Alongside vLLM

vLLM is the default self-hosted engine. SGLang is added for specific workload characteristics that vLLM serves less efficiently:

| Capability | vLLM | SGLang |
|---|---|---|
| Free-form generation (summaries, narratives) | Excellent | Good |
| Structured JSON output (constrained decoding) | Good — guided decode overhead ~15–30% | Excellent — native, minimal overhead |
| Shared-prefix KV caching | Block-level prefix cache | RadixAttention — fine-grained radix tree |
| Embedding model serving | Yes (`/v1/embeddings`) | No — use vLLM for all embedding endpoints |
| RHOAI / Red Hat validated image | Yes (`quay.io/modh/vllm`) | No — custom `ServingRuntime` required |
| AWQ quantization | Yes | Limited — verify current SGLang release notes |
| FP8 / GPTQ quantization | Yes | Yes |
| Multi-step inference programs | Not native | Yes (`sgl.function` Python frontend) |
| Production maturity (2026-Q2) | High | Medium-High |

**Rule:** vLLM is the default for all inference workloads. SGLang is deployed only where constrained JSON output or RadixAttention provides a measurable advantage for insurance use cases. Never replace the embedding or reranking `InferenceService` with SGLang — vLLM is the correct engine for those endpoints.

### SGLang Differentiators for Insurance Workloads

**RadixAttention**

Insurance agent workloads share long common prefixes: the same policy document chunk appears in many sequential requests from the same investigator session; the same underwriting rule set is prefixed to every quote explanation request. SGLang's radix tree reuses these prefix KV states across requests at a finer granularity than vLLM's block-aligned cache, reducing time-to-first-token for prefix-heavy workloads by 30–60%.

**Native Constrained Decoding**

FNOL extraction and coverage determination require structured output: claim detail JSON, coverage eligibility flags, risk factor arrays. vLLM's `guided_json` parameter invokes Outlines decoding with measurable per-token overhead; SGLang integrates constraint-guided generation at the engine level. For schemas with 20+ fields, SGLang's structured output throughput is 2–4× higher than vLLM's guided decode mode.

**sgl.function Programs (Optional, Async Only)**

Multi-step extraction (parse → validate → enrich) can be expressed as `sgl.function` programs that run entirely within the SGLang server process, avoiding round-trip latency between steps. This is optional and applies only to the async FNOL extraction pipeline — not the sync agent path.

### SGLang Configuration Reference

SGLang is launched as a Python module. Key parameter names differ from vLLM:

```bash
python -m sglang.launch_server \
  --model-path /mnt/models \
  --served-model-name redwood-extraction-llm \
  --tensor-parallel-size 1 \
  --mem-fraction-static 0.85 \
  --max-prefill-tokens 16384 \
  --schedule-conservativeness 0.9 \
  --host 0.0.0.0 \
  --port 8000
```

| Parameter | Value | Rationale |
|---|---|---|
| `--model-path` | Internal registry path | Model weights location (same air-gapped pull pattern as vLLM) |
| `--served-model-name` | Canonical alias | Agent env var references alias, not model path |
| `--tensor-parallel-size` | 1 or 2 | Match to GPU allocation per `InferenceService` |
| `--mem-fraction-static` | `0.85` | Equivalent to vLLM's `--gpu-memory-utilization`; static KV cache allocation |
| `--max-prefill-tokens` | `16384` | Max tokens processed in a single prefill batch; tune to available VRAM |
| `--schedule-conservativeness` | `0.9` | Memory reservation margin; increase toward 1.0 if OOM risk is observed |
| `--enable-torch-compile` | (flag only) | torch.compile kernel optimization — A100/H100 only; skip on V100 or A10G |

**Quantization Note (2026-Q2):** SGLang FP8 quantization is production-ready on A100/H100. AWQ support is present but less battle-tested than vLLM. For memory-constrained nodes where AWQ is required, prefer vLLM. Verify current SGLang release notes before deploying quantized models on SGLang.

### KServe InferenceService — SGLang

SGLang does not have a pre-built KServe `ServingRuntime` on RHOAI (2026-Q2). On EKS and AKS, use a `RawDeployment` InferenceService with a custom container:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: redwood-extraction-llm
  namespace: inference
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    containers:
      - name: kserve-container
        image: lmsysorg/sglang:v0.4.1-cu124   # pin specific tag; never use :latest
        command: ["python", "-m", "sglang.launch_server"]
        args:
          - "--model-path=/mnt/models"
          - "--served-model-name=redwood-extraction-llm"
          - "--tensor-parallel-size=1"
          - "--mem-fraction-static=0.85"
          - "--max-prefill-tokens=16384"
          - "--host=0.0.0.0"
          - "--port=8080"
        ports:
          - containerPort: 8080
            protocol: TCP
        resources:
          requests:
            nvidia.com/gpu: "1"
            memory: "20Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: "1"
            memory: "24Gi"
        volumeMounts:
          - name: model-dir
            mountPath: /mnt/models
    volumes:
      - name: model-dir
        persistentVolumeClaim:
          claimName: redwood-extraction-model-pvc
    tolerations:
      - key: "inference"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

**On RHOAI:** SGLang requires a custom `ServingRuntime` resource. Incidents with this runtime are not covered under RHOAI support contracts. Default to vLLM on RHOAI. Only deploy SGLang on RHOAI if the platform team has explicitly approved the custom runtime and accepted the support boundary.

### Environment Variables for SGLang Endpoint

The agent factory (`get_llm()`) requires no changes. Agent nodes that perform structured extraction reference a second env var set pointing to the SGLang endpoint:

```bash
# Primary LLM — vLLM (long-form generation, investigator summaries)
LLM_BASE_URL=http://vllm-llm-service.inference.svc.cluster.local:8000/v1
LLM_MODEL=redwood-llm
LLM_API_KEY=none

# Extraction LLM — SGLang (structured JSON output)
EXTRACTION_LLM_BASE_URL=http://sglang-extraction-service.inference.svc.cluster.local:8000/v1
EXTRACTION_LLM_MODEL=redwood-extraction-llm
EXTRACTION_LLM_API_KEY=none
```

Agent nodes that require structured output call via `EXTRACTION_LLM_BASE_URL`. All other nodes use `LLM_BASE_URL`. Both endpoints use the same OpenAI client SDK — the extraction endpoint adds `response_format={"type": "json_schema", ...}` on the client side.

### DNS Naming — SGLang Service

```bash
# Structured extraction engine (SGLang)
sglang-extraction-service.inference.svc.cluster.local:8000
```

The SGLang service lives in the same `inference` namespace as vLLM services. The same `AuthorizationPolicy` (agents and rag namespaces only), mTLS, and `PeerAuthentication: STRICT` apply.

---

## 15. Customer-Based Use Cases — vLLM vs SGLang

Four customer archetypes define the realistic deployment scenarios for the Redwood inference stack. Each maps to a specific inference engine configuration based on regulatory standing, workload shape, and substrate constraints.

---

### Archetype 1: State-Mandated Auto Fund (Regulated, Air-Gapped, RHOAI)

**Profile:** State-operated auto insurance fund under direct DOI oversight. No internet egress permitted. OpenShift AI (RHOAI) on bare-metal or private cloud. Every model inference call is subject to regulatory examination.

**LLM Use Cases:**
- FNOL narrative inconsistency detection (async batch, nightly runs)
- Investigator copilot: long-form case summary generation
- Claim denial letter drafting (long-form; reviewed by human before send)

**Inference Engine: vLLM only**

| Decision Factor | Details |
|---|---|
| Platform | RHOAI — vLLM `ServingRuntime` is built-in and Red Hat supported |
| Model storage | ODF (internal S3-compatible) — no egress required |
| Quantization | AWQ 4-bit — GPU quota is limited; must fit 13B model on 1× A100 40GB |
| Air-gapped support | Full — model pre-staged during release pipeline; no runtime egress |
| SGLang status | Not deployed — no Red Hat validated image; compliance team has not approved custom runtime |

```bash
LLM_BASE_URL=http://vllm-llm-service.inference.svc.cluster.local:8000/v1
LLM_MODEL=redwood-llm        # Mistral-7B-Instruct AWQ on single A100 40GB
LLM_API_KEY=none
```

All FNOL extraction uses structured prompting with vLLM's `guided_json` parameter and Pydantic validation on the output. The guided decode overhead (~15–30%) is acceptable because FNOL extraction is async — there is no sync SLA for this path.

---

### Archetype 2: Regional P&C Carrier (Semi-Regulated, AWS EKS, Dual-Engine)

**Profile:** Mid-size regional P&C carrier on AWS EKS with GPU nodepool. Sensitive model outputs stay self-hosted; the security team prefers self-hosted for all LLM traffic during the initial deployment phase. Internet egress is allowed but not used for LLM inference.

**LLM Use Cases:**
- Quote explanation generation — strict JSON: `reason_codes[]`, `risk_factors[]`, `premium_delta`
- FNOL intake detail extraction — strict JSON: `loss_type`, `vehicle_damage`, `injury_claim`, `witness_count`
- Investigator copilot: case narrative summary (long-form, free text)
- Policy coverage Q&A: semi-structured (`coverage_applies: bool` + `rationale: str`)

**Inference Engine: vLLM (generative) + SGLang (structured extraction)**

| InferenceService | Engine | Use Case | Model |
|---|---|---|---|
| `redwood-llm` | vLLM | Investigator summaries, copilot long-form | Mistral-7B-Instruct FP16 |
| `redwood-extraction-llm` | SGLang | FNOL extraction, quote explanation JSON | Mistral-7B-Instruct FP16 |
| `redwood-embedding` | vLLM | BGE-M3 embeddings for RAG | BGE-M3 |

**Why dual-engine:**
The quote explanation node has an 8s sync SLA. vLLM's `guided_json` decode overhead pushed p95 latency to ~2.8s for the LLM call alone on a 20-field output schema. SGLang's native constrained decoding holds p95 at ~1.6s for the same schema. FNOL extraction is async but runs at high volume (thousands of claims/hour); SGLang's 2–4× throughput advantage on structured output reduces GPU-hours required. Investigator copilot generates 800–1200 token free-form summaries where vLLM PagedAttention handles concurrent sessions more efficiently than SGLang.

```bash
LLM_BASE_URL=http://vllm-llm-service.inference.svc.cluster.local:8000/v1
LLM_MODEL=redwood-llm
EXTRACTION_LLM_BASE_URL=http://sglang-extraction-service.inference.svc.cluster.local:8000/v1
EXTRACTION_LLM_MODEL=redwood-extraction-llm
EMBEDDING_BASE_URL=http://vllm-embedding-service.inference.svc.cluster.local:8000/v1
EMBEDDING_MODEL=redwood-embedding
```

---

### Archetype 3: Digital-Native InsurTech (Cloud-Native, Azure, Hybrid)

**Profile:** InsurTech startup offering instant digital quotes. Azure-native. Time-to-market is the primary constraint; self-hosted inference is not required but is evaluated for cost optimization at scale. Primary LLM traffic can egress to Azure OpenAI.

**LLM Use Cases:**
- High-volume quote explanation: 50,000+ quotes/day, structured JSON output required
- Coverage chatbot: conversational, free-form responses to policyholder questions
- Document parsing: policy PDF to structured coverage summary

**Inference Engine: Azure OpenAI (primary) + SGLang on AKS (cost-optimization path)**

| Path | Engine | When Used | Rationale |
|---|---|---|---|
| Azure OpenAI GPT-4o | Cloud API | Primary — all production traffic | No GPU management; managed SLA |
| SGLang on AKS GPU nodepool | Self-hosted | Cost optimization at >40K structured requests/day | Self-hosted TCO below Azure OpenAI per-token cost at this volume |

**Cost decision trigger:** At quote volumes exceeding 40,000/day, Azure OpenAI per-token cost at standard pricing exceeds the amortized cost of a dedicated `Standard_NC24ads_A100_v4` AKS node running SGLang. Re-evaluate the break-even quarterly as Azure OpenAI pricing tiers and reservation discounts change.

**SGLang rationale over vLLM for this archetype:** ~85% of the workload is structured output (quote JSON, coverage determination JSON). SGLang's throughput advantage for constrained decoding is the dominant factor. There is no RHOAI constraint and no AWQ requirement — the A100 80GB fits 7B FP16 with headroom.

```bash
# Azure OpenAI (primary)
LLM_BASE_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>/
LLM_MODEL=gpt-4o
LLM_API_KEY=${AZURE_OPENAI_KEY}

# SGLang on AKS (cost-optimization path — activated by EXTRACTION_BACKEND env var)
EXTRACTION_LLM_BASE_URL=http://sglang-extraction-service.inference.svc.cluster.local:8000/v1
EXTRACTION_LLM_MODEL=redwood-extraction-llm
EXTRACTION_LLM_API_KEY=none
EXTRACTION_BACKEND=azure_openai   # switch to: sglang when cost threshold is crossed
```

Traffic routing between Azure OpenAI and SGLang is controlled by `EXTRACTION_BACKEND` read in `get_extraction_llm()` — no agent code changes required to switch backends.

---

### Archetype 4: National Multi-Line Carrier (Multi-Substrate, Hybrid Engine Strategy)

**Profile:** Top-10 P&C carrier with multiple lines of business across substrates. Life/health lines on RHOAI (regulatory requirement: data never leaves private cloud). P&C lines on AWS EKS with controlled internet egress. Bedrock available for non-sensitive P&C use cases.

**Use Cases by Line of Business:**

| Line | Use Case | Volume | Latency Requirement |
|---|---|---|---|
| Auto P&C | FNOL structured extraction | 200K claims/month | Async |
| Auto P&C | Quote explanation JSON | 500K quotes/month | Sync, 8s |
| Auto P&C | Investigator copilot summary | 15K sessions/month | Sync, 10s |
| Life | Policy coverage analysis (structured) | 50K/month | Async |
| Health | Prior auth determination (structured) | 80K/month | Async |
| Shared | Embedding (RAG for all lines) | 2M/day | <30ms |

**Inference Engine Strategy by Substrate:**

| Substrate | Line | Engine | Rationale |
|---|---|---|---|
| RHOAI (private cloud) | Life, Health | vLLM only | No SGLang custom runtime approval; Red Hat support boundary |
| EKS (P&C) | Auto investigator copilot | vLLM | Long-form free-form generation; PagedAttention handles concurrent sessions |
| EKS (P&C) | Auto FNOL extraction + quote JSON | SGLang | High-volume structured output; 2–4× throughput advantage justifies separate `InferenceService` |
| EKS (shared) | Embedding (all RAG) | vLLM | SGLang does not serve `/v1/embeddings`; vLLM is the only correct engine here |
| Bedrock (P&C optional) | Low-stakes coverage Q&A chatbot | Bedrock (Claude Haiku) | Sub-cent cost per interaction; eliminates GPU requirement for conversational path |

**Scale rationale for SGLang on EKS P&C:** At 500K structured quote explanations/month (~17K/day), SGLang's throughput advantage over vLLM guided decode reduces the required GPU replicas from 3 to 2 at peak, saving approximately $4K/month in A100 40GB spot instance cost. The operational cost of a second `InferenceService` is justified only at this volume — below ~10K structured requests/day, vLLM guided decode is adequate and the simpler single-engine deployment is preferred.

**Multi-substrate env var matrix:**

```bash
# RHOAI substrate (life/health) — vLLM only
LLM_BASE_URL=http://vllm-llm-service.inference.svc.cluster.local:8000/v1
LLM_MODEL=redwood-llm-life
EMBEDDING_BASE_URL=http://vllm-embedding-service.inference.svc.cluster.local:8000/v1
EMBEDDING_MODEL=redwood-embedding

# EKS substrate (P&C) — vLLM + SGLang
LLM_BASE_URL=http://vllm-llm-service.inference.svc.cluster.local:8000/v1
LLM_MODEL=redwood-llm-pc
EXTRACTION_LLM_BASE_URL=http://sglang-extraction-service.inference.svc.cluster.local:8000/v1
EXTRACTION_LLM_MODEL=redwood-extraction-llm
EMBEDDING_BASE_URL=http://vllm-embedding-service.inference.svc.cluster.local:8000/v1
EMBEDDING_MODEL=redwood-embedding
```

Different substrates receive different Kubernetes `ConfigMap` / secret values at deploy time. Agent and RAG code is identical across substrates — only env vars change.

---

*Architecture version: 2026-Q2 | Applies to: `enterprise-rag-platform/`, `fnol-claims-multi-agent-system/`, `intelligent-underwriting-platform/`*
*See also: [DEPLOYMENT_TOPOLOGIES.md](./DEPLOYMENT_TOPOLOGIES.md) · [Multi_Agent_Architecture.md](./Multi_Agent_Architecture.md) · [DECISION_LOG.md](./DECISION_LOG.md)*
