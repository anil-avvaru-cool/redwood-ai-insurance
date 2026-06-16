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
│  │ Azure OpenAI     │      │ KServe InferenceService       │ │
│  │ Anthropic API    │      │ RHOAI / EKS / AKS             │ │
│  └──────────────────┘      └───────────────────────────────┘ │
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
| LLM serving | KServe + vLLM on GPU nodepool | KServe + vLLM on GPU nodepool | RHOAI KServe + vLLM ServingRuntime |
| vLLM container image | `vllm/vllm-openai:latest` (pinned tag) | `vllm/vllm-openai:latest` (pinned) | `quay.io/modh/vllm:rhoai-2.16` |
| Model storage | S3 (VPC endpoint) | Azure Blob (private endpoint) | ODF S3-compatible (internal) |
| Embedding model | Same KServe / vLLM pattern | Same | Same |
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

*Architecture version: 2026-Q2 | Applies to: `enterprise-rag-platform/`, `fnol-claims-multi-agent-system/`, `intelligent-underwriting-platform/`*
*See also: [DEPLOYMENT_TOPOLOGIES.md](./DEPLOYMENT_TOPOLOGIES.md) · [Multi_Agent_Architecture.md](./Multi_Agent_Architecture.md) · [DECISION_LOG.md](./DECISION_LOG.md)*
