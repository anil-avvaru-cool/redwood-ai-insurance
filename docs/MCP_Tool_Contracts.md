# MCP Tool Contracts
### Redwood AI Insurance — Agent Tool Layer

> **Scope:** Defines the MCP (Model Context Protocol) server tools consumed by
> DocumentVerificationAgent and SubrogationAgent. Each tool is an independently
> versioned, testable unit that replaces a direct Python import inside a graph node.
> Tools with cross-agent reuse potential are identified in the Shared Tools section.

---

## Table of Contents
1. [Tool Design Rules](#1-tool-design-rules)
2. [DocumentVerificationAgent Tools](#2-documentverificationagent-tools)
   - [exif_analyzer](#21-exif_analyzer)
   - [perceptual_hash_checker](#22-perceptual_hash_checker)
   - [repair_estimate_validator](#23-repair_estimate_validator)
   - [narrative_consistency_checker](#24-narrative_consistency_checker)
3. [SubrogationAgent Tools](#3-subrogationagent-tools)
   - [police_report_api](#31-police_report_api)
   - [telematics_crash_query](#32-telematics_crash_query)
   - [evidence_collector](#33-evidence_collector)
   - [recovery_demand_queue](#34-recovery_demand_queue)
4. [Shared Tool Registry](#4-shared-tool-registry)
5. [Versioning and Breaking Change Policy](#5-versioning-and-breaking-change-policy)
6. [Error Contract](#6-error-contract)

---

## 1. Tool Design Rules

- **Tools are pure I/O boundaries.** A tool receives a typed input, calls an external
  system or runs a bounded computation, and returns a typed output. No LangGraph
  state is imported inside a tool.
- **Tools are node-agnostic.** A node calls a tool by name and version. The tool
  does not know which graph or node invoked it.
- **Tools fail fast.** On missing required config (`os.environ["KEY"]`) or
  upstream API failure, tools raise — they do not return degraded results silently.
- **Tools are substrate-agnostic.** External API credentials are injected via
  environment variables. Tools never hard-code endpoints.
- **Versioning:** Each tool carries a `version` field in its manifest. Minor versions
  are backwards-compatible. Major versions require explicit opt-in by callers.

---

## 2. DocumentVerificationAgent Tools

Node-to-tool mapping for `DocumentVerificationAgent`:

| Node | Tool(s) Called |
|---|---|
| `photo_auth_node` | `exif_analyzer`, `perceptual_hash_checker` |
| `estimate_check_node` | `repair_estimate_validator` |
| `narrative_check_node` | `narrative_consistency_checker` |

---

### 2.1 `exif_analyzer`

**Purpose:** Extracts and validates EXIF metadata from a submitted photo to detect
signs of forgery: GPS coordinate mismatches, timestamp anomalies, or device
fingerprint inconsistencies relative to the claimed incident.

**Owner:** `doc-verification-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class ExifAnalyzerInput(BaseModel):
    document_uri: str          # presigned S3/blob URI to the image
    claim_id: str
    incident_date: date        # expected date of loss — compared to EXIF timestamp
    incident_zip_code: str     # expected location — compared to EXIF GPS
```

#### Output Schema

```python
class ExifAnalyzerOutput(BaseModel):
    exif_present: bool
    gps_coords: tuple[float, float] | None   # (lat, lon) from EXIF
    exif_timestamp: datetime | None
    device_model: str | None
    timestamp_delta_days: float | None       # |EXIF date − incident_date|
    gps_distance_miles: float | None         # distance from incident_zip centroid
    flags: list[str]                         # e.g. ["TIMESTAMP_MISMATCH", "GPS_OUTSIDE_INCIDENT_AREA"]
    forged: bool                             # True if any flag present
```

#### Flag Definitions

| Flag | Trigger Condition |
|---|---|
| `NO_EXIF` | Image has no EXIF block — stripped metadata is itself a signal |
| `TIMESTAMP_MISMATCH` | `timestamp_delta_days > EXIF_TIMESTAMP_TOLERANCE_DAYS` (config) |
| `GPS_OUTSIDE_INCIDENT_AREA` | `gps_distance_miles > EXIF_GPS_RADIUS_MILES` (config) |
| `DEVICE_FINGERPRINT_MISMATCH` | Device model inconsistent with prior submissions on this policy |

#### Error Cases

| Condition | Behavior |
|---|---|
| URI not accessible (403/404) | Raise `ToolInputError("document_uri not accessible")` |
| Image format not supported | Raise `ToolInputError("unsupported image format: {fmt}")` |
| External geocoding service unavailable | Raise `ToolDependencyError("geocoding service unavailable")` |

---

### 2.2 `perceptual_hash_checker`

**Purpose:** Computes a perceptual hash (pHash) of the submitted image and queries
the claim image index to detect reused or duplicate photos across claims on the
same or different policies.

**Owner:** `doc-verification-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class PerceptualHashInput(BaseModel):
    document_uri: str
    claim_id: str
    policy_id: str
```

#### Output Schema

```python
class PerceptualHashOutput(BaseModel):
    phash: str                              # hex-encoded 64-bit pHash
    duplicate_claim_ids: list[str]          # claims where phash distance < threshold
    duplicate_policy_ids: list[str]         # policies associated with duplicates
    reuse_detected: bool
    max_similarity_score: float | None      # 0.0 (no match) – 1.0 (identical)
```

#### Similarity Threshold

`PHASH_SIMILARITY_THRESHOLD` in `config.py`. Default: `0.95`. Hashes with
similarity ≥ threshold are considered duplicates.

#### Error Cases

| Condition | Behavior |
|---|---|
| URI not accessible | Raise `ToolInputError` |
| Image too small to hash (< 64px) | Return `reuse_detected=False`, `phash=None`, flag in logs |
| Hash index service unavailable | Raise `ToolDependencyError("phash index unavailable")` |

---

### 2.3 `repair_estimate_validator`

**Purpose:** Validates a submitted repair estimate against regional historical
actuals for the vehicle make/model/year. Flags estimates that deviate materially
from the regional distribution — both high (inflation) and low (staged minor damage
concealing total loss).

**Owner:** `doc-verification-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class RepairEstimateInput(BaseModel):
    estimate_amount_usd: float
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    damage_description: str                 # free-text from FNOL
    incident_zip_code: str                  # drives regional actuals lookup
```

#### Output Schema

```python
class RepairEstimateOutput(BaseModel):
    regional_p25_usd: float
    regional_median_usd: float
    regional_p75_usd: float
    deviation_pct: float                    # (estimate − median) / median
    flagged: bool
    flag_direction: str | None              # "HIGH" | "LOW" | None
    comparable_count: int                   # number of comparables in regional dataset
```

#### Flag Conditions

- `flagged=True, flag_direction="HIGH"` when `deviation_pct > ESTIMATE_HIGH_DEVIATION_PCT` (config)
- `flagged=True, flag_direction="LOW"` when `deviation_pct < -ESTIMATE_LOW_DEVIATION_PCT` (config)
- `comparable_count < ESTIMATE_MIN_COMPARABLES` (config): return result with
  `flagged=False` and a `low_confidence` annotation — do not flag on thin data

#### Error Cases

| Condition | Behavior |
|---|---|
| `vehicle_year` outside supported range | Raise `ToolInputError` |
| `incident_zip_code` not in regional dataset | Return result with `comparable_count=0`, `flagged=False` |
| Actuals DB unavailable | Raise `ToolDependencyError` |

---

### 2.4 `narrative_consistency_checker`

**Purpose:** Rule-based check that the `damage_description` in the FNOL is
consistent with the declared `loss_type`. Detects cases where the narrative
describes a loss event incompatible with the claimed coverage (e.g., flood damage
claimed as collision, or theft narrative inconsistent with comprehensive coverage).

**Owner:** `doc-verification-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class NarrativeCheckInput(BaseModel):
    loss_type: str                          # e.g. "COLLISION", "COMPREHENSIVE", "LIABILITY"
    damage_description: str
    declared_damages: list[str]             # structured damage codes from FNOL
    coverage_types: list[str]               # active coverages on the policy
```

#### Output Schema

```python
class NarrativeCheckOutput(BaseModel):
    consistent: bool
    inconsistency_reasons: list[str]        # human-readable rule violations
    coverage_mismatch: bool                 # damage type not covered by active coverages
    confidence: float                       # 0.0–1.0; low if description is ambiguous
```

#### Implementation Note

This tool is rule-based (not LLM-based). Rules are defined in
`config/narrative_rules.yaml` — a versioned file that maps `(loss_type, damage_keywords)`
pairs to consistency verdicts. LLM narrative analysis for the FNOL async path
(`narrative_llm_node`) is separate and not part of this tool.

---

## 3. SubrogationAgent Tools

Node-to-tool mapping for `SubrogationAgent`:

| Node | Tool(s) Called |
|---|---|
| `liability_scoring_node` | `police_report_api`, `telematics_crash_query` |
| `evidence_assembly_node` | `evidence_collector` |
| `subrogation_referral_node` | `recovery_demand_queue` |

---

### 3.1 `police_report_api`

**Purpose:** Fetches a structured police report by reference ID. Extracts at-fault
party determination, fault percentage, citations issued, and report metadata.
Used by SubrogationAgent for liability assessment and potentially by FNOL's
async evidence path.

**Owner:** `claims-data-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class PoliceReportInput(BaseModel):
    police_report_ref: str          # report ID or external reference number
    claim_id: str                   # for audit logging
```

#### Output Schema

```python
class PoliceReportOutput(BaseModel):
    report_found: bool
    at_fault_party: str | None      # "INSURED" | "THIRD_PARTY" | "SHARED" | "UNDETERMINED"
    insured_at_fault_pct: float | None   # 0.0–1.0; insured's share of fault
    confidence: float | None             # derived from report completeness
    report_date: date | None
    citations_issued: list[str]          # citation codes
    jurisdiction: str | None             # state code
    narrative_summary: str | None        # officer narrative excerpt (max 500 chars)
```

#### Error Cases

| Condition | Behavior |
|---|---|
| Report reference not found | Return `report_found=False`, all other fields `None` |
| External report API unavailable | Raise `ToolDependencyError("police report API unavailable")` |
| Report access denied (jurisdiction restriction) | Raise `ToolInputError("report access denied: {jurisdiction}")` |

---

### 3.2 `telematics_crash_query`

**Purpose:** Fetches crash event data from the telematics provider by crash
reference ID. Provides an independent at-fault signal based on vehicle kinematics
at impact time — corroborates or contradicts the police report.

**Owner:** `claims-data-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class TelematicsCrashInput(BaseModel):
    telematics_crash_ref: str       # crash event ID from the telematics provider
    claim_id: str
```

#### Output Schema

```python
class TelematicsCrashOutput(BaseModel):
    crash_found: bool
    impact_timestamp: datetime | None
    speed_at_impact_mph: float | None
    impact_force_g: float | None
    direction_of_impact: str | None     # "FRONT" | "REAR" | "SIDE" | "ROLLOVER"
    insured_at_fault_pct: float | None  # kinematic-derived fault estimate
    confidence: float | None
    data_quality_score: float | None    # 0.0–1.0; signal completeness
```

#### Error Cases

| Condition | Behavior |
|---|---|
| Crash ref not found | Return `crash_found=False`, all other fields `None` |
| Telematics provider unavailable | Raise `ToolDependencyError` |
| Data older than `TELEMATICS_MAX_AGE_DAYS` (config) | Return result with `data_quality_score` penalized |

---

### 3.3 `evidence_collector`

**Purpose:** Retrieves and scores the completeness of evidence associated with a
settled claim: photos, witness statements, police report, telematics data. Produces
a structured evidence package and a `evidence_score` representing completeness.

**Owner:** `claims-data-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class EvidenceCollectorInput(BaseModel):
    claim_id: str
    policy_id: str
    evidence_types: list[str]       # ["PHOTOS", "POLICE_REPORT", "WITNESS_STATEMENTS", "TELEMATICS"]
```

#### Output Schema

```python
class EvidenceCollectorOutput(BaseModel):
    evidence_refs: list[str]            # URIs to collected evidence artifacts
    evidence_score: float               # 0.0–1.0; completeness score
    present_types: list[str]            # evidence types that were found
    missing_types: list[str]            # requested types not found
    collection_notes: list[str]         # e.g. "POLICE_REPORT: jurisdiction redaction applied"
```

#### Scoring Rules

`evidence_score` is computed as a weighted sum defined in `config/evidence_weights.yaml`.
Default weights: `PHOTOS=0.25`, `POLICE_REPORT=0.35`, `WITNESS_STATEMENTS=0.20`,
`TELEMATICS=0.20`. Weights sum to 1.0.

---

### 3.4 `recovery_demand_queue`

**Purpose:** Enqueues a recovery demand to the subrogation recovery team's case
management system. Returns a `referral_id` for tracking. Called only after
HITL approval (high-value) or automatic routing (below high-value threshold).

**Owner:** `subrogation-mcp-server`
**Version:** `1.0`

#### Input Schema

```python
class RecoveryDemandInput(BaseModel):
    claim_id: str
    policy_id: str
    recovery_amount_usd: float
    at_fault_party: str             # third-party name or entity ID
    at_fault_party_insurer: str | None
    evidence_refs: list[str]
    liability_confidence: float
    approved_by: str | None         # senior handler ID if HITL-approved; None if auto-routed
    approval_notes: str | None
```

#### Output Schema

```python
class RecoveryDemandOutput(BaseModel):
    referral_id: str                        # case management system ID
    queued_at: datetime
    estimated_processing_days: int          # SLA estimate from case management system
    demand_letter_template_id: str | None   # pre-selected letter template if available
```

#### Error Cases

| Condition | Behavior |
|---|---|
| Case management system unavailable | Raise `ToolDependencyError("recovery queue unavailable")` |
| `recovery_amount_usd <= 0` | Raise `ToolInputError("recovery_amount_usd must be positive")` |
| `approved_by` required but missing for high-value demand | Raise `ToolInputError("high-value demand requires approved_by")` |

---

## 4. Shared Tool Registry

Tools that are candidates for use by more than one agent are tracked here.
Cross-agent tool use requires coordination on versioning and breaking changes.

| Tool | Primary Owner | Current Callers | Potential Additional Callers | Shared MCP Server |
|---|---|---|---|---|
| `police_report_api` | `claims-data-mcp-server` | `SubrogationAgent / liability_scoring_node` | `FNOLAgent / evidence_request_node` (async path) | Yes |
| `evidence_collector` | `claims-data-mcp-server` | `SubrogationAgent / evidence_assembly_node` | `FNOLAgent / investigator_copilot_node` | Yes |
| `narrative_consistency_checker` | `doc-verification-mcp-server` | `DocumentVerificationAgent / narrative_check_node` | `FNOLAgent / narrative_llm_node` (rule-based pre-filter) | Possible — evaluate before adding caller |
| `telematics_crash_query` | `claims-data-mcp-server` | `SubrogationAgent / liability_scoring_node` | `FNOLAgent / feature_enrichment_node` (async) | Possible |
| `exif_analyzer` | `doc-verification-mcp-server` | `DocumentVerificationAgent / photo_auth_node` | `FNOLAgent / image_analysis_node` | Possible — image_analysis_node currently uses ViT; evaluate overlap |
| `repair_estimate_validator` | `doc-verification-mcp-server` | `DocumentVerificationAgent / estimate_check_node` | No current candidates | No |
| `perceptual_hash_checker` | `doc-verification-mcp-server` | `DocumentVerificationAgent / photo_auth_node` | No current candidates | No |
| `recovery_demand_queue` | `subrogation-mcp-server` | `SubrogationAgent / subrogation_referral_node` | No current candidates | No |

### MCP Server Grouping

Three MCP servers cover all tools above. Tools are grouped by data access domain,
not by agent — so cross-agent sharing is a server-level config, not a code change.

| MCP Server | Tools | Owned By |
|---|---|---|
| `doc-verification-mcp-server` | `exif_analyzer`, `perceptual_hash_checker`, `repair_estimate_validator`, `narrative_consistency_checker` | DocVerification team |
| `claims-data-mcp-server` | `police_report_api`, `telematics_crash_query`, `evidence_collector` | Claims platform team |
| `subrogation-mcp-server` | `recovery_demand_queue` | Subrogation team |

---

## 5. Versioning and Breaking Change Policy

### Version format: `MAJOR.MINOR`

- **MINOR bump** — backwards-compatible: new optional output fields, relaxed
  input validation, new flag codes. Callers do not need to update.
- **MAJOR bump** — breaking: renamed fields, removed fields, changed required
  inputs, changed error contract. Callers must opt in explicitly by updating the
  tool version in their node configuration.

### Rollout procedure for MAJOR version bump

1. New version is deployed alongside the old version (parallel MCP server routes).
2. Caller nodes are updated and tested in a non-production environment.
3. Old version is deprecated with a 30-day sunset window.
4. Old version is removed after all callers have migrated.

### Tool version is pinned per agent deployment

Each agent's deployment configuration specifies the tool version it calls:

```yaml
# doc-verification-agent/config/tools.yaml
tools:
  exif_analyzer: "1.0"
  perceptual_hash_checker: "1.0"
  repair_estimate_validator: "1.0"
  narrative_consistency_checker: "1.0"
```

Version upgrades are explicit configuration changes — not automatic on server deploy.

---

## 6. Error Contract

All tools raise from one of three exception classes. Graph nodes catch these at
the node boundary and write structured error state rather than letting exceptions
propagate to the LangGraph runtime.

```python
class ToolInputError(Exception):
    """Caller passed invalid or missing input. Do not retry."""

class ToolDependencyError(Exception):
    """Upstream dependency (external API, DB) is unavailable. Retry is appropriate."""

class ToolAuthError(Exception):
    """Credentials missing or expired. Do not retry — alert ops."""
```

### Node error handling pattern

```python
async def photo_auth_node(state: DocVerificationState) -> dict:
    try:
        exif_result = await mcp_client.call_tool(
            "exif_analyzer",
            ExifAnalyzerInput(
                document_uri=state["doc_refs"][0],
                claim_id=state["claim_id"],
                incident_date=state["claim_context"]["incident_date"],
                incident_zip_code=state["claim_context"]["incident_zip_code"],
            )
        )
    except ToolInputError as e:
        return {"photo_auth_result": {"error": str(e), "forged": False, "flags": ["INPUT_ERROR"]}}
    except ToolDependencyError as e:
        return {"photo_auth_result": {"error": str(e), "forged": False, "flags": ["SERVICE_UNAVAILABLE"]}}
    return {"photo_auth_result": exif_result.model_dump()}
```

Nodes never re-raise tool exceptions into the graph. `ToolDependencyError` results
are written to state with a `SERVICE_UNAVAILABLE` flag — `doc_decision_node` treats
unavailable checks as inconclusive (not PASS or FLAG) and writes an audit annotation.

---

*Document version: 2026-Q2 | Applies to: `doc-verification-mcp-server/`, `claims-data-mcp-server/`, `subrogation-mcp-server/`*
