# Verifiable requirements

| ID | Requirement | Evidence |
|---|---|---|
| QWEN-MODEL-001 | Production execution uses exactly 28 layers, H=1024, I=3072, NH=16, NKV=8, HD=128, V=151936, and at most 40960 configured positions. | Manifest validation and full-ROM test |
| QWEN-ROM-001 | All native BF16 weights are immutable ROM constants. | RTL structure assertion and ROM hash |
| QWEN-HOST-001 | The runtime host supplies token IDs and control only. | Top-level port audit |
| QWEN-PY-001 | The production Python runtime imports only standard-library modules. | AST import audit in tests |
| QWEN-FWD-001 | Python executes every Qwen3 dense forward operation and emits all vocabulary logits. | Full-model trace and logits artifact |
| QWEN-RTL-001 | RTL executes the same complete schedule without `real`, DPI arithmetic, or omitted layers. | Lint, elaboration, and full trace |
| QWEN-NUM-001 | Python and RTL implement the same versioned arithmetic profile. | Primitive known-answer vectors |
| QWEN-EQ-001 | Every mandatory checkpoint and every final logit agree under the selected profile. | Comparison report |
| QWEN-MEM-001 | ROM and SRAM organizations are selectable behind stable interfaces. | Configuration sweep |
| QWEN-CMP-001 | Reports compare passing designs by cycles, utilization, memory traffic, area, and energy and label estimates. | Pareto report |
| QWEN-VIZ-001 | A dependency-free browser view displays all layers, computation stages, memory activity, and first mismatch. | Browser smoke test |
| QWEN-PORT-001 | The reference and simulator workflows run on Linux and macOS. | Platform instructions and CI |

A tiny fixture is permitted for fast unit and controller tests. It is never
accepted as evidence for `QWEN-MODEL-001`, `QWEN-FWD-001`, or `QWEN-EQ-001`.

## Proof status

Fast tests use shape-inferred operator fixtures and may not monkey-patch the
production `Geometry`. A full-checkpoint proof additionally requires the
published Qwen3-0.6B Safetensors file, a validated/hash-bound BF16 ROM, a
complete 28-layer trace, and an independent implementation producing every
mandatory checkpoint and all 151936 logits.

The repository now contains the candidate `qwen-ref-v2` activation/SRAM
rounding contract, generated transcendental tables, and operator vectors.
It does not contain the full-checkpoint payloads or a compliant RTL arithmetic
implementation. In particular, the current MAC flushes subnormals and is not
v2-compliant. Unit tests therefore establish the Python contract and artifact
integrity; they are not evidence that `QWEN-FWD-001` or `QWEN-EQ-001` has
passed.

[PLAN_REAL_RTL_FORWARD.md](PLAN_REAL_RTL_FORWARD.md) records the phased plan
that closes these open requirements for statements of up to 128 tokens.
