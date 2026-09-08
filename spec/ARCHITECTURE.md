# Qwen3-0.6B chip laboratory architecture

## Invariants

The production model is always the complete unquantized Qwen3-0.6B dense
decoder:

| Parameter | Value |
|---|---:|
| decoder layers | 28 |
| hidden width | 1024 |
| intermediate width | 3072 |
| query heads | 16 |
| key/value heads | 8 |
| head dimension | 128 |
| vocabulary | 151936 |
| maximum configured positions | 40960 |
| weight format | native BF16 |
| tied embedding/head | yes |

Architectural experiments may alter throughput, storage organization, and
latency. They may not reduce these model dimensions or omit operations.

## Runtime trust boundary

The host supplies token IDs, sequence length, start, and trace controls. It
cannot address or modify the weight store. Model weights are manufactured ROM
constants. In simulation, a never-written `$readmemh` array or a read-only
memory-mapped image represents that ROM.

Activations, Q/K/V values, partial sums, and control state reside in explicitly
modeled on-chip SRAM or registers. No runtime DRAM weight path exists.

## Stable interfaces

The controller targets four technology-neutral interfaces:

1. `weight_rom_if`: read-only request/response with configurable width, banks,
   ports, and latency.
2. `scratch_sram_if`: read/write request/response with configurable banks,
   ports, latency, and conflict reporting.
3. `compute_if`: reusable BF16 operand and FP32 accumulation datapath.
4. `trace_if`: non-functional event stream carrying stage, layer, token,
   address, data, stalls, and counters.

Foundry memory macros are integrated through adapters behind the first two
interfaces. Changing a macro must not change the Qwen schedule.

## Numerical contract

- ROM words are IEEE-754 bfloat16.
- Each multiply consumes BF16 operands.
- Products and accumulations round through binary32 in a fixed, serially
  specified order. Parallel implementations must reproduce the same reduction
  tree selected by their configuration.
- All activation and KV state committed to scratch SRAM is BF16. Accumulators
  and mandatory pre-commit checkpoints are binary32. Each operator checkpoint
  precedes its BF16 scratch write; the next operator consumes the committed
  BF16 value. In particular, residual operands and softmax input scores are
  scratch reads, not unrounded checkpoint values.
- Norm, exp, SiLU, softmax, and RoPE use the frozen `qwen-ref-v2` profile in
  `spec/arithmetic/qwen-ref-v2/`. Runtime inference uses only integer-addressed,
  hash-pinned binary32 ROMs and explicit binary32 operations: no nonlinear
  operation calls the host C math library. Reciprocal square root and negative
  exponential use linear table interpolation; canonical positions 0 through
  127 use a direct RoPE cosine/sine ROM.
- `qwen-ref-v2` preserves subnormals, rounds to nearest ties-to-even, uses
  canonical quiet NaNs, prohibits fused multiply-add, and fixes serial
  increasing-index reductions. Scratch activations are BF16; checkpoints and
  accumulators are binary32.
- A configuration is correct only when its selected arithmetic profile matches
  the Python oracle at every mandatory checkpoint and across every final logit.

The current `qwen_bf16_mac_array` is not yet v2-compliant: it flushes
subnormals and has different NaN/infinity behavior. Its existing normal-value
tests are bring-up tests only. The RTL arithmetic phase must close these gaps;
they do not alter the v2 contract.

## Required forward schedule

`embedding -> 28 * (input RMSNorm -> QKV -> Q/K RMSNorm -> RoPE -> causal GQA
-> O projection -> residual -> post-attention RMSNorm -> gate/up -> SiLU ->
down projection -> residual) -> final RMSNorm -> tied full-vocabulary head`

The published checkpoint contains both `model.embed_tokens.weight` and
`lm_head.weight`, despite declaring tied embeddings. Validation requires both
canonical BF16 tensors and verifies that they are bit-identical. The flat ROM
stores one copy and exposes `lm_head.weight` as a read-only alias.

Python traces include every schedule boundary above plus per-token causal
attention scores and softmax probabilities. Trace tensors are rectangular,
carry shape and element count, and are hashed over their complete FP32 payload.

## Comparison policy

Correctness is a gate, not an optimization metric. Designs that pass are
compared using:

- cycle count and latency,
- useful MAC utilization,
- ROM and SRAM traffic, stalls, and bank conflicts,
- memory and compute area estimates,
- energy estimates and, when available, measured implementation power,
- throughput and Pareto dominance.

Reports must name the RTL commit, ROM manifest hash, arithmetic profile, design
configuration, simulator/synthesis tool versions, and whether each metric is
measured or estimated.
