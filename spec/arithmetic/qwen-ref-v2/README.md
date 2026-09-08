# qwen-ref-v2 arithmetic profile

Status: Phase 1 contract candidate; freeze follows review.

`qwen-ref-v2` is the sole bit-exact reference profile. All bit patterns in this
document use IEEE-754 interchange encoding. Multi-byte artifacts use little
endian order.

## Formats and rounding

- Weights are native BF16. Loading a BF16 weight is exact.
- Every activation or KV value persisted in scratch is BF16. Embeddings enter
  scratch as BF16. Projection, norm, RoPE, attention, MLP, and residual
  operators produce a binary32 pre-commit value; the scratch write converts it
  to BF16 RNE. A later consumer reads that BF16 value and widens it exactly.
- Accumulators and mandatory checkpoint values are binary32. Checkpoints named
  below observe operator results before the BF16 scratch commit unless the
  event explicitly says otherwise. Thus a checkpoint can differ from the value
  subsequently read from scratch.
- Every model multiplier input explicitly marked as an operand is converted to
  BF16 with round-to-nearest, ties-to-even (RNE), then widened exactly to
  binary32. This applies to linear/dot operands, RMSNorm input and weight,
  RoPE input, SiLU input, SwiGLU `up`, and attention probability/value.
- A BF16 product is computed as binary32 and every product and serial
  accumulator addition is rounded to binary32 RNE. Reductions run in increasing
  logical index order from positive zero. There is no fused multiply-add.
- Point operations state each binary32 rounding point in the Python source.
  Division is an integer implementation of correctly rounded binary32 RNE;
  it does not depend on the host floating-point divider.
- BF16 conversion preserves signed zero and infinity. Any NaN is changed to the
  quiet BF16 pattern `0x7fc0`. Binary32 operations similarly use canonical
  quiet NaN `0x7fc00000`.
- Binary32 and BF16 subnormals are preserved (no flush-to-zero). Exact zero
  signs follow IEEE-754. Overflow rounds to signed infinity.
- Checkpoint weights and runtime model inputs are required to be finite.
  A domain fault raises `ValueError` in Python before the destination scratch
  write or downstream checkpoint. RTL must expose an equivalent command fault
  and suppress that destination write; the exact control signal is deferred to
  the arithmetic-engine integration phase. Faults never clamp NaN or infinity
  into a finite model value.

## Scratch commit and checkpoint order

For a valid operation, the following ordering is normative. A fault suppresses
the operation's externally emitted checkpoint as well as its destination write:

1. An operator reads BF16 scratch operands and computes binary32 intermediates
   and output.
2. Its mandatory checkpoint captures that binary32 output.
3. If the value persists, it is converted to BF16 RNE and written to scratch.
4. The next operator consumes the widened committed BF16 value.

In particular, Q/K/V projection results commit before Q/K norm or attention;
Q/K norm commits before RoPE; RoPE Q/K and V commit into the KV/attention
workspace; gate/up commit before SiLU/SwiGLU; and attention/MLP updates commit
before residual addition. Both residual operands are BF16 scratch reads. The
binary32 residual sum is checkpointed, then committed to BF16 as the next
hidden state.

More generally, every mandatory stage output that a later model stage consumes
is treated as scratch-persisted and therefore commits to BF16 after its
binary32 checkpoint. Final logits are streamed binary32 outputs and do not
commit to activation scratch. `embedding` is the sole post-commit checkpoint:
it records the BF16 ROM/scratch value widened exactly to binary32.

Attention scores have a deliberate two-view rule. `attention_scores` records
the scaled binary32 dot-product result. Every score is then committed to BF16,
and max subtraction and softmax consume those committed BF16 scores.

The exponential is **not** a scratch value. After the row max, each
`exp_negative(BF16(score) − max)` and the serial sum of those exponentials
stay binary32. The divide uses that binary32 exponential as the dividend.
Writing `exp(score − max)` to BF16 SRAM and then dividing the rounded
exponential is a spec violation: it changes probabilities for every row
longer than one. A one-token prompt cannot catch it (`P ≡ 1` per head).
T4 (full-statement compare) on a single token is therefore not a softmax-commit test.

`attention_softmax` records those binary32 probabilities before their BF16
commit. The value-weighted reduction then consumes committed BF16
probabilities and values.

## Nonlinear ROMs

Committed table words in `qwen_ref/fp_tables.py` are normative payload.
`tables-manifest.json` is the only table count/hash authority and is suitable
for Python and future RTL generators. The generator writes both artifacts,
accepts explicit output paths, and must reproduce their bytes exactly. Hashes
cover each sequence of little-endian uint32 words.

### Reciprocal square root

`reciprocal_sqrt(x)` accepts positive finite binary32 `x`. It decomposes `x`
exactly into a power of four and a normalized mantissa in `[1,4)`. A ROM stores
binary32 `1/sqrt(1 + i/256)` for `i=0..768`. The low mantissa bits select a
linear interpolation fraction. Difference, multiply, add, and final power-of-
two scaling each round to binary32 RNE.

RMSNorm serially sums squares of BF16 inputs, divides by the vector length,
adds binary32 epsilon `1e-6`, obtains scale with this reciprocal-square-root
engine, then computes `(BF16(x) * scale) * BF16(weight)` with a binary32 round
after each multiply. Attention scaling uses the same engine on head dimension
128.

### Negative exponential and stable softmax

`exp_negative(x)` accepts only finite binary32 `x <= 0`; positive values, NaN,
and either infinity are domain faults. A ROM stores binary32
`exp(-i/256)` for `i=0..4096`. Values in `[-16,0]` use linear interpolation,
rounding the scale, difference, multiply, and add to binary32. Values below
`-16` return positive zero; exactly `-16` uses the nonzero final ROM entry.
This clipping is part of the arithmetic result and is not a fault.

Softmax requires a nonempty row of finite committed BF16 scores. A binary32
pre-commit score that rounds to BF16 infinity faults before trace publication,
destination commit, or max reduction. Softmax finds the serial row maximum,
subtracts it from every committed score, uses `exp_negative`, serially sums
those **binary32** exponentials in source order, and performs correctly
rounded binary32 division of each **binary32** exponential by that sum. The
exponentials are not committed to BF16 between `exp_negative` and the
divide. A zero, NaN, or infinite denominator is a domain fault. Thus for a
valid row at least one exponential is exactly one. A row of length 1 always
yields probability 1 and does not exercise the divide.

SiLU first commits `x` to BF16. NaN or infinity after that conversion is a
domain fault, including finite binary32 values that overflow during BF16
commit. Valid inputs use the stable branches
`x/(1+exp(-x))` for nonnegative `x` and `x*exp(x)/(1+exp(x))` otherwise.
The SiLU result is checkpointed in binary32 and committed to BF16. SwiGLU
requires finite committed BF16 SiLU and `up` operands, multiplies them, and
faults if the binary32 product is non-finite.

### RoPE

The profile accepts only integer positions 0 through 127 (booleans are not
integers for this interface), head width 128, and theta 1,000,000. There is no
wrapping, saturation, or extrapolation. Invalid parameters fault before any
table access or output commit. A forward command starts at position zero and
therefore accepts 1 through 128 tokens; token 129 faults before weight access.
A direct ROM contains a binary32 cosine/sine pair for
every `(position, rotate-half pair)`. It is addressed as
`((position * 64 + pair) * 2 + {cos=0,sin=1})`. RoPE rounds its BF16 input
products and output additions to binary32. Other positions, widths, or theta
values are outside this profile and are rejected.

The table generator uses a 110-digit Decimal context. Exponential, logarithm,
and square root use Decimal operations; sine and cosine use range reduction
and a Decimal Taylor series. Runtime inference does not call Decimal or
platform `libm`.

## Mandatory event names

Events are emitted in schedule order:

`embedding`; for each layer `L`, `layer.L.input_norm`, `q_proj`, `k_proj`,
`v_proj`, `q_norm`, `k_norm`, `q_rope`, `k_rope`; for each token `T`,
`layer.L.token.T.attention_scores` and `attention_softmax`; then
`layer.L.causal_gqa`, `o_proj`, `attention_residual`, `post_norm`, `gate_proj`,
`up_proj`, `silu`, `swiglu`, `down_proj`, and `output`; finally `final_norm`
and `logits`.

Every event is binary32, includes shape and element count, and hashes the full
little-endian payload. Implementations must match every element, not only the
hash or sampled values.

## RTL implementation status

`qwen_bf16_mac_array` is qwen-ref-v2 compliant: it preserves subnormals, rounds
ties to even, canonicalizes quiet NaNs, and reduces serially in increasing
index order regardless of `MAC_LANES`.

## RTL-consumable vectors

`vectors.json` is the shared operator-vector interchange artifact. Arrays are
row-major, exact expected BF16/binary32 values are hexadecimal interchange
words, and every case names its operand and output format. Decimal source
values are converted to the declared operand format before use. RTL benches
must consume this artifact (or a byte-for-byte generated translation), not
copy expected values into an independent source.
