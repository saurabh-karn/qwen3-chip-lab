# qwen_ref

This package is a dependency-free, deliberately slow Qwen3-0.6B reference for
checking RTL. Its production geometry is fixed at L28/H1024/I3072/NH16/NKV8/
HD128/V151936. Tiny dimensions are supported only by the standalone operators.

Examples:

```sh
python3 -m qwen_ref inspect /path/to/model
python3 -m qwen_ref pack-rom /path/to/model weights.bf16
python3 -m qwen_ref tokenize /path/to/tokenizer.json "hello"
python3 -m qwen_ref forward /path/to/model 1,2,3 --trace trace.ndjson
```

`pack-rom` emits one immutable, little-endian BF16 layout in
`manifest.ROM_LAYOUT`, plus a JSON manifest containing source, layout, and ROM
SHA-256 hashes. The CLI verifies that manifest and ROM hash before inference.
`FlatBF16ROM` and checkpoint readers expose the canonical Hugging Face tensor
names to `model.forward`; the ROM maps the bit-identical `lm_head.weight` and
`model.embed_tokens.weight` checkpoint tensors to one tied storage region.

Numerical policy:

* Weight and operator operands are rounded to BF16 by explicit
  round-to-nearest-even bit conversion.
* Products, additions, division, square root, and transcendental outputs are
  rounded to FP32 at defined points.
* Linear and attention dot products round after every multiply and add. This is
  an oracle policy, not a claim that every optimized framework uses the same
  reduction tree.

The tokenizer implements and validates the published Qwen3 NFC + Split +
ByteLevel graph. It intentionally rejects other tokenizer graphs instead of
silently approximating them.

Known limitations: the pure Python full model is impractically slow for normal
serving; there is no KV-cache generation loop; and `exp`, `sin`, and `cos`
still come from the host math library rather than shared versioned
approximation tables. Consequently the current arithmetic is reproducible for
a fixed Python/libm build but is not a cross-platform bit-exact RTL contract.
The HTTP API is a local integration hook, not a hardened public server.
