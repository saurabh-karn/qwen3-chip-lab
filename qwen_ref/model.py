"""Qwen3 forward operators and canonical full-model forward pass."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

from .config import QWEN3_0_6B as G
from .fp import (
    bf16,
    bits_f32,
    dot_f32_bf16,
    exp_negative,
    f32,
    f32_add,
    f32_bits,
    f32_div,
    f32_mul,
    reciprocal_sqrt,
)
from .fp_tables import ROPE_COS_SIN_F32_BITS, ROPE_PAIRS, ROPE_POSITIONS
from .trace import Progress, TraceWriter

Vector = list[float]
Matrix = list[Vector]


class WeightSource(Protocol):
    def __getitem__(self, name: str) -> Sequence[float]: ...


def linear(vector: Sequence[float], weight: Sequence[float], out_features: int) -> Vector:
    """Bias-free row-major linear transform with BF16 operands."""
    if out_features <= 0 or len(weight) % out_features:
        raise ValueError("invalid linear weight dimensions")
    in_features = len(weight) // out_features
    if len(vector) != in_features:
        raise ValueError(f"linear expected {in_features} inputs, got {len(vector)}")
    return [
        dot_f32_bf16(vector, weight[row * in_features:(row + 1) * in_features])
        for row in range(out_features)
    ]


def rms_norm(
    vector: Sequence[float], weight: Sequence[float], epsilon: float = 1.0e-6
) -> Vector:
    if len(vector) != len(weight) or not vector:
        raise ValueError("RMSNorm vectors must have equal nonzero length")
    squares = 0.0
    for value in vector:
        operand = bf16(value)
        squares = f32_add(squares, f32_mul(operand, operand))
    mean = f32_div(squares, float(len(vector)))
    scale = reciprocal_sqrt(f32_add(mean, epsilon))
    return [f32_mul(f32_mul(bf16(x), scale), bf16(w)) for x, w in zip(vector, weight)]


def qk_norm(heads: Sequence[Sequence[float]], weight: Sequence[float]) -> Matrix:
    return [rms_norm(head, weight) for head in heads]


def rope(
    heads: Sequence[Sequence[float]], position: int, theta: float = 1_000_000.0
) -> Matrix:
    """Apply canonical Qwen rotate-half RoPE from the direct v2 ROM."""
    if isinstance(position, bool) or not isinstance(position, int):
        raise ValueError("RoPE position must be an integer")
    if not 0 <= position < ROPE_POSITIONS:
        raise ValueError(f"qwen-ref-v2 RoPE position must be in [0, {ROPE_POSITIONS})")
    if theta != 1_000_000.0:
        raise ValueError("qwen-ref-v2 fixes RoPE theta at 1000000")
    result: Matrix = []
    for head in heads:
        if len(head) != 2 * ROPE_PAIRS:
            raise ValueError("qwen-ref-v2 fixes RoPE head dimension at 128")
        half = len(head) // 2
        output = [0.0] * len(head)
        for index in range(half):
            table_offset = (position * ROPE_PAIRS + index) * 2
            cosine = bits_f32(ROPE_COS_SIN_F32_BITS[table_offset])
            sine = bits_f32(ROPE_COS_SIN_F32_BITS[table_offset + 1])
            first, second = bf16(head[index]), bf16(head[index + half])
            output[index] = f32_add(f32_mul(first, cosine), -f32_mul(second, sine))
            output[index + half] = f32_add(
                f32_mul(second, cosine), f32_mul(first, sine)
            )
        result.append(output)
    return result


def causal_gqa(
    queries: Sequence[Sequence[Sequence[float]]],
    keys: Sequence[Sequence[Sequence[float]]],
    values: Sequence[Sequence[Sequence[float]]],
    checkpoint: Callable[[int, Matrix, Matrix], None] | None = None,
) -> list[Matrix]:
    """Causal grouped-query attention for complete Q/K/V sequences."""
    if not queries or len(queries) != len(keys) or len(keys) != len(values):
        raise ValueError("Q/K/V sequences must have equal nonzero length")
    q_heads = len(queries[0])
    kv_heads = len(keys[0])
    if not q_heads or not kv_heads or q_heads % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    head_dim = len(queries[0][0])
    if not head_dim:
        raise ValueError("attention head dimension must be nonzero")
    for name, sequence, heads in (
        ("queries", queries, q_heads),
        ("keys", keys, kv_heads),
        ("values", values, kv_heads),
    ):
        if any(
            len(token) != heads or any(len(head) != head_dim for head in token)
            for token in sequence
        ):
            raise ValueError(f"{name} have inconsistent head dimensions")
    groups = q_heads // kv_heads
    scale = reciprocal_sqrt(float(head_dim))
    outputs: list[Matrix] = []
    for position, token_queries in enumerate(queries):
        token_output: Matrix = []
        token_scores: Matrix = []
        token_probabilities: Matrix = []
        for q_head, query in enumerate(token_queries):
            kv_head = q_head // groups
            precommit_scores = [
                f32_mul(dot_f32_bf16(query, keys[source][kv_head]), scale)
                for source in range(position + 1)
            ]
            # Scores are checkpointed before commit, then softmax reads the
            # BF16 score row back from scratch.
            scores = _commit_vector(precommit_scores)
            if any(f32_bits(score) & 0x7F80_0000 == 0x7F80_0000 for score in scores):
                raise ValueError("softmax requires finite committed BF16 scores")
            maximum = max(scores)
            exponentials = [
                exp_negative(f32_add(score, -maximum)) for score in scores
            ]
            denominator = 0.0
            for value in exponentials:
                denominator = f32_add(denominator, value)
            if not 0.0 < denominator < float("inf"):
                raise ValueError("softmax denominator must be positive and finite")
            probabilities = [f32_div(value, denominator) for value in exponentials]
            committed_probabilities = _commit_vector(probabilities)
            token_scores.append(precommit_scores)
            token_probabilities.append(probabilities)
            attended = []
            for component in range(head_dim):
                acc = 0.0
                for source, probability in enumerate(committed_probabilities):
                    acc = f32_add(
                        acc,
                        f32_mul(bf16(probability), bf16(values[source][kv_head][component])),
                    )
                attended.append(acc)
            token_output.append(attended)
        if checkpoint:
            checkpoint(position, token_scores, token_probabilities)
        outputs.append(token_output)
    return outputs


def silu(vector: Sequence[float]) -> Vector:
    output = []
    for value in vector:
        x = bf16(value)
        if f32_bits(x) & 0x7F80_0000 == 0x7F80_0000:
            raise ValueError("SiLU requires finite BF16 inputs")
        if x >= 0.0:
            sigmoid = f32_div(1.0, f32_add(1.0, exp_negative(-x)))
        else:
            exponential = exp_negative(x)
            sigmoid = f32_div(exponential, f32_add(1.0, exponential))
        output.append(f32_mul(x, sigmoid))
    return output


def swiglu(gate: Sequence[float], up: Sequence[float]) -> Vector:
    if len(gate) != len(up):
        raise ValueError("SwiGLU vectors differ in length")
    return _swiglu_product(silu(gate), up)


def _swiglu_product(activated: Sequence[float], up: Sequence[float]) -> Vector:
    if len(activated) != len(up):
        raise ValueError("SwiGLU vectors differ in length")
    output = []
    for gate_value, up_value in zip(activated, up):
        committed_gate = bf16(gate_value)
        committed_up = bf16(up_value)
        if any(
            f32_bits(value) & 0x7F80_0000 == 0x7F80_0000
            for value in (committed_gate, committed_up)
        ):
            raise ValueError("SwiGLU requires finite committed BF16 inputs")
        result = f32_mul(committed_gate, committed_up)
        if f32_bits(result) & 0x7F80_0000 == 0x7F80_0000:
            raise ValueError("SwiGLU produced a non-finite FP32 pre-commit value")
        output.append(result)
    return output


def _commit_vector(vector: Sequence[float]) -> Vector:
    """Commit an FP32 operator result to BF16 scratch storage."""
    return [bf16(value) for value in vector]


def _commit_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [_commit_vector(row) for row in matrix]


def _heads(vector: Sequence[float], count: int, width: int) -> Matrix:
    if len(vector) != count * width:
        raise ValueError("projection has incorrect head dimensions")
    return [list(vector[i * width:(i + 1) * width]) for i in range(count)]


def _residual(left: Sequence[float], right: Sequence[float]) -> Vector:
    if len(left) != len(right):
        raise ValueError("residual vectors differ in length")
    # Both operands are read from BF16 scratch. The returned sum is the FP32
    # pre-commit checkpoint; the caller performs the BF16 write explicitly.
    return [f32_add(bf16(a), bf16(b)) for a, b in zip(left, right)]


def _weight(weights: WeightSource | Mapping[str, Sequence[float]], name: str) -> Sequence[float]:
    return weights[name]


def forward(
    token_ids: Sequence[int],
    weights: WeightSource | Mapping[str, Sequence[float]],
    trace: TraceWriter | None = None,
    progress: Progress | None = None,
) -> Matrix:
    """Run canonical Qwen3-0.6B and return tied full-vocabulary logits."""
    if not token_ids:
        raise ValueError("at least one token is required")
    if len(token_ids) > ROPE_POSITIONS:
        raise ValueError(
            f"sequence exceeds qwen-ref-v2 limit of {ROPE_POSITIONS} tokens"
        )
    embedding = _weight(weights, "model.embed_tokens.weight")
    if len(embedding) != G.vocab_size * G.hidden_size:
        raise ValueError("embedding has non-canonical shape")
    hidden: Matrix = []
    for token in token_ids:
        if not 0 <= token < G.vocab_size:
            raise ValueError(f"token id out of range: {token}")
        start = token * G.hidden_size
        hidden.append([bf16(x) for x in embedding[start:start + G.hidden_size]])
    if trace:
        trace.event("embedding", hidden)

    for layer in range(G.layers):
        prefix = f"model.layers.{layer}"
        normalized = [
            rms_norm(row, _weight(weights, f"{prefix}.input_layernorm.weight"))
            for row in hidden
        ]
        if trace:
            trace.event(f"layer.{layer}.input_norm", normalized, layer=layer)
        queries, keys, values = [], [], []
        normalized_queries, normalized_keys = [], []
        raw_queries, raw_keys, raw_values = [], [], []
        traced_queries: list[Matrix] = []
        traced_keys: list[Matrix] = []
        for position, row in enumerate(normalized):
            q = _heads(
                linear(row, _weight(weights, f"{prefix}.self_attn.q_proj.weight"), G.q_size),
                G.attention_heads, G.head_dim,
            )
            k = _heads(
                linear(row, _weight(weights, f"{prefix}.self_attn.k_proj.weight"), G.kv_size),
                G.kv_heads, G.head_dim,
            )
            v = _heads(
                linear(row, _weight(weights, f"{prefix}.self_attn.v_proj.weight"), G.kv_size),
                G.kv_heads, G.head_dim,
            )
            raw_queries.append(q)
            raw_keys.append(k)
            raw_values.append(v)
            q_normalized = qk_norm(q, _weight(weights, f"{prefix}.self_attn.q_norm.weight"))
            k_normalized = qk_norm(k, _weight(weights, f"{prefix}.self_attn.k_norm.weight"))
            normalized_queries.append(q_normalized)
            normalized_keys.append(k_normalized)
            q_rotated = rope(q_normalized, position, G.rope_theta)
            k_rotated = rope(k_normalized, position, G.rope_theta)
            queries.append(_commit_matrix(q_rotated))
            keys.append(_commit_matrix(k_rotated))
            values.append(_commit_matrix(v))
            # Trace the FP32 values before their BF16 scratch commits.
            traced_queries.append(q_rotated)
            traced_keys.append(k_rotated)
        if trace:
            trace.event(f"layer.{layer}.q_proj", raw_queries, layer=layer)
            trace.event(f"layer.{layer}.k_proj", raw_keys, layer=layer)
            trace.event(f"layer.{layer}.v_proj", raw_values, layer=layer)
            trace.event(f"layer.{layer}.q_norm", normalized_queries, layer=layer)
            trace.event(f"layer.{layer}.k_norm", normalized_keys, layer=layer)
            trace.event(f"layer.{layer}.q_rope", traced_queries, layer=layer)
            trace.event(f"layer.{layer}.k_rope", traced_keys, layer=layer)
        def attention_checkpoint(
            position: int, scores: Matrix, probabilities: Matrix
        ) -> None:
            if trace:
                trace.event(
                    f"layer.{layer}.token.{position}.attention_scores",
                    scores,
                    layer=layer,
                    token=position,
                )
                trace.event(
                    f"layer.{layer}.token.{position}.attention_softmax",
                    probabilities,
                    layer=layer,
                    token=position,
                )

        attended = causal_gqa(queries, keys, values, attention_checkpoint if trace else None)
        if trace:
            trace.event(f"layer.{layer}.causal_gqa", attended, layer=layer)
        attention_output = [
            linear(
                [component for head in token_heads for component in head],
                _weight(weights, f"{prefix}.self_attn.o_proj.weight"),
                G.hidden_size,
            )
            for token_heads in attended
        ]
        if trace:
            trace.event(f"layer.{layer}.o_proj", attention_output, layer=layer)
        attention_residual = [
            _residual(row, update) for row, update in zip(hidden, attention_output)
        ]
        if trace:
            trace.event(
                f"layer.{layer}.attention_residual", attention_residual, layer=layer
            )
        hidden = [_commit_vector(row) for row in attention_residual]
        post_norm = [
            rms_norm(row, _weight(weights, f"{prefix}.post_attention_layernorm.weight"))
            for row in hidden
        ]
        if trace:
            trace.event(f"layer.{layer}.post_norm", post_norm, layer=layer)
        mlp_output = []
        gate_outputs, up_outputs, silu_outputs, swiglu_outputs = [], [], [], []
        for row in post_norm:
            gate = linear(row, _weight(weights, f"{prefix}.mlp.gate_proj.weight"), G.intermediate_size)
            up = linear(row, _weight(weights, f"{prefix}.mlp.up_proj.weight"), G.intermediate_size)
            activated = silu(gate)
            gated = _swiglu_product(activated, up)
            gate_outputs.append(gate)
            up_outputs.append(up)
            silu_outputs.append(activated)
            swiglu_outputs.append(gated)
            mlp_output.append(
                linear(
                    gated,
                    _weight(weights, f"{prefix}.mlp.down_proj.weight"),
                    G.hidden_size,
                )
            )
        if trace:
            trace.event(f"layer.{layer}.gate_proj", gate_outputs, layer=layer)
            trace.event(f"layer.{layer}.up_proj", up_outputs, layer=layer)
            trace.event(f"layer.{layer}.silu", silu_outputs, layer=layer)
            trace.event(f"layer.{layer}.swiglu", swiglu_outputs, layer=layer)
            trace.event(f"layer.{layer}.down_proj", mlp_output, layer=layer)
        layer_output = [
            _residual(row, update) for row, update in zip(hidden, mlp_output)
        ]
        if trace:
            trace.event(f"layer.{layer}.output", layer_output, layer=layer)
        hidden = [_commit_vector(row) for row in layer_output]
        if progress:
            progress(layer + 1, G.layers, f"layer {layer + 1}/{G.layers}")

    hidden = [rms_norm(row, _weight(weights, "model.norm.weight")) for row in hidden]
    if trace:
        trace.event("final_norm", hidden)
    lm_head = _weight(weights, "lm_head.weight")
    if len(lm_head) != G.vocab_size * G.hidden_size:
        raise ValueError("language-model head has non-canonical shape")
    logits = [linear(row, lm_head, G.vocab_size) for row in hidden]
    if trace:
        trace.event("logits", logits)
    return logits
