"""Canonical flat BF16 ROM layout and reproducible manifest generation."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Iterable, Mapping

from .config import QWEN3_0_6B as G
from .fp import ARITHMETIC_PROFILE, bf16_bits
from .safetensors import SafeTensor, SafeTensorFile, ShardedSafeTensors


@dataclass(frozen=True, slots=True)
class TensorLayout:
    hf_name: str
    shape: tuple[int, ...]
    byte_offset: int
    byte_length: int


def _storage_definitions() -> Iterable[tuple[str, tuple[int, ...]]]:
    yield "model.embed_tokens.weight", (G.vocab_size, G.hidden_size)
    for layer in range(G.layers):
        p = f"model.layers.{layer}"
        yield f"{p}.input_layernorm.weight", (G.hidden_size,)
        yield f"{p}.self_attn.q_proj.weight", (G.q_size, G.hidden_size)
        yield f"{p}.self_attn.q_norm.weight", (G.head_dim,)
        yield f"{p}.self_attn.k_proj.weight", (G.kv_size, G.hidden_size)
        yield f"{p}.self_attn.k_norm.weight", (G.head_dim,)
        yield f"{p}.self_attn.v_proj.weight", (G.kv_size, G.hidden_size)
        yield f"{p}.self_attn.o_proj.weight", (G.hidden_size, G.q_size)
        yield f"{p}.post_attention_layernorm.weight", (G.hidden_size,)
        yield f"{p}.mlp.gate_proj.weight", (G.intermediate_size, G.hidden_size)
        yield f"{p}.mlp.up_proj.weight", (G.intermediate_size, G.hidden_size)
        yield f"{p}.mlp.down_proj.weight", (G.hidden_size, G.intermediate_size)
    yield "model.norm.weight", (G.hidden_size,)


def _checkpoint_definitions() -> Iterable[tuple[str, tuple[int, ...]]]:
    yield "lm_head.weight", (G.vocab_size, G.hidden_size)
    yield from _storage_definitions()


def _make_layout() -> tuple[TensorLayout, ...]:
    result = []
    offset = 0
    for name, shape in _storage_definitions():
        count = 1
        for extent in shape:
            count *= extent
        length = count * 2
        result.append(TensorLayout(name, shape, offset, length))
        offset += length
    return tuple(result)


ROM_LAYOUT = _make_layout()
_rom_by_name = {entry.hf_name: entry for entry in ROM_LAYOUT}
_rom_by_name["lm_head.weight"] = _rom_by_name["model.embed_tokens.weight"]
ROM_BY_HF_NAME: Mapping[str, TensorLayout] = MappingProxyType(_rom_by_name)
TIED_LOGITS_HF_NAME = "lm_head.weight"


def layout_sha256() -> str:
    payload = json.dumps(
        [
            {
                "hf_name": item.hf_name,
                "shape": item.shape,
                "byte_offset": item.byte_offset,
                "byte_length": item.byte_length,
            }
            for item in ROM_LAYOUT
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


LAYOUT_SHA256 = layout_sha256()
ROM_BYTES = ROM_LAYOUT[-1].byte_offset + ROM_LAYOUT[-1].byte_length


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(
    checkpoint: SafeTensorFile | ShardedSafeTensors,
) -> None:
    expected_names = set()
    for name, shape in _checkpoint_definitions():
        expected_names.add(name)
        try:
            actual = checkpoint[name]
        except KeyError as exc:
            raise ValueError(f"missing tensor {name}") from exc
        if actual.shape != shape:
            raise ValueError(
                f"{name}: expected {shape}, got {actual.shape}"
            )
        if actual.dtype != "BF16":
            raise ValueError(f"{name}: canonical checkpoint dtype is BF16, got {actual.dtype}")
    actual_names = (
        set(checkpoint.tensors)
        if isinstance(checkpoint, SafeTensorFile)
        else set(checkpoint.weight_map)
    )
    if actual_names != expected_names:
        extras = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        raise ValueError(f"non-canonical tensor set; missing={missing}, extra={extras}")

    head = checkpoint["lm_head.weight"]
    embedding = checkpoint["model.embed_tokens.weight"]
    chunk_size = 1024 * 1024
    for offset in range(0, head.end - head.start, chunk_size):
        count = min(chunk_size, head.end - head.start - offset)
        if (
            head.owner.mapping[head.start + offset:head.start + offset + count]
            != embedding.owner.mapping[
                embedding.start + offset:embedding.start + offset + count
            ]
        ):
            raise ValueError("lm_head.weight is not bit-identical to tied embeddings")


def _write_tensor_bf16(output: BinaryIO, tensor: SafeTensor) -> None:
    if tensor.dtype == "BF16":
        view = memoryview(tensor.owner.mapping)[tensor.start:tensor.end]
        try:
            output.write(view)
        finally:
            view.release()
        return
    buffer = bytearray()
    for value in tensor:
        buffer += struct.pack("<H", bf16_bits(value))
        if len(buffer) >= 1024 * 1024:
            output.write(buffer)
            buffer.clear()
    output.write(buffer)


def pack_rom(
    checkpoint: SafeTensorFile | ShardedSafeTensors, output_path: str | Path
) -> str:
    """Write canonical BF16 ROM and return its SHA-256."""
    validate_checkpoint(checkpoint)
    output_path = Path(output_path)
    with output_path.open("wb") as output:
        for item in ROM_LAYOUT:
            if output.tell() != item.byte_offset:
                raise AssertionError("ROM layout offset mismatch")
            _write_tensor_bf16(output, checkpoint[item.hf_name])
    return sha256_file(output_path)


def manifest_document(
    rom_sha256: str, source_hashes: Mapping[str, str] | None = None
) -> dict[str, object]:
    return {
        "format": "qwen_ref.flat-bf16-rom.v1",
        "model": "Qwen3-0.6B",
        "dtype": "BF16",
        "endianness": "little",
        "arithmetic_profile": ARITHMETIC_PROFILE,
        "geometry": asdict(G),
        "layout_sha256": LAYOUT_SHA256,
        "rom_sha256": rom_sha256,
        "rom_bytes": ROM_BYTES,
        "tied_logits_hf_name": TIED_LOGITS_HF_NAME,
        "rom_aliases": {"lm_head.weight": "model.embed_tokens.weight"},
        "source_sha256": dict(sorted((source_hashes or {}).items())),
        "tensors": [
            {
                "hf_name": item.hf_name,
                "shape": list(item.shape),
                "byte_offset": item.byte_offset,
                "byte_length": item.byte_length,
            }
            for item in ROM_LAYOUT
        ],
    }


def write_manifest(
    path: str | Path, rom_sha256: str, source_hashes: Mapping[str, str] | None = None
) -> None:
    document = manifest_document(rom_sha256, source_hashes)
    Path(path).write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
