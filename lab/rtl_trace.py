"""Rebuild qwen_ref NDJSON / f32le traces from the Verilator commit log."""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from qwen_ref.fp import bits_f32
from qwen_ref.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[1]
EVENT_MAP = json.loads(
    (ROOT / "spec" / "trace_event_map.json").read_text(encoding="utf-8")
)
GEOM = EVENT_MAP["geometry"]
HIDDEN = int(GEOM["hidden"])
INTERMEDIATE = int(GEOM["intermediate"])
Q_HEADS = int(GEOM["q_heads"])
KV_HEADS = int(GEOM["kv_heads"])
HEAD_DIM = int(GEOM["head_dim"])
VOCAB = int(GEOM["vocab"])
Q_WIDTH = Q_HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM

KIND_COMMIT = 0
KIND_LOGIT = 1
KIND_SILU = 2
KIND_TOKEN_DONE = 3
KIND_META = 4
KIND_STAGE_ENTRY = 5


def _u8(stream) -> int:
    data = stream.read(1)
    if len(data) != 1:
        raise EOFError
    return data[0]


def _u16(stream) -> int:
    data = stream.read(2)
    if len(data) != 2:
        raise EOFError
    return struct.unpack("<H", data)[0]


def _u32(stream) -> int:
    data = stream.read(4)
    if len(data) != 4:
        raise EOFError
    return struct.unpack("<I", data)[0]


def _u64(stream) -> int:
    data = stream.read(8)
    if len(data) != 8:
        raise EOFError
    return struct.unpack("<Q", data)[0]


def read_commit_log(path: str | Path) -> dict[str, Any]:
    commits: list[tuple[int, int, int, int, int]] = []
    silu: list[tuple[int, int, int, int]] = []
    logits: list[tuple[int, int, int]] = []
    tokens_done: list[int] = []
    stage_entries: list[tuple[int, int, int, int, int, int]] = []
    meta: dict[str, int] = {}
    with Path(path).open("rb") as stream:
        magic = stream.read(4)
        if magic != b"QTRC":
            raise ValueError(f"{path} is not a Qwen RTL commit log")
        version = _u32(stream)
        if version != 1:
            raise ValueError(f"unsupported commit-log version {version}")
        while True:
            try:
                kind = _u8(stream)
            except EOFError:
                break
            if kind == KIND_META:
                meta = {
                    "cycle_count": _u64(stream),
                    "rom_read_count": _u64(stream),
                    "sram_read_count": _u64(stream),
                    "sram_write_count": _u64(stream),
                    "mac_count": _u64(stream),
                    "stall_count": _u64(stream),
                    "argmax": _u32(stream),
                }
                continue
            if kind == KIND_STAGE_ENTRY:
                stage = _u8(stream)
                layer = _u8(stream)
                _u16(stream)  # reserved
                cycle = _u64(stream)
                rom_reads = _u64(stream)
                macs = _u64(stream)
                stage_entries.append((stage, layer, cycle, rom_reads, macs))
                continue
            stage = _u8(stream)
            layer = _u8(stream)
            position = _u16(stream)
            index = _u32(stream)
            bits = _u32(stream)
            if kind == KIND_COMMIT:
                commits.append((stage, layer, position, index, bits))
            elif kind == KIND_LOGIT:
                logits.append((position, index, bits))
            elif kind == KIND_SILU:
                silu.append((layer, position, index, bits))
            elif kind == KIND_TOKEN_DONE:
                tokens_done.append(position)
            else:
                raise ValueError(f"unknown commit kind {kind} (stage={stage})")
    return {
        "commits": commits,
        "silu": silu,
        "logits": logits,
        "tokens_done": tokens_done,
        "stage_entries": stage_entries,
        "meta": meta,
    }


def _bits_to_f32(bits: int) -> float:
    return bits_f32(bits & 0xFFFFFFFF)


def _reshape_heads(flat: list[float], heads: int) -> list[list[float]]:
    if len(flat) != heads * HEAD_DIM:
        raise ValueError("head layout length mismatch")
    return [flat[head * HEAD_DIM:(head + 1) * HEAD_DIM] for head in range(heads)]


def _reshape_seq(values: dict[int, dict[int, int]], width: int) -> list[list[float]]:
    if not values:
        return []
    seq = max(values) + 1
    rows = []
    for token in range(seq):
        row = values.get(token, {})
        missing = [index for index in range(width) if index not in row]
        if missing:
            raise ValueError(
                f"incomplete tensor row token={token} missing {len(missing)}/{width}"
            )
        rows.append([_bits_to_f32(row[index]) for index in range(width)])
    return rows


def _reshape_seq_heads(
    values: dict[int, dict[int, int]], heads: int
) -> list[list[list[float]]]:
    width = heads * HEAD_DIM
    return [_reshape_heads(row, heads) for row in _reshape_seq(values, width)]


def _reshape_heads_context(
    values: dict[int, int], token: int
) -> list[list[float]]:
    ctx = token + 1
    expected = Q_HEADS * ctx
    if len(values) != expected:
        missing = [i for i in range(expected) if i not in values]
        raise ValueError(
            f"attention row for token {token} missing {len(missing)} elements"
        )
    return [
        [_bits_to_f32(values[head * ctx + source]) for source in range(ctx)]
        for head in range(Q_HEADS)
    ]


def _event_name(template: str, layer: int, token: int) -> str:
    return template.replace("{layer}", str(layer)).replace("{token}", str(token))


def _fields(spec: dict[str, Any], layer: int, token: int) -> dict[str, int]:
    fields: dict[str, int] = {}
    if spec.get("layer", True):
        fields["layer"] = layer
    if spec.get("per_token"):
        fields["token"] = token
    return fields


def _emit_mapped(
    writer: TraceWriter,
    spec: dict[str, Any],
    layer: int,
    buckets: dict[int, dict[int, int]],
) -> None:
    layout = spec["layout"]
    if spec.get("per_token"):
        for token, indexed in sorted(buckets.items()):
            writer.event(
                _event_name(spec["event"], layer, token),
                _reshape_heads_context(indexed, token),
                **_fields(spec, layer, token),
            )
        return
    if layout == "seq_hidden":
        tensor: Any = _reshape_seq(buckets, HIDDEN)
    elif layout == "seq_intermediate":
        tensor = _reshape_seq(buckets, INTERMEDIATE)
    elif layout == "seq_q_heads":
        tensor = _reshape_seq_heads(buckets, Q_HEADS)
    elif layout == "seq_kv_heads":
        tensor = _reshape_seq_heads(buckets, KV_HEADS)
    elif layout == "seq_vocab":
        tensor = _reshape_seq(buckets, VOCAB)
    else:
        raise ValueError(f"unknown layout {layout}")
    writer.event(
        _event_name(spec["event"], layer, -1),
        tensor,
        **_fields(spec, layer, -1),
    )


def write_rtl_trace(
    commit_log: str | Path,
    ndjson_path: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, int]:
    payload = read_commit_log(commit_log)
    grouped: dict[tuple[str, int], dict[int, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    stages = EVENT_MAP["stages"]
    for stage, layer, position, index, bits in payload["commits"]:
        spec = stages.get(str(stage))
        if spec is None:
            continue
        if "split" in spec:
            for part in spec["split"]:
                start = int(part.get("index_start", 0))
                end = int(part.get("index_end", index + 1))
                if start <= index < end:
                    grouped[(part["event"], layer)][position][index - start] = bits
                    break
            continue
        grouped[(spec["event"], layer if spec.get("layer", True) else -1)][
            position
        ][index] = bits
    silu_spec = EVENT_MAP["silu_event"]
    for layer, position, index, bits in payload["silu"]:
        grouped[(silu_spec["event"], layer)][position][index] = bits
    logit_spec = EVENT_MAP["logit_event"]
    for position, index, bits in payload["logits"]:
        grouped[(logit_spec["event"], -1)][position][index] = bits

    writer = TraceWriter(ndjson_path, checkpoint_dir)
    try:
        specs_by_event = {}
        for stage_spec in EVENT_MAP["stages"].values():
            if "split" in stage_spec:
                for part in stage_spec["split"]:
                    specs_by_event[part["event"]] = part
            else:
                specs_by_event[stage_spec["event"]] = stage_spec
        specs_by_event[silu_spec["event"]] = silu_spec
        specs_by_event[logit_spec["event"]] = logit_spec
        for (template, layer), buckets in sorted(grouped.items(), key=str):
            spec = specs_by_event[template]
            _emit_mapped(writer, spec, layer if layer >= 0 else 0, buckets)
    finally:
        writer.close()
    return payload["meta"]


STAGE_NAMES = {
    0: "IDLE", 1: "EMBED", 2: "INPUT_NORM", 3: "Q_PROJ", 4: "K_PROJ",
    5: "V_PROJ", 6: "Q_NORM", 7: "K_NORM", 8: "ROPE", 9: "ATTN_SCORE",
    10: "SOFTMAX", 11: "ATTN_VALUE", 12: "O_PROJ", 13: "ATTN_RESIDUAL",
    14: "POST_NORM", 15: "GATE_PROJ", 16: "UP_PROJ", 17: "SILU",
    18: "DOWN_PROJ", 19: "MLP_RESIDUAL", 20: "NEXT_LAYER",
    21: "FINAL_NORM", 22: "LM_HEAD", 23: "ARGMAX", 24: "DONE",
}


def stage_breakdown(commit_log: str | Path) -> dict[str, Any]:
    """Measured per-stage cycle spans from stage-entry instrumentation.

    Each kind-5 record stamps the cycle counter when the controller enters a
    stage; consecutive entries of the same stage across a layer walk are
    summed, and the final entry runs to the meta cycle_count. Requires a
    trace produced by an instrumented build; older logs return {}.
    """
    payload = read_commit_log(commit_log)
    entries = payload.get("stage_entries") or []
    if not entries:
        return {}
    total_cycles = int(payload["meta"]["cycle_count"])
    per_stage: dict[str, dict[str, int]] = {}
    for i, (stage, _layer, cycle, rom, macs) in enumerate(entries):
        end = entries[i + 1][2] if i + 1 < len(entries) else total_cycles
        span = max(0, end - cycle)
        name = STAGE_NAMES.get(stage, f"S{stage}")
        row = per_stage.setdefault(name, {"entries": 0, "cycles": 0, "rom_reads": 0, "macs": 0})
        row["entries"] += 1
        row["cycles"] += span
        row["rom_reads"] += rom - entries[i - 1][3] if i else 0
        row["macs"] += macs - entries[i - 1][4] if i else 0
    return {
        "total_cycles": total_cycles,
        "stages": [
            {"stage": name, **row} for name, row in per_stage.items()
        ],
    }


def iter_mandatory_names(token_count: int) -> Iterator[str]:
    profile = json.loads(
        (
            ROOT / "spec" / "arithmetic" / "qwen-ref-v2" / "profile.json"
        ).read_text(encoding="utf-8")
    )
    for template in profile["mandatory_events"]:
        if "{layer}" in template and "{token}" in template:
            for layer in range(int(GEOM["layers"])):
                for token in range(token_count):
                    yield template.replace("{layer}", str(layer)).replace(
                        "{token}", str(token)
                    )
        elif "{layer}" in template:
            for layer in range(int(GEOM["layers"])):
                yield template.replace("{layer}", str(layer))
        else:
            yield template
