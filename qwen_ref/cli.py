"""Command-line entry points for checkpoint inspection, ROM packing and inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import (
    LAYOUT_SHA256,
    ROM_BYTES,
    pack_rom,
    sha256_file,
    validate_checkpoint,
    write_manifest,
)
from .model import forward
from .rom import FlatBF16ROM
from .safetensors import SafeTensorFile, ShardedSafeTensors, open_checkpoint
from .server import serve
from .tokenizer import Tokenizer
from .trace import TraceWriter


def _source_hashes(checkpoint: SafeTensorFile | ShardedSafeTensors) -> dict[str, str]:
    if isinstance(checkpoint, SafeTensorFile):
        return {checkpoint.path.name: sha256_file(checkpoint.path)}
    return {
        file.path.name: sha256_file(file.path)
        for file in checkpoint.files.values()
    }


def _progress(done: int, total: int, message: str) -> None:
    print(f"[{done}/{total}] {message}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qwen_ref")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="validate a canonical checkpoint")
    inspect.add_argument("checkpoint")
    pack = commands.add_parser("pack-rom", help="pack canonical flat BF16 ROM")
    pack.add_argument("checkpoint")
    pack.add_argument("rom")
    pack.add_argument("--manifest")
    tokenize = commands.add_parser("tokenize", help="encode text with tokenizer.json")
    tokenize.add_argument("tokenizer")
    tokenize.add_argument("text")
    run = commands.add_parser("forward", help="run full canonical ROM forward")
    run.add_argument("rom")
    run.add_argument("token_ids", help="comma-separated integer token ids")
    run.add_argument("--manifest", help="ROM manifest (default: ROM path plus .json)")
    run.add_argument("--trace")
    run.add_argument("--checkpoints")
    http = commands.add_parser("serve", help="serve tokenize/forward JSON hooks")
    http.add_argument("rom")
    http.add_argument("--manifest", help="ROM manifest (default: ROM path plus .json)")
    http.add_argument("--tokenizer")
    http.add_argument("--host", default="127.0.0.1")
    http.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "tokenize":
        print(json.dumps(Tokenizer(args.tokenizer).encode(args.text)))
        return 0
    if args.command == "inspect":
        with open_checkpoint(args.checkpoint) as checkpoint:
            validate_checkpoint(checkpoint)
        print(json.dumps({"ok": True, "layout_sha256": LAYOUT_SHA256, "rom_bytes": ROM_BYTES}))
        return 0
    if args.command == "pack-rom":
        with open_checkpoint(args.checkpoint) as checkpoint:
            sources = _source_hashes(checkpoint)
            digest = pack_rom(checkpoint, args.rom)
        manifest = args.manifest or f"{args.rom}.json"
        write_manifest(manifest, digest, sources)
        print(json.dumps({"rom_sha256": digest, "manifest": manifest}))
        return 0
    if args.command == "forward":
        token_ids = [int(value) for value in args.token_ids.split(",") if value]
        trace = TraceWriter(args.trace, args.checkpoints) if args.trace else None
        try:
            manifest = args.manifest or f"{args.rom}.json"
            with FlatBF16ROM(args.rom, manifest) as rom:
                logits = forward(token_ids, rom, trace=trace, progress=_progress)
            print(json.dumps(logits))
        finally:
            if trace:
                trace.close()
        return 0
    if args.command == "serve":
        manifest = args.manifest or f"{args.rom}.json"
        rom = FlatBF16ROM(args.rom, manifest)
        tokenizer = Tokenizer(args.tokenizer) if args.tokenizer else None

        def infer(payload: dict[str, object]) -> object:
            raw_ids = payload["token_ids"]
            if not isinstance(raw_ids, list) or not all(isinstance(x, int) for x in raw_ids):
                raise ValueError("token_ids must be an integer list")
            return {"logits": forward(raw_ids, rom, progress=_progress)}

        routes = {"/forward": infer}
        if tokenizer:
            routes["/tokenize"] = lambda payload: {
                "token_ids": tokenizer.encode(str(payload["text"]))
            }
        try:
            serve(routes, args.host, args.port)
        finally:
            rom.close()
        return 0
    raise AssertionError("unreachable")
