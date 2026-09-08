"""Multi-hardware bit-exact verification harness.

One contract, four backends:

  mask_rom  unfused forward controller + mmap'd mask ROM   (FUSED_SCHEDULE=0)
  gpu       fused streaming-GEMM schedule                  (FUSED_SCHEDULE=1)
  tpu       MXU-style weight-stationary tile engine        (FUSED_SCHEDULE=2)
  lpu       weight-stationary interleaved engine           (FUSED_SCHEDULE=3)

Every backend consumes the same token file and ROM image and produces the
same artifacts (rtl.ndjson + checkpoints) that lab.compare already knows how
to check bit-for-bit against the Python oracle. `run_backend` drives one
backend; the orchestrator (lab/hw_verify.py) fans all four out in parallel
and reduces to one verdict per backend.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "qwen_chip" / "sim"
RUNS_ROOT = ROOT / "evidence" / "runs"


@dataclass(frozen=True)
class Backend:
    """One hardware backend: how to build it and where its binary lands.

    A backend is (schedule, memory scenario). The schedule selects the
    binary (FUSED_SCHEDULE baked in at build time); the memory scenario is
    a runtime plusarg on the same binary:

      memory="rom"  direct on-die ROM port (the mask-ROM chip's reality;
                    also the canonical per-engine measurement condition)
      memory="hbm"  +EXT_MEM_LAT=<ext_mem_lat> on the shared binary — the
                    same datapath paying HBM-class response latency. Data
                    words pass through unmodified, so bit-exactness is
                    unchanged by construction; only cycle timing moves.
    """

    key: str              # e.g. "gpu_hbm"
    label: str            # human-readable name for reports
    fused: int            # FUSED_SCHEDULE value baked into the build
    memory: str = "rom"   # "rom" | "hbm"
    ext_mem_lat: int = 0  # +EXT_MEM_LAT cycles when memory == "hbm"

    @property
    def obj_dir(self) -> Path:
        return SIM_DIR / f"obj_dir_fused{self.fused}"

    @property
    def binary(self) -> Path:
        return self.obj_dir / "Vtb_qwen3_full_forward"

    def build(self) -> Path:
        subprocess.run(
            ["make", "-C", str(SIM_DIR), "all", f"FUSED={self.fused}"],
            check=True,
            cwd=ROOT,
        )
        if not self.binary.is_file():
            raise FileNotFoundError(
                f"{self.key}: build did not produce {self.binary}"
            )
        return self.binary

    def work_dir(self, run_key: str) -> Path:
        return RUNS_ROOT / f"{run_key}_{self.key}"


# HBM-class response latency for the memory="hbm" scenarios, in cycles at
# the harness's 500 MHz lab clock (~80 ns). Own reference-design choice [E]:
# no published per-request HBM latency figure exists; the capacity figure
# used for utilization reporting (3.35 TB/s HBM3e) is [P], docs/references.md
# SS6. The binding constraint under this latency is the engines' own
# prefetch depth (measured), not a bandwidth cap — at 2 ports x 32 B the
# request stream is ~105x below HBM capacity, so a cap could never bind.
EXT_MEM_LATENCY_CYCLES = 40

BACKENDS: dict[str, Backend] = {
    # --- canonical per-engine measurements (direct ROM port) ---
    "mask_rom": Backend(
        key="mask_rom",
        label="Mask-ROM chip (unfused controller)",
        fused=0,
    ),
    "mask_rom_fused": Backend(
        key="mask_rom_fused",
        label="Mask-ROM chip (fused 2D schedule)",
        fused=1,
    ),
    "gpu": Backend(
        key="gpu",
        label="GPU tile (fused streaming GEMM)",
        fused=1,
    ),
    "tpu": Backend(
        key="tpu",
        label="TPU MXU (weight-stationary tile engine)",
        fused=2,
    ),
    "lpu": Backend(
        key="lpu",
        label="LPU (weight-stationary interleaved)",
        fused=3,
    ),
    # --- HBM-machine scenarios (same binaries, +EXT_MEM_LAT) ---
    "gpu_hbm": Backend(
        key="gpu_hbm",
        label="GPU tile (HBM latency)",
        fused=1,
        memory="hbm",
        ext_mem_lat=EXT_MEM_LATENCY_CYCLES,
    ),
    "tpu_hbm": Backend(
        key="tpu_hbm",
        label="TPU MXU (HBM latency)",
        fused=2,
        memory="hbm",
        ext_mem_lat=EXT_MEM_LATENCY_CYCLES,
    ),
    "lpu_hbm": Backend(
        key="lpu_hbm",
        label="LPU (HBM latency)",
        fused=3,
        memory="hbm",
        ext_mem_lat=EXT_MEM_LATENCY_CYCLES,
    ),
}


def run_backend(
    backend: Backend,
    token_ids: list[int],
    rom: Path,
    work: Path,
) -> dict[str, Any]:
    """Run one backend on the token list; return its rtl_meta + artifacts.

    Writes tokens.txt, runs the binary, and regenerates the NDJSON trace +
    checkpoints via lab.rtl_trace. All four engines emit the same commit-log
    format, so the same decoder applies.
    """
    from lab.rtl_trace import write_rtl_trace

    work.mkdir(parents=True, exist_ok=True)
    (work / "rtl").mkdir(exist_ok=True)
    token_file = work / "tokens.txt"
    token_file.write_text("\n".join(str(t) for t in token_ids) + "\n")

    trace_dir = work / "rtl"
    command = [
        str(backend.binary),
        f"+ROM_FILE={rom}",
        f"+TOKEN_FILE={token_file}",
        f"+TRACE_DIR={trace_dir}",
    ]
    if backend.memory == "hbm":
        command.append(f"+EXT_MEM_LAT={backend.ext_mem_lat}")
    subprocess.run(command, check=True, cwd=trace_dir)

    log = trace_dir / "rtl_commits.bin"
    if not log.is_file():
        raise FileNotFoundError(f"{backend.key}: commit log missing: {log}")
    meta = write_rtl_trace(log, work / "rtl.ndjson", work / "rtl_checkpoints")
    return {
        "backend": backend.key,
        "rtl_meta": meta,
        "memory": backend.memory,
        "ext_mem_lat": backend.ext_mem_lat if backend.memory == "hbm" else 0,
        "work": str(work),
    }
