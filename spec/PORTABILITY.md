# Linux and macOS verification

The production Python path uses only Python 3.11+ standard-library modules.
CI executes its unit suite and import audit on both Ubuntu and macOS.

```sh
python3 -m unittest discover -s tests -v
python3 -m qwen_ref inspect /path/to/Qwen3-0.6B
python3 -m qwen_ref pack-rom /path/to/Qwen3-0.6B qwen3.bf16rom
python3 -m qwen_ref forward qwen3.bf16rom 9707
```

RTL development tools:

| Platform | Icarus | Verilator | Yosys |
|---|---|---|---|
| Ubuntu | `apt install iverilog verilator yosys` | same | same |
| macOS/Homebrew | `brew install icarus-verilog verilator yosys` | same | same |

Then run:

```sh
make test-rtl
make -C qwen_chip/tests all
make synth
```

The committed RTL has no DPI, C-extension, absolute-path, or
platform-endianness dependency. ROM files are explicitly little-endian BF16;
foundry adapters own any physical macro word-lane conversion.
