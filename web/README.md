# Chip Trace Lab

Dependency-free browser visualizer for comparing Python reference and RTL traces
across all 28 Qwen decoder layers.

## Run

Serve this directory with any static file server:

```sh
python3 -m http.server 8000 --directory web
```

Open `http://localhost:8000`. Opening `index.html` directly also works, but
browser `file://` restrictions prevent fixture fetches, so the UI uses its
equivalent embedded sample.

No trace is uploaded or persisted. Parsing and rendering happen in the browser.

## Accepted trace shapes

The importer accepts JSON arrays, `{records: [...]}`, `{events: [...]}`,
`{trace: [...]}`, `{steps: [...]}`, `{cycles: [...]}`, layer-keyed objects, and
one JSON object per line (NDJSON/JSONL). Field aliases include:

| Concept | Recognized fields |
|---|---|
| Layer | `layer`, `layer_id`, `layer_idx`, `decoder_layer` |
| Cycle | `cycle`, `clk`, `timestamp`, `step`, `idx` |
| Stage | `stage`, `fsm_stage`, `state`, `op`, `name`, `section` |
| Value | `tensor`, `output`, `sample`, `values`, `value` |
| Attention | `attention`, `attn`, `attention_weights`, `heatmap` |
| Pipeline | `pipeline_stages`, `pipe`, or `pipeline.stages` |
| Memory | `memory_events`, a `traffic` array, or one traffic object |

Example:

```json
{"cycle":42,"layer":3,"stage":"q_proj","fsm_stage":"MAC","tensor":[0.25,-0.5],"shape":[1,2],"pipeline_utilization":0.75,"stall_cycles":1,"bank_conflicts":0,"active_lanes":192,"memory_events":[{"memory":"ROM","operation":"read","bank":2,"address":"0x1000","bytes":128}]}
```

`stages` may be an array of strings or objects when one record summarizes a
layer. Stage names are normalized into the 12 displayed architectural stages.

## Fixtures and tests

- `fixtures/python_trace.ndjson`: 28-layer reference sample
- `fixtures/rtl_trace.ndjson`: 28-layer RTL sample with a deliberate mismatch
  at layer 7 / score
- `fixtures/design_config.json`: baseline memory and compute configuration
- “Run self-tests” checks parsers, aliases, comparison tolerance, mismatch
  indexing, and sample layer coverage in the browser.

## Metric provenance and limits

- Imported RTL cycle, utilization, stall, conflict, area, and power fields are
  labeled **measured**. If area or power is absent, the measured-row plot uses
  documented sample placeholders (4.2 mm² / 2.6 W) only to remain visible.
- Every configuration-control result is labeled **estimated**. The compact
  analytical model is for design-space exploration, not synthesis, place and
  route, SRAM compilation, or cycle-accurate resimulation.
- Pairing uses layer, normalized stage, and occurrence order. Traces with
  different event ordering should include the same number of events per
  layer/stage or be pre-aligned.
- Tensor comparison operates on numeric samples present in the trace, not full
  tensors omitted by the producer. It uses absolute tolerance only.
- Large files are parsed in memory on the main browser thread. The values table
  renders 500 rows, traffic renders 1,000 rows, and tensor cards render 40
  records to keep interaction responsive; aggregate counts still use all rows.
- Attention rendering displays the first head if the trace contains a
  head-by-query-by-key tensor.
