(() => {
  "use strict";

  const SHARE_DEMO = document.documentElement.hasAttribute("data-share-demo");
  // Machine id: --tier T4. Lab name: this sentence, all 28 layers, all logits.
  const FULL_COMPARE = "full-statement compare";
  const tensorCache = new Map();

  async function apiFetch(url, opts) {
    const res = await fetch(url, opts);
    if (!SHARE_DEMO && res.status === 401) {
      location.replace("/login");
      throw new Error("login required");
    }
    return res;
  }

  const OP_ROWS = [
    ["input_norm"],
    ["q_proj", "k_proj", "v_proj"],
    ["q_norm", "k_norm"],
    ["q_rope", "k_rope"],
    ["attention_scores", "attention_softmax", "causal_gqa"],
    ["o_proj", "attention_residual"],
    ["post_norm"],
    ["gate_proj", "up_proj"],
    ["silu", "swiglu"],
    ["down_proj", "output"]
  ];
  const OP_LABEL = {
    input_norm: "input RMSNorm",
    q_proj: "Q proj",
    k_proj: "K proj",
    v_proj: "V proj",
    q_norm: "Q RMSNorm",
    k_norm: "K RMSNorm",
    q_rope: "Q RoPE",
    k_rope: "K RoPE",
    attention_scores: "attn scores",
    attention_softmax: "softmax",
    causal_gqa: "causal GQA",
    o_proj: "O proj",
    attention_residual: "attn residual",
    post_norm: "post RMSNorm",
    gate_proj: "gate proj",
    up_proj: "up proj",
    silu: "SiLU",
    swiglu: "SwiGLU",
    down_proj: "down proj",
    output: "MLP residual"
  };
  const OP_EQ = {
    embedding: "h = E[token_id]",
    input_norm: "y = (x / RMS(x)) ⊙ γ",
    q_proj: "Q = x W_Q",
    k_proj: "K = x W_K",
    v_proj: "V = x W_V",
    q_norm: "Q̂ = RMSNorm(Q) ⊙ γ_q",
    k_norm: "K̂ = RMSNorm(K) ⊙ γ_k",
    q_rope: "Q̃ = RoPE(Q̂, pos)",
    k_rope: "K̃ = RoPE(K̂, pos)",
    attention_scores: "S = (Q̃ · K̃ᵀ) / √128",
    attention_softmax: "pᵢ = exp(sᵢ−m) / Σ exp  ·  exp stays FP32",
    causal_gqa: "C = P Ṽ   (GQA 16→8)",
    o_proj: "u = concat(C) W_O",
    attention_residual: "x ← x + u",
    post_norm: "y = (x / RMS(x)) ⊙ γ",
    gate_proj: "g = x W_g",
    up_proj: "v = x W_u",
    silu: "σ̃(g) = g · sigmoid(g)",
    swiglu: "h = σ̃(g) ⊙ v",
    down_proj: "z = h W_d",
    output: "x ← x + z",
    final_norm: "y = (x / RMS(x)) ⊙ γ",
    logits: "ℓ = y Eᵀ   (tied embed)",
    argmax: "t* = argmax(ℓ)",
    residual: "x ← embed or previous decoder"
  };
  const OP_SHAPE = {
    embedding: "[seq, 1024]",
    input_norm: "[seq, 1024]",
    q_proj: "[seq, 16, 128]",
    k_proj: "[seq, 8, 128]",
    v_proj: "[seq, 8, 128]",
    q_norm: "[seq, 16, 128]",
    k_norm: "[seq, 8, 128]",
    q_rope: "[seq, 16, 128]",
    k_rope: "[seq, 8, 128]",
    attention_scores: "[16 heads, ctx]",
    attention_softmax: "[16 heads, ctx]",
    causal_gqa: "[seq, 16, 128]",
    o_proj: "[seq, 1024]",
    attention_residual: "[seq, 1024]",
    post_norm: "[seq, 1024]",
    gate_proj: "[seq, 3072]",
    up_proj: "[seq, 3072]",
    silu: "[seq, 3072]",
    swiglu: "[seq, 3072]",
    down_proj: "[seq, 1024]",
    output: "[seq, 1024]",
    final_norm: "[seq, 1024]",
    logits: "[seq, 151936]",
    argmax: "scalar token id",
    residual: "[seq, 1024]"
  };
  const DAG_OPS = OP_ROWS.flat();
  const OP_CLASS = {
    embedding: "embed",
    residual: "resid",
    input_norm: "norm",
    q_proj: "proj",
    k_proj: "proj",
    v_proj: "proj",
    o_proj: "proj",
    gate_proj: "proj",
    up_proj: "proj",
    down_proj: "proj",
    q_norm: "norm",
    k_norm: "norm",
    post_norm: "norm",
    final_norm: "norm",
    q_rope: "rope",
    k_rope: "rope",
    attention_scores: "attn",
    attention_softmax: "attn",
    causal_gqa: "attn",
    silu: "act",
    swiglu: "act",
    attention_residual: "resid",
    output: "resid",
    logits: "head",
    argmax: "head"
  };
  const DAG_SPAN = {
    residual: "1 / -1",
    input_norm: "1 / -1",
    q_proj: "1 / 3",
    k_proj: "3 / 5",
    v_proj: "5 / 7",
    q_norm: "1 / 3",
    k_norm: "3 / 5",
    q_rope: "1 / 3",
    k_rope: "3 / 5",
    attention_scores: "1 / 3",
    attention_softmax: "3 / 5",
    causal_gqa: "5 / 7",
    o_proj: "1 / 4",
    attention_residual: "4 / 7",
    post_norm: "1 / -1",
    gate_proj: "1 / 4",
    up_proj: "4 / 7",
    silu: "1 / 4",
    swiglu: "4 / 7",
    down_proj: "1 / 4",
    output: "4 / 7"
  };
  const DAG_EDGES = [
    { from: "residual", to: "input_norm", lab: "x" },
    { from: "input_norm", to: "q_proj", lab: "x" },
    { from: "input_norm", to: "k_proj", lab: "x" },
    { from: "input_norm", to: "v_proj", lab: "x" },
    { from: "q_proj", to: "q_norm", lab: "Q" },
    { from: "k_proj", to: "k_norm", lab: "K" },
    { from: "q_norm", to: "q_rope", lab: "Q̂" },
    { from: "k_norm", to: "k_rope", lab: "K̂" },
    { from: "q_rope", to: "attention_scores", lab: "Q̃" },
    { from: "k_rope", to: "attention_scores", lab: "K̃" },
    { from: "attention_scores", to: "attention_softmax", lab: "S" },
    { from: "attention_softmax", to: "causal_gqa", lab: "P" },
    { from: "v_proj", to: "causal_gqa", lab: "V", kind: "skip-right" },
    { from: "causal_gqa", to: "o_proj", lab: "C" },
    { from: "o_proj", to: "attention_residual", lab: "u" },
    { from: "residual", to: "attention_residual", lab: "x", kind: "skip-left" },
    { from: "attention_residual", to: "post_norm", lab: "x" },
    { from: "post_norm", to: "gate_proj", lab: "x" },
    { from: "post_norm", to: "up_proj", lab: "x" },
    { from: "gate_proj", to: "silu", lab: "g" },
    { from: "silu", to: "swiglu", lab: "σ̃(g)" },
    { from: "up_proj", to: "swiglu", lab: "v" },
    { from: "swiglu", to: "down_proj", lab: "h" },
    { from: "down_proj", to: "output", lab: "z" },
    { from: "attention_residual", to: "output", lab: "x", kind: "skip-left" }
  ];
  const OP_FEEDS = {
    embedding: { in: [{ kind: "token", label: "token_id" }], w: "E [151936×1024] ROM" },
    input_norm: { in: [{ op: "residual", label: "x" }], w: "γ_in [1024] ROM" },
    q_proj: { in: [{ op: "input_norm", label: "x" }], w: "W_Q [1024×2048] ROM" },
    k_proj: { in: [{ op: "input_norm", label: "x" }], w: "W_K [1024×1024] ROM" },
    v_proj: { in: [{ op: "input_norm", label: "x" }], w: "W_V [1024×1024] ROM" },
    q_norm: { in: [{ op: "q_proj", label: "Q" }], w: "γ_q [128] ROM" },
    k_norm: { in: [{ op: "k_proj", label: "K" }], w: "γ_k [128] ROM" },
    q_rope: { in: [{ op: "q_norm", label: "Q̂" }, { kind: "pos", label: "pos" }], w: null },
    k_rope: { in: [{ op: "k_norm", label: "K̂" }, { kind: "pos", label: "pos" }], w: null },
    attention_scores: { in: [{ op: "q_rope", label: "Q̃" }, { op: "k_rope", label: "K̃" }], w: null },
    attention_softmax: { in: [{ op: "attention_scores", label: "S" }], w: null },
    causal_gqa: { in: [{ op: "attention_softmax", label: "P" }, { op: "v_proj", label: "V" }], w: null },
    o_proj: { in: [{ op: "causal_gqa", label: "C" }], w: "W_O [2048×1024] ROM" },
    attention_residual: { in: [{ op: "residual", label: "x" }, { op: "o_proj", label: "u" }], w: null },
    post_norm: { in: [{ op: "attention_residual", label: "x" }], w: "γ_post [1024] ROM" },
    gate_proj: { in: [{ op: "post_norm", label: "x" }], w: "W_g [1024×3072] ROM" },
    up_proj: { in: [{ op: "post_norm", label: "x" }], w: "W_u [1024×3072] ROM" },
    silu: { in: [{ op: "gate_proj", label: "g" }], w: null },
    swiglu: { in: [{ op: "silu", label: "σ̃(g)" }, { op: "up_proj", label: "v" }], w: null },
    down_proj: { in: [{ op: "swiglu", label: "h" }], w: "W_d [3072×1024] ROM" },
    output: { in: [{ op: "attention_residual", label: "x" }, { op: "down_proj", label: "z" }], w: null },
    final_norm: { in: [{ op: "last_output", label: "x" }], w: "γ_f [1024] ROM" },
    logits: { in: [{ op: "final_norm", label: "y" }], w: "Eᵀ tied ROM" },
    argmax: { in: [{ op: "logits", label: "ℓ" }], w: null }
  };
  const EVENT_STAGE = {
    embedding: "EMBED",
    input_norm: "INPUT_NORM",
    q_proj: "Q_PROJ",
    k_proj: "K_PROJ",
    v_proj: "V_PROJ",
    q_norm: "Q_NORM",
    k_norm: "K_NORM",
    q_rope: "ROPE",
    k_rope: "ROPE",
    attention_scores: "ATTN_SCORE",
    attention_softmax: "SOFTMAX",
    causal_gqa: "ATTN_VALUE",
    o_proj: "O_PROJ",
    attention_residual: "ATTN_RESIDUAL",
    post_norm: "POST_NORM",
    gate_proj: "GATE_PROJ",
    up_proj: "UP_PROJ",
    silu: "SILU",
    swiglu: "SILU",
    down_proj: "DOWN_PROJ",
    output: "MLP_RESIDUAL",
    final_norm: "FINAL_NORM",
    logits: "LM_HEAD",
    argmax: "ARGMAX"
  };
  const STAGE_OP = {
    IDLE: { global: "embedding" },
    EMBED: { global: "embedding" },
    INPUT_NORM: { op: "input_norm" },
    Q_PROJ: { op: "q_proj" },
    K_PROJ: { op: "k_proj" },
    V_PROJ: { op: "v_proj" },
    Q_NORM: { op: "q_norm" },
    K_NORM: { op: "k_norm" },
    ROPE: { op: "q_rope", extra: "k_rope" },
    ATTN_SCORE: { op: "attention_scores" },
    SOFTMAX: { op: "attention_softmax" },
    ATTN_VALUE: { op: "causal_gqa" },
    O_PROJ: { op: "o_proj" },
    ATTN_RESIDUAL: { op: "attention_residual" },
    POST_NORM: { op: "post_norm" },
    GATE_PROJ: { op: "gate_proj" },
    UP_PROJ: { op: "up_proj" },
    SILU: { op: "silu", extra: "swiglu" },
    DOWN_PROJ: { op: "down_proj" },
    MLP_RESIDUAL: { op: "output" },
    NEXT_LAYER: { op: "output" },
    FINAL_NORM: { global: "final_norm" },
    LM_HEAD: { global: "logits" },
    ARGMAX: { global: "argmax" },
    DONE: { global: "argmax" }
  };
  const STAGE_CALC = {
    EMBED: "token-id → 1024-d row of E",
    INPUT_NORM: "RMS + γ over H=1024 × 28 layers",
    Q_PROJ: "x W_Q · GEMM 1024×2048 × 28",
    K_PROJ: "x W_K · GEMM 1024×1024 × 28",
    V_PROJ: "x W_V · GEMM 1024×1024 × 28",
    Q_NORM: "RMSNorm per Q head, 16×128 × 28",
    K_NORM: "RMSNorm per K head, 8×128 × 28",
    ROPE: "rotate Q̂ and K̂ at pos × 28",
    ATTN_SCORE: "(Q̃·K̃ᵀ)/√128, 16 heads × ctx × 28",
    SOFTMAX: "softmax over ctx per head × 28",
    ATTN_VALUE: "P Ṽ, GQA 16→8 × 28",
    O_PROJ: "concat(C) W_O · GEMM 2048×1024 × 28",
    ATTN_RESIDUAL: "x ← x + u, H=1024 × 28",
    POST_NORM: "RMS + γ over H=1024 × 28",
    GATE_PROJ: "x W_g · GEMM 1024×3072 × 28",
    UP_PROJ: "x W_u · GEMM 1024×3072 × 28",
    SILU: "g·sigmoid(g) then ⊙ v, 3072 × 28",
    DOWN_PROJ: "h W_d · GEMM 3072×1024 × 28",
    MLP_RESIDUAL: "x ← x + z, H=1024 × 28",
    FINAL_NORM: "RMS + γ over H=1024, once",
    LM_HEAD: "y Eᵀ · GEMM 1024×151936 (tied embed)",
    ARGMAX: "argmax over 151936 logits"
  };
  const STAGE_ID_NAME = {
    0: "IDLE", 1: "EMBED", 2: "INPUT_NORM", 3: "Q_PROJ", 4: "K_PROJ", 5: "V_PROJ",
    6: "Q_NORM", 7: "K_NORM", 8: "ROPE", 9: "ATTN_SCORE", 10: "SOFTMAX",
    11: "ATTN_VALUE", 12: "O_PROJ", 13: "ATTN_RESIDUAL", 14: "POST_NORM",
    15: "GATE_PROJ", 16: "UP_PROJ", 17: "SILU", 18: "DOWN_PROJ", 19: "MLP_RESIDUAL",
    20: "NEXT_LAYER", 21: "FINAL_NORM", 22: "LM_HEAD", 23: "ARGMAX", 24: "DONE"
  };
  const STAGE_SKIP = new Set(["IDLE", "NEXT_LAYER", "DONE"]);
  const LAYER_WALK = [
    "INPUT_NORM", "Q_PROJ", "K_PROJ", "V_PROJ", "Q_NORM", "K_NORM", "ROPE",
    "ATTN_SCORE", "SOFTMAX", "ATTN_VALUE", "O_PROJ", "ATTN_RESIDUAL", "POST_NORM",
    "GATE_PROJ", "UP_PROJ", "SILU", "DOWN_PROJ", "MLP_RESIDUAL"
  ];
  const STAGE_BLOCK = {
    IDLE: "host",
    EMBED: "rom",
    INPUT_NORM: "rmsnorm",
    Q_PROJ: "mac",
    K_PROJ: "mac",
    V_PROJ: "mac",
    Q_NORM: "rmsnorm",
    K_NORM: "rmsnorm",
    ROPE: "rope",
    ATTN_SCORE: "mac",
    SOFTMAX: "softmax",
    ATTN_VALUE: "mac",
    O_PROJ: "mac",
    ATTN_RESIDUAL: "sram-hidden",
    POST_NORM: "rmsnorm",
    GATE_PROJ: "mac",
    UP_PROJ: "mac",
    SILU: "swiglu",
    DOWN_PROJ: "mac",
    MLP_RESIDUAL: "sram-hidden",
    NEXT_LAYER: "fsm",
    FINAL_NORM: "rmsnorm",
    LM_HEAD: "mac",
    ARGMAX: "logits",
    DONE: "logits",
    ERROR: "fsm"
  };
  const SRAM_STAGE = {
    ATTN_RESIDUAL: "sram-hidden",
    MLP_RESIDUAL: "sram-hidden",
    EMBED: "sram-hidden",
    K_PROJ: "sram-kv",
    V_PROJ: "sram-kv",
    Q_PROJ: "sram-ws",
    ATTN_SCORE: "sram-ws",
    SOFTMAX: "sram-ws",
    GATE_PROJ: "sram-ws",
    UP_PROJ: "sram-ws",
    SILU: "sram-ws",
    DOWN_PROJ: "sram-ws"
  };

  // Chip inspector: static identity per data-block, plus per-stage dynamics.
  const CHIP_BLOCKS = {
    host: {
      title: "host I/O",
      module: "tb_qwen3_full_forward",
      detail: "Supplies token IDs, position, and control only. Weights never cross this interface.",
      now: (ctx) => {
        const s = ctx.rtl || {};
        if (s.busy) return `driving position ${s.position ?? 0} · token handshake ${s.stage || "?"}`;
        return ctx.py && ctx.py.phase === "python"
          ? `idle · waiting on ${ctx.py.message || "Python oracle"}`
          : "cmd idle";
      }
    },
    fsm: {
      title: "controller FSM",
      module: "qwen3_forward_controller",
      detail: "One qwen_stage_e walk per token: 28 layers muxed over one cluster, then final norm and the tied lm_head.",
      now: (ctx) => {
        const s = ctx.rtl || {};
        if (!s.stage || s.stage === "IDLE") return "IDLE";
        const layer = typeof s.layer === "number" && s.layer >= 0 ? ` · layer ${s.layer + 1}/28` : "";
        return `${s.stage}${layer} · cycle ${fmt(s.cycle)}`;
      }
    },
    rom: {
      title: "weight ROM · u_rom",
      module: "qwen_mmap_rom  (sim) · on-die mask ROM (silicon)",
      detail: "Silicon, not a file size: 1,192,099,840 bytes × 8 = 9,536,798,720 bits. Mask ROM is 1 transistor per bit → 9.54 billion transistors. N4-class NOR cell 0.018 µm²/bit → 171.7 mm² of array. A 193i reticle field is ~26×32 mm ≈ 830 mm²; usable die after scribe is 600 mm², so this ROM is 29% of one reticle die. 8 banks × 256b is the read port (16 BF16s/beat), not the capacity. Host cannot write it. Sim: mmap of artifacts/qwen3.bf16rom.",
      now: (ctx) => {
        const s = ctx.rtl || {};
        const w = ctx.weightText || "no weight read this stage";
        return `reading ${w} · ${fmt(s.rom_reads)} ROM beats so far`;
      }
    },
    sram: {
      title: "scratch SRAM · u_sram",
      module: "qwen3_sram_top",
      detail: "8 banks × 256b R/W. Regions: live hidden, per-layer K/V cache (persistent across tokens), and an overlaid projection/MLP workspace.",
      now: (ctx) => {
        const s = ctx.rtl || {};
        const region = ctx.sramRegion ? `region ${ctx.sramRegion}` : "no SRAM traffic this stage";
        return `${region} · ${fmt(s.sram_reads)} R / ${fmt(s.sram_writes)} W so far`;
      }
    },
    "sram-hidden": {
      title: "SRAM · hidden activations",
      module: "qwen3_sram_top",
      detail: "Live hidden state and residuals. Untagged by layer: each layer reads the previous layer's committed BF16 output.",
      now: (ctx) => ctx.sramRegion === "sram-hidden" ? "active: hidden act read/write" : "idle"
    },
    "sram-kv": {
      title: "SRAM · K/V cache",
      module: "qwen3_sram_top",
      detail: "Post-RoPE K and V persist across tokens: K is written per (layer, position), V transposed for the value pass.",
      now: (ctx) => ctx.sramRegion === "sram-kv" ? "active: K/V cache write" : "idle"
    },
    "sram-ws": {
      title: "SRAM · workspace",
      module: "qwen3_sram_top",
      detail: "Overlaid scratch for projections, attention scores/probs, and the MLP intermediate.",
      now: (ctx) => ctx.sramRegion === "sram-ws" ? "active: workspace traffic" : "idle"
    },
    mac: {
      title: "MAC array · qwen_bf16_mac_array",
      module: "qwen_bf16_mac_array",
      detail: "16 lanes, BF16×BF16 products, serial binary32 accumulation per qwen-ref-v2 (round-ties-to-even, subnormals preserved).",
      now: (ctx) => {
        const s = ctx.rtl || {};
        if (!ctx.stageIsMac) return "idle";
        const reduce = s.reduce_index != null ? `reduce word ${s.reduce_index}` : "";
        const out = s.output_index != null ? `row ${s.output_index}` : "";
        return [out, reduce, `${fmt(s.macs)} MACs`].filter(Boolean).join(" · ");
      }
    },
    rmsnorm: {
      title: "RMSNorm engine · u_norm",
      module: "qwen3_rmsnorm_engine",
      detail: "y = (x / RMS(x)) ⊙ γ with the pinned ε; reciprocal square root comes from the table-backed NLIN unit.",
      now: (ctx) => ctx.stage === "INPUT_NORM" || ctx.stage === "POST_NORM" || ctx.stage === "FINAL_NORM"
        ? `active: ${ctx.stage}` : (ctx.stage === "Q_NORM" || ctx.stage === "K_NORM" ? `active: ${ctx.stage}` : "idle")
    },
    softmax: {
      title: "softmax engine · u_softmax",
      module: "qwen3_softmax_engine",
      detail: "qwen-ref-v2 row softmax. Pass 0: max of BF16 scores. Pass 1: FP32 exp(x − max) and a serial FP32 sum — the exponential is not written to SRAM. Pass 2: reread the BF16 scores, exp again in FP32, divide by that FP32 sum, then commit P to BF16. A 1-token row is P ≡ 1 and does not test the divide. Dividing a BF16 exponential is not this profile.",
      now: (ctx) => ctx.stage === "SOFTMAX" ? "active: 3-pass row softmax" : "idle"
    },
    rope: {
      title: "RoPE engine · u_rope",
      module: "qwen3_rope_engine",
      detail: "Rotate-half RoPE for positions 0-127 from the direct cos/sin ROM (theta 1e6, head dim 128).",
      now: (ctx) => ctx.stage === "ROPE" ? "active: rotating Q and K" : "idle"
    },
    swiglu: {
      title: "SwiGLU engine · u_swiglu",
      module: "qwen3_swiglu_engine",
      detail: "SiLU(gate) ⊙ up with the pinned table-backed sigmoid; emits both the SiLU value and the product.",
      now: (ctx) => ctx.stage === "SILU" ? "active: SiLU + product" : "idle"
    },
    logits: {
      title: "lm_head stream",
      module: "qwen3_forward_controller",
      detail: "Tied embedding rows streamed as the head; 151936 logits reduced to argmax in flight, never buffered.",
      now: (ctx) => {
        const s = ctx.rtl || {};
        if (ctx.stage === "LM_HEAD") return `row ${fmt(s.output_index)} / 151,936`;
        if (ctx.stage === "ARGMAX" || ctx.stage === "DONE") {
          return state.argmax != null ? `argmax ${state.argmax} ${state.argmaxText ? JSON.stringify(state.argmaxText) : ""}` : "argmax ready";
        }
        return "idle";
      }
    }
  };

  // Which network op's tensor flows through a chip block at a given stage.
  const BLOCK_STAGE_OP = {
    rom: (stage) => (STAGE_OP[stage] || {}).op || (STAGE_OP[stage] || {}).global || null,
    sram: (stage) => (STAGE_OP[stage] || {}).op || (STAGE_OP[stage] || {}).global || null,
    "sram-hidden": (stage) => (stage === "ATTN_RESIDUAL" || stage === "MLP_RESIDUAL" || stage === "EMBED")
      ? ((STAGE_OP[stage] || {}).op || "embedding") : null,
    "sram-kv": (stage) => (stage === "K_PROJ" || stage === "V_PROJ") ? (STAGE_OP[stage] || {}).op : null,
    "sram-ws": (stage) => (STAGE_OP[stage] || {}).op || null,
    mac: (stage) => (STAGE_OP[stage] || {}).op || null,
    rmsnorm: (stage) => (STAGE_OP[stage] || {}).op || null,
    softmax: (stage) => (stage === "SOFTMAX") ? "attention_softmax" : null,
    rope: (stage) => (stage === "ROPE") ? "q_rope" : null,
    swiglu: (stage) => (stage === "SILU") ? "silu" : null,
    logits: (stage) => (stage === "LM_HEAD") ? "logits" : (stage === "ARGMAX" || stage === "DONE") ? "argmax" : null,
    fsm: (stage) => (STAGE_OP[stage] || {}).op || (STAGE_OP[stage] || {}).global || null,
    host: () => null
  };

  const $ = (id) => document.getElementById(id);
  const state = {
    jobId: null,
    workDir: null,
    runStartedAt: null,
    inspectorBlock: null,
    lastStage: null,
    lastRtlStatus: null,
    lastPyStatus: null,
    poll: null,
    playTimer: null,
    playing: false,
    userPaused: false,
    index: 0,
    timeline: [],
    mismatch: new Set(),
    tokenIds: [],
    tokenTexts: [],
    argmax: null,
    argmaxText: "",
    prompt: "",
    predictedText: "",
    rtlSignedOff: false,
    replayOnly: false,
    viewLayer: 0,
    lastHotOp: null,
    layerSeen: new Set(),
    dagObs: null,
    runFinished: false,
    progressPeak: 0,
    slide: 0,
    stageLoaded: false,
    stageSched: "fused",
    stageBreakdown: null,
    walks: null,
    walkSched: "fused",
    lastScenario: null,
    compare: null,
    modal: { rec: null, source: "python", offset: 0, total: 0 }
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function fmtVal(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    return n.toPrecision(4);
  }

  function sampleLine(vals, n) {
    const take = n || 4;
    if (!vals || !vals.length) return "—";
    return vals.slice(0, take).map(fmtVal).join("  ") + (vals.length > take ? " …" : "");
  }

  function shapeText(rec, op) {
    if (rec && rec.shape && rec.shape.length) return "[" + rec.shape.join(" × ") + "]";
    return OP_SHAPE[op] || "";
  }

  function recMapThrough(index) {
    const map = new Map();
    const end = Math.min(index, state.timeline.length - 1);
    for (let i = 0; i <= end; i++) {
      const rec = state.timeline[i];
      map.set(nodeId(rec.event, rec.layer), rec);
    }
    return map;
  }

  function resolveFeed(feed, layer, map) {
    if (!feed || !feed.op) return null;
    if (feed.op === "residual") {
      const id = layer === 0 ? "g-embedding" : `L${layer - 1}-output`;
      return map.get(id) || null;
    }
    if (feed.op === "last_output") return map.get("L27-output") || null;
    if (typeof layer === "number" && layer >= 0) return map.get(`L${layer}-${feed.op}`) || null;
    return map.get("g-" + feed.op) || null;
  }

  function feedText(feed, rec, map) {
    if (feed.kind === "token") {
      const pos = rec.token == null || rec.token < 0 ? 0 : rec.token;
      const id = state.tokenIds[pos];
      return `token_id = ${id == null ? "—" : id}   pos = ${pos}`;
    }
    if (feed.kind === "pos") {
      const pos = rec.token == null || rec.token < 0 ? 0 : rec.token;
      return `pos = ${pos}`;
    }
    const src = resolveFeed(feed, rec.layer, map);
    const shp = src ? shapeText(src, feed.op) : (OP_SHAPE[feed.op] || "");
    return `${feed.label} ${shp}  ${sampleLine(src && src.values)}`;
  }

  // Compact node body: label only. Equation, shape, weight source, and the
  // live sample move to a hover tooltip so the DAG fits a small area.
  function neuronInner(op, title) {
    const tip = [
      OP_LABEL[op] || op,
      OP_EQ[op] || "",
      `out ${OP_SHAPE[op] || ""}`,
      (OP_FEEDS[op] && OP_FEEDS[op].w) ? `W ${OP_FEEDS[op].w}` : ""
    ].filter(Boolean).join("\n");
    return `<span class="soma" aria-hidden="true"></span>` +
      `<b>${title || OP_LABEL[op] || op}</b>` +
      `<span class="tip" role="tooltip">${escapeHtml(tip)}</span>`;
  }

  function showCompute(rec) {
    if (!rec) return;
    const op = opKey(rec.event);
    const map = recMapThrough(state.index);
    const feeds = OP_FEEDS[op] || { in: [], w: null };
    $("computeName").textContent = rec.event;
    $("computeEq").textContent = OP_EQ[op] || rec.event;
    const n = rec.elements != null ? rec.elements : "";
    $("computeShape").textContent = `out ${shapeText(rec, op)}${n ? ` · ${n} F32 · first ${Math.min(16, (rec.values || []).length)} shown` : ""}`;
    $("computeIns").innerHTML = (feeds.in || []).map((feed) =>
      `<div><em>in</em> ${feedText(feed, rec, map)}</div>`
    ).join("") + (feeds.w ? `<div><em>W</em> ${feeds.w}</div>` : "");
    const vals = rec.values || [];
    $("computeVals").innerHTML = vals.map((v) => `<span>${fmtVal(v)}</span>`).join("") ||
      "<span class='muted'>tensor samples load from the Python/RTL checkpoint</span>";
    if ($("dieEq")) $("dieEq").textContent = OP_EQ[op] || rec.event;
    if ($("dieIo")) {
      const firstIn = (feeds.in || [])[0];
      $("dieIo").textContent = (firstIn ? "in " + feedText(firstIn, rec, map) + "  ·  " : "") +
        "out " + sampleLine(vals, 6);
    }
  }

  function paintOpSample(el, rec, map) {
    if (!el || !rec) return;
    const op = opKey(rec.event);
    const tip = el.querySelector(".tip");
    if (!tip) return;
    const feeds = OP_FEEDS[op] || { in: [], w: null };
    const lines = [
      OP_LABEL[op] || op,
      OP_EQ[op] || "",
      `out ${shapeText(rec, op)}`,
      (feeds.w ? `W ${feeds.w}` : null),
      `sample ${sampleLine(rec.values, 4)}`
    ].filter(Boolean);
    tip.textContent = lines.join("\n");
  }

  function fillDagLayer(layer, map) {
    const L = typeof layer === "number" && layer >= 0 ? layer : state.viewLayer || 0;
    map = map || recMapThrough(state.timeline.length - 1);
    DAG_OPS.forEach((op) => {
      paintOpSample($("dag-" + op), map.get(`L${L}-${op}`), map);
    });
    const res = L === 0 ? map.get("g-embedding") : map.get(`L${L - 1}-output`);
    const port = $("dag-residual");
    if (port && res) {
      const tip = port.querySelector(".tip");
      if (tip) tip.textContent =
        `x in — ${shapeText(res, L === 0 ? "embedding" : "output")}\nsample ${sampleLine(res.values, 4)}`;
    }
    requestAnimationFrame(drawDagEdges);
  }

  function fillTensorPreviews() {
    const map = recMapThrough(state.timeline.length - 1);
    ["embedding", "final_norm", "logits"].forEach((op) => {
      paintOpSample($("g-" + op), map.get("g-" + op), map);
    });
    fillDagLayer(state.viewLayer, map);
    paintOutput();
  }

  function fmt(n) {
    if (n === undefined || n === null || n === "") return "—";
    const v = Number(n);
    if (!Number.isFinite(v)) return String(n);
    return v.toLocaleString("en-US");
  }

  function setBadge(el, text, kind) {
    el.textContent = text;
    el.className = "badge" + (kind ? " " + kind : "");
  }

  function opKey(event) {
    const name = String(event || "");
    if (name === "embedding" || name === "final_norm" || name === "logits" || name === "argmax") return name;
    const parts = name.split(".");
    return parts[parts.length - 1];
  }

  function nodeId(event, layer) {
    const op = opKey(event);
    if (op === "embedding" || op === "final_norm" || op === "logits" || op === "argmax") return "g-" + op;
    if (typeof layer === "number" && layer >= 0) return `L${layer}-${op}`;
    return "g-" + op;
  }

  function eventStage(event) {
    return EVENT_STAGE[opKey(event)] || "IDLE";
  }

  const GPU_OP_SHORT = {
    residual: "x in",
    input_norm: "RMS",
    q_proj: "Q",
    k_proj: "K",
    v_proj: "V",
    q_norm: "Qn",
    k_norm: "Kn",
    q_rope: "Qθ",
    k_rope: "Kθ",
    attention_scores: "scores",
    attention_softmax: "softmax",
    causal_gqa: "GQA",
    o_proj: "O",
    attention_residual: "+attn",
    post_norm: "post",
    gate_proj: "gate",
    up_proj: "up",
    silu: "SiLU",
    swiglu: "SwiGLU",
    down_proj: "down",
    output: "+MLP"
  };

  function dagNode(op, title, extraClass, prefix) {
    const span = DAG_SPAN[op] || "1 / -1";
    const p = prefix || "dag-";
    const cls = OP_CLASS[op] || "other";
    const circ = p === "g2-" ? ` gpu-circ cls-${cls}` : "";
    const shown = p === "g2-" ? (GPU_OP_SHORT[op] || title) : title;
    return `<button type="button" class="op${extraClass ? " " + extraClass : ""}${circ}" id="${p}${op}" data-dag-op="${op}" data-op="${op}" data-cls="${cls}" style="grid-column:${span}">${neuronInner(op, shown)}</button>`;
  }

  function decoderDagOps(prefix) {
    return [
      dagNode("residual", "x in", "port", prefix),
      dagNode("input_norm", null, null, prefix),
      dagNode("q_proj", null, null, prefix), dagNode("k_proj", null, null, prefix), dagNode("v_proj", null, null, prefix),
      dagNode("q_norm", null, null, prefix), dagNode("k_norm", null, null, prefix),
      dagNode("q_rope", null, null, prefix), dagNode("k_rope", null, null, prefix),
      dagNode("attention_scores", null, null, prefix), dagNode("attention_softmax", null, null, prefix), dagNode("causal_gqa", null, null, prefix),
      dagNode("o_proj", null, null, prefix), dagNode("attention_residual", null, null, prefix),
      dagNode("post_norm", null, null, prefix),
      dagNode("gate_proj", null, null, prefix), dagNode("up_proj", null, null, prefix),
      dagNode("silu", null, null, prefix), dagNode("swiglu", null, null, prefix),
      dagNode("down_proj", null, null, prefix), dagNode("output", null, null, prefix)
    ].join("");
  }

  // Token-serial decode: the chip walks L00..L27 once per token. The trace
  // is layer-major within one token's pass, so a pass boundary is where the
  // layer index resets to 0 (or a global op like logits/final_norm appears).
  function passInfo() {
    if (!state.timeline.length) return null;
    const idx = Math.min(state.index, state.timeline.length - 1);
    let pass = 0;
    let prevLayer = -1;
    for (let i = 0; i <= idx; i++) {
      const rec = state.timeline[i];
      const layer = typeof rec.layer === "number" ? rec.layer : -1;
      if (layer === 0 && prevLayer > 0) pass += 1;
      if (layer >= 0) prevLayer = layer;
    }
    // Total passes = number of layer-0 resets + 1, capped by token count.
    let total = 1;
    prevLayer = -1;
    for (const rec of state.timeline) {
      const layer = typeof rec.layer === "number" ? rec.layer : -1;
      if (layer === 0 && prevLayer > 0) total += 1;
      if (layer >= 0) prevLayer = layer;
    }
    return { pass: Math.min(pass, total - 1), total };
  }

  function bindDagNodes(layer) {
    state.viewLayer = layer;
    const pass = passInfo();
    const passHtml = pass ? `<span class="pass-chip">pass ${pass.pass + 1}/${pass.total} · layer-major</span>` : "";
    const title = $("dagTitle");
    if (title) {
      title.innerHTML = `decoder ${String(layer).padStart(2, "0")} · 28 layers share this graph${passHtml}`;
    }
    const res = $("dag-residual");
    if (res) res.setAttribute("data-node", layer === 0 ? "g-embedding" : `L${layer - 1}-output`);
    DAG_OPS.forEach((op) => {
      const el = $("dag-" + op);
      if (el) el.setAttribute("data-node", `L${layer}-${op}`);
    });
  }

  function dagAnchor(el, dag, which) {
    const soma = el.classList.contains("gpu-circ") ? el.querySelector(".soma") : null;
    const er = (soma || el).getBoundingClientRect();
    const dr = dag.getBoundingClientRect();
    const x = er.left - dr.left + er.width / 2;
    const left = er.left - dr.left;
    const right = er.right - dr.left;
    const top = er.top - dr.top;
    const bot = er.bottom - dr.top;
    const midY = top + er.height / 2;
    const pad = soma ? 0 : 4;
    if (which === "left") return { x: left, y: midY };
    if (which === "right") return { x: right, y: midY };
    if (which === "top") return { x, y: top - (soma ? 0 : pad) };
    return { x, y: bot + (soma ? 0 : 2) };
  }

  function routeEdge(a, b, kind, width) {
    const fx = (n) => n.toFixed(1);
    if (kind === "skip-left") {
      const gx = 16;
      return {
        d: `M ${fx(a.x)} ${fx(a.y)} H ${fx(gx)} V ${fx(b.y)} H ${fx(b.x)}`,
        lab: { x: gx + 8, y: (a.y + b.y) / 2, anchor: "start" }
      };
    }
    if (kind === "skip-right") {
      const gx = width - 16;
      return {
        d: `M ${fx(a.x)} ${fx(a.y)} H ${fx(gx)} V ${fx(b.y)} H ${fx(b.x)}`,
        lab: { x: gx - 8, y: (a.y + b.y) / 2, anchor: "end" }
      };
    }
    const gap = b.y - a.y;
    const busY = a.y + (gap > 20 ? Math.min(12, gap * 0.45) : Math.max(6, gap * 0.4));
    return {
      d: `M ${fx(a.x)} ${fx(a.y)} V ${fx(busY)} H ${fx(b.x)} V ${fx(b.y)}`,
      lab: { x: b.x, y: (busY + b.y) / 2 - 1, anchor: "middle" }
    };
  }

  function restoreEdgeLive(svg, dag) {
    const root = dag || $("dag");
    const edges = svg || $("dagEdges");
    if (!root || !edges) return;
    root.querySelectorAll(".op.active, .op.on-path").forEach((el) => {
      const op = el.getAttribute("data-op");
      edges.querySelectorAll(`[data-to="${op}"]`).forEach((e) => e.classList.add("live"));
    });
  }

  function drawDagEdges(opts) {
    const dag = (opts && opts.dag) || $("dag");
    const svg = (opts && opts.svg) || $("dagEdges");
    const prefix = (opts && opts.prefix) || "dag-";
    const marker = (opts && opts.marker) || "arr";
    if (!dag || !svg) return;
    const w = Math.max(1, dag.clientWidth);
    const h = Math.max(1, dag.clientHeight);
    svg.setAttribute("width", String(w));
    svg.setAttribute("height", String(h));
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const live = marker + "Live";
    const defs = `<defs>
      <marker id="${marker}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M0,1 L9.5,5 L0,9 z" fill="#0f172a"/>
      </marker>
      <marker id="${live}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9.5" markerHeight="9.5" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M0,1 L9.5,5 L0,9 z" fill="#0f766e"/>
      </marker>
    </defs>`;
    const paths = DAG_EDGES.map((edge) => {
      const src = dag.querySelector("#" + prefix + edge.from);
      const dst = dag.querySelector("#" + prefix + edge.to);
      if (!src || !dst) return "";
      const kind = edge.kind || "fwd";
      const a = dagAnchor(src, dag, kind === "skip-left" ? "left" : kind === "skip-right" ? "right" : "bottom");
      const b = dagAnchor(dst, dag, kind === "skip-left" ? "left" : kind === "skip-right" ? "right" : "top");
      const routed = routeEdge(a, b, kind, w);
      const cls = kind === "fwd" ? "edge" : "edge skip";
      return `<path class="${cls}" data-from="${edge.from}" data-to="${edge.to}" d="${routed.d}" marker-end="url(#${marker})"/>` +
        `<text class="edge-lab" data-from="${edge.from}" data-to="${edge.to}" text-anchor="${routed.lab.anchor}" x="${routed.lab.x}" y="${routed.lab.y}">${escapeHtml(edge.lab)}</text>`;
    }).join("");
    svg.innerHTML = defs + paths;
    restoreEdgeLive(svg, dag);
  }

  function drawNnDagEdges() {
    drawDagEdges({ dag: $("nnDag"), svg: $("nnDagEdges"), prefix: "nn-", marker: "nnArr" });
  }

  // Hover tooltips: fixed-position at the cursor so they never affect
  // layout. One listener on the network container covers all nodes.
  function bindTooltips() {
    bindTooltipsOn($("network"));
    bindTooltipsOn($("gpuNet"));
  }

  function bindTooltipsOn(net) {
    if (!net || net.dataset.tipBound) return;
    net.dataset.tipBound = "1";
    net.addEventListener("mousemove", (ev) => {
      const node = ev.target.closest(".op, .net-node");
      const tip = node && node.querySelector(".tip");
      net.querySelectorAll(".tip.live").forEach((t) => {
        if (t !== tip) t.classList.remove("live");
      });
      if (!tip) return;
      tip.classList.add("live");
      const pad = 14;
      const w = Math.min(tip.offsetWidth, 260);
      const x = Math.min(ev.clientX + pad, window.innerWidth - w - 8);
      const y = Math.min(ev.clientY + pad, window.innerHeight - tip.offsetHeight - 8);
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    });
    net.addEventListener("mouseleave", () => {
      net.querySelectorAll(".tip.live").forEach((t) => t.classList.remove("live"));
    });
  }

  function renderNetwork() {
    const dagOps = decoderDagOps("dag-");
    const rail = Array.from({ length: 28 }, (_, i) =>
      `<button type="button" class="layer-dot" data-layer="${i}" data-jump-layer="${i}">L${String(i).padStart(2, "0")}</button>`
    ).join("");
    $("network").innerHTML = `
      <p class="rail-caption">Layer-major walk: all tokens pass L00, then all pass L01, … L27 — one lit token at a time, K/V cached for the later tokens. The chip's measured trace follows this order; the rail lights top-to-bottom once per layer, not once per token.</p>
      <button type="button" class="op net-node wide" id="g-embedding" data-node="g-embedding" data-op="embedding">${neuronInner("embedding", "embedding")}</button>
      <div class="flow-edge"><i>↓</i><span>h feeds decoder x</span></div>
      <div class="layer-rail" id="layerRail">${rail}</div>
      <article class="decoder-graph" id="decoderGraph">
        <p class="dag-title" id="dagTitle">decoder 00 · 28 layers share this graph</p>
        <p class="dag-legend">Solid arrow = tensor in. Dashed = residual skip. Click a node for the full checkpoint.</p>
        <div class="dag" id="dag">
          <svg class="dag-edges" id="dagEdges" aria-hidden="true"></svg>
          <div class="dag-nodes">${dagOps}</div>
        </div>
      </article>
      <div class="flow-edge"><i>↓</i><span>last decoder h → final RMSNorm</span></div>
      <button type="button" class="op net-node wide" id="g-final_norm" data-node="g-final_norm" data-op="final_norm">${neuronInner("final_norm", "final RMSNorm")}</button>
      <div class="flow-edge"><i>↓</i><span>y → lm_head</span></div>
      <button type="button" class="op net-node wide" id="g-logits" data-node="g-logits" data-op="logits">${neuronInner("logits", "lm_head")}</button>
      <div class="flow-edge"><i>↓</i><span>ℓ → argmax</span></div>
      <button type="button" class="op net-node wide" id="g-argmax" data-node="g-argmax" data-op="argmax">${neuronInner("argmax", "argmax")}</button>
    `;
    bindDagNodes(0);
    bindTooltips();
    if (state.dagObs) state.dagObs.disconnect();
    const dag = $("dag");
    if (dag && typeof ResizeObserver !== "undefined") {
      state.dagObs = new ResizeObserver(() => drawDagEdges());
      state.dagObs.observe(dag);
    }
    requestAnimationFrame(drawDagEdges);
  }

  function drawGpuDagEdges() {
    drawDagEdges({ dag: $("gpuDag"), svg: $("gpuDagEdges"), prefix: "g2-", marker: "g2Arr" });
  }

  const GPU_NET_PIPE_SHORT = {
    input_rmsnorm: "RMS",
    qkv_proj: "QKV",
    qk_norm: "QKn",
    rope: "RoPE",
    attn_scores: "QKᵀ",
    softmax: "softmax",
    attn_value: "P×V",
    o_proj: "O proj",
    attn_residual: "+ attn",
    post_rmsnorm: "RMS",
    gate_up_proj: "gate/up",
    silu_swiglu: "SwiGLU",
    down_proj: "down",
    mlp_residual: "+ MLP"
  };

  function gpuNetPipeStep(stage) {
    const ops = GPU_STAGE_OPS[stage] || [];
    const op = ops[0] || stage;
    const cls = OP_CLASS[op] || "other";
    const lab = GPU_NET_PIPE_SHORT[stage] || GPU_STAGE_TBL_SHORT[stage] || stage;
    const tip = GPU_STAGE_LABEL[stage] || stage;
    return `<button type="button" class="gpu-net-chip cls-${cls}" data-gpu-net-stage="${stage}" data-cls="${cls}" tabindex="-1" title="${tip}"><i aria-hidden="true"></i><span>${lab}</span></button>`;
  }

  function renderGpuNetwork() {
    const root = $("gpuNet");
    if (!root) return;
    if (root.dataset.ready === "4") return;
    const rail = Array.from({ length: 28 }, (_, i) =>
      `<span class="gpu-net-ldot" data-gpu-layer="${i}" title="L${String(i).padStart(2, "0")}"></span>`
    ).join("");
    const wide = (id, op, title) => {
      const cls = OP_CLASS[op] || "other";
      return `<button type="button" class="op net-node wide gpu-circ cls-${cls}" id="${id}" data-op="${op}" data-cls="${cls}" tabindex="-1">${neuronInner(op, title)}</button>`;
    };
    const attnStages = GPU_STAGE_ORDER.slice(0, 9);
    const mlpStages = GPU_STAGE_ORDER.slice(9);
    root.innerHTML =
      `<div class="gpu-net-track">` +
        wide("g2g-embedding", "embedding", "embed") +
        `<div class="gpu-net-layers" id="gpuLayerRail" aria-label="28 decoder layers">${rail}</div>` +
        `<article class="gpu-net-block" id="gpuDecoderGraph">` +
          `<p class="gpu-net-block-k" id="gpuDagTitle">L?? · decoder ×28</p>` +
          `<div class="gpu-net-row">` +
            `<span class="gpu-net-row-k">attn</span>` +
            `<div class="gpu-net-chips">${attnStages.map(gpuNetPipeStep).join("")}</div>` +
          `</div>` +
          `<div class="gpu-net-row">` +
            `<span class="gpu-net-row-k">mlp</span>` +
            `<div class="gpu-net-chips">${mlpStages.map(gpuNetPipeStep).join("")}</div>` +
          `</div>` +
        `</article>` +
        `<div class="gpu-net-tail">` +
          wide("g2g-final_norm", "final_norm", "final") +
          wide("g2g-logits", "logits", "lm_head") +
          wide("g2g-argmax", "argmax", "argmax") +
        `</div>` +
      `</div>`;
    root.dataset.ready = "4";
    root.querySelectorAll("button").forEach((b) => { b.tabIndex = -1; });
    bindTooltipsOn(root);
    if (state.gpuDagObs) {
      state.gpuDagObs.disconnect();
      state.gpuDagObs = null;
    }
  }

  function renderNnMap() {
    const root = $("nnMap");
    if (!root || root.dataset.ready === "1") return;
    const rail = Array.from({ length: 28 }, (_, i) =>
      `<span class="layer-dot" data-layer="${i}">L${String(i).padStart(2, "0")}</span>`
    ).join("");
    const wide = (id, op, title) =>
      `<button type="button" class="op net-node wide" id="${id}" data-op="${op}" tabindex="-1">${neuronInner(op, title)}</button>`;
    root.innerHTML =
      wide("nn-embed", "embedding", "embedding") +
      `<div class="flow-edge"><i>↓</i><span>h feeds decoder x</span></div>` +
      `<div class="layer-rail">${rail}</div>` +
      `<article class="decoder-graph current">` +
        `<div class="dag" id="nnDag">` +
          `<svg class="dag-edges" id="nnDagEdges" aria-hidden="true"></svg>` +
          `<div class="dag-nodes">${decoderDagOps("nn-")}</div>` +
        `</div>` +
      `</article>` +
      `<div class="flow-edge"><i>↓</i><span>last decoder h → final RMSNorm</span></div>` +
      wide("nn-final_norm", "final_norm", "final RMSNorm") +
      `<div class="flow-edge"><i>↓</i><span>y → lm_head</span></div>` +
      wide("nn-logits", "logits", "lm_head") +
      `<div class="flow-edge"><i>↓</i><span>ℓ → argmax</span></div>` +
      wide("nn-argmax", "argmax", "argmax");
    root.dataset.ready = "1";
    root.querySelectorAll("button").forEach((b) => { b.tabIndex = -1; });
    if (state.nnDagObs) state.nnDagObs.disconnect();
    const dag = $("nnDag");
    if (dag && typeof ResizeObserver !== "undefined") {
      state.nnDagObs = new ResizeObserver(() => drawNnDagEdges());
      state.nnDagObs.observe(dag);
    }
  }

  function renderBanks() {
    const banks = (id, n) => {
      $(id).innerHTML = Array.from({ length: n }, (_, i) =>
        `<div class="bank" title="bank ${i}"><b>B${i}</b></div>`
      ).join("");
    };
    banks("romBanks", 8);
    banks("sramBanks", 8);
    $("macLanes").innerHTML = Array.from({ length: 16 }, (_, i) =>
      `<div class="pe" title="lane ${i}"><span>a${i}</span><em>×</em><span>w${i}</span></div>`
    ).join("");
    $("layerStrip").innerHTML = Array.from({ length: 28 }, (_, i) =>
      `<div class="layer" data-layer="${i}" role="listitem">L${String(i).padStart(2, "0")}</div>`
    ).join("");
  }

  function paintTokens(ids, activeIndex) {
    if (!ids || !ids.length) {
      $("tokenList").innerHTML = `<span class="muted">Token IDs appear as soon as the statement is tokenized.</span>`;
      return;
    }
    const texts = state.tokenTexts || [];
    $("tokenList").innerHTML = ids.map((id, i) => {
      const piece = texts[i];
      const shown = piece == null || piece === "" ? "" : `<small>${escapeHtml(JSON.stringify(piece))}</small>`;
      return `<span class="token${i === activeIndex ? " active" : ""}" title="position ${i}">${id}${shown}</span>`;
    }).join("");
  }

  function predictedSentence() {
    const prompt = state.prompt || (state.tokenTexts || []).join("");
    const nxt = state.argmaxText || "";
    if (state.predictedText) return { prompt, next: nxt, full: state.predictedText };
    if (!nxt) return { prompt, next: "", full: prompt };
    return { prompt, next: nxt, full: prompt + nxt };
  }

  function paintOutput() {
    const el = $("outToken");
    const sent = $("predSentence");
    if (!el || !sent) return;
    const { prompt, next, full } = predictedSentence();
    if (state.argmax == null) {
      el.innerHTML = "";
      sent.className = "pred-inline muted";
      sent.textContent = "Predicted sentence appears here after argmax.";
      return;
    }
    const quoted = next == null ? "" : JSON.stringify(next);
    el.innerHTML = `<span class="token out">${state.argmax}<small>${escapeHtml(quoted)}</small></span>`;
    sent.className = "pred-inline";
    sent.innerHTML = escapeHtml(prompt) + (next ? `<mark>${escapeHtml(next)}</mark>` : "");
    if ($("argmaxDetail")) $("argmaxDetail").textContent = `argmax ${state.argmax}  ${quoted}`;
    const node = $("g-argmax");
    if (node) {
      const tip = node.querySelector(".tip");
      if (tip) tip.textContent = `argmax(ℓ)\nt* = ${state.argmax} ${quoted}\n${full}`;
    }
  }

  function resolveStage(stage) {
    if (stage == null || stage === "") return "IDLE";
    if (typeof stage === "number" || /^[0-9]+$/.test(String(stage).trim())) {
      return STAGE_ID_NAME[Number(stage)] || "IDLE";
    }
    const raw = String(stage).trim();
    if (STAGE_OP[raw]) return raw;
    const upper = raw.toUpperCase();
    if (STAGE_OP[upper]) return upper;
    return EVENT_STAGE[raw] || EVENT_STAGE[raw.toLowerCase()] || upper;
  }

  function layerIndex(layer) {
    const n = Number(layer);
    return Number.isFinite(n) ? n : NaN;
  }

  function dagAncestors(op) {
    const seen = new Set();
    const walk = (node) => {
      DAG_EDGES.forEach((edge) => {
        if (edge.to !== node || seen.has(edge.from)) return;
        seen.add(edge.from);
        walk(edge.from);
      });
    };
    if (op) walk(op);
    seen.delete(op);
    return seen;
  }

  function clearActive() {
    const root = document.querySelector(".slide-replica") || document;
    root.querySelectorAll(
      ".op.active, .net-node.active, .op.on-path, .net-node.on-path, " +
      ".layer-dot.active, .layer.active, .decoder-graph.current, " +
      ".blk.active, .token.active, [data-wire].live, " +
      ".edge.live, .edge.live-out, .edge-lab.live, .tip.live"
    ).forEach((el) => {
      el.classList.remove("active", "on-path", "current", "live", "live-out");
    });
  }

  function markMismatches() {
    document.querySelectorAll(".mismatch").forEach((el) => el.classList.remove("mismatch"));
    state.mismatch.forEach((event) => {
      const match = /^layer\.(\d+)/.exec(event);
      const layer = match ? Number(match[1]) : -1;
      const op = opKey(event);
      if (layer === state.viewLayer) {
        const el = $("dag-" + op);
        if (el) el.classList.add("mismatch");
      }
      if (layer >= 0) {
        document.querySelectorAll(`[data-layer="${layer}"]`).forEach((el) => el.classList.add("mismatch"));
      }
      const id = nodeId(event, layer);
      const gel = document.getElementById(id);
      if (gel) gel.classList.add("mismatch");
    });
  }

  function activeBlocks(stage) {
    const blocks = new Set();
    if (STAGE_BLOCK[stage]) blocks.add(STAGE_BLOCK[stage]);
    if (SRAM_STAGE[stage]) blocks.add(SRAM_STAGE[stage]);
    if (["Q_PROJ", "K_PROJ", "V_PROJ", "O_PROJ", "GATE_PROJ", "UP_PROJ", "DOWN_PROJ", "LM_HEAD", "EMBED"].includes(stage)) {
      blocks.add("rom");
      blocks.add("mac");
    }
    if (stage === "LM_HEAD" || stage === "ARGMAX" || stage === "DONE") blocks.add("logits");
    if (stage && stage !== "IDLE") blocks.add("fsm");
    return blocks;
  }

  function paintChip(stage, mismatched) {
    const blocks = activeBlocks(stage);
    blocks.forEach((name) => {
      document.querySelectorAll(`[data-block="${name}"]`).forEach((el) => {
        el.classList.add("active");
        if (mismatched) el.classList.add("mismatch");
      });
    });
    document.querySelectorAll("[data-wire]").forEach((el) => {
      const need = el.getAttribute("data-wire").split(/\s+/);
      el.classList.toggle("live", need.some((name) => blocks.has(name)));
    });
    $("fsmName").textContent = stage === "LM_HEAD" ? "LM_HEAD · argmax in flight" : (stage || "IDLE");
    $("mStage").textContent = stage === "LM_HEAD" ? "LM_HEAD · argmax in flight" : (stage || "IDLE");
    state.lastStage = stage || "IDLE";
    if (state.inspectorBlock) paintInspector(state.inspectorBlock);
  }

  // ---- Chip inspector ------------------------------------------------------
  // Click any block in the schematic to see what it is doing at the current
  // scrub/live step: identity, live indices, and the tensor flowing through.

  function inspectorContext(block) {
    const rec = state.timeline[state.index] || null;
    const stage = resolveStage(rec ? (rec.stage || eventStage(rec.event)) : (state.lastStage || "IDLE"));
    const rtl = state.lastRtlStatus || {};
    const py = state.lastPyStatus || {};
    const opFn = BLOCK_STAGE_OP[block];
    const op = opFn ? opFn(stage) : null;
    const map = recMapThrough(state.timeline.length - 1);
    let rec2 = null;
    if (op) {
      if (rec && opKey(rec.event) === op) {
        rec2 = rec;
      } else if (rec && typeof rec.layer === "number" && rec.layer >= 0) {
        rec2 = map.get(`L${rec.layer}-${op}`) || null;
      } else {
        rec2 = map.get(`g-${op}`) || null;
      }
    }
    const feeds = op ? (OP_FEEDS[op] || null) : null;
    const macStages = new Set(["Q_PROJ", "K_PROJ", "V_PROJ", "ATTN_SCORE", "ATTN_VALUE",
      "O_PROJ", "GATE_PROJ", "UP_PROJ", "DOWN_PROJ", "LM_HEAD"]);
    return {
      rec: rec2 || rec,
      op,
      stage,
      rtl,
      py,
      weightText: feeds ? feeds.w : null,
      sramRegion: SRAM_STAGE[stage] || null,
      stageIsMac: macStages.has(stage)
    };
  }

  function paintInspector(block) {
    const body = $("chipInspectorBody");
    if (!body) return;
    // The card is always visible; unknown blocks fall back to the FSM so the
    // schematic height never changes when the selection clears.
    const meta = CHIP_BLOCKS[block] || CHIP_BLOCKS.fsm;
    const ctx = inspectorContext(block);
    const rec = ctx.rec;
    const now = meta.now(ctx);
    const dataRows = [];
    if (rec && rec.values && rec.values.length) {
      dataRows.push(`<div class="ci-row"><em>tensor</em> ${escapeHtml(rec.event)} ` +
        `${shapeText(rec, ctx.op || opKey(rec.event))}</div>`);
      dataRows.push(`<div class="ci-vals">${rec.values.slice(0, 8).map((v) => `<span>${fmtVal(v)}</span>`).join("")}</div>`);
    } else if (ctx.op) {
      dataRows.push(`<div class="ci-row"><em>tensor</em> ${escapeHtml(ctx.op)} — checkpoint not in loaded timeline yet</div>`);
    } else {
      dataRows.push(`<div class="ci-row muted">no tensor routed through this block at ${escapeHtml(ctx.stage)}</div>`);
    }
    if (block === "rom") {
      dataRows.unshift(
        `<div class="ci-store">` +
          `<div class="ci-row"><em>bits</em> 1,192,099,840 B × 8 = 9,536,798,720</div>` +
          `<div class="ci-row"><em>Tx</em> 1T/bit mask ROM = 9.54 billion transistors</div>` +
          `<div class="ci-row"><em>area</em> 9.54B × 0.018 µm² (N4) = 171.7 mm² array</div>` +
          `<div class="ci-row"><em>reticle</em> 171.7 / 600 mm² usable die = 0.29 of one reticle (field ~26×32 mm)</div>` +
        `</div>`
      );
    }
    const canOpenModal = !!(rec && rec.event);
    $("chipInspectorTitle").textContent = meta.title;
    $("chipInspectorModule").textContent = meta.module;
    $("chipInspectorDetail").textContent = meta.detail;
    $("chipInspectorNow").textContent = now;
    body.innerHTML = dataRows.join("");
    const tensorBtn = $("chipInspectorTensor");
    tensorBtn.hidden = false;
    tensorBtn.disabled = !canOpenModal;
    tensorBtn.onclick = canOpenModal ? () => openTensorModal(rec) : null;
  }

  function openInspector(block, reveal) {
    state.inspectorBlock = block;
    const el = $("chipInspector");
    if (el && reveal !== false) el.classList.add("is-open");
    paintInspector(block);
  }

  function closeInspector() {
    state.inspectorBlock = "fsm";
    const el = $("chipInspector");
    if (el) el.classList.remove("is-open");
    paintInspector("fsm");
  }

  function activateDagOp(op, cls) {
    if (!op) return;
    const klass = cls || "active";
    document.querySelectorAll(`#network [data-dag-op="${op}"]`).forEach((el) => {
      el.classList.add(klass);
    });
    const el = $("dag-" + op);
    if (el) el.classList.add(klass);
    if (klass === "active") {
      document.querySelectorAll(`#dagEdges [data-to="${op}"]`).forEach((e) => e.classList.add("live"));
    }
  }

  function paintNetwork(event, layer) {
    const op = opKey(event);
    const layerN = layerIndex(layer);
    const hasLayer = layerN >= 0 && layerN < 28;
    const global = ["embedding", "final_norm", "logits", "argmax"].includes(op);
    const decoderOp = DAG_OPS.includes(op) || op === "residual";
    if (hasLayer) {
      if (layerN !== state.viewLayer) {
        state.lastHotOp = null;
        state.layerSeen = new Set();
        bindDagNodes(layerN);
        fillDagLayer(layerN);
      }
      const wrap = $("decoderGraph");
      if (wrap) wrap.classList.add("current");
      document.querySelectorAll(`[data-layer="${layerN}"]`).forEach((el) => el.classList.add("active"));
    }
    if (global) {
      const gel = $("g-" + op);
      if (gel) gel.classList.add("active");
      return;
    }
    if (decoderOp) {
      const extras = [];
      if (op === "q_rope") extras.push("k_rope");
      if (op === "silu") extras.push("swiglu");
      state.layerSeen.add(op);
      extras.forEach((extra) => state.layerSeen.add(extra));
      const trail = new Set(state.layerSeen);
      dagAncestors(op).forEach((a) => trail.add(a));
      extras.forEach((extra) => dagAncestors(extra).forEach((a) => trail.add(a)));
      extras.forEach((extra) => trail.delete(extra));
      trail.delete(op);
      trail.forEach((prev) => activateDagOp(prev, "on-path"));
      activateDagOp(op);
      extras.forEach((extra) => activateDagOp(extra));
      const hot = $("dag-" + op);
      if (hot && state.lastHotOp !== op) {
        state.lastHotOp = op;
        const pane = hot.closest(".network");
        if (pane) {
          const pr = pane.getBoundingClientRect();
          const hr = hot.getBoundingClientRect();
          if (hr.top < pr.top || hr.bottom > pr.bottom) {
            hot.scrollIntoView({ block: "nearest", inline: "nearest" });
          }
        }
      }
    } else {
      const gel = $("g-" + op);
      if (gel) gel.classList.add("active");
    }
  }

  function paintCursor({ event, layer, token, stage }) {
    const st = (() => {
      const fromStage = stage ? resolveStage(stage) : null;
      if (fromStage && STAGE_BLOCK[fromStage]) return fromStage;
      return resolveStage(eventStage(event));
    })();
    const mismatched = event ? state.mismatch.has(event) : false;
    clearActive();
    markMismatches();
    paintNetwork(event, layer);
    paintChip(st, mismatched);
    const layerN = layerIndex(layer);
    $("mLayer").textContent = layerN >= 0 ? String(layerN) : "—";
    // Token highlight: the timeline carries the token being processed.
    // During decode the chip walks all tokens through one layer at a time
    // (layer-major), so the lit token is the one whose turn it is — not
    // "all tokens re-passing this layer".
    if (token !== undefined && token !== null && token !== -1) {
      paintTokens(state.tokenIds, token);
    } else if (state.tokenIds.length) {
      const rec = state.timeline[state.index];
      paintTokens(state.tokenIds, rec && typeof rec.token === "number" ? rec.token : -1);
    }
  }

  function paintFromStage(stage, layer) {
    const name = resolveStage(stage);
    const spec = STAGE_OP[name] || {};
    const layerN = layerIndex(layer);
    if (spec.global) {
      paintCursor({ event: spec.global, layer: -1, stage: name });
      if (name === "LM_HEAD") {
        const arg = $("g-argmax");
        if (arg) {
          arg.classList.add("on-path");
          const tip = arg.querySelector(".tip");
          const best = (state.lastRtlStatus || {}).argmax;
          if (tip && best != null) {
            tip.textContent = `argmax in flight — reduced per logit row, never stored\nbest so far t* = ${best}`;
          }
        }
      }
      if (name === "ARGMAX" || name === "DONE") {
        const logits = $("g-logits");
        if (logits) logits.classList.add("on-path");
        const arg = $("g-argmax");
        if (arg) arg.classList.add("active");
      }
      if (layerN >= 0) {
        const strip = document.querySelector(`.layer[data-layer="${layerN}"]`);
        if (strip) strip.classList.add("active");
        $("mLayer").textContent = String(layerN);
      }
      return;
    }
    paintCursor({ event: spec.op || "embedding", layer: layerN, stage: name });
    if (spec.extra) activateDagOp(spec.extra);
    // Always mark this layer's completed prefix. Q/K/V are DAG siblings, so
    // landing on V_PROJ does not light Q via ancestors. Polling also skips
    // short fused GEMMs (Q ~3.7M cyc, K ~1.9M, V ~12.8M).
    lightWalkPrefix(name);
  }

  function lightWalkPrefix(current) {
    const b = LAYER_WALK.indexOf(resolveStage(current));
    if (b <= 0) return;
    LAYER_WALK.slice(0, b).forEach((st) => {
      const spec = STAGE_OP[st] || {};
      if (spec.op) {
        state.layerSeen.add(spec.op);
        activateDagOp(spec.op, "on-path");
      }
      if (spec.extra) {
        state.layerSeen.add(spec.extra);
        activateDagOp(spec.extra, "on-path");
      }
    });
  }

  function setTimeline(events) {
    state.timeline = Array.isArray(events) ? events : [];
    const last = state.timeline[state.timeline.length - 1];
    if (
      last
      && opKey(last.event) === "logits"
      && !state.timeline.some((rec) => opKey(rec.event) === "argmax")
      && state.argmax != null
    ) {
      state.timeline.push({
        event: "argmax",
        layer: -1,
        token: last.token,
        stage: "ARGMAX",
        values: [state.argmax]
      });
    }
    const slider = $("playSlider");
    slider.max = String(Math.max(state.timeline.length - 1, 0));
    if (state.index >= state.timeline.length) state.index = Math.max(state.timeline.length - 1, 0);
    slider.value = String(state.index);
    fillTensorPreviews();
  }

  function showIndex(index, scroll) {
    if (!state.timeline.length) {
      $("playLabel").textContent = "event 0 / 0";
      $("playEvent").textContent = "Waiting for a forward timeline.";
      return;
    }
    state.index = Math.max(0, Math.min(index, state.timeline.length - 1));
    $("playSlider").value = String(state.index);
    const rec = state.timeline[state.index];
    $("playLabel").textContent = `event ${state.index + 1} / ${state.timeline.length}`;
    $("playEvent").textContent = rec.event + (rec.layer >= 0 ? ` · layer ${rec.layer}` : "");
    paintCursor(rec);
    showCompute(rec);
    if (scroll === false) return;
  }

  function jumpLayer(layer) {
    state.userPaused = true;
    stopPlay();
    bindDagNodes(layer);
    fillDagLayer(layer);
    const idx = state.timeline.findIndex((rec) => rec.layer === layer);
    if (idx >= 0) showIndex(idx);
    else markMismatches();
  }

  function seekNode(node) {
    const idx = state.timeline.findIndex((rec) => nodeId(rec.event, rec.layer) === node);
    if (idx >= 0) {
      state.userPaused = true;
      stopPlay();
      showIndex(idx);
      openTensorModal(state.timeline[idx]);
      return;
    }
    if (node === "g-argmax") {
      state.userPaused = true;
      stopPlay();
      openTensorModal({ event: "argmax", layer: -1, shape: [], elements: 1, values: [state.argmax] });
    }
  }

  function coordOf(index, shape) {
    if (!shape || !shape.length) return String(index);
    const coords = [];
    let n = index;
    for (let d = shape.length - 1; d >= 0; d--) {
      const dim = Number(shape[d]) || 1;
      coords.unshift(n % dim);
      n = Math.floor(n / dim);
    }
    return "[" + coords.join(",") + "]";
  }

  function isAttentionEvent(rec) {
    if (!rec) return false;
    const op = opKey(rec.event);
    return (op === "attention_scores" || op === "attention_softmax") &&
      rec.shape && rec.shape.length === 2 && Number(rec.shape[0]) > 1;
  }

  function heatColor(v, lo, hi) {
    const span = hi > lo ? (v - lo) / (hi - lo) : 0;
    const t = Math.max(0, Math.min(1, isFinite(span) ? span : 0));
    // teal (low) -> white (mid) -> red (high)
    const stops = [
      [15, 118, 110], [224, 242, 254], [254, 226, 226], [185, 28, 28]
    ];
    const pos = Math.max(0, Math.min(stops.length - 1.000001, t * (stops.length - 1)));
    const seg = Math.floor(pos);
    const f = pos - seg;
    const c = stops[seg].map((base, i) =>
      Math.round(base + f * (stops[seg + 1][i] - base)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  function renderHeatmap(values, shape, diffValues) {
    const heads = Number(shape[0]) || 1;
    const ctx = Number(shape[1]) || 1;
    const finite = values.filter((v) => Number.isFinite(v));
    const lo = finite.length ? Math.min(...finite) : 0;
    const hi = finite.length ? Math.max(...finite) : 1;
    const diffSet = new Set();
    if (diffValues && diffValues.length === values.length) {
      values.forEach((v, i) => {
        if (v !== diffValues[i]) diffSet.add(i);
      });
    }
    const cells = [];
    for (let h = 0; h < heads; h++) {
      cells.push(`<div class="hm-rowlabel">h${h}</div>`);
      for (let k = 0; k < ctx; k++) {
        const i = h * ctx + k;
        const v = values[i];
        const shown = Number.isFinite(v) ? v.toPrecision(3) : "—";
        const diff = diffSet.has(i) ? " diff" : "";
        cells.push(
          `<div class="hm-cell${diff}" title="head ${h}, key ${k}: ${shown}"` +
          ` style="background:${heatColor(v, lo, hi)}">${shown}</div>`
        );
      }
    }
    const diffNote = diffSet.size
      ? ` · <span class="hm-diff-note">${diffSet.size} element(s) differ from RTL (outlined)</span>`
      : "";
    return `<div class="hm-note">rows = query heads · columns = key positions` +
      ` · range ${fmtVal(lo)} … ${fmtVal(hi)}${diffNote}</div>` +
      `<div class="hm-grid" style="grid-template-columns:32px repeat(${ctx}, minmax(44px, 1fr))">` +
      `<div class="hm-rowlabel"></div>` +
      Array.from({ length: ctx }, (_, k) => `<div class="hm-collabel">k${k}</div>`).join("") +
      cells.join("") +
      `</div>`;
  }

  function closeModal() {
    $("tensorModal").hidden = true;
    state.modal.rec = null;
  }

  async function openTensorModal(rec) {
    if (!rec) return;
    state.modal.rec = rec;
    state.modal.offset = 0;
    $("tensorModal").hidden = false;
    $("modalTitle").textContent = rec.event;
    $("modalEq").textContent = OP_EQ[opKey(rec.event)] || rec.event;
    document.querySelectorAll(".modal-tabs .ghost").forEach((btn) => {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-src") === state.modal.source ? "true" : "false");
    });
    await loadModalPage();
  }

  async function loadModalPage() {
    const rec = state.modal.rec;
    if (!rec) return;
    const op = opKey(rec.event);
    if (op === "argmax") {
      $("modalMeta").textContent = "decoded next token from argmax(ℓ)";
      $("modalPage").textContent = "";
      $("modalGrid").innerHTML =
        `<span><b>t*</b>${state.argmax == null ? "—" : state.argmax}</span>` +
        `<span><b>decode</b>${escapeHtml(JSON.stringify(state.argmaxText || ""))}</span>` +
        `<span><b>sentence</b>${escapeHtml(predictedSentence().full)}</span>`;
      return;
    }
    const page = (rec.elements || 0) > 8192 ? 1024 : 4096;
    $("modalGrid").innerHTML = `<span class="muted">loading ${rec.event}…</span>`;
    try {
      if (isAttentionEvent(rec)) {
        const other = state.modal.source === "python" ? "rtl" : "python";
        const [main, diff] = await Promise.all([
          SHARE_DEMO
            ? loadShareTensor(rec, state.modal.source, 0, rec.elements || 0)
            : loadLiveTensor(rec, state.modal.source, 0, rec.elements || 0),
          SHARE_DEMO
            ? loadShareTensor(rec, other, 0, rec.elements || 0).catch(() => null)
            : loadLiveTensor(rec, other, 0, rec.elements || 0).catch(() => null)
        ]);
        state.modal.total = main.elements || 0;
        const shape = main.shape && main.shape.length ? main.shape : rec.shape;
        $("modalMeta").textContent =
          `${main.source} heatmap  ${shapeText({ shape }, op)}  ·  ${main.elements} F32` +
          (diff ? `  ·  outlined cells differ from ${diff.source}` : "");
        $("modalPage").textContent = `0 / ${main.elements}`;
        $("modalGrid").innerHTML = renderHeatmap(
          main.values || [], shape, diff ? diff.values : null
        );
        return;
      }
      const body = SHARE_DEMO
        ? await loadShareTensor(rec, state.modal.source, state.modal.offset, page)
        : await loadLiveTensor(rec, state.modal.source, state.modal.offset, page);
      state.modal.total = body.elements || 0;
      const shape = body.shape && body.shape.length ? body.shape : rec.shape;
      const start = body.offset || 0;
      $("modalMeta").textContent =
        `${body.source}  ${shapeText({ shape }, op)}  ·  ${body.elements} F32` +
        `  ·  showing ${start}–${start + body.values.length - 1}`;
      $("modalPage").textContent = `${start} / ${body.elements}`;
      $("modalGrid").innerHTML = (body.values || []).map((v, i) =>
        `<span><b>${coordOf(start + i, shape)}</b>${fmtVal(v)}</span>`
      ).join("") || `<span class="muted">empty tensor</span>`;
    } catch (err) {
      $("modalMeta").textContent = "";
      $("modalGrid").innerHTML = `<span class="muted">${escapeHtml(err.message || "checkpoint missing")}</span>`;
    }
  }

  async function loadLiveTensor(rec, source, offset, limit) {
    const params = new URLSearchParams({
      event: rec.event,
      source,
      offset: String(offset),
      limit: String(limit)
    });
    if (state.workDir) params.set("work_dir", state.workDir);
    const res = await apiFetch(`/api/tensor?${params}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || res.statusText);
    }
    return res.json();
  }

  async function loadShareTensor(rec, source, offset, limit) {
    const key = source + ":" + rec.event;
    let arr = tensorCache.get(key);
    if (!arr) {
      const safe = String(rec.event).replace(/[/.]/g, "_") + ".f32le";
      const res = await fetch(`data/${source}_checkpoints/${safe}`);
      if (!res.ok) throw new Error(`no ${source} checkpoint for ${rec.event}`);
      arr = new Float32Array(await res.arrayBuffer());
      tensorCache.set(key, arr);
    }
    const start = Math.max(0, offset);
    const values = Array.from(arr.subarray(start, start + limit));
    return {
      event: rec.event,
      source,
      shape: rec.shape || [],
      elements: arr.length,
      offset: start,
      values
    };
  }

  function stopPlay() {
    if (state.playTimer) {
      clearTimeout(state.playTimer);
      state.playTimer = null;
    }
    state.playing = false;
    $("playBtn").textContent = "Play";
  }

  function playDelayFor(rec) {
    const op = opKey(rec && rec.event);
    if (op === "attention_scores" || op === "attention_softmax") return 50;
    if (op === "q_norm" || op === "k_norm" || op === "q_rope" || op === "k_rope") return 520;
    if (op === "logits" || op === "argmax") return 700;
    return 280;
  }

  function startPlay() {
    if (!state.timeline.length || state.live) return;
    state.playing = true;
    state.userPaused = false;
    $("playBtn").textContent = "Pause";
    if (state.playTimer) clearTimeout(state.playTimer);
    const tick = () => {
      if (!state.playing) return;
      if (state.index >= state.timeline.length - 1) {
        stopPlay();
        return;
      }
      showIndex(state.index + 1);
      state.playTimer = setTimeout(tick, playDelayFor(state.timeline[state.index]));
    };
    state.playTimer = setTimeout(tick, playDelayFor(state.timeline[state.index]));
  }

  function fmtElapsed(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
  }

  // Fraction of one token's fused RTL work already done (0..1). Weighted by
  // measured stage cycles so the bar does not reset when the FSM returns to
  // INPUT_NORM on the next layer, or EMBED on the next token.
  function rtlTokenWorkFrac(rtl) {
    const cyc = FUSED_STAGE_CYCLES;
    const layerCyc = LAYER_WALK.reduce((sum, st) => sum + (cyc[st] || 0), 0);
    const tokenCyc = (cyc.EMBED || 0) + 28 * layerCyc
      + (cyc.FINAL_NORM || 0) + (cyc.LM_HEAD || 0) + (cyc.ARGMAX || 0);
    const stage = rtl.stage || "EMBED";
    const layer = Math.max(0, Math.min(27, Number(rtl.layer) || 0));
    let done = 0;
    if (stage === "EMBED" || stage === "IDLE") {
      done = 0.25 * (cyc.EMBED || 0);
    } else if (stage === "NEXT_LAYER") {
      done = (cyc.EMBED || 0) + (layer + 1) * layerCyc;
    } else if (LAYER_WALK.includes(stage)) {
      done = (cyc.EMBED || 0) + layer * layerCyc;
      const idx = LAYER_WALK.indexOf(stage);
      for (let i = 0; i < idx; i += 1) done += cyc[LAYER_WALK[i]] || 0;
      done += 0.45 * (cyc[stage] || 0);
    } else if (stage === "FINAL_NORM") {
      done = (cyc.EMBED || 0) + 28 * layerCyc + 0.5 * (cyc.FINAL_NORM || 0);
    } else if (stage === "LM_HEAD") {
      const rows = Math.min(1, Math.max(0, Number(rtl.output_index) || 0) / 151936);
      done = (cyc.EMBED || 0) + 28 * layerCyc + (cyc.FINAL_NORM || 0)
        + rows * (cyc.LM_HEAD || 0);
    } else if (stage === "ARGMAX" || stage === "DONE") {
      done = tokenCyc;
    } else {
      done = (cyc.EMBED || 0) + layer * layerCyc;
    }
    return Math.min(1, done / Math.max(1, tokenCyc));
  }

  function rtlJobFrac(rtl) {
    const n = Math.max(1, Number(rtl.token_count) || state.tokenIds.length || 1);
    const t = Math.max(0, Number(rtl.token_index) || 0);
    if (t >= n) return 1;
    return Math.min(1, (t + rtlTokenWorkFrac(rtl)) / n);
  }

  // Python 0–50%, then RTL 50–100% (or 0–100% for signed-off infer).
  // RTL fill is (finished tokens + work inside this token) / N.
  function paintRunProgress(body) {
    const wrap = $("runProgress");
    if (!wrap) return;
    const py = (body && body.python_status) || {};
    const rtl = (body && body.rtl_status) || {};
    const status = body && body.status;
    const running = status === "running";
    const phase = (body && body.phase) || py.phase;
    const infer = (body && body.mode === "rtl_infer")
      || (body && body.evidence && body.evidence.mode === "rtl_infer");
    const labelEl = $("runProgressLabel");
    const elapsedEl = $("runProgressElapsed");
    if (running) {
      state.runFinished = false;
    } else if (status === "completed" || status === "cached") {
      state.runFinished = true;
    }
    const track = document.querySelector(".run-progress-track");
    if (!running && !state.runFinished) {
      wrap.hidden = true;
      if (labelEl) { labelEl.hidden = true; labelEl.textContent = ""; }
      if (elapsedEl) { elapsedEl.hidden = true; elapsedEl.textContent = ""; }
      state.progressPeak = 0;
      if (track) track.setAttribute("aria-valuenow", "0");
      return;
    }
    if (!running) {
      wrap.hidden = false;
      $("runProgressPy").style.width = infer ? "0%" : "50%";
      $("runProgressRtl").style.width = infer ? "100%" : "50%";
      if (labelEl) {
        labelEl.hidden = false;
        const oracle = body && body.mode === "oracle_replay";
        labelEl.textContent = infer
          ? "RTL complete · ROM signed off"
          : oracle
            ? "Python cached · playback"
            : `${FULL_COMPARE} complete · Python + chip`;
      }
      if (elapsedEl) {
        elapsedEl.hidden = false;
        elapsedEl.textContent = state.runStartedAt != null
          ? `elapsed ${fmtElapsed(Date.now() - state.runStartedAt)}`
          : "";
      }
      if (track) {
        track.setAttribute("aria-valuemin", "0");
        track.setAttribute("aria-valuemax", "100");
        track.setAttribute("aria-valuenow", "100");
      }
      return;
    }
    if (state.runStartedAt == null) {
      state.runStartedAt = Date.now();
      state.progressPeak = 0;
    }
    wrap.hidden = false;
    let pyFrac = 0;
    let rtlFrac = 0;
    let label = "";
    if (infer || phase === "rtl" || rtl.busy) {
      pyFrac = infer ? 0 : 1;
      rtlFrac = rtlJobFrac(rtl);
      const tok = (rtl.token_index ?? 0) + 1;
      const ntok = rtl.token_count || state.tokenIds.length || "?";
      if (rtl.stage === "LM_HEAD") {
        const rows = Math.max(0, rtl.output_index ?? 0);
        label = `token ${tok}/${ntok} · lm_head ${rows.toLocaleString("en-US")} / 151,936`;
      } else if (rtl.stage === "ARGMAX" || rtl.stage === "DONE") {
        label = `token ${tok}/${ntok} · argmax${rtl.argmax != null ? ` ${rtl.argmax}` : ""}`;
      } else {
        label = `token ${tok}/${ntok} · layer ${(rtl.layer ?? 0) + 1}/28 · ${rtl.stage || "…"}`;
      }
    } else {
      const done = py.layers_done != null ? py.layers_done : 0;
      const total = py.layers_total != null ? py.layers_total : 28;
      pyFrac = Math.min(1, done / total);
      label = `Python oracle · layer ${done}/${total}`;
    }
    const raw = infer ? rtlFrac : (pyFrac * 0.5 + rtlFrac * 0.5);
    const peak = Math.max(state.progressPeak || 0, raw);
    state.progressPeak = peak;
    const pyW = infer ? 0 : pyFrac * 50;
    const rtlW = infer ? peak * 100 : Math.max(0, peak * 100 - pyW);
    $("runProgressPy").style.width = `${Math.max(0, pyW).toFixed(1)}%`;
    $("runProgressRtl").style.width = `${Math.max(0, rtlW).toFixed(1)}%`;
    if (track) {
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", String(Math.round(peak * 100)));
    }
    if (labelEl) { labelEl.hidden = false; labelEl.textContent = label; }
    if (elapsedEl) {
      elapsedEl.hidden = false;
      elapsedEl.textContent = `elapsed ${fmtElapsed(Date.now() - state.runStartedAt)}`;
    }
  }

  function setRunWhy(body) {
    const el = $("runWhy");
    if (!el) return;
    // The status line is hidden while idle; it appears only when there is
    // something live to say (the meta line carries idle state).
    el.hidden = true;
    if (SHARE_DEMO) {
      el.hidden = false;
      el.textContent = "Recorded full-statement compare — you are scrubbing a finished run.";
      return;
    }
    const py = (body && body.python_status) || {};
    const rtl = (body && body.rtl_status) || {};
    const status = body && body.status;
    const phase = (body && body.phase) || py.phase;
    if (status === "cached" && body && body.mode === "rtl_infer") {
      el.hidden = false;
      el.textContent = "ROM signed off. This sentence ran RTL only — host sent token IDs, weights stayed in ROM, Python was not re-run.";
      return;
    }
    if (status === "cached" && body && body.mode === "oracle_replay") {
      el.hidden = false;
      el.textContent = "Cached Python forward. The graph is playback of the saved walk — Python and Verilator are not running again.";
      return;
    }
    if (status === "cached") {
      el.hidden = false;
      el.textContent = "Cached. This sentence already finished a full-statement compare — the graph is playback, Python and the chip are not running again.";
      return;
    }
    if (status === "partial") {
      el.hidden = false;
      el.textContent = "Cached Python checkpoints. The graph is playback — not a new forward.";
      return;
    }
    if (status === "running" && body && body.mode === "rtl_infer") {
      el.hidden = false;
      el.textContent = "ROM is signed off. Running the chip only — token IDs in, next token out. No Python oracle, no full-statement compare.";
      return;
    }
    if (status === "running" && (phase === "python" || py.phase === "python")) {
      const done = py.layers_done != null ? py.layers_done : 0;
      const total = py.layers_total != null ? py.layers_total : 28;
      const msg = py.message ? ` ${py.message}.` : "";
      el.hidden = false;
      el.textContent = `Now: Python oracle, layer ${done}/${total}.${msg} Slow because this is bit-exact scalar Python over the full 0.6B — no NumPy or PyTorch GEMM. Every MAC rounds like the chip. Verilator has not started.`;
      return;
    }
    if (status === "running" && (phase === "rtl" || rtl.busy)) {
      el.hidden = false;
      const tok = rtl.token_index != null
        ? ` Token ${rtl.token_index + 1}/${rtl.token_count ?? "?"} of the prompt.`
        : "";
      const oracle = body && body.python_argmax_text
        ? ` Python already predicts ${JSON.stringify(body.python_argmax_text)}.`
        : "";
      el.textContent = `Now: Verilator RTL.${tok}${oracle} Mid-prompt argmax is discarded; the predicted word is the last position only. Cycle-accurate 16-lane sim, not a GPU kernel.`;
      return;
    }
    if (status === "running") {
      el.hidden = false;
      el.textContent = "Full-statement compare running — Python first (bit-exact, no BLAS), then the Verilator chip. Both are slow on purpose.";
      return;
    }
    if (status === "failed") {
      el.hidden = false;
      el.textContent = "This full-statement compare failed. A live run is still the two slow forwards described below.";
      return;
    }
    if (body && (body.evidence || (body.timeline && body.timeline.length))) {
      el.hidden = false;
      el.textContent = "Full-statement compare finished. Scrubbing only reads checkpoints — it does not re-run Python or the chip.";
      return;
    }
    el.textContent = state.rtlSignedOff
      ? "Idle. ROM is signed off — Run chip sends token IDs only. A full-statement compare (T4) runs only if you force a re-verify."
      : "Idle. Compare full statement runs Python, then the chip, on this sentence (all 28 layers, every checkpoint, all logits). After one pass, later sentences can be chip-only.";
  }

  function applyMetrics(rtl, python) {
    const status = rtl || {};
    const py = python || {};
    state.lastRtlStatus = rtl || state.lastRtlStatus;
    state.lastPyStatus = python || state.lastPyStatus;
    $("mCycle").textContent = fmt(status.cycle);
    $("mToken").textContent = status.token_index === undefined
      ? (state.tokenIds.length ? `0 / ${state.tokenIds.length}` : "—")
      : `${status.token_index} / ${status.token_count ?? "?"}`;
    $("mRom").textContent = fmt(status.rom_reads);
    $("mSram").textContent = status.sram_reads === undefined
      ? "—"
      : `${fmt(status.sram_reads)} / ${fmt(status.sram_writes)}`;
    $("mMacs").textContent = fmt(status.macs);
    $("mStalls").textContent = fmt(status.stalls);
    $("hostDetail").textContent = status.busy
      ? `position ${status.position ?? 0}`
      : (py.phase === "python" ? (py.message || "Python oracle") : "cmd idle");
    if (status.argmax != null && (status.stage === "LM_HEAD" || status.stage === "ARGMAX" || status.stage === "DONE")) {
      const last = (status.token_count || 0) > 0
        && status.token_index >= (status.token_count - 1);
      const node = $("g-argmax");
      if (node && last) {
        node.classList.add(status.stage === "LM_HEAD" ? "on-path" : "active");
        const tip = node.querySelector(".tip");
        if (tip) {
          tip.textContent = status.stage === "LM_HEAD"
            ? `argmax in flight on the last prompt token\nbest so far t* = ${status.argmax}`
            : `argmax(ℓ)\nt* = ${status.argmax}`;
        }
      }
      // host_token_out is the previous position until this token's DONE.
      // Never replace the oracle sentence with a mid-prompt ',' / ' Question'.
      if (last && (status.stage === "ARGMAX" || status.stage === "DONE")) {
        state.argmax = status.argmax;
        paintOutput();
      }
    }
  }

  function bitsF32(hex) {
    const u = parseInt(hex, 16);
    if (!Number.isFinite(u)) return null;
    return new Float32Array(new Uint32Array([u]).buffer)[0];
  }

  function applyEvidence(evidence) {
    if (!evidence) return;
    const comparison = evidence.comparison || {};
    const passed = evidence.passed === true || comparison.passed === true;
    const prompt = evidence.text || "this prompt";
    const n = comparison.mismatch_count;
    const decoded = state.argmaxText ? JSON.stringify(state.argmaxText) : "";
    const sentence = predictedSentence().full;
    if (evidence.mode === "rtl_infer" || evidence.compared === false) {
      setBadge($("compareBadge"), "ROM signed off · RTL only", "ok");
      $("compareDetail").textContent =
        `ROM already signed off by a full-statement compare. This sentence ran the chip only (no Python oracle). Next token ${evidence.rtl_argmax} ${decoded}. Predicted: ${sentence}`;
      $("mismatchBox").hidden = true;
      return;
    }
    if (passed) {
      setBadge($("compareBadge"),
        evidence.fused_schedule ? "full-statement match · fused" : "full-statement match · unfused", "ok");
      $("compareDetail").textContent =
        `Full-statement compare passed for “${prompt}”: every checkpoint matched. Next token ${evidence.python_argmax} ${decoded}. ` +
        `Predicted: ${sentence}`;
      $("mismatchBox").hidden = true;
      return;
    }
    setBadge($("compareBadge"), "real mismatch", "bad");
    const first = comparison.first_mismatch;
    const key = first
      ? ((Array.isArray(first.key) ? first.key[0] : first.event) || "checkpoint")
      : "checkpoint";
    let scaleNote = "";
    const fe = first && first.first_element;
    if (fe && fe.python_bits && fe.rtl_bits) {
      const pv = bitsF32(fe.python_bits);
      const rv = bitsF32(fe.rtl_bits);
      if (pv != null && rv != null && Math.abs(rv) > 0) {
        scaleNote = ` First element Python ${pv.toPrecision(4)} vs RTL ${rv.toPrecision(4)}.`;
        const scaled = rv * (1 / Math.sqrt(128));
        if (Math.abs(scaled - pv) / Math.max(Math.abs(pv), 1e-12) < 1e-5) {
          scaleNote += " Ratio is 1/√128: this run’s RTL committed raw Q·K; Python stored the scaled attention score.";
        }
      }
    }
    const mismatchNames = (comparison.mismatches || []).map((m) => {
      if (Array.isArray(m.key)) return String(m.key[0] || "");
      return String(m.event || "");
    });
    const l0Softmax = mismatchNames.some((k) => k.indexOf("layer.0.") === 0 && k.indexOf("attention_softmax") >= 0);
    const l0Scores = mismatchNames.some((k) => k.indexOf("layer.0.") === 0 && k.indexOf("attention_scores") >= 0);
    if (l0Softmax && !l0Scores) {
      scaleNote +=
        " Layer 0 attention scores matched; softmax did not. qwen-ref-v2 keeps exp(score − max) in FP32 until after the divide — it is not written to BF16 SRAM. A one-token compare cannot catch that (each head’s P is 1). On a longer sentence the wrong P poisons every later token and the next-token argmax is not the oracle’s.";
    }
    const sameArg =
      evidence.python_argmax != null && evidence.python_argmax === evidence.rtl_argmax;
    $("compareDetail").textContent = first
      ? `This is a real bit mismatch from the last full-statement compare (“${prompt}”): ${n} checkpoint(s), first ${key}.` +
        scaleNote +
        (sameArg
          ? ` Next-token argmax still agreed (${evidence.python_argmax}), so the predicted sentence can look correct.`
          : "") +
        ` Predicted: ${sentence}. A job that is still running has not been compared yet.`
      : (evidence.message || "Verify did not pass.");
    $("mismatchBox").hidden = false;
    $("mismatchBox").textContent = JSON.stringify(first || comparison, null, 2);
  }

  function ingest(body, { autoplay } = {}) {
    if (body.rtl_signed_off != null) state.rtlSignedOff = !!body.rtl_signed_off;
    if (body.job_id) state.jobId = body.job_id;
    if (body.work_dir) state.workDir = body.work_dir;
    if (body.token_ids) state.tokenIds = body.token_ids;
    if (body.token_texts) state.tokenTexts = body.token_texts;
    if (body.text) state.prompt = body.text;
    else if (body.prompt_text) state.prompt = body.prompt_text;
    else if (body.evidence && body.evidence.text) state.prompt = body.evidence.text;
    if (body.predicted_text) state.predictedText = body.predicted_text;
    if (body.python_argmax != null) state.argmax = body.python_argmax;
    else if (body.rtl_argmax != null) state.argmax = body.rtl_argmax;
    else if (body.evidence && body.evidence.python_argmax != null) state.argmax = body.evidence.python_argmax;
    if (body.python_argmax_text != null) state.argmaxText = body.python_argmax_text;
    else if (body.rtl_argmax_text != null) state.argmaxText = body.rtl_argmax_text;
    if (body.mismatch_events) state.mismatch = new Set(body.mismatch_events);
    if (body.timeline) setTimeline(body.timeline);
    markMismatches();
    paintOutput();
    applyMetrics(body.rtl_status, body.python_status);
    setRunWhy(body);
    paintRunProgress(body);
    if (body.evidence) {
      applyEvidence(body.evidence);
    } else if (body.status === "running") {
      setBadge($("compareBadge"), "not compared yet");
      $("compareDetail").textContent =
        "This job has not finished both forwards. Python vs RTL numbers appear only after Verilator completes — the mismatch text from a previous Hello run is not about this job.";
      $("mismatchBox").hidden = true;
    }
    const rtl = body.rtl_status || {};
    // DONE is the handshake after the 1-cycle ARGMAX. Exclude it and the
    // DAG never lights argmax — polling cannot catch ST_ARGMAX itself.
    const liveRtl = body.phase === "rtl" && rtl.stage && rtl.stage !== "IDLE";
    const livePy = body.phase === "python" || (body.python_status && body.python_status.phase === "python" && body.status === "running");
    if (liveRtl) {
      state.live = true;
      stopPlay();
      paintFromStage(rtl.stage, rtl.layer);
      applyMetrics(rtl, body.python_status);
      return;
    }
    if (livePy) {
      state.live = true;
      stopPlay();
      if (state.timeline.length) showIndex(state.timeline.length - 1);
      return;
    }
    state.live = false;
    if (autoplay && state.timeline.length && !state.userPaused) {
      showIndex(0);
      startPlay();
    } else if (state.timeline.length) {
      showIndex(state.index);
    }
  }

  async function refreshReady() {
    if (SHARE_DEMO) {
      setBadge($("readyBadge"), "public demo · recorded compare", "ok");
      $("verifyBtn").disabled = true;
      $("verifyBtn").textContent = "Recorded run";
      $("verifyText").readOnly = true;
      $("formError").hidden = true;
      setRunWhy({ status: "cached" });
      return;
    }
    try {
      const res = await apiFetch("/api/ready");
      const body = await res.json();
      if (body.ready) {
        const signed = !!body.rtl_signed_off;
        state.rtlSignedOff = signed;
        state.replayOnly = false;
        setBadge($("readyBadge"), signed ? "ROM signed off · RTL only" : "ROM + tokenizer ready", "ok");
        const btn = $("verifyBtn");
        if (btn && !SHARE_DEMO) btn.textContent = signed ? "Run chip" : "Compare full statement";
        $("formError").hidden = true;
      } else if (body.replay_only || body.tokenizer) {
        state.replayOnly = true;
        setBadge($("readyBadge"), "recorded replay", "ok");
        const btn = $("verifyBtn");
        if (btn) btn.textContent = "Replay statement";
        $("formError").hidden = true;
      } else {
        setBadge($("readyBadge"), "ROM missing", "bad");
        $("formError").hidden = false;
        $("formError").textContent = body.error || "Packed ROM or tokenizer is missing.";
      }
    } catch (_) {
      setBadge($("readyBadge"), "lab offline", "bad");
    }
  }

  function stopPoll() {
    if (state.poll) {
      clearTimeout(state.poll);
      state.poll = null;
    }
  }

  async function pollJob(jobId) {
    try {
      const res = await apiFetch(`/api/verify/${jobId}`);
      const body = await res.json();
      state.tokenIds = body.token_ids || state.tokenIds;
      paintTokens(state.tokenIds, body.rtl_status && body.rtl_status.token_index);
      ingest(body, { autoplay: false });
      if (body.status === "running") {
        setBadge($("runBadge"), body.phase === "rtl" || body.mode === "rtl_infer" ? "RTL running" : "Python oracle", "run");
        $("verifyBtn").disabled = true;
        state.poll = setTimeout(() => pollJob(jobId), 400);
        return;
      }
      $("verifyBtn").disabled = false;
      stopPoll();
      paintRunProgress(body);
      $("formError").hidden = true;
      if (body.evidence || (body.timeline && body.timeline.length)) {
        const infer = body.mode === "rtl_infer" || (body.evidence && body.evidence.mode === "rtl_infer");
        const ok = infer || (body.evidence && body.evidence.passed);
        setBadge($("runBadge"), infer ? "chip complete · signed-off ROM" : (ok ? "full-statement compare complete" : "mismatch, playing through"),
          ok ? "ok" : "bad");
        ingest(body, { autoplay: true });
        return;
      }
      setBadge($("runBadge"), "failed", "bad");
      $("formError").hidden = false;
      $("formError").textContent = body.message || body.stderr || body.error || "verify failed";
    } catch (err) {
      $("verifyBtn").disabled = false;
      setBadge($("runBadge"), "poll error", "bad");
      state.poll = setTimeout(() => pollJob(jobId), 1500);
    }
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (SHARE_DEMO) return;
    const text = $("verifyText").value.trim();
    if (!text) return;
    $("formError").hidden = true;
    $("verifyBtn").disabled = true;
    setBadge($("runBadge"), "tokenizing…", "run");
    $("runWhy").textContent = state.rtlSignedOff
      ? "Tokenizing. ROM is signed off — RTL only, no Python compare."
      : "Tokenizing. Next: Python oracle (bit-exact, no NumPy/PyTorch), then the chip — full-statement compare.";
    setBadge($("compareBadge"), "not compared yet");
    $("compareDetail").textContent = state.rtlSignedOff
      ? "ROM is signed off. This run sends token IDs into RTL only — Python is not re-run."
      : "This job has not finished both forwards. Python vs RTL numbers appear only after Verilator completes.";
    $("mismatchBox").hidden = true;
    stopPlay();
    state.userPaused = false;
    state.mismatch = new Set();
    state.workDir = null;
    state.runStartedAt = null;
    state.progressPeak = 0;
    paintRunProgress({ status: "running", phase: "starting", python_status: {} });
    state.prompt = text;
    state.predictedText = "";
    state.argmax = null;
    state.argmaxText = "";
    paintOutput();
    try {
      const tok = await apiFetch("/api/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const tokBody = await tok.json();
      if (!tok.ok) throw new Error(tokBody.detail || "tokenize failed");
      state.tokenIds = tokBody.token_ids;
      if (tokBody.token_texts) state.tokenTexts = tokBody.token_texts;
      paintTokens(state.tokenIds, 0);

      const res = await apiFetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, tier: "T4" })
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
      state.tokenIds = body.token_ids || state.tokenIds;
      paintTokens(state.tokenIds, 0);
      if (body.status === "cached" || body.status === "partial" ||
          (body.timeline && body.timeline.length && body.status !== "running")) {
        applyReplay(body);
        return;
      }
      setBadge($("runBadge"), "running", "run");
      ingest(body, { autoplay: false });
      pollJob(body.job_id);
    } catch (err) {
      $("verifyBtn").disabled = false;
      $("formError").hidden = false;
      $("formError").textContent = err.message;
      setBadge($("runBadge"), "error", "bad");
    }
  }

  async function loadLatest() {
    try {
      const res = await (SHARE_DEMO ? fetch("data/evidence.json") : apiFetch("/api/evidence"));
      const body = await res.json();
      if (!body.evidence && !(body.timeline && body.timeline.length)) return;
      if (body.evidence && body.evidence.work_dir) state.workDir = body.evidence.work_dir;
      state.tokenIds = body.token_ids || (body.evidence && body.evidence.token_ids) || [];
      if (body.evidence && body.evidence.text && (SHARE_DEMO || !$("verifyText").value)) {
        $("verifyText").value = body.evidence.text;
      }
      paintTokens(state.tokenIds, 0);
      setBadge($("runBadge"), body.evidence && body.evidence.passed ? "full-statement compare complete" : "mismatch, playing through",
        body.evidence && body.evidence.passed ? "ok" : "bad");
      state.runFinished = true;
      ingest(body, { autoplay: true });
    } catch (_) { /* no prior evidence */ }
  }

  function applyReplay(body) {
    if (body.token_ids) state.tokenIds = body.token_ids;
    if (body.token_texts) state.tokenTexts = body.token_texts;
    paintTokens(state.tokenIds, (body.rtl_status && body.rtl_status.token_index) || 0);
    if (body.status === "cached" || (body.evidence && body.evidence.passed && body.status !== "running")) {
      $("verifyBtn").disabled = false;
      const infer = body.mode === "rtl_infer" || (body.evidence && body.evidence.mode === "rtl_infer");
      const oracle = body.mode === "oracle_replay";
      setBadge($("runBadge"), infer
        ? "RTL complete · signed-off ROM"
        : oracle
          ? "cached Python · playback"
          : "full-statement compare · cached replay", "ok");
      state.runFinished = true;
      ingest(body, { autoplay: true });
      return "done";
    }
    if (body.status === "running") {
      $("verifyBtn").disabled = true;
      setBadge($("runBadge"), body.phase === "rtl" || body.mode === "rtl_infer" ? "RTL running" : "Python oracle", "run");
      ingest(body, { autoplay: false });
      return "running";
    }
    if (body.status === "partial" && body.timeline && body.timeline.length) {
      $("verifyBtn").disabled = false;
      setBadge($("runBadge"), "replaying saved checkpoints", "run");
      ingest(body, { autoplay: true });
      return "done";
    }
    if (body.timeline && body.timeline.length) {
      $("verifyBtn").disabled = false;
      setBadge($("runBadge"), body.evidence && body.evidence.passed ? "full-statement compare · cached replay" : "playing through",
        body.evidence && body.evidence.passed ? "ok" : "");
      ingest(body, { autoplay: true });
      return "done";
    }
    return "miss";
  }

  async function pollReplay(text) {
    try {
      const res = await apiFetch("/api/replay?text=" + encodeURIComponent(text) + "&tier=T4");
      const body = await res.json();
      const kind = applyReplay(body);
      if (kind === "running") {
        state.poll = setTimeout(() => pollReplay(text), 400);
        return;
      }
      $("verifyBtn").disabled = false;
      stopPoll();
    } catch (_) {
      state.poll = setTimeout(() => pollReplay(text), 1500);
    }
  }

  async function loadCachedReplay() {
    if (SHARE_DEMO) {
      await loadLatest();
      return;
    }
    const text = $("verifyText").value.trim();
    if (!text) {
      await loadLatest();
      return;
    }
    try {
      const res = await apiFetch("/api/replay?text=" + encodeURIComponent(text) + "&tier=T4");
      const body = await res.json();
      if (!res.ok || body.status === "miss") return;
      const kind = applyReplay(body);
      if (kind === "running") pollReplay(text);
    } catch (_) { /* no cache for this prompt */ }
  }

  // ---- Boot ----------------------------------------------------------------
  openInspector("fsm", false);
  renderNetwork();
  renderNnMap();
  renderBanks();
  document.querySelectorAll(".deck-dot").forEach((d, i) =>
    d.addEventListener("click", () => showSlide(i)));
  document.addEventListener("keydown", (ev) => {
    if (ev.target && ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;
    if (!$("tensorModal").hidden) return;
    if (ev.key === "ArrowRight") showSlide(state.slide + 1);
    if (ev.key === "ArrowLeft") showSlide(state.slide - 1);
  });
  if ($("sbFused")) $("sbFused").addEventListener("click", () => {
    state.walkSched = "fused"; renderProfile();
    $("sbFused").setAttribute("aria-pressed", "true");
    $("sbUnfused").setAttribute("aria-pressed", "false");
  });
  if ($("sbUnfused")) $("sbUnfused").addEventListener("click", () => {
    state.walkSched = "unfused"; renderProfile();
    $("sbUnfused").setAttribute("aria-pressed", "true");
    $("sbFused").setAttribute("aria-pressed", "false");
  });
  if ($("cfgFused")) $("cfgFused").addEventListener("click", () => {
    state.walkSched = "fused"; renderProfile();
    if ($("sbFused")) $("sbFused").setAttribute("aria-pressed", "true");
    if ($("sbUnfused")) $("sbUnfused").setAttribute("aria-pressed", "false");
  });
  if ($("cfgUnfused")) $("cfgUnfused").addEventListener("click", () => {
    state.walkSched = "unfused"; renderProfile();
    if ($("sbUnfused")) $("sbUnfused").setAttribute("aria-pressed", "true");
    if ($("sbFused")) $("sbFused").setAttribute("aria-pressed", "false");
  });
  window.addEventListener("resize", () => requestAnimationFrame(() => {
    drawDagEdges();
    drawGpuDagEdges();
    if (state.slideId === "eqs") drawNnDagEdges();
  }));
  $("verifyForm").addEventListener("submit", onSubmit);
  $("playBtn").addEventListener("click", () => {
    if (state.playing) {
      state.userPaused = true;
      stopPlay();
    } else {
      startPlay();
    }
  });
  $("stepBack").addEventListener("click", () => {
    state.userPaused = true;
    stopPlay();
    showIndex(state.index - 1);
  });
  $("stepFwd").addEventListener("click", () => {
    state.userPaused = true;
    stopPlay();
    showIndex(state.index + 1);
  });
  $("playSlider").addEventListener("input", (event) => {
    state.userPaused = true;
    stopPlay();
    showIndex(Number(event.target.value));
  });
  $("network").addEventListener("click", (event) => {
    const jump = event.target.closest("[data-jump-layer]");
    if (jump) {
      jumpLayer(Number(jump.getAttribute("data-jump-layer")));
      return;
    }
    const node = event.target.closest("[data-node]");
    if (node && node.getAttribute("data-node")) seekNode(node.getAttribute("data-node"));
  });
  // Chip inspector: click any block (or SRAM region tag) in the schematic.
  $("chip").addEventListener("click", (event) => {
    const block = event.target.closest("[data-block]");
    if (block && block.getAttribute("data-block")) {
      openInspector(block.getAttribute("data-block"));
      return;
    }
    if (event.target.closest("#chipInspector")) return;
  });
  $("chipInspectorClose").addEventListener("click", closeInspector);
  function setChipView(view) {
    const pnr = view === "pnr";
    if ($("chipViewRtl")) $("chipViewRtl").setAttribute("aria-pressed", String(!pnr));
    if ($("chipViewPnr")) $("chipViewPnr").setAttribute("aria-pressed", String(pnr));
    if ($("chip")) $("chip").hidden = pnr;
    if ($("asicPnr")) $("asicPnr").hidden = !pnr;
    if ($("chipViewLabel")) $("chipViewLabel").textContent = pnr ? "P&R floorplan [E]" : "RTL";
  }
  if ($("chipViewRtl")) $("chipViewRtl").addEventListener("click", () => setChipView("rtl"));
  if ($("chipViewPnr")) $("chipViewPnr").addEventListener("click", () => setChipView("pnr"));
  $("compute").addEventListener("click", () => {
    if (state.timeline[state.index]) openTensorModal(state.timeline[state.index]);
  });
  $("modalClose").addEventListener("click", closeModal);
  $("tensorModal").addEventListener("click", (event) => {
    if (event.target === $("tensorModal")) closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if ($("nnMechModal") && !$("nnMechModal").hidden) {
      closeNnMech();
      return;
    }
    if (!$("tensorModal").hidden) closeModal();
  });
  $("modalPy").addEventListener("click", () => {
    state.modal.source = "python";
    state.modal.offset = 0;
    document.querySelectorAll(".modal-tabs .ghost").forEach((btn) => {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-src") === "python" ? "true" : "false");
    });
    loadModalPage();
  });
  $("modalRtl").addEventListener("click", () => {
    state.modal.source = "rtl";
    state.modal.offset = 0;
    document.querySelectorAll(".modal-tabs .ghost").forEach((btn) => {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-src") === "rtl" ? "true" : "false");
    });
    loadModalPage();
  });
  $("modalPrev").addEventListener("click", () => {
    const rec = state.modal.rec;
    const page = (rec && rec.elements > 8192) ? 1024 : 4096;
    state.modal.offset = Math.max(0, state.modal.offset - page);
    loadModalPage();
  });
  $("modalNext").addEventListener("click", () => {
    const rec = state.modal.rec;
    const page = (rec && rec.elements > 8192) ? 1024 : 4096;
    if (state.modal.offset + page < (state.modal.total || 0)) state.modal.offset += page;
    loadModalPage();
  });

  // ---- Platform compare (analytical) --------------------------------------
  function fmtSci(x, unit) {
    const v = Number(x);
    if (!Number.isFinite(v)) return "—";
    if (v >= 1e9) return (v / 1e9).toFixed(1) + " G" + unit;
    if (v >= 1e6) return (v / 1e6).toFixed(1) + " M" + unit;
    if (v >= 1e3) return (v / 1e3).toFixed(1) + " k" + unit;
    if (v < 1e-3 && v > 0) return (v * 1e6).toFixed(1) + " u" + unit;
    return v.toFixed(3) + " " + unit;
  }

  function fmtJ(j) {
    const v = Number(j);
    if (!Number.isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1) return v.toFixed(2) + " J";
    if (a >= 1e-3) return (v * 1e3).toFixed(1) + " mJ";
    if (a >= 1e-6) return (v * 1e6).toFixed(1) + " µJ";
    return (v * 1e9).toFixed(1) + " nJ";
  }

  function fmtMs(s) {
    const v = Number(s);
    if (!Number.isFinite(v)) return "—";
    if (v >= 1) return v.toFixed(2) + " s";
    return (v * 1e3).toFixed(1) + " ms";
  }

  function fmtBytes(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return "—";
    if (v >= 1e12) return (v / 1e12).toFixed(2) + " TB";
    if (v >= 1e9) return (v / 1e9).toFixed(2) + " GB";
    if (v >= 1e6) return (v / 1e6).toFixed(1) + " MB";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + " kB";
    return v.toFixed(0) + " B";
  }

  // ---- Slide deck -----------------------------------------------------------
  // Slide identity is its data-slide-id, never its ordinal: slides can be
  // merged, reordered or removed by editing markup alone and every hook below
  // keeps firing on the right slide.
  const slideEls = () => Array.from(document.querySelectorAll(".slide"));
  const slideIds = () => slideEls().map((s) => s.dataset.slideId || "");

  function showSlide(n) {
    const els = slideEls();
    const nSlides = els.length;
    if (!nSlides) return;
    state.slide = ((n % nSlides) + nSlides) % nSlides;
    const id = els[state.slide].dataset.slideId || "";
    state.slideId = id;
    const on = (...ids) => ids.includes(id);

    els.forEach((s, i) => { s.hidden = i !== state.slide; });
    document.querySelectorAll(".deck-dot").forEach((d, i) => {
      const active = i === state.slide;
      d.classList.toggle("active", active);
      if (active) {
        d.setAttribute("aria-current", "step");
        if (typeof d.scrollIntoView === "function") {
          d.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
        }
      } else {
        d.removeAttribute("aria-current");
      }
    });

    document.body.classList.add("deck");
    slideIds().forEach((sid) => {
      if (sid) document.body.classList.toggle(`on-${sid}`, sid === id);
    });

    if (on("replica", "eqs") && !state.stageLoaded) { loadStageBreakdown(); loadWalk(); }
    if (on("eqs")) {
      renderNnMath();
      paintNnHeat();
      loadNnKpis();
      startNnAnim();
      requestAnimationFrame(() => {
        drawNnDagEdges();
        paintNnWhere();
      });
    } else {
      stopNnAnim();
    }
    if (on("replica")) requestAnimationFrame(drawDagEdges);
    if (on("gpu")) {
      if (!gpuAnim.introShown) {
        gpuSetStory(true);
        gpuAnim.introShown = true;
      }
      startGpuAnim();
      renderGpuMath();
      requestAnimationFrame(drawGpuDagEdges);
    }
    else stopGpuAnim();
    if (!on("cost")) {
      document.querySelectorAll(".slide-spec details.fold, .slide-scale details.fold").forEach((el) => {
        el.open = false;
      });
    }
    if (on("2d")) startD2Anim();
    else stopD2Anim();
    if (on("moe")) startMoeAnim();
    else stopMoeAnim();
    if (on("kpis")) {
      loadKpiTable();
      loadMachPlayer();
    } else {
      machStop();
    }
    if (on("workload")) loadWorkloadTable();
    if (on("packages")) loadKpiTable();  // same payload feeds the package facts
    if (on("packages") && window.HwPackages) {
      const wall = $("hwWall");
      // idempotent: only build the six packages the first time the slide opens
      if (wall && !wall.dataset.built) {
        // ordered by how far a weight must travel: off-package -> inside the
        // stacks -> already on the die
        window.HwPackages.mountAll(wall,
          ["h100", "b200", "tpu", "pim", "lpu", "cim", "maskrom", "rom2d"]);
        wall.dataset.built = "1";
      }
    }
    if (on("scaling")) {
      // one slide, three axes: load all three so switching tabs is instant
      loadArrayScaling();
      loadPrecisionScaling();
      loadMultiDieScaling();
    }
    if (on("cost")) {
      renderSpecMath();
      bindSpecSlider();
      paintSpecSlider(Number(($("specSlider") || {}).value || 3));
      paintRoadmapLive();
      paintRoadmapBars();
    }
    window.scrollTo({ top: 0 });
    const hash = `#slide-${state.slide + 1}`;
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }
  }

  // ---- Slide 1 · measured per-stage breakdown -------------------------------
  async function loadStageBreakdown() {
    try {
      const res = await apiFetch("/api/stage_breakdown");
      if (!res.ok) return;
      state.stageBreakdown = await res.json();
      state.stageLoaded = true;
      renderStageChart();
      paintNnHeat();
    } catch (_) { /* optional */ }
  }

  // ---- Slide 1 · the walk (per-layer, measured) ------------------------------
  async function loadWalk() {
    try {
      const [f, u, cmp] = await Promise.all([
        apiFetch("/api/walk?schedule=fused").then((r) => r.ok ? r.json() : null),
        apiFetch("/api/walk?schedule=unfused").then((r) => r.ok ? r.json() : null),
        apiFetch("/api/compare").then((r) => r.ok ? r.json() : null),
      ]);
      state.walks = { fused: f, unfused: u };
      state.walkSched = state.walkSched || "fused";
      if (cmp) state.compare = cmp;
      renderWalk();
      renderProfile();
      renderStageFormulas();
      renderKpiSlide();
    } catch (_) { /* optional */ }
  }

  function renderWalk() {
    const wrap = $("walkStrip");
    const w = (state.walks || {})[state.walkSched || "fused"];
    if (!wrap || !w || !Array.isArray(w.layers) || !w.layers.length) return;
    const clock = 500e6;
    const maxCyc = Math.max(...w.layers.map((l) => l.cycles));
    const rows = w.layers.map((l) => {
      const top = l.stages[0];
      const ms = l.cycles / clock * 1e3;
      const cum = l.start_cycle / clock * 1e3;
      return `<div class="walk-row" title="L${String(l.layer).padStart(2, "0")} · ${l.cycles.toLocaleString("en-US")} cycles · top: ${escapeHtml(top.stage)} ${top.cycles.toLocaleString("en-US")}">` +
        `<span class="wk-layer">L${String(l.layer).padStart(2, "0")}</span>` +
        `<span class="wk-bar"><span style="width:${(100 * l.cycles / maxCyc).toFixed(1)}%"></span></span>` +
        `<span class="wk-cyc">${(l.cycles / 1e6).toFixed(2)}M cyc · ${ms.toFixed(1)} ms</span>` +
        `<span class="wk-top">bottleneck: <b>${escapeHtml(top.stage)}</b> ${(100 * top.cycles / l.cycles).toFixed(0)}%</span>` +
        `<span class="wk-cum">@ ${cum.toFixed(0)} ms</span></div>`;
    }).join("");
    wrap.innerHTML =
      `<div class="walk-head"><span>layer</span><span>cycles (measured)</span><span>time</span><span>dominant stage</span><span>cumulative</span></div>` + rows;
  }

  function fmtTransistors(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return "—";
    if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    return fmt(v);
  }

  function stageFormula(stage) {
    const meta = STAGE_OP[stage] || {};
    const op = meta.op || meta.global;
    const eq = (op && OP_EQ[op]) || "—";
    const extra = meta.extra && OP_EQ[meta.extra] ? " · " + OP_EQ[meta.extra] : "";
    return eq + extra;
  }

  function macsForStage(stage) {
    const sb = state.stageBreakdown && state.stageBreakdown.schedules
      && state.stageBreakdown.schedules[state.walkSched || "fused"];
    if (!sb) return null;
    const row = (sb.stages || []).find((r) => r.stage === stage);
    return row ? row.macs : null;
  }

  function renderProfile() {
    const head = $("profileHead");
    const chart = $("profileChart");
    const w = (state.walks || {})[state.walkSched || "fused"];
    if (!head || !chart || !w) {
      renderStageFormulas();
      return;
    }
    const isFused = (state.walkSched || "fused") === "fused";
    const clock = 500e6;
    const totalMs = w.total_cycles / clock * 1e3;
    const sb = state.stageBreakdown && state.stageBreakdown.schedules.fused;
    const energyByStage = {};
    if (sb) {
      for (const r of sb.stages) {
        energyByStage[r.stage] = r.mac_energy_j + r.rom_energy_j;
      }
    }
    const top = w.stages.slice(0, 3);
    head.innerHTML =
      `<div class="key-card"><span class="kc-num">${(w.total_cycles / 1e6).toFixed(1)}M</span><span class="kc-cap">cycles for the full pass — measured, sums exactly</span></div>` +
      `<div class="key-card"><span class="kc-num">${totalMs.toFixed(1)} ms</span><span class="kc-cap">per token @ 500 MHz, 0 stalls — measured</span></div>` +
      `<div class="key-card"><span class="kc-num">${top[0].stage}</span><span class="kc-cap">biggest component: ${(100 * top[0].cycles / w.total_cycles).toFixed(0)}% of all cycles (${top[0].cycles.toLocaleString("en-US")})</span></div>` +
      `<div class="key-card"><span class="kc-num">${top[1].stage}</span><span class="kc-cap">#2: ${(100 * top[1].cycles / w.total_cycles).toFixed(0)}% · then ${top[2].stage} ${(100 * top[2].cycles / w.total_cycles).toFixed(0)}%</span></div>`;
    const maxCyc = Math.max(...w.stages.map((r) => r.cycles), 1);
    const rows = w.stages.map((r) => {
      const pct = (100 * r.cycles / maxCyc).toFixed(1);
      const share = (100 * r.cycles / w.total_cycles).toFixed(1);
      const e = energyByStage[r.stage];
      const eTxt = e != null ? ` · ${fmtJ(e)}` : "";
      return `<div class="ss-row"><span class="ss-name" title="${escapeHtml(r.stage)}">${escapeHtml(r.stage)}</span>` +
        `<span class="ss-bar"><span style="width:${pct}%"></span></span>` +
        `<span class="ss-val">${r.cycles.toLocaleString("en-US")} cyc · ${share}%${eTxt}</span></div>`;
    }).join("");
    chart.innerHTML =
      `<div class="sc-total">showing <strong>${isFused ? "FUSED" : "UNFUSED"}</strong> schedule · bottlenecks: ` +
      `<strong>${escapeHtml(top[0].stage)}</strong> (${(100 * top[0].cycles / w.total_cycles).toFixed(0)}%) → ` +
      `<strong>${escapeHtml(top[1].stage)}</strong> (${(100 * top[1].cycles / w.total_cycles).toFixed(0)}%) → ` +
      `<strong>${escapeHtml(top[2].stage)}</strong> (${(100 * top[2].cycles / w.total_cycles).toFixed(0)}%). ` +
      (isFused
        ? `The unfused schedule spends the same math across 295.4M cycles (5.18× more) — see docs/fused_vs_unfused.md for the code-level diff.`
        : `The fused schedule does this exact math in 57.0M cycles (5.18× less) by streaming GEMM stages and computing full-width norms sum-once — see docs/fused_vs_unfused.md.`) +
      `</div>` + rows;
    renderStageFormulas();
  }

  function renderStageFormulas() {
    const body = $("stageFormulaBody");
    const lead = $("stageLead");
    const keyline = $("stageKeyline");
    const chart = $("cfgProfileChart");
    const w = (state.walks || {})[state.walkSched || "fused"];
    if (!w) return;
    const isFused = (state.walkSched || "fused") === "fused";
    const clock = 500e6;
    const stages = (w.stages || []).filter((r) => !STAGE_SKIP.has(r.stage));
    const top = stages[0] || w.stages[0];
    const second = stages[1] || w.stages[1];
    const totalMs = w.total_cycles / clock * 1e3;
    const asic = state.compare && state.compare.platforms && state.compare.platforms[0];
    const tr = asic && asic.metrics;
    const topShare = (100 * top.cycles / w.total_cycles).toFixed(0);
    const topMs = top.cycles / clock * 1e3;
    if (lead) {
      lead.innerHTML =
        `<strong>${escapeHtml(top.stage)}</strong> takes the most time: ${topShare}% of the measured ` +
        `${isFused ? "fused" : "unfused"} pass (${top.cycles.toLocaleString("en-US")} cycles, ${topMs.toFixed(1)} ms @ 500 MHz) — ` +
        `<code>${escapeHtml(stageFormula(top.stage))}</code>. ` +
        (second
          ? `Next is <strong>${escapeHtml(second.stage)}</strong> (${(100 * second.cycles / w.total_cycles).toFixed(0)}%). `
          : "") +
        `Time here is RTL cycle count, not FLOP rank.`;
    }
    if (keyline) {
      const macTotal = stages.reduce((s, r) => s + (macsForStage(r.stage) || 0), 0);
      keyline.innerHTML =
        `<div class="key-card"><span class="kc-num">${(w.total_cycles / 1e6).toFixed(1)}M</span><span class="kc-cap">cycles · measured ${isFused ? "fused" : "unfused"} RTL, sums exactly</span></div>` +
        `<div class="key-card"><span class="kc-num">${escapeHtml(top.stage)}</span><span class="kc-cap">longest stage · ${topShare}% · ${topMs.toFixed(1)} ms @ 500 MHz</span></div>` +
        `<div class="key-card"><span class="kc-num">${macTotal ? fmtSci(macTotal, "") : "—"}</span><span class="kc-cap">MACs counted on this schedule (RTL counter)</span></div>` +
        `<div class="key-card"><span class="kc-num">${tr ? fmtTransistors(tr.transistors) : "—"}</span><span class="kc-cap">on-die transistors · est · 6T ROM ${tr ? fmtTransistors(tr.rom_transistors) : "—"} + SRAM ${tr ? fmtTransistors(tr.sram_transistors) : "—"} + compute ${tr ? fmtTransistors(tr.compute_transistors) : "—"}</span></div>`;
    }
    if (body) {
      body.innerHTML = stages.map((r, i) => {
        const share = (100 * r.cycles / w.total_cycles).toFixed(1);
        const ms = r.cycles / clock * 1e3;
        const macs = macsForStage(r.stage);
        const hot = i === 0 ? " class=\"hot\"" : "";
        return `<tr${hot}>` +
          `<td>${escapeHtml(r.stage)}</td>` +
          `<td class="eq">${escapeHtml(stageFormula(r.stage))}</td>` +
          `<td class="calc">${escapeHtml(STAGE_CALC[r.stage] || "")}</td>` +
          `<td class="num">${macs ? macs.toLocaleString("en-US") : "—"}</td>` +
          `<td class="num">${r.cycles.toLocaleString("en-US")}</td>` +
          `<td class="num">${ms >= 1 ? ms.toFixed(1) + " ms" : (ms * 1e3).toFixed(0) + " µs"}</td>` +
          `<td class="num">${share}%</td></tr>`;
      }).join("");
    }
    if (chart) {
      const maxCyc = Math.max(...stages.map((r) => r.cycles), 1);
      const rows = stages.map((r) => {
        const pct = (100 * r.cycles / maxCyc).toFixed(1);
        const share = (100 * r.cycles / w.total_cycles).toFixed(1);
        const macs = macsForStage(r.stage);
        const macTxt = macs ? ` · ${macs.toLocaleString("en-US")} MAC` : "";
        return `<div class="ss-row"><span class="ss-name" title="${escapeHtml(r.stage)}">${escapeHtml(r.stage)}</span>` +
          `<span class="ss-bar"><span style="width:${pct}%"></span></span>` +
          `<span class="ss-val">${r.cycles.toLocaleString("en-US")} cyc · ${share}%${macTxt}</span></div>`;
      }).join("");
      chart.innerHTML =
        `<div class="sc-total">${totalMs.toFixed(1)} ms/token @ 500 MHz · 0 stalls. ` +
        `Longest: <strong>${escapeHtml(top.stage)}</strong> (${topShare}%). ` +
        (tr
          ? `Transistors ${fmtTransistors(tr.transistors)} est (ROM cells dominate compute).`
          : "") +
        `</div>` + rows;
    }
    const fusedBtn = $("cfgFused");
    const unfusedBtn = $("cfgUnfused");
    if (fusedBtn) fusedBtn.setAttribute("aria-pressed", isFused ? "true" : "false");
    if (unfusedBtn) unfusedBtn.setAttribute("aria-pressed", isFused ? "false" : "true");
  }

  function fracEq(num, den) {
    if (!den) return `<span class="eq-line">${num}</span>`;
    return `<span class="frac"><span class="fn">${num}</span><span class="fd">${den}</span></span>`;
  }

  function renderKpiSlide() {
    const uWalk = (state.walks || {}).unfused;
    const fWalk = (state.walks || {}).fused;
    const clock = 500e6;
    const lanes = 16;
    const uCyc = (uWalk && uWalk.total_cycles) || 295368736;
    const fCyc = (fWalk && fWalk.total_cycles) || 57025007;
    const fMacs = 608945152;
    const fRom = 38086208;
    const macPj = 1.2, romPj = 0.8;
    const uT = uCyc / clock;
    const fT = fCyc / clock;
    const fToks = 1 / fT;
    const fEJ = fMacs * macPj / 1e12 + fRom * 256 * romPj / 1e12;
    const macUtil = 100 * fMacs / (fCyc * lanes);
    const ratio = uCyc / fCyc;

    const keyline = $("kpiKeyline");
    if (keyline) {
      keyline.innerHTML =
        `<div class="key-card"><span class="kc-num">${fToks.toFixed(1)}</span><span class="kc-cap">tok/s fused · 1 / ${(fT * 1e3).toFixed(0)} ms · measured 16-lane die</span></div>` +
        `<div class="key-card"><span class="kc-num">${(fT * 1e3).toFixed(0)} ms</span><span class="kc-cap">TPOT = cycles / 500 MHz</span></div>` +
        `<div class="key-card"><span class="kc-num">${ratio.toFixed(2)}×</span><span class="kc-cap">unfused is slower (naive re-reduce) · not the chip</span></div>` +
        `<div class="key-card"><span class="kc-num">0</span><span class="kc-cap">stalls per token · measured</span></div>`;
    }

    const blurb = $("kernelBlurb");
    if (blurb) {
      blurb.innerHTML =
        `<strong>Unfused</strong> re-reads the whole activation row for every output element — including RMSNorm, which redoes the same 1024-sum 1024 times. ` +
        `<strong>${(uCyc / 1e6).toFixed(0)}M cycles</strong> (${(uT * 1e3).toFixed(0)} ms). ` +
        `<strong>Fused</strong> streams GEMM and holds the norm sum once: <strong>${(fCyc / 1e6).toFixed(0)}M cycles</strong> (${(fT * 1e3).toFixed(0)} ms). ` +
        `Same bit-exact math. V_PROJ stays unfused (transposed V-cache).`;
    }

    const kpiBody = $("kpiBody");
    if (kpiBody) {
      kpiBody.innerHTML =
        `<tr><td>Tokens / sec</td><td>${fracEq("1", "time")}</td>` +
        `<td class="num">${fToks.toFixed(1)}<div class="prov">measured</div></td></tr>` +
        `<tr><td>TPOT</td><td>${fracEq("cycles", "500 MHz")}</td>` +
        `<td class="num">${(fT * 1e3).toFixed(0)} ms<div class="prov">measured</div></td></tr>` +
        `<tr><td>Energy / token</td><td>${fracEq("MAC pJ + ROM pJ", "1 token")}</td>` +
        `<td class="num">${(fEJ * 1e3).toFixed(1)} mJ<div class="prov est">est</div></td></tr>` +
        `<tr><td>MAC utilization</td><td>${fracEq("MAC ops", "cycles × 16")}</td>` +
        `<td class="num">${macUtil.toFixed(0)}%<div class="prov">measured</div></td></tr>` +
        `<tr><td>Users</td><td>${fracEq("N_max", "one FSM")}</td>` +
        `<td class="num">1<div class="prov">this die</div></td></tr>`;
    }

    const why = $("kpiWhySlow");
    if (why) {
      why.innerHTML =
        `${fCyc.toLocaleString("en-US")} / 5×10⁸ Hz = ${(fT * 1e3).toFixed(0)} ms → ${fToks.toFixed(1)} tok/s. ` +
        `Energy ≈ ${fMacs.toLocaleString("en-US")} × 1.2 pJ + ${fRom.toLocaleString("en-US")} × 256 × 0.8 pJ = ${(fEJ * 1e3).toFixed(1)} mJ (28nm-class switching, no leakage). ` +
        `At 4096 lanes (Configure) the same math is ~256× more MAC throughput if still compute-bound — ~${(fToks * (4096 / 16)).toFixed(0)} tok/s.`;
    }
  }

  function renderStageChart() {
    // The old per-stage toggle chart was replaced by the walk + profile
    // panels; keep the function as a no-op guard for stale listeners.
    const wrap = $("stageChart");
    if (!wrap || !state.stageBreakdown) return;
    renderProfile();
  }

  // ---- Slide 2 · multi-die visualization ------------------------------------
  function renderDieViz(d) {
    const wrap = $("dieViz");
    if (!wrap) return;
    const mc = d.multi_chip;
    const n = Math.min(mc.dies, 64);
    const cols = Math.ceil(Math.sqrt(n));
    const tileSize = n <= 4 ? 120 : n <= 16 ? 84 : n <= 36 ? 62 : 46;
    const bwPct = Math.min(100, 100 * mc.feed_bps / mc.tier_bw_per_die);
    const tiles = Array.from({ length: n }, (_, i) =>
      `<div class="die-tile${mc.bandwidth_limited ? " bwl" : ""}" ` +
      `style="width:${tileSize}px;height:${tileSize}px" ` +
      `title="die ${i}: ${mc.lanes_per_die} lanes · ${fmtBytes(mc.weight_slice_b)} weights · ${fmtBytes(mc.kv_slice_b)} KV">` +
      `<b>${mc.lanes_per_die >= 1000 ? (mc.lanes_per_die / 1000) + "k" : mc.lanes_per_die}</b>` +
      `<span>lanes</span>` +
      `</div>`).join("");
    // The alternative people would otherwise build: one big GPU-style die
    // with a unified cache hierarchy and every lane reaching across it.
    const gpuLanes = Math.min(d.lanes, 2048);
    wrap.innerHTML =
      `<h4>this design vs the default · ${n} die${n > 1 ? "s" : ""} × ${mc.lanes_per_die} lanes</h4>` +
      `<div class="arch-compare">` +
      `<div class="arch-col">` +
      `<p class="arch-name ours">this chip · NUMA slices</p>` +
      `<div class="die-grid" style="grid-template-columns:repeat(${cols}, ${tileSize + 8}px)">${tiles}</div>` +
      `<div class="arch-notes">` +
      `<div class="pc-row"><span>memory</span><span>no cache hierarchy · weights fixed per die · KV local</span></div>` +
      `<div class="pc-row"><span>schedule</span><span>static FSM · 0 stalls (measured)</span></div>` +
      `<div class="pc-row"><span>per-die tier BW</span><span>${(mc.feed_bps / 1e12).toFixed(2)} / ${(mc.tier_bw_per_die / 1e12).toFixed(2)} TB/s ${mc.bandwidth_limited ? "· LIMITED" : "· " + mc.bw_headroom.toFixed(1) + "× headroom"}</span></div>` +
      `<div class="pc-row"><span>capacity/die</span><span>${fmtBytes(mc.weight_slice_b + mc.kv_slice_b)} / ${fmtBytes(mc.capacity_b)} ${mc.capacity_ok ? "ok" : "OVERFLOW"}</span></div>` +
      `<div class="pc-row"><span>collectives</span><span>${mc.dies > 1 ? (mc.allreduce_s_per_token * 1e6).toFixed(1) + " µs/token (" + (mc.allreduce_share * 100).toFixed(2) + "%)" : "none (1 die)"}</span></div>` +
      `</div></div>` +
      `<div class="arch-col">` +
      `<p class="arch-name gpu">the default · GPU-style monolith</p>` +
      `<div class="gpu-die">` +
      `<div class="gpu-l2">unified L2 · 200–400 cyc UMA latency</div>` +
      `<div class="gpu-sm-grid">${Array.from({ length: 16 }, () =>
        `<div class="gpu-sm">SM</div>`).join("")}</div>` +
      `<div class="gpu-hbm">shared HBM · every lane crosses the fabric</div>` +
      `</div>` +
      `<div class="arch-notes">` +
      `<div class="pc-row"><span>memory</span><span>thick cache hierarchy · UMA contention · 60–70% realized BW</span></div>` +
      `<div class="pc-row"><span>schedule</span><span>warp occupancy · needs batch to hide latency</span></div>` +
      `<div class="pc-row"><span>at batch 1</span><span>pipeline idles — the mechanism breaks down</span></div>` +
      `<div class="pc-row"><span>kernel launches</span><span>per op · host round-trips between stages</span></div>` +
      `<div class="pc-row"><span>same model</span><span>H100: 198 mJ vs this chip's 1.19 mJ @ 0.6B (measured cycles, N4-class energy)</span></div>` +
      `</div></div>` +
      `</div>` +
      `<p class="muted small">Each tile is one reticle-class die: MAC lanes + a permanent weight slice + a KV shard, attached 1:1 to the ${escapeHtml(d.tier)} tier — the NUMA pattern Jalapeño validated. The alternative puts every lane behind one shared cache hierarchy; at batch 1 that mechanism has nothing to hide latency with.</p>`;
  }

  // ---- Slide 3 · analysis ----------------------------------------------------
  function updateConfigKeyline(d) {
    const c = d.chip.metrics;
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    set("kcLat", fmtMs(c.latency_s));
    set("kcEnergy", fmtJ(c.total_energy_j));
    set("kcDies2", `${d.multi_chip.dies} × ${d.multi_chip.lanes_per_die}`);
    set("kcBw", c.bandwidth_limited ? "BW-limited" : "fits BW");
  }

  function renderAnalysis(d) {
    const cards = $("analysisCards");
    if (!cards) return;
    const c = d.chip.metrics;
    const mem = d.memory;
    const mc = d.multi_chip;
    const lanes = d.lanes;
    const computeTr = lanes * 4200 * 2 + 24000;
    const kvTr = Math.round(mem.kv_bytes_at_ctx * 8 / 6);
    const cardsHtml =
      `<div class="sm-card"><h4>transistors <span class="prov-badge est">est</span></h4>` +
      `<div class="pc-row"><span>compute lanes</span><span>${(computeTr / 1e6).toFixed(1)} M</span></div>` +
      `<div class="pc-row"><span>KV SRAM (6T)</span><span>${(kvTr / 1e9).toFixed(2)} B</span></div>` +
      `<div class="pc-row"><span>total on-die</span><span><strong>${((computeTr + kvTr) / 1e9).toFixed(2)} B</strong></span></div>` +
      `<div class="pc-row"><span>weights (in tier)</span><span>not on-die</span></div>` +
      `<div class="pc-row"><span>area</span><span>${c.area_mm2.toFixed(1)} mm² · ${mc.dies} die(s)</span></div></div>` +
      `<div class="sm-card"><h4>energy / token <span class="prov-badge est">est</span></h4>` +
      `<div class="pc-row"><span>compute (MACs)</span><span>${fmtJ(c.compute_energy_j)}</span></div>` +
      `<div class="pc-row"><span>weights (tier)</span><span>${fmtJ(c.weight_energy_j)}</span></div>` +
      `<div class="pc-row"><span>KV</span><span>${fmtJ(c.kv_energy_j)}</span></div>` +
      `<div class="pc-row"><span>total</span><span><strong>${fmtJ(c.total_energy_j)}</strong></span></div>` +
      `<div class="pc-row"><span>power</span><span>${c.power_w.toFixed(0)} W</span></div></div>` +
      `<div class="sm-card"><h4>time / token</h4>` +
      `<div class="pc-row"><span>cycles</span><span>${c.cycles.toLocaleString("en-US")}</span></div>` +
      `<div class="pc-row"><span>latency</span><span><strong>${fmtMs(c.latency_s)}</strong></span></div>` +
      `<div class="pc-row"><span>bandwidth</span><span>${c.bandwidth_limited ? "LIMITED" : "not limiting"}</span></div>` +
      `<div class="pc-row"><span>calibration</span><span>0.21% vs measured</span></div></div>`;
    cards.innerHTML = cardsHtml;
    const setK = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    setK("kaTr", ((computeTr + kvTr) / 1e9).toFixed(2) + " B");
    setK("kaEnergy", fmtJ(c.total_energy_j));
    setK("kaArea", c.area_mm2.toFixed(1) + " mm² × " + mc.dies);
    const h100m = d.rivals[0].metrics;
    setK("kaH100", h100m.total_energy_j ? (h100m.total_energy_j / c.total_energy_j).toFixed(1) + "×" : "—");
    setK("tkTr", ((computeTr + kvTr) / 1e9).toFixed(2) + " B");
    setK("tkEnergy3", fmtJ(c.total_energy_j));
    const st = $("analysisStages");
    if (st) {
      const maxCyc = Math.max(...d.chip.stages.map((r) => r.cycles), 1);
      const total = d.chip.stages.reduce((s, x) => s + x.cycles, 0);
      st.innerHTML = `<h4>per-stage cycles · calibrated model</h4>` +
        d.chip.stages.slice().sort((a, b) => b.cycles - a.cycles).map((r) => {
          const pct = (100 * r.cycles / maxCyc).toFixed(1);
          const share = (100 * r.cycles / total).toFixed(1);
          return `<div class="ss-row"><span class="ss-name">${escapeHtml(r.stage)}</span>` +
            `<span class="ss-bar"><span style="width:${pct}%"></span></span>` +
            `<span class="ss-val">${Math.round(r.cycles).toLocaleString("en-US")} · ${share}%</span></div>`;
        }).join("");
    }
    renderRivalsChart(d, "analysisRivals");
  }

  function renderRivalsChart(d, targetId) {
    const wrap = $(targetId);
    if (!wrap) return;
    const chip = d.chip.metrics;
    const rows = d.rivals.filter((r) => r.feasible !== false && r.metrics.latency_s);
    const infeasible = d.rivals.filter((r) => r.feasible === false);
    const maxLat = Math.max(chip.latency_s, ...rows.map((r) => r.metrics.latency_s));
    const maxE = Math.max(chip.total_energy_j, ...rows.map((r) => r.metrics.total_energy_j));
    const bar = (v, max, color) =>
      `<span class="cc-bar"><span style="width:${(100 * v / max).toFixed(1)}%;background:${color}"></span></span>`;
    const row = (name, lat, e, color, note) =>
      `<div class="cc-row"><span class="cc-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>` +
      `<span class="cc-cell">${bar(lat, maxLat, color)}<em>${fmtMs(lat)}</em></span>` +
      `<span class="cc-cell">${bar(e, maxE, color)}<em>${fmtJ(e)}</em></span>` +
      `<span class="cc-vs">${note}</span></div>`;
    const html = rows.map((r) => {
      const m = r.metrics;
      const spd = (m.latency_s / chip.latency_s).toFixed(2) + "× chip latency";
      const eff = (m.total_energy_j / chip.total_energy_j).toFixed(1) + "× chip energy";
      return row(r.platform, m.latency_s, m.total_energy_j, "#94a3b8", spd + " · " + eff);
    }).join("");
    const inf = infeasible.map((r) =>
      `<div class="cc-row infeasible"><span class="cc-name">${escapeHtml(r.platform)}</span>` +
      `<span class="cc-vs" colspan="3">${escapeHtml(r.note || "infeasible")}</span></div>`).join("");
    wrap.innerHTML =
      `<h4>same model · chip vs accelerators <span class="prov-badge est">est</span></h4>` +
      `<div class="cc-head"><span></span><span>latency</span><span>energy/token</span><span>delta</span></div>` +
      row(`this chip (${d.lanes} lanes × ${d.multi_chip.dies})`, chip.latency_s, chip.total_energy_j, "#0f766e", "reference") +
      html + inf;
  }

  // ---- Slide 4 · verdict ------------------------------------------------------
  async function renderVerdict() {
    const d = state.lastScenario;
    const fixed = $("verdictFixed");
    if (fixed && d) {
      const mem = d.memory;
      const rom = mem.rom;
      let verdict, limits;
      if (d.preset === "qwen3_06b") {
        verdict = "Fixed weights WIN at this scale. 0.6B BF16 = 1.19 GB fits one die's ROM; weights are read for ~0.08 pJ/bit instead of streamed from HBM at ~3.5 pJ/bit — a ~40× memory-energy advantage: 1.19 mJ vs 198 mJ on H100 (measured cycles + ROM reads, N4-class energy constants).";
        limits = "Limit: ROM area grows linearly with params. One reticle die holds ~2.8 GB of ROM — this is the entire fixed-weight regime.";
      } else if (d.preset === "moe_1t") {
        verdict = `Fixed weights DO NOT scale to 1T: ${fmtBytes(mem.weight_bytes_stored)} would need ${rom.dies_needed.toLocaleString("en-US")} reticle dies (${rom.verdict}). The buildable design keeps the fixed-weight MECHANISM per die — each die owns a permanent weight slice — and shards the model across ${d.multi_chip.dies} dies with a ${d.tier} tier for capacity.`;
        limits = `Limits: (1) capacity — ${fmtBytes(mem.weight_bytes_stored)} needs ${d.multi_chip.dies} dies minimum; (2) bandwidth — per-die demand ${(d.multi_chip.feed_bps / 1e12).toFixed(2)} TB/s vs ${(d.multi_chip.tier_bw_per_die / 1e12).toFixed(2)} TB/s attached; (3) numerics — INT4/INT8 weights are approximate, never the bit-exact target.`;
      } else {
        verdict = `Fixed weights are INFEASIBLE for dense 1T: ${fmtBytes(mem.weight_bytes_stored)} = ${rom.dies_needed.toLocaleString("en-US")} reticle dies of pure ROM. Dense 1T exists only as a streaming design — and then the GPU's HBM parity erases the latency advantage (see the rivals chart).`;
        limits = "Limit: dense 1T BF16 is bandwidth-bound everywhere; the chip's remaining edge is energy per MAC and the deterministic schedule, not memory.";
      }
      fixed.innerHTML =
        `<h4>(a) should weights be fixed (ROM)?</h4>` +
        `<p class="verdict-main">${verdict}</p>` +
        `<p class="muted small">${limits}</p>` +
        `<div class="pc-row"><span>ROM area if fixed</span><span>${rom.area_mm2_if_rom} mm² (${(rom.area_mm2_if_rom / 600).toFixed(0)} reticle dies)</span></div>` +
        `<div class="pc-row"><span>active weights / token</span><span>${fmtBytes(mem.weight_bytes_active_per_token)}</span></div>`;
    }
    const rivals = $("verdictRivals");
    if (rivals && d) {
      const chip = d.chip.metrics;
      const h100 = d.rivals[0].metrics;
      const b200 = d.rivals[1].metrics;
      const lpu = d.rivals[3];
      const lines = [];
      if (h100.latency_s) lines.push(`vs H100: ${(h100.latency_s / chip.latency_s).toFixed(2)}× the chip's latency, ${(h100.total_energy_j / chip.total_energy_j).toFixed(1)}× its energy`);
      if (b200.latency_s) lines.push(`vs B200: ${(b200.latency_s / chip.latency_s).toFixed(2)}× latency, ${(b200.total_energy_j / chip.total_energy_j).toFixed(1)}× energy`);
      if (lpu.feasible === false) lines.push(`LPU: infeasible — ${lpu.note}`);
      rivals.innerHTML =
        `<h4>(b) chip vs GPU / TPU / LPU at this configuration</h4>` +
        `<p class="verdict-main">${lines.join(" · ")}</p>` +
        `<p class="muted small">Key indicators: per-stage cycles (slide 1, measured; slide 3, calibrated), energy per token, and die count. The chip's structural advantages — weights read once from a fixed slice, deterministic schedule with 0 stalls, no kernel launches — hold at every scale; the GPU's HBM bandwidth parity erases the latency edge exactly when the design becomes bandwidth-bound.</p>`;
      renderRivalsChart(d, "verdictRivalsChart");
    }
    const kvWrap = $("verdictKv");
    if (kvWrap) {
      try {
        const res = await apiFetch("/api/scale1t?preset=qwen3_06b&lanes=16&ctx=1&schedule=fused&weight_bits=16&tier=sram");
        if (res.ok) {
          const kd = await res.json();
          const ks = kd.kv_story;
          if (ks) {
            const rows = ks.rows.map((r) =>
              `<tr><td>${r.ctx.toLocaleString("en-US")}</td><td>${fmtBytes(r.kv_bytes)}</td>` +
              `<td>${r.fits_sram ? "SRAM" : "HBM"}</td><td>${r.kv_over_chip_energy.toFixed(1)}×</td></tr>`).join("");
            kvWrap.innerHTML =
              `<h4>the 0.6B chip's one HBM consumer: KV</h4>` +
              `<p class="muted small">Weights never leave ROM, so HBM exists only for the KV cache. It overtakes the whole measured chip at ctx ${ks.kv_energy_overtakes_chip_at_ctx.toLocaleString("en-US")}.</p>` +
              `<table class="plat-table"><thead><tr><th>ctx</th><th>KV</th><th>tier</th><th>vs chip energy</th></tr></thead><tbody>${rows}</tbody></table>`;
          }
        }
      } catch (_) { kvWrap.innerHTML = ""; }
    }
    const ext = $("verdictExternal");
    if (ext) {
      ext.innerHTML =
        `<h4>external cross-check · Jalapeño (OpenAI, 2026)</h4>` +
        `<p class="muted small">OpenAI's published inference-chip architecture independently reaches the same four conclusions this lab derives from the measured 0.6B chip:</p>` +
        `<div class="pc-row"><span>minimal memory hierarchy</span><span>they cut the L2 entirely (UMA costs 200–400 cycles on Blackwell); our chip never has one — weights stream once, activations live in slice-local SRAM</span></div>` +
        `<div class="pc-row"><span>batch-1 philosophy</span><span>their "shorten distance + prefetch" beats warp-occupancy 700–1,459 vs 169–535 tok/s/user; our deterministic 0-stall schedule is the statically-scheduled version of the same idea</span></div>` +
        `<div class="pc-row"><span>keep KV local</span><span>they reject PD-disaggregation because moving KV costs power and sync; our model shows KV is the whole HBM story for the 0.6B chip past ctx ~2k</span></div>` +
        `<div class="pc-row"><span>MXFP4-class weights</span><span>their 64×64 MXFP4 tensor units (group-of-32 scales) match our INT4 variant's group-scale model; precision right-sizing moves the bandwidth wall</span></div>` +
        `<div class="pc-row"><span>NUMA slices</span><span>their 64 core slices pair 1:1 with HBM stacks; our multi-die plan pairs each die with its weight slice + KV shard + tier attachment</span></div>` +
        `<p class="muted small">Divergence: Jalapeño adds out-of-order superscalar cores per slice to hide latency dynamically; our schedule is static by construction (bit-exactness requirement), which buys determinism at the cost of their generality. Their 9-month RTL-to-tapeout via LLM-agent + XLS flow is the toolchain path for scaling this lab's schedule to 1T geometry.</p>`;
    }
    if (d) renderCrossoverInto(d.economics, d.context, d.kv_bits, "verdictCrossover");
  }

  function renderCrossoverInto(econ, ctx, kvBits, targetId) {
    const wrap = $(targetId);
    if (!wrap) return;
    const w = 720, h = 240;
    const padL = 56, padR = 16, padT = 14, padB = 34;
    const iw = w - padL - padR, ih = h - padT - padB;
    const wPerTok = econ.weight_traffic_per_token_b;
    const kvPerTok = econ.kv_traffic_per_token_b / Math.max(ctx, 1);
    const xMax = 1e6;
    const yMax = Math.max(wPerTok, kvPerTok * xMax) * 1.3;
    const lx = (c) => padL + iw * Math.log10(Math.max(c, 1)) / Math.log10(xMax);
    const ly = (b) => padT + ih * (1 - Math.log10(Math.max(b, 1e3)) / Math.log10(yMax));
    const xctx = lx(ctx);
    const kvAtCtx = kvPerTok * ctx;
    const kvPath = `M ${lx(1)} ${ly(kvPerTok)} L ${lx(xMax)} ${ly(kvPerTok * xMax)}`;
    const wPath = `M ${lx(1)} ${ly(wPerTok)} L ${lx(xMax)} ${ly(wPerTok)}`;
    const cross = wPerTok / Math.max(kvPerTok, 1);
    const marks = [1, 1e3, 4e3, 131072, 1e6].map((c) =>
      `<line x1="${lx(c)}" y1="${padT}" x2="${lx(c)}" y2="${padT + ih}" stroke="#d5dde3" stroke-width="1"/>` +
      `<text x="${lx(c)}" y="${h - 12}" text-anchor="middle" font-size="9" fill="#5b6b76" font-family="var(--mono)">${c >= 1e6 ? "1M" : c.toLocaleString("en-US")}</text>`
    ).join("");
    const yMarks = [1e6, 1e9, 1e10, 1e11].filter((b) => b < yMax).map((b) =>
      `<line x1="${padL}" y1="${ly(b)}" x2="${padL + iw}" y2="${ly(b)}" stroke="#eef2f5" stroke-width="1"/>` +
      `<text x="${padL - 6}" y="${ly(b) + 3}" text-anchor="end" font-size="9" fill="#5b6b76" font-family="var(--mono)">${fmtBytes(b)}</text>`
    ).join("");
    wrap.innerHTML =
      `<h4>HBM traffic per token · weights vs KV${kvBits === 8 ? " (KV INT8)" : ""}</h4>` +
      `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="weights versus KV traffic by context">` +
      `<rect x="${lx(cross)}" y="${padT}" width="${padL + iw - lx(cross)}" height="${ih}" fill="#fee4e255"/>` +
      `<path d="${wPath}" stroke="#0f766e" stroke-width="2.5" fill="none"/>` +
      `<path d="${kvPath}" stroke="#b54708" stroke-width="2.5" fill="none" stroke-dasharray="6 4"/>` +
      marks + yMarks +
      `<circle cx="${xctx}" cy="${ly(kvAtCtx)}" r="5" fill="#15202b"/>` +
      `<text x="${xctx}" y="${ly(kvAtCtx) - 10}" text-anchor="middle" font-size="10" fill="#15202b" font-family="var(--mono)">you: ctx ${ctx.toLocaleString("en-US")}</text>` +
      `<text x="${lx(cross)}" y="${padT + 12}" text-anchor="middle" font-size="10" fill="#b42318" font-family="var(--mono)">KV wins ≥ ${Math.round(cross).toLocaleString("en-US")} ctx</text>` +
      `<text x="${padL + 8}" y="${ly(wPerTok) - 6}" font-size="10" fill="#0f766e" font-family="var(--mono)">weights ${fmtBytes(wPerTok)}/token (flat)</text>` +
      `<text x="${lx(xMax) - 8}" y="${ly(kvPerTok * xMax) - 6}" text-anchor="end" font-size="10" fill="#b54708" font-family="var(--mono)">KV ${fmtBytes(kvPerTok)}/token/ctx (linear)</text>` +
      `</svg>` +
      `<p class="muted small">Log scale. KV crosses weights at ctx ≈ ${Math.round(cross).toLocaleString("en-US")}; red region is KV-bound.</p>`;
  }

  // Crossover chart: per-token HBM traffic for weights (flat) vs KV (linear
  // in ctx), log-x from 1 to 1M tokens. The crossing point is the context
  // where the KV cache, not the weights, owns the memory system.
  function renderCrossover(econ, ctx, kvBits) {
    const wrap = $("scaleCrossover");
    if (!wrap) return;
    const w = 720, h = 240;
    const padL = 56, padR = 16, padT = 14, padB = 34;
    const iw = w - padL - padR, ih = h - padT - padB;
    const wPerTok = econ.weight_traffic_per_token_b;
    const kvPerTok = econ.kv_traffic_per_token_b / Math.max(ctx, 1);
    const xMax = 1e6;
    const yMax = Math.max(wPerTok, kvPerTok * xMax) * 1.3;
    const lx = (c) => padL + iw * Math.log10(Math.max(c, 1)) / Math.log10(xMax);
    const ly = (b) => padT + ih * (1 - Math.log10(Math.max(b, 1e3)) / Math.log10(yMax));
    const xctx = lx(ctx);
    const kvAtCtx = kvPerTok * ctx;
    // Weight line + KV line + shaded region where KV > weights.
    const kvPath = `M ${lx(1)} ${ly(kvPerTok)} L ${lx(xMax)} ${ly(kvPerTok * xMax)}`;
    const wPath = `M ${lx(1)} ${ly(wPerTok)} L ${lx(xMax)} ${ly(wPerTok)}`;
    const cross = wPerTok / Math.max(kvPerTok, 1);
    const marks = [1, 1e3, 4e3, 131072, 1e6].map((c) =>
      `<line x1="${lx(c)}" y1="${padT}" x2="${lx(c)}" y2="${padT + ih}" stroke="#d5dde3" stroke-width="1"/>` +
      `<text x="${lx(c)}" y="${h - 12}" text-anchor="middle" font-size="9" fill="#5b6b76" font-family="var(--mono)">${c >= 1e6 ? "1M" : c.toLocaleString("en-US")}</text>`
    ).join("");
    const yMarks = [1e6, 1e9, 1e10, 1e11].filter((b) => b < yMax).map((b) =>
      `<line x1="${padL}" y1="${ly(b)}" x2="${padL + iw}" y2="${ly(b)}" stroke="#eef2f5" stroke-width="1"/>` +
      `<text x="${padL - 6}" y="${ly(b) + 3}" text-anchor="end" font-size="9" fill="#5b6b76" font-family="var(--mono)">${fmtBytes(b)}</text>`
    ).join("");
    wrap.innerHTML =
      `<h4>HBM traffic per token · weights vs KV cache${kvBits === 8 ? " (KV INT8)" : ""}</h4>` +
      `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="weights versus KV traffic by context">` +
      `<rect x="${lx(cross)}" y="${padT}" width="${padL + iw - lx(cross)}" height="${ih}" fill="#fee4e255"/>` +
      `<path d="${wPath}" stroke="#0f766e" stroke-width="2.5" fill="none"/>` +
      `<path d="${kvPath}" stroke="#b54708" stroke-width="2.5" fill="none" stroke-dasharray="6 4"/>` +
      marks + yMarks +
      `<circle cx="${xctx}" cy="${ly(kvAtCtx)}" r="5" fill="#15202b"/>` +
      `<text x="${xctx}" y="${ly(kvAtCtx) - 10}" text-anchor="middle" font-size="10" fill="#15202b" font-family="var(--mono)">you: ctx ${ctx.toLocaleString("en-US")}</text>` +
      `<text x="${lx(cross)}" y="${padT + 12}" text-anchor="middle" font-size="10" fill="#b42318" font-family="var(--mono)">KV wins ≥ ${Math.round(cross).toLocaleString("en-US")} ctx</text>` +
      `<text x="${padL + 8}" y="${ly(wPerTok) - 6}" font-size="10" fill="#0f766e" font-family="var(--mono)">weights ${fmtBytes(wPerTok)}/token (flat)</text>` +
      `<text x="${lx(xMax) - 8}" y="${ly(kvPerTok * xMax) - 6}" text-anchor="end" font-size="10" fill="#b54708" font-family="var(--mono)">KV ${fmtBytes(kvPerTok)}/token/ctx (linear)</text>` +
      `</svg>` +
      `<p class="muted small">Log scale. The dashed KV line crosses the weight line at ctx ≈ ${Math.round(cross).toLocaleString("en-US")} — before that, streaming weights dominates HBM traffic; after it, reading the cache back does. Red region: KV-bound.</p>`;
  }

  // Config corner chart: four buildable 128k-class configurations, latency
  // and energy side by side, with the H100 reference line.
  async function loadConfigCorner() {
    const wrap = $("scaleConfigs");
    if (!wrap) return;
    const preset = $("scalePreset").value;
    const lanes = $("scaleLanes").value;
    const ctx = $("scaleCtx").value;
    const tier = $("scaleTier").value;
    const schedule = $("scaleSchedule").value;
    const configs = [
      { label: "BF16 W · BF16 KV", weight_bits: 16, kv_bits: 16 },
      { label: "BF16 W · INT8 KV", weight_bits: 16, kv_bits: 8 },
      { label: "INT8 W · INT8 KV", weight_bits: 8, kv_bits: 8 },
      { label: "INT4 W · INT8 KV", weight_bits: 4, kv_bits: 8 },
    ];
    try {
      const results = await Promise.all(configs.map((c) =>
        apiFetch(`/api/scale1t?preset=${preset}&lanes=${lanes}&ctx=${ctx}&schedule=${schedule}&tier=${tier}&weight_bits=${c.weight_bits}&kv_bits=${c.kv_bits}`)
        .then((r) => (r.ok ? r.json() : null))));
      const rows = results.map((d, i) => d ? { ...d, label: configs[i].label } : null);
      const valid = rows.filter(Boolean);
      if (!valid.length) { wrap.innerHTML = ""; return; }
      const h100 = valid[0].rivals[0];
      const maxLat = Math.max(...valid.map((r) => r.chip.metrics.latency_s),
        h100.metrics.latency_s || 0);
      const maxE = Math.max(...valid.map((r) => r.chip.metrics.total_energy_j),
        h100.metrics.total_energy_j || 0);
      const bar = (v, max, color) =>
        `<span class="cc-bar"><span style="width:${(100 * v / max).toFixed(1)}%;background:${color}"></span></span>`;
      const rowsHtml = rows.map((r) => {
        if (!r) return "";
        const m = r.chip.metrics;
        const h = r.rivals[0].metrics;
        const spd = h.latency_s && m.latency_s ? (h.latency_s / m.latency_s).toFixed(2) + "× vs H100" : "—";
        const eff = h.total_energy_j && m.total_energy_j ? (h.total_energy_j / m.total_energy_j).toFixed(1) + "× energy" : "—";
        return `<div class="cc-row"><span class="cc-name">${escapeHtml(r.label)}</span>` +
          `<span class="cc-cell">${bar(m.latency_s, maxLat, "#0f766e")}<em>${fmtMs(m.latency_s)}</em></span>` +
          `<span class="cc-cell">${bar(m.total_energy_j, maxE, "#b54708")}<em>${fmtJ(m.total_energy_j)}</em></span>` +
          `<span class="cc-dies">${r.multi_chip.dies} dies</span>` +
          `<span class="cc-vs">${spd} · ${eff}</span></div>`;
      }).join("");
      const hRow = h100.metrics.latency_s ?
        `<div class="cc-row h100"><span class="cc-name">H100 SXM (ref)</span>` +
        `<span class="cc-cell">${bar(h100.metrics.latency_s, maxLat, "#94a3b8")}<em>${fmtMs(h100.metrics.latency_s)}</em></span>` +
        `<span class="cc-cell">${bar(h100.metrics.total_energy_j, maxE, "#94a3b8")}<em>${fmtJ(h100.metrics.total_energy_j)}</em></span>` +
        `<span class="cc-dies">—</span><span class="cc-vs">baseline</span></div>` : "";
      wrap.innerHTML = `<h4>Configuration corner @ ctx ${Number(ctx).toLocaleString("en-US")} · ${escapeHtml(valid[0].label.split("·")[0].trim())} class</h4>` +
        `<div class="cc-head"><span></span><span>latency</span><span>energy/token</span><span>dies</span><span>vs H100</span></div>` +
        rowsHtml + hRow;
    } catch (_) { wrap.innerHTML = ""; }
  }

  async function loadScale1t() {
    if (!$("scalePreset")) return;
    const preset = $("scalePreset").value;
    const lanes = $("scaleLanes").value;
    const ctx = $("scaleCtx").value;
    const bits = $("scaleBits").value;
    const tier = $("scaleTier").value;
    const schedule = $("scaleSchedule").value;
    const dies = $("scaleDies").value;
    const kvBits = $("scaleKvBits").value;
    const params = new URLSearchParams({ preset, lanes, ctx, schedule,
      weight_bits: bits, tier, kv_bits: kvBits });
    if (dies) params.set("dies", dies);
    try {
      const res = await apiFetch(`/api/scale1t?${params}`);
      if (!res.ok) return;
      const d = await res.json();
      state.lastScenario = d;
      const mem = d.memory;
      const chip = d.chip.metrics;
      $("scaleMemory").innerHTML =
        `<div class="sm-card"><h4>weights</h4>` +
        `<div class="pc-row"><span>stored</span><span>${fmtBytes(mem.weight_bytes_stored)} (${((mem.params_stored||0)/1e12).toFixed(2)}T params)</span></div>` +
        `<div class="pc-row"><span>active / token</span><span>${fmtBytes(mem.weight_bytes_active_per_token)}</span></div>` +
        `<div class="pc-row"><span>as ROM</span><span>${mem.rom.verdict} · ${mem.rom.dies_needed.toLocaleString("en-US")} dies</span></div>` +
        `<div class="pc-row"><span>tier</span><span>${d.tier} · ${chip.bandwidth_limited ? "BW-limited" : "fits BW"}</span></div></div>` +
        `<div class="sm-card"><h4>KV cache @ ctx ${d.context}</h4>` +
        `<div class="pc-row"><span>size</span><span>${fmtBytes(mem.kv_bytes_at_ctx)}</span></div>` +
        `<div class="pc-row"><span>SRAM area</span><span>${mem.kv_sram_area_mm2} mm²</span></div>` +
        `<div class="pc-row"><span>verdict</span><span>${mem.kv_verdict}</span></div></div>` +
        `<div class="sm-card"><h4>chip (est.)</h4>` +
        `<div class="pc-row"><span>latency</span><span>${fmtMs(chip.latency_s)}</span></div>` +
        `<div class="pc-row"><span>energy</span><span>${fmtJ(chip.total_energy_j)}</span></div>` +
        `<div class="pc-row"><span>power</span><span>${chip.power_w.toFixed(0)} W</span></div>` +
        `<div class="pc-row"><span>area</span><span>${chip.area_mm2.toFixed(1)} mm²</span></div></div>`;
      const econ = d.economics;
      const mc = d.multi_chip;
      const kvShare = (100 * econ.kv_traffic_share).toFixed(econ.kv_traffic_share < 0.001 ? 3 : 1);
      $("scaleEcon").innerHTML =
        `<div class="sm-card"><h4>HBM traffic per token</h4>` +
        `<div class="pc-row"><span>weights (streamed)</span><span>${fmtBytes(econ.weight_traffic_per_token_b)}</span></div>` +
        `<div class="pc-row"><span>KV (read back)</span><span>${fmtBytes(econ.kv_traffic_per_token_b)} · ${kvShare}%</span></div>` +
        `<div class="pc-row"><span>KV overtakes weights</span><span>ctx ≈ ${econ.kv_overtakes_weights_at_ctx.toLocaleString("en-US")}</span></div>` +
        `<div class="pc-row"><span>energy/token (tier)</span><span>${fmtJ(econ.weight_energy_per_token_j + econ.kv_energy_per_token_j)}</span></div></div>` +
        `<div class="sm-card"><h4>memory bill (${d.tier})</h4>` +
        `<div class="pc-row"><span>weights capacity</span><span>${fmtBytes(econ.capacity.weights_b)} · $${Math.round(econ.capacity.weights_usd).toLocaleString("en-US")}</span></div>` +
        `<div class="pc-row"><span>KV capacity</span><span>${fmtBytes(econ.capacity.kv_b)} · $${econ.capacity.kv_usd.toFixed(2)}</span></div>` +
        `<div class="pc-row"><span>total</span><span>$${Math.round(econ.capacity.usd).toLocaleString("en-US")} @ $${econ.capacity.usd_per_gb}/GB</span></div></div>` +
        `<div class="sm-card"><h4>multi-chip plan</h4>` +
        `<div class="pc-row"><span>dies</span><span>${mc.dies} × ${mc.lanes_per_die} lanes</span></div>` +
        `<div class="pc-row"><span>per-die demand</span><span>${(mc.feed_bps/1e12).toFixed(2)} TB/s vs ${(mc.tier_bw_per_die/1e12).toFixed(2)} TB/s</span></div>` +
        `<div class="pc-row"><span>bandwidth</span><span>${mc.bandwidth_limited ? "LIMITED" : "ok · " + mc.bw_headroom.toFixed(1) + "x headroom"}</span></div>` +
        `<div class="pc-row"><span>capacity/die</span><span>${fmtBytes(mc.weight_slice_b + mc.kv_slice_b)} / ${fmtBytes(mc.capacity_b)} ${mc.capacity_ok ? "ok" : "OVERFLOW"}</span></div></div>`;
      renderDieViz(d);
      updateConfigKeyline(d);
      if (d.kv_story) {
        const ks = d.kv_story;
        const rows = ks.rows.map((r) =>
          `<tr><td>${r.ctx.toLocaleString("en-US")}</td>` +
          `<td>${fmtBytes(r.kv_bytes)}</td>` +
          `<td>${r.fits_sram ? "yes" : "no → HBM"}</td>` +
          `<td>${fmtJ(r.kv_read_energy_j + r.spill_energy_j)}</td>` +
          `<td>${r.kv_over_chip_energy.toFixed(1)}×</td></tr>`).join("");
        $("scaleKvStory").innerHTML =
          `<h4>KV-cache story · measured 0.6B chip (weights never leave ROM)</h4>` +
          `<p class="muted small">On-chip SRAM holds KV to ctx ${ks.sram_max_ctx_tokens.toLocaleString("en-US")}. ` +
          `KV energy overtakes the whole measured chip (MAC+ROM, ${fmtJ(ks.measured_chip_energy_j)}) at ctx ${ks.kv_energy_overtakes_chip_at_ctx.toLocaleString("en-US")} — past that, the cache, not the math, is the cost.</p>` +
          `<table class="plat-table"><thead><tr><th>ctx</th><th>KV size</th><th>fits SRAM</th><th>KV energy/token</th><th>vs whole chip</th></tr></thead><tbody>${rows}</tbody></table>`;
      } else {
        $("scaleKvStory").innerHTML = "";
      }
      renderCrossover(d.economics, d.context, d.kv_bits);
      loadConfigCorner();
      const maxCyc = Math.max(...d.chip.stages.map((r) => r.cycles), 1);
      const rows = d.chip.stages.slice().sort((a, b) => b.cycles - a.cycles)
        .map((r) => {
          const pct = (100 * r.cycles / maxCyc).toFixed(1);
          const share = (100 * r.cycles /
            d.chip.stages.reduce((s, x) => s + x.cycles, 0)).toFixed(1);
          return `<div class="ss-row"><span class="ss-name">${escapeHtml(r.stage)}</span>` +
            `<span class="ss-bar"><span style="width:${pct}%"></span></span>` +
            `<span class="ss-val">${Math.round(r.cycles).toLocaleString("en-US")} · ${share}%</span></div>`;
        }).join("");
      $("scaleStages").innerHTML = `<h4>per-stage cycles (calibrated)</h4>${rows}`;
      const tbody = $("scaleTable").querySelector("tbody");
      const chipLat = chip.latency_s, chipE = chip.total_energy_j;
      const trs = [`<tr><td>chip (this model)</td><td>${fmtMs(chipLat)}</td><td>${fmtJ(chipE)}</td><td>—</td></tr>`];
      d.rivals.forEach((r) => {
        const m = r.metrics || {};
        if (r.feasible === false) {
          trs.push(`<tr><td>${escapeHtml(r.platform)}</td><td colspan="3" class="muted">infeasible — ${(r.note || "").replace(/</g, "&lt;")}</td></tr>`);
          return;
        }
        const spd = m.latency_s && chipLat ? (m.latency_s / chipLat).toFixed(2) + "x slower" : "—";
        const eff = m.total_energy_j && chipE ? (m.total_energy_j / chipE).toFixed(1) + "x energy" : "—";
        trs.push(`<tr><td>${escapeHtml(r.platform)}</td><td>${fmtMs(m.latency_s)}</td><td>${fmtJ(m.total_energy_j)}</td><td>${spd} · ${eff}</td></tr>`);
      });
      tbody.innerHTML = trs.join("");
      if (d.validation) {
        $("scaleValidation").textContent =
          `Calibration: cycle model reproduces the measured 0.6B fused run within ` +
          `${Math.abs(d.validation.fused_error_pct)}% and unfused within ` +
          `${Math.abs(d.validation.unfused_error_pct)}%. ` +
          `Measured: ${d.validation.fused_measured.toLocaleString("en-US")} / ` +
          `${d.validation.unfused_measured.toLocaleString("en-US")} cycles.`;
      } else {
        $("scaleValidation").textContent =
          "Per-stage constants calibrated on the measured 0.6B chip (both schedules within 0.25%); 1T geometry and N4 constants are analytical estimates.";
      }
    } catch (_) { /* scaling panel is optional */ }
  }

  // ---- Slide 2 · weight-storage architecture comparison ---------------------
  async function loadArchCompare() {
    const table = $("archTable");
    if (!table) return;
    try {
      const res = await apiFetch("/api/arch");
      if (!res.ok) return;
      const d = await res.json();
      const order = ["hbm", "chiplet", "cim", "wafer"];
      const bits = [4, 8, 16];
      const names = { hbm: "HBM streaming", chiplet: "3D ROM chiplet",
                      cim: "analog CIM", wafer: "wafer-scale ROM" };
      const byKey = {};
      d.forEach((v) => { byKey[`${v.arch}_${v.weight_bits}`] = v; });
      const rows = order.map((a) => {
        const cells = bits.map((b) => {
          const v = byKey[`${a}_${b}`];
          if (!v) return "<td>—</td>";
          const c = v.chip.metrics;
          return `<td>${fmtMs(c.latency_s)} · ${fmtJ(c.total_energy_j)} · ${v.dies} dies</td>`;
        }).join("");
        return `<tr><td>${names[a]}</td>${cells}</tr>`;
      }).join("");
      table.innerHTML =
        `<table class="plat-table"><thead><tr><th>architecture</th>` +
        `<th>MXFP4-class (4b)</th><th>INT8 (8b)</th><th>BF16 (16b)</th></tr></thead>` +
        `<tbody>${rows}</tbody></table>` +
        `<p class="muted small">Each cell: latency · energy/token · die count for the 1T MoE at ctx 1, 4096 lanes, calibrated on the measured 0.6B constants. CIM wins on energy (no weight fetch); chiplet matches HBM latency at a fraction of the energy; wafer trades energy for capacity in one substrate.</p>`;
    } catch (_) { /* optional */ }
  }

  function bindArchDiagrams() {
    const frame = $("archDiagram");
    if (!frame) return;
    document.querySelectorAll(".arch-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".arch-tab").forEach((t) =>
          t.classList.toggle("active", t === tab));
        showDiagram(tab.getAttribute("data-diagram"));
      });
    });
    showDiagram("comparison");
  }

  function showDiagram(name) {
    const frame = $("archDiagram");
    if (!frame) return;
    frame.innerHTML = `<img src="/api/arch/diagram/${encodeURIComponent(name)}" alt="${escapeHtml(name)} diagram" />`;
  }

  function bindScaleControls() {
    ["scalePreset", "scaleLanes", "scaleCtx", "scaleBits", "scaleTier",
     "scaleSchedule", "scaleDies", "scaleKvBits"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", loadScale1t);
    });
    const btn = $("scaleRun");
    if (btn) btn.addEventListener("click", loadScale1t);
  }

  async function loadPlatformCompare() {
    const grid = $("platformGrid");
    const table = $("platformTable");
    if (!table) return;  // analysis table lives on the appendix; absent is fine
    const tbody = table.querySelector("tbody");
    if (!tbody) return;
    if (!grid) {
      // platformGrid moved off the deck; still fill the analysis table.
      try {
        const res = await apiFetch("/api/compare");
        if (!res.ok) return;
        const d = await res.json();
        const rows = [
          ["total FLOPs", (p) => fmtSci(p.metrics.total_flops, "")],
          ["latency", (p) => fmtSci(p.metrics.latency_s, "s")],
          ["compute energy", (p) => fmtSci(p.metrics.compute_energy_j, "J")],
          ["weight-move energy", (p) => fmtSci(p.metrics.weight_energy_j, "J")],
          ["total energy", (p) => fmtSci(p.metrics.total_energy_j, "J")],
          ["transistors", (p) => fmtSci(p.metrics.transistors, "")],
          ["area", (p) => p.metrics.area_mm2 ? p.metrics.area_mm2 + " mm²" : "needs PDK"],
        ];
        tbody.innerHTML = rows.map(([label, get]) =>
          `<tr><td>${label}</td>` + d.platforms.map((p) => `<td>${get(p)}</td>`).join("") + `</tr>`
        ).join("");
        if (d.measured_baseline) {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>measured full-statement compare (unfused)</td>` +
            `<td>${fmt(d.measured_baseline.cycles)} cycles · ${fmt(d.measured_baseline.macs)} MACs</td>` +
            `<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>`;
          tbody.appendChild(tr);
        }
        if (d.measured_fused) {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>measured full-statement compare (fused)</td>` +
            `<td>${fmt(d.measured_fused.cycles)} cycles · ${fmt(d.measured_fused.macs)} MACs</td>` +
            `<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>`;
          tbody.appendChild(tr);
        }
      } catch (_) { /* optional */ }
      return;
    }
    try {
      const res = await apiFetch("/api/compare");
      if (!res.ok) return;
      const d = await res.json();
      const cards = d.platforms.map((p) => {
        const m = p.metrics;
        return `<div class="plat-card"><h3>${escapeHtml(p.platform)}</h3>` +
          `<div class="pc-row"><span>latency</span><span>${fmtSci(m.latency_s, "s")}</span></div>` +
          `<div class="pc-row"><span>energy</span><span>${fmtSci(m.total_energy_j, "J")}</span></div>` +
          `<div class="pc-row"><span>transistors</span><span>${fmtSci(m.transistors, "")}</span></div>` +
          `<div class="pc-row"><span>weights moved</span><span>${fmtSci(m.weight_bytes, "B")}</span></div>` +
          `</div>`;
      });
      grid.innerHTML = cards.join("");
      const rows = [
        ["total FLOPs", (p) => fmtSci(p.metrics.total_flops, "")],
        ["latency", (p) => fmtSci(p.metrics.latency_s, "s")],
        ["compute energy", (p) => fmtSci(p.metrics.compute_energy_j, "J")],
        ["weight-move energy", (p) => fmtSci(p.metrics.weight_energy_j, "J")],
        ["total energy", (p) => fmtSci(p.metrics.total_energy_j, "J")],
        ["transistors", (p) => fmtSci(p.metrics.transistors, "")],
        ["area", (p) => p.metrics.area_mm2 ? p.metrics.area_mm2 + " mm²" : "needs PDK"],
      ];
      tbody.innerHTML = rows.map(([label, get]) =>
        `<tr><td>${label}</td>` + d.platforms.map((p) => `<td>${get(p)}</td>`).join("") + `</tr>`
      ).join("");
      if (d.measured_baseline) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>measured full-statement compare (unfused)</td>` +
          `<td>${fmt(d.measured_baseline.cycles)} cycles · ${fmt(d.measured_baseline.macs)} MACs</td>` +
          `<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>`;
        tbody.appendChild(tr);
      }
      if (d.measured_fused) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>measured full-statement compare (fused)</td>` +
          `<td>${fmt(d.measured_fused.cycles)} cycles · ${fmt(d.measured_fused.macs)} MACs</td>` +
          `<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>`;
        tbody.appendChild(tr);
      }
    } catch (_) { /* compare panel is optional */ }
  }

  const NN_KPI_REF = {
    // Fused 0.6B chip: MEASURED cycles (57,025,007) and measured ROM reads
    // (38,086,208 words x 256 b) priced at N4-class constants (0.45 pJ/MAC,
    // 0.08 pJ/bit) -> 0.41 + 0.78 = 1.19 mJ. Energy is analytical; the
    // cycle/MAC counts are measured (bit-exact full-statement compares).
    fusedE: 0.001191,
    fusedLat: 0.114,
    unfusedLat: 0.591,
    unfusedE: 0.001191 * (295.4 / 57.0)
  };
  const NN_KPI_FALLBACK = {
    gpu: { eJ: 0.197505027, latS: 0.0003566766997014925 },
    tpu: { eJ: 0.12387358, latS: 0.00074679184 },
    // LPU: 750 W TDP x 0.6 utilization over the FULL latency (compute +
    // weight-SRAM stream), per lab/compute_metrics.py lpu_estimate.
    lpu: { eJ: 0.187423, latS: 0.000406 },
    // Analog CIM, 0.6B BF16 (4 passes): lab/scale_1t cim_estimate at
    // N4-class constants. ADC/DAC per-op pricing; arrays hold W.
    cim: { eJ: 0.00036, latS: 0.00025 },
    // HBM-PIM: in-bank GEMV at ~2 pJ/bit over the 1.19 GB weight stream +
    // bank-ALU energy; still a DRAM walk every token.
    pim: { eJ: 0.0195, latS: 0.0003125 }
  };
  const NN_SIDE_META = {
    gpu: { name: "NVIDIA H100 SXM", short: "H100", color: "#1d4ed8", pj: 5, w: "HBM" },
    tpu: { name: "Google TPU v5e", short: "TPU v5e", color: "#b45309", pj: 5, w: "HBM" },
    lpu: { name: "Groq LPU", short: "LPU", color: "#6d28d9", pj: 0.5, w: "SRAM" },
    cim: { name: "Analog CIM", short: "CIM", color: "#be123c", pj: 0, w: "array" },
    pim: { name: "HBM-PIM", short: "PIM", color: "#0e7490", pj: 2, w: "DRAM" },
    unfused: { name: "This chip · unfused", short: "unfused", color: "#c2410c", pj: 0.8, w: "ROM" },
    fused: { name: "This chip · fused", short: "fused", color: "#0f766e", pj: 0.8, w: "ROM" }
  };
  const NN_SIDE_ORDER = ["gpu", "tpu", "lpu", "cim", "pim", "unfused", "fused"];
  /* Die transistors. GPU/LPU published-class; TPU est from 300–350 mm²;
     ASIC = 1T/bit mask ROM for 1.19 GB BF16 (9.5B) + 8B SRAM workspace
     (compute is ~0.16M). 6T was the SRAM-class overcount. */
  const NN_TX = {
    gpu: { label: "80B", note: "814 mm² · HBM stacks not in the 80B" },
    tpu: { label: "~32B est", note: "300–350 mm² · Google unpublished · +16 GB HBM" },
    lpu: { label: "26.8B", note: "~725 mm² · 14nm · no HBM" },
    cim: { label: "~26B", note: "1-transistor analog cells · ADC/DAC extra · 600 mm² class · multi-level cell = the research risk" },
    pim: { label: "~20B+", note: "logic ~20B · DRAM 1T1C counted in Tx × energy" },
    unfused: { label: "~18B", note: "9.5B ROM (1T/bit mask ROM) · 8B SRAM · same die as fused" },
    fused: { label: "~18B", note: "9.5B ROM (1T/bit mask ROM) · 8B SRAM · fusion adds 0 transistors" }
  };

  function nnLat(s) {
    const v = Number(s);
    if (!Number.isFinite(v)) return "—";
    if (v >= 1) return v.toFixed(2) + " s";
    if (v >= 0.01) return (v * 1e3).toFixed(0) + " ms";
    return (v * 1e3).toFixed(2) + " ms";
  }

  function nnToks(s) {
    const t = 1 / Number(s);
    if (!Number.isFinite(t)) return "—";
    if (t >= 100) return "~" + Math.round(t).toLocaleString("en-US") + " tok/s";
    if (t >= 10) return t.toFixed(1) + " tok/s";
    return t.toFixed(2) + " tok/s";
  }

  function nnVs(j, fusedE) {
    const x = j / fusedE;
    if (!Number.isFinite(x)) return "—";
    // Anything at or under this chip used to collapse to "least", which
    // labelled analog CIM (0.36 mJ) as tying us when it actually wins. Say so.
    if (x >= 0.95 && x <= 1.05) return "this chip";
    if (x < 0.95) {
      const r = 1 / x;
      return r.toFixed(r >= 10 ? 0 : 1) + "× less";
    }
    const n = x.toFixed(x >= 10 ? 0 : 1);
    return n + "× more";
  }

  function nnNiceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / p;
    const m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
    return m * p;
  }

  function nnLineChartSvg(labels, energy, toks, colors, eTexts, tTexts, txTexts) {
    const W = 920, H = 420, L = 62, R = 148, T = 36, B = 52;
    const iw = W - L - R, ih = H - T - B;
    const tMin = 1, tMax = 10000, eMax = 200;
    const xAt = (t) => L + iw * ((Math.log10(Math.max(Number(t), tMin)) - Math.log10(tMin)) /
      (Math.log10(tMax) - Math.log10(tMin)));
    const yAt = (e) => T + ih * (1 - Number(e) / eMax);
    let g = "";
    [0, 50, 100, 150, 200].forEach((v) => {
      const y = yAt(v);
      g += `<line class="nn-grid" x1="${L}" y1="${y.toFixed(1)}" x2="${(L + iw).toFixed(1)}" y2="${y.toFixed(1)}"/>`;
      g += `<text class="nn-yl" x="${L - 8}" y="${(y + 3.5).toFixed(1)}">${v}</text>`;
    });
    [1, 10, 100, 1000, 10000].forEach((v) => {
      const x = xAt(v);
      const lab = v >= 1000 ? (v / 1000) + "k" : String(v);
      g += `<line class="nn-grid" x1="${x.toFixed(1)}" y1="${T}" x2="${x.toFixed(1)}" y2="${(T + ih).toFixed(1)}"/>`;
      g += `<text class="nn-xl" x="${x.toFixed(1)}" y="${H - 22}">${lab}</text>`;
    });
    const split = xAt(100);
    g += `<line class="nn-split-line" x1="${split.toFixed(1)}" y1="${T}" x2="${split.toFixed(1)}" y2="${(T + ih).toFixed(1)}"/>`;
    g += `<text class="nn-region" text-anchor="middle" x="${((L + split) / 2).toFixed(1)}" y="${T + 14}">this 16-lane die</text>`;
    g += `<text class="nn-region" text-anchor="middle" x="${((split + L + iw) / 2).toFixed(1)}" y="${T + 14}">production parts</text>`;
    g += `<line class="nn-axis" x1="${L}" y1="${(T + ih).toFixed(1)}" x2="${(L + iw).toFixed(1)}" y2="${(T + ih).toFixed(1)}"/>`;
    g += `<line class="nn-axis" x1="${L}" y1="${T}" x2="${L}" y2="${(T + ih).toFixed(1)}"/>`;
    g += `<text class="nn-panel-k" x="${L}" y="${T - 12}">energy / token · mJ</text>`;
    g += `<text class="nn-panel-k" text-anchor="middle" x="${L + iw / 2}" y="${H - 6}">throughput · tok/s · log  →  faster</text>`;
    const iU = labels.indexOf("unfused");
    const iF = labels.indexOf("fused");
    if (iU >= 0 && iF >= 0) {
      g += `<line class="nn-fuse-link" x1="${xAt(toks[iU]).toFixed(1)}" y1="${yAt(energy[iU]).toFixed(1)}"` +
        ` x2="${xAt(toks[iF]).toFixed(1)}" y2="${yAt(energy[iF]).toFixed(1)}"/>`;
    }
    const off = [
      { dx: 10, dy: -36, a: "start" },
      { dx: -12, dy: -36, a: "end" },
      { dx: 12, dy: 16, a: "start" },
      { dx: -14, dy: 18, a: "end" },
      { dx: 12, dy: -28, a: "start" },
      { dx: 10, dy: -36, a: "start" },
      { dx: 10, dy: 16, a: "start" }
    ];
    toks.forEach((t, i) => {
      const x = xAt(t), y = yAt(energy[i]);
      const o = off[i] || { dx: 10, dy: -12, a: "start" };
      const tx = (txTexts && txTexts[i]) ? txTexts[i] : "";
      g += `<circle class="nn-dot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="8" fill="${colors[i]}"/>`;
      g += `<text class="nn-xl" text-anchor="${o.a}" x="${(x + o.dx).toFixed(1)}" y="${(y + o.dy).toFixed(1)}">${escapeHtml(labels[i])}</text>`;
      g += `<text class="nn-pt" text-anchor="${o.a}" x="${(x + o.dx).toFixed(1)}" y="${(y + o.dy + 13).toFixed(1)}">${escapeHtml(eTexts[i])} · ${escapeHtml(tTexts[i])}/s</text>`;
      if (tx) {
        g += `<text class="nn-tx" text-anchor="${o.a}" x="${(x + o.dx).toFixed(1)}" y="${(y + o.dy + 26).toFixed(1)}">${escapeHtml(tx)} Tx</text>`;
      }
    });
    return `<svg class="nn-chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Energy, throughput, and die transistors for one token">${g}</svg>`;
  }

  function paintNnKpis(d) {
    const fusedE = NN_KPI_REF.fusedE;
    const nums = {
      gpu: { ...NN_KPI_FALLBACK.gpu },
      tpu: { ...NN_KPI_FALLBACK.tpu },
      lpu: { ...NN_KPI_FALLBACK.lpu },
      cim: { ...NN_KPI_FALLBACK.cim },
      pim: { ...NN_KPI_FALLBACK.pim },
      unfused: { eJ: NN_KPI_REF.unfusedE, latS: NN_KPI_REF.unfusedLat },
      fused: { eJ: fusedE, latS: NN_KPI_REF.fusedLat }
    };
    if (d && Array.isArray(d.rivals)) {
      for (const r of d.rivals) {
        const n = String(r.platform || r.name || "").toLowerCase();
        const m = r.metrics || {};
        let key = null;
        if (n.includes("h100")) key = "gpu";
        else if (n.includes("tpu")) key = "tpu";
        else if (n.includes("groq") || /\blpu\b/.test(n)) key = "lpu";
        if (!key || r.feasible === false) continue;
        if (Number.isFinite(m.total_energy_j)) nums[key].eJ = m.total_energy_j;
        if (Number.isFinite(m.latency_s)) nums[key].latS = m.latency_s;
      }
    }
    const energy = [];
    const toks = [];
    const pkgFacts = {};
    const colors = [];
    const eTexts = [];
    const tTexts = [];
    const txTexts = [];
    const labels = [];
    for (const side of NN_SIDE_ORDER) {
      const v = nums[side];
      const meta = NN_SIDE_META[side];
      const vs = nnVs(v.eJ, fusedE);
      const eTxt = (side === "unfused" ? "~" : "") + fmtJ(v.eJ) + (side === "unfused" ? " est" : "");
      const latTxt = `${nnLat(v.latS)} · ${nnToks(v.latS)}`;
      const eEl = document.querySelector(`[data-kpis="${side}"] [data-kpi="e"]`);
      const lEl = document.querySelector(`[data-kpis="${side}"] [data-kpi="lat"]`);
      const tEl = document.querySelector(`[data-kpis="${side}"] [data-kpi="tr"]`);
      const nEl = document.querySelector(`[data-kpis="${side}"] [data-kpi="tr-note"]`);
      if (eEl) eEl.textContent = `${eTxt} · ${vs}`;
      if (lEl) lEl.textContent = latTxt;
      if (tEl && NN_TX[side]) tEl.textContent = NN_TX[side].label;
      if (nEl && NN_TX[side]) nEl.textContent = NN_TX[side].note;
      const eCell = document.querySelector(`[data-kpi-cell="${side}-e"]`);
      const lCell = document.querySelector(`[data-kpi-cell="${side}-lat"]`);
      if (eCell) {
        eCell.textContent = eTxt + " · " + vs;
        eCell.classList.toggle("win", side === "fused");
      }
      if (lCell) lCell.textContent = latTxt;
      // feed the physical-package gallery from this same computation, so the
      // 3D cards can never disagree with the chart or the KPI table
      pkgFacts[side] = {
        store: `${meta.w}${meta.pj ? ` \u00b7 ~${meta.pj} pJ/bit` : ""}`,
        energy: `${eTxt} \u00b7 ${vs}`,
        tx: (NN_TX[side] && NN_TX[side].label) || "",
        areaNote: (NN_TX[side] && NN_TX[side].note) || ""
      };
      energy.push(v.eJ * 1e3);
      toks.push(1 / v.latS);
      colors.push(meta.color);
      eTexts.push(eTxt);
      tTexts.push(nnToks(v.latS).replace(/^~/, "").replace(" tok/s", ""));
      txTexts.push((NN_TX[side] && NN_TX[side].label) || "");
      labels.push(meta.short);
    }
    if (window.HwPackages) window.HwPackages.setFacts(pkgFacts);
    const eChart = $("nnChartE");
    if (eChart) eChart.innerHTML = nnLineChartSvg(labels, energy, toks, colors, eTexts, tTexts, txTexts);
  }

  async function loadNnKpis() {
    const cached = state.lastScenario && state.lastScenario.preset === "qwen3_06b"
      ? state.lastScenario : null;
    paintNnKpis(cached);
    if (cached) return;
    try {
      const res = await apiFetch("/api/scale1t?preset=qwen3_06b&lanes=16&ctx=1&schedule=fused&weight_bits=16&tier=sram");
      if (!res.ok) return;
      paintNnKpis(await res.json());
    } catch (_) { /* HTML fallbacks already on the slide */ }
  }

  // ---- Slides 5-9 · evidence-JSON loaders (workload, KPIs, scaling) ----------

  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const fmtMJ = (j) => {
    const mj = j * 1e3;
    if (mj >= 100) return mj.toFixed(0) + " mJ";
    if (mj >= 10) return mj.toFixed(1) + " mJ";
    if (mj >= 1) return mj.toFixed(2) + " mJ";
    return (mj * 1e3).toFixed(0) + " µJ";
  };
  const fmtLat = (s) => {
    const ms = s * 1e3;
    if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
    if (ms >= 1) return ms.toFixed(1) + " ms";
    if (ms >= 0.01) return ms.toFixed(3) + " ms";
    return (ms * 1e3).toFixed(0) + " µs";
  };
  const fmtToks = (t) => {
    if (t >= 1000) return Math.round(t).toLocaleString("en-US");
    if (t >= 10) return t.toFixed(0);
    return t.toFixed(1);
  };

  const WL_STAGE_LABEL = {
    embedding: "Embedding lookup",
    input_rmsnorm: "Input RMSNorm",
    qkv_proj: "Q/K/V projection (Q = yW)",
    qk_norm: "Q/K RMSNorm",
    rope: "RoPE",
    attention_scores: "Attention scores",
    attention_softmax: "Softmax",
    causal_gqa: "Causal GQA",
    o_proj: "O projection",
    attention_residual: "Attention residual",
    post_rmsnorm: "Post RMSNorm",
    gate_up_proj: "Gate/up projection",
    silu_swiglu: "SiLU · SwiGLU",
    down_proj: "Down projection",
    output_residual: "Output residual",
    final_rmsnorm: "Final RMSNorm",
    lm_head: "LM head (all logits)",
  };

  let deckCache = {};
  async function deckEvidence(name) {
    if (deckCache[name]) return deckCache[name];
    try {
      const res = await apiFetch(`/api/deck/${name}`);
      if (!res.ok) return null;
      deckCache[name] = await res.json();
      return deckCache[name];
    } catch (_) {
      return null;
    }
  }

  async function loadWorkloadTable() {
    const d = await deckEvidence("exec_models");
    const rows = d && d.workload && d.workload.stages_detail;
    const tbody = $("workloadRows");
    if (!tbody || !rows) return;
    tbody.innerHTML = rows.map((r) => {
      const label = WL_STAGE_LABEL[r.stage] || r.stage;
      const macs = r.macs_per_call > 0 ? fmtInt(r.macs_per_call) : "—";
      const wbytes = r.weight_bytes_per_call > 0 ? fmtInt(r.weight_bytes_per_call) : "—";
      return `<tr><td>${label}</td><td>${r.count}</td><td>${macs}</td><td>${wbytes}</td><td>${r.total_macs > 0 ? fmtInt(r.total_macs) : "—"}</td></tr>`;
    }).join("");
    const calls = rows.reduce((a, r) => a + r.count, 0);
    const macs = rows.reduce((a, r) => a + r.total_macs, 0);
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    set("wlTotalCalls", fmtInt(calls));
    set("wlTotalMacs", fmtInt(macs));
    set("wlStatMacs", fmtInt(d.workload.total_macs));
    set("wlStatWbytes", (d.workload.weight_bytes / 1e9).toFixed(2) + " GB");
  }

  const KPI_MACHINE = [
    { key: "h100", where: "HBM (off-chip), streamed every token" },
    { key: "b200", where: "HBM (off-chip), streamed every token" },
    { key: "tpu_v5e", where: "HBM → MXU weight FIFO" },
    { key: "lpu", where: "On-chip SRAM (230 MB)" },
    { key: "pim", where: "DRAM banks, MACs beside sense amps" },
    { key: "cim", where: "Analog cells (W is the conductance)" },
    { key: "rom", where: "Mask ROM, burned in at fabrication" },
  ];
  // Three evidence classes, because "simulated" was hiding a real difference.
  //   measured      - RTL counters from a Verilator run, bit-exact vs the
  //                   Python oracle. Only this chip.
  //   rtl-dataflow  - the ENERGY/LATENCY number here is analytical (datasheet
  //                   TDP + HBM pJ/bit), but this machine's DATAFLOW is
  //                   implemented in our RTL and measured cycle-by-cycle in
  //                   the player below: our MAC array and nonlinear engines,
  //                   their weight path. Not a vendor netlist - deliberately.
  //   analytical    - no RTL anywhere: datasheet or literature plus a formula.
  const KPI_PROV = {
    measured: { cls: "kp-measured", label: "measured" },
    analytical_estimate: { cls: "kp-sim", label: "analytical" },
    published: { cls: "kp-cited", label: "cited" },
  };

  // Platforms whose dataflow exists as real RTL (qwen_chip/rtl), measured in
  // machine_timelines. The schedule name is the FUSED build that implements it.
  const KPI_RTL_DATAFLOW = {
    h100: "gpu",       // qwen3_stream_gemm.sv  - streaming GEMM, weights from memory
    tpu_v5e: "tpu",    // qwen3_tpu_engine.sv   - MXU weight-stationary tile
    lpu: "lpu",        // qwen3_lpu_engine.sv   - SRAM-resident weights
    rom: "mask_rom_fused",
  };

  const fmtMacRate = (v) => {
    if (!(v > 0)) return "—";
    if (v >= 1e15) return `${(v / 1e15).toFixed(1)}P`;
    if (v >= 1e12) return `${(v / 1e12).toFixed(0)}T`;
    if (v >= 1e9) return `${(v / 1e9).toFixed(1)}G`;
    return `${(v / 1e6).toFixed(0)}M`;
  };

  async function loadKpiTable() {
    const d = await deckEvidence("exec_models");
    const tbody = $("kpiRows");
    if (!d || !d.platforms) return;
    const romE = d.platforms.rom.metrics.total_energy_j;
    const romW = d.platforms.rom.metrics.weight_energy_j;

    // Sorted by throughput (fastest first), then by energy per token
    // (cheapest first) as the tie-break.
    const ordered = KPI_MACHINE
      .map((m) => ({ m, p: d.platforms[m.key] }))
      .filter((r) => r.p)
      .sort((a, b) =>
        (b.p.metrics.throughput_toks - a.p.metrics.throughput_toks) ||
        (a.p.metrics.total_energy_j - b.p.metrics.total_energy_j));

    const ratio = (v, isSelf, unit) => {
      if (isSelf) return "1× (this chip)";
      if (!(v > 0)) return "—";
      return v >= 1
        ? `${v.toFixed(v < 10 ? 1 : 0)}× more${unit}`
        : `${(1 / v).toFixed(1)}× less${unit}`;
    };

    if (tbody) {
      tbody.innerHTML = ordered.map(({ m, p }) => {
        const mt = p.metrics;
        const prov = KPI_PROV[p.provenance.kind] || KPI_PROV.analytical_estimate;
        const self = m.key === "rom";
        const sched = KPI_RTL_DATAFLOW[m.key];
        // An analytical number whose dataflow is nonetheless real RTL gets its
        // own badge - it is stronger evidence than a pure formula.
        const rtlTag = (!self && sched)
          ? `<span class="kp-tag kp-rtl" title="Dataflow implemented in qwen_chip/rtl and measured in the player below (schedule: ${sched}). Our MAC array, their weight path - not a vendor netlist.">RTL dataflow</span>`
          : "";
        // The weight walk is bottom-up on every row, so it is the only
        // like-for-like column: the totals mix a TDP heuristic (GPU/TPU/LPU)
        // with counter-based physics (PIM/CIM/this chip).
        return `<tr${self ? ' class="kp-self"' : ""}>
          <td class="kp-name">${p.platform}</td>
          <td>${m.where}</td>
          <td data-provenance="${p.provenance.kind}">${fmtMJ(mt.total_energy_j)}</td>
          <td class="kp-walk">${fmtMJ(mt.weight_energy_j)}</td>
          <td class="kp-used">${
            mt.peak_utilization == null
              ? '<span class="kp-used-n">—</span><span class="kp-used-d">no published peak</span>'
              : `<span class="kp-used-n">${(mt.peak_utilization * 100).toFixed(mt.peak_utilization < 0.1 ? 2 : 1)}%</span>` +
                `<span class="kp-used-d">of ${fmtMacRate(mt.peak_macs_per_s)} MAC/s</span>`
          }</td>
          <td data-provenance="${p.provenance.kind}">${fmtToks(mt.throughput_toks)}</td>
          <td>${ratio(romW > 0 ? mt.weight_energy_j / romW : 0, self, "")}</td>
          <td><span class="kp-tag ${prov.cls}">${prov.label}</span>${rtlTag}</td>
        </tr>`;
      }).join("");
    }

    // Feed the physical-package gallery from this same payload.
    if (window.HwPackages) {
      const facts = {};
      ordered.forEach(({ m, p }) => {
        const sched = KPI_RTL_DATAFLOW[m.key];
        const evidence = p.provenance.kind === "measured"
          ? "measured RTL"
          : (sched ? "RTL dataflow" : "analytical");
        facts[m.key] = {
          energy: fmtMJ(p.metrics.total_energy_j),
          toks: fmtToks(p.metrics.throughput_toks) + " tok/s",
          vs: ratio(romW > 0 ? p.metrics.weight_energy_j / romW : 0, m.key === "rom", " weight walk"),
          evidence
        };
      });
      window.HwPackages.setFacts(facts);
    }

    const noteEl = $("kpiEnergyNote");
    const romProv = (d.platforms.rom || {}).provenance || {};
    if (noteEl) {
      noteEl.innerHTML =
        `Sorted by throughput, then energy per token. Cycles and counters are measured; ` +
        `energy is those counters priced at <strong>${romProv.energy_constants || "assumed per-node constants"}</strong> ` +
        `&mdash; the lab die itself is 28&nbsp;nm, so this column is a node projection, not a measured wall-plug figure. ` +
        `<strong>Weight walk</strong> is bottom-up on every row and is the only like-for-like column: the ` +
        `<em>energy/token</em> totals for the GPU, TPU and LPU rows are dominated by a TDP&times;0.6 power-budget ` +
        `heuristic (61&ndash;97% of their number), while the PIM, CIM and mask-ROM rows are counted from physics. ` +
        `Rival rows are pinned at their published node; only the mask-ROM row follows the node above. ` +
        `<strong>Peak used</strong> is achieved MAC rate over the machine's own datasheet ceiling &mdash; ` +
        `it shows how much of its silicon actually does work on this batch-1 workload. Only the mask-ROM ` +
        `row's figure is measured; for the modelled machines it is derived from the same datasheet numbers ` +
        `that set their latency, so it reports where the model puts the bottleneck. HBM-PIM lands at ~100% ` +
        `because its published compute and in-stack bandwidth are matched, making utilisation 1.0 by ` +
        `construction rather than by measurement. Analog CIM publishes no FLOPS figure, so it has no ` +
        `comparable ceiling. <strong>Evidence classes:</strong> <em>measured</em> = RTL counters, bit-exact against the Python ` +
        `oracle. <em>RTL dataflow</em> = the number here is analytical, but that machine's weight path is ` +
        `implemented in our RTL (qwen3_stream_gemm / qwen3_tpu_engine / qwen3_lpu_engine) and measured ` +
        `cycle-by-cycle in the player below &mdash; our MAC array and nonlinear engines, their dataflow. ` +
        `We deliberately hold no vendor netlist. <em>analytical</em> = datasheet or literature plus a formula, ` +
        `no RTL: B200, HBM-PIM and analog CIM (which can never have RTL &mdash; it computes in physics).`;
    }
  }

  // ---- Machine trace replay player (slide 7) ----
  // Replays the measured RTL commit logs behind each row of the table above
  // (evidence/runs/hw_1tok_*, parsed by lab/machine_timelines.py). The same
  // 16-lane MAC array runs every schedule; only the memory path changes.
  // Energy is priced client-side from measured counters with the same
  // measured-chip constants the /api/stage_breakdown endpoint uses.
  const MACH_TABS = [
    ["mask_rom", "Mask-ROM · unfused"],
    ["mask_rom_fused", "Mask-ROM · fused"],
    ["gpu", "GPU tile"],
    ["gpu_hbm", "GPU + HBM latency"],
    ["tpu", "TPU-style"],
    ["tpu_hbm", "TPU + HBM latency"],
    ["lpu", "LPU-style"],
    ["lpu_hbm", "LPU + HBM latency"]
  ];
  const MACH_PLAY_MS = 24000;
  const machGroup = (stage) => {
    if (stage === "EMBED") return "embed";
    if (stage === "INPUT_NORM" || stage === "POST_NORM" || stage === "Q_NORM" ||
      stage === "K_NORM" || stage === "FINAL_NORM") return "norm";
    if (stage === "Q_PROJ" || stage === "K_PROJ" || stage === "V_PROJ" ||
      stage === "O_PROJ" || stage === "GATE_PROJ" || stage === "UP_PROJ" ||
      stage === "DOWN_PROJ") return "proj";
    if (stage === "ROPE") return "rope";
    if (stage === "ATTN_SCORE" || stage === "ATTN_VALUE" || stage === "SOFTMAX") return "attn";
    if (stage === "SILU") return "act";
    if (stage === "ATTN_RESIDUAL" || stage === "MLP_RESIDUAL") return "resid";
    if (stage === "LM_HEAD" || stage === "ARGMAX") return "head";
    return "other";
  };
  const machJ = (v) => v >= 1e-3 ? `${(v * 1e3).toFixed(2)} mJ` : `${(v * 1e6).toFixed(1)} µJ`;

  // Per-machine narrative, computed from the measured trace counters — never
  // hand-typed. Every line states cycles, MAC deltas, or latency ratios that
  // reconcile with the segment sums and the table above.
  function machStory(key) {
    const m = machData.machines[key];
    const fused = machData.machines.mask_rom_fused;
    const k = machData.energy_constants;
    const clock = machData.clock_hz || 500e6;
    const memJ = m.meta.rom_read_count * k.rom_read_bits * k.rom_pj_per_bit / 1e12;
    const macJ = m.meta.mac_count * k.mac_pj / 1e12;
    const memPct = memJ + macJ > 0 ? (memJ / (memJ + macJ)) * 100 : 0;
    const ms = m.meta.cycle_count / clock * 1e3;
    const fusedMs = fused.meta.cycle_count / clock * 1e3;
    const ratio = fusedMs > 0 ? ms / fusedMs : 0;
    const extraMacs = m.meta.mac_count - fused.meta.mac_count;
    const cycM = (m.meta.cycle_count / 1e6).toFixed(1);
    let line;
    if (key === "mask_rom") {
      line = `The blocking schedule waits on every ROM read: ${cycM}M cycles — ${ratio.toFixed(1)}× the fused pass —` +
        (extraMacs > 0
          ? ` and executes ${(extraMacs / 1e6).toFixed(1)}M extra MACs (+${(extraMacs / fused.meta.mac_count * 100).toFixed(1)}%) for the same bit-exact result.`
          : " for the same bit-exact result.");
    } else if (key === "mask_rom_fused") {
      line = `Streaming overlaps the 2-cycle ROM read: ${cycM}M cycles for one token — the measured anchor every other schedule is compared against.`;
    } else if (m.ext_mem_lat > 0) {
      line = `Same counters with HBM-class latency (+${m.ext_mem_lat} cycles per request): identical energy, ${ratio.toFixed(1)}× the fused time — the round trip costs time, not joules, on this die.`;
    } else if (key === "gpu") {
      line = `GPU-tile scheduling on the same array: ${cycM}M cycles, same MACs, same energy — the schedule changes, the math does not.`;
    } else if (key === "tpu") {
      line = `Weight-stationary tiles: reuse pays at batch > 1 — at batch 1 it is a schedule change, not a traffic cut. ${cycM}M cycles.`;
    } else if (key === "lpu") {
      line = `Whole-matrix SRAM staging: ROM traffic drops to zero after the bulk load, which costs ${cycM}M cycles up front at batch 1.`;
    } else {
      line = `${cycM}M cycles · ${(m.meta.mac_count / 1e6).toFixed(1)}M MACs · ${m.meta.rom_read_count.toLocaleString("en-US")} ROM reads.`;
    }
    return { line, memPct };
  }

  let machData = null;
  let machState = null;

  async function loadMachPlayer() {
    if (machData) return;
    const d = await deckEvidence("machine_timelines");
    if (!d || !d.machines) return;
    machData = d;
    const wrap = $("machPlayer");
    if (!wrap) return;
    wrap.hidden = false;
    machBuildTabs();
    // Default to the fused anchor — the measured baseline every other
    // schedule in the story is compared against.
    const defaultKey = d.machines.mask_rom_fused ? "mask_rom_fused"
      : (MACH_TABS.find(([key]) => d.machines[key])?.[0] || Object.keys(d.machines)[0]);
    machSelect(defaultKey);
  }

  function machBuildTabs() {
    const tabs = $("machTabs");
    if (!tabs) return;
    tabs.innerHTML = MACH_TABS
      .filter(([key]) => machData.machines[key])
      .map(([key, label]) =>
        `<button type="button" class="mp-tab" role="tab" data-mach="${key}" aria-pressed="false">${label}</button>`)
      .join("");
    tabs.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-mach]");
      if (btn) machSelect(btn.dataset.mach);
    });
  }

  // Per-machine schematic configs, grounded in the mechanisms slide's RTL
  // facts (qwen3_forward_controller schedules, qwen3_tpu_engine w_tile,
  // qwen3_lpu_engine S_STAGE staging, +EXT_MEM_LAT round trips). The renderer
  // rebuilds the strip per machine; paint logic reuses the same element ids.
  const MACH_SCHEMATICS = {
    mask_rom: {
      mem: { name: "ROM", cap: "on-die · 2-cyc read" },
      wire: { label: "ROM port · blocking", mode: "blocking" },
      extra: null,
      macIdle: "lane idle · waiting on read",
      watch: "Watch the MAC lane wait — every read blocks the next multiply.",
    },
    mask_rom_fused: {
      mem: { name: "ROM", cap: "on-die · 2-cyc read" },
      wire: { label: "ROM port · streaming", mode: "stream" },
      extra: { kind: "fifo", label: "FIFO", cap: "request every cycle" },
      macIdle: "idle",
      watch: "Watch the FIFO — the 2-cycle latency disappears into the stream.",
    },
    gpu: {
      mem: { name: "ROM", cap: "on-die · 2-cyc read" },
      wire: { label: "ROM port · streaming", mode: "stream" },
      extra: { kind: "fifo", label: "FIFO", cap: "tile-wave issuance" },
      macIdle: "idle",
      watch: "Watch the tile waves — GPU-style issuance on the same array.",
    },
    gpu_hbm: {
      mem: { name: "ROM + ext mem", cap: "+40 cyc latency / req" },
      wire: { label: "ext mem · HBM-class", mode: "stream" },
      extra: { kind: "fifo", label: "FIFO", cap: "tile-wave issuance" },
      macIdle: "idle",
      watch: "Watch the wire stall — +40 cycles per request, same energy.",
    },
    tpu: {
      mem: { name: "ROM", cap: "one tile" },
      wire: { label: "tile load · once", mode: "burst" },
      extra: { kind: "reg", label: "w_tile reg", cap: "reused rows" },
      macIdle: "idle",
      watch: "Watch the tile register — reuse pays only at batch > 1.",
    },
    tpu_hbm: {
      mem: { name: "ROM + ext mem", cap: "+40 cyc latency / req" },
      wire: { label: "ext mem · HBM-class", mode: "burst" },
      extra: { kind: "reg", label: "w_tile reg", cap: "reused rows" },
      macIdle: "idle",
      watch: "Watch the tile load stall — same reuse, HBM-class round trip.",
    },
    lpu: {
      mem: { name: "ROM", cap: "stage once" },
      wire: { label: "bulk stage · upfront", mode: "burst" },
      extra: { kind: "sram", label: "w_sram", cap: "zero traffic after" },
      macIdle: "idle",
      watch: "Watch the staging burst — ROM goes silent after the bulk load.",
    },
    lpu_hbm: {
      mem: { name: "ROM + ext mem", cap: "+40 cyc latency / req" },
      wire: { label: "ext mem · HBM-class", mode: "burst" },
      extra: { kind: "sram", label: "w_sram", cap: "zero traffic after" },
      macIdle: "idle",
      watch: "Watch the staging burst stall — same silence, slower arrival.",
    },
  };

  function machRenderSchematic(key) {
    const sch = document.querySelector("#machPlayer .mp-sch");
    const cfg = MACH_SCHEMATICS[key];
    if (!sch || !cfg) return;
    const mach = machData ? machData.machines[key] : null;
    const extra = cfg.extra
      ? `<div class="mp-blk mp-extra mp-x-${cfg.extra.kind}" id="machExtra"><b>${cfg.extra.label}</b><small>${cfg.extra.cap}</small></div>`
      : "";
    const bubble = mach && mach.ext_mem_lat > 0
      ? `<span class="mp-bubble">+${mach.ext_mem_lat} cyc/req</span>`
      : "";
    sch.innerHTML =
      `<div class="mp-blk mp-host"><b>host</b><small>token IDs</small></div>` +
      `<div class="mp-wire${cfg.wire.mode === "blocking" ? " mp-blocking" : ""}" id="machWire">` +
      `<i id="machPkt"></i>${bubble}<span id="machWireLab">${cfg.wire.label}</span></div>` +
      `<div class="mp-blk mp-mem" id="machMem"><b id="machMemName">${cfg.mem.name}</b><small id="machMemCap">${cfg.mem.cap}</small></div>` +
      extra +
      `<div class="mp-blk mp-mac" id="machMacBlk"><b>16-lane MAC array</b><small id="machMacCap">idle</small></div>`;
  }

  function machSelect(key) {
    const m = machData.machines[key];
    if (!m) return;
    machStop();
    const segs = m.segments;
    const pre = { rom: [0], mac: [0] };
    for (let i = 0; i < segs.length; i++) {
      pre.rom.push(pre.rom[i] + segs[i].rom);
      pre.mac.push(pre.mac[i] + segs[i].macs);
    }
    const span = (s) => Math.max(1, s.c1 - s.c0);
    const maxRomRate = Math.max(...segs.map((s) => s.rom / span(s)));
    const maxMacRate = Math.max(...segs.map((s) => s.macs / span(s)));
    machState = {
      key, m, segs, pre, maxRomRate, maxMacRate,
      total: m.meta.cycle_count, cycle: 0, playing: false, raf: 0, lastTs: 0
    };
    const bits = machData.energy_constants.rom_read_bits;
    const memTotal = m.meta.rom_read_count * bits * machData.energy_constants.rom_pj_per_bit / 1e12;
    const macTotal = m.meta.mac_count * machData.energy_constants.mac_pj / 1e12;
    machState.memTotal = memTotal;
    machState.macTotal = macTotal;
    machRenderSchematic(key);
    const cfg = MACH_SCHEMATICS[key];
    const watchEl = $("machWatch");
    if (watchEl && cfg) watchEl.textContent = cfg.watch;
    const ms = m.meta.cycle_count / (machData.clock_hz || 500e6) * 1e3;
    if ($("machCap")) {
      $("machCap").textContent =
        `${m.label} · ${m.meta.cycle_count.toLocaleString("en-US")} cycles · ${ms.toFixed(1)} ms/token @ 500 MHz` +
        (m.meta.mac_count !== machData.machines.mask_rom_fused.meta.mac_count
          ? ` · ${(m.meta.mac_count / 1e6).toFixed(1)}M MACs` : "");
    }
    const story = machStory(key);
    const storyEl = $("machStory");
    if (storyEl) storyEl.textContent = story.line;
    const shareEl = $("machShare");
    if (shareEl) {
      shareEl.innerHTML =
        `<b>${story.memPct.toFixed(0)}%</b> of this schedule's energy is moving weights out of storage — the rest is the multiply itself.`;
    }
    document.querySelectorAll("#machTabs .mp-tab").forEach((btn) => {
      btn.setAttribute("aria-pressed", String(btn.dataset.mach === key));
    });
    const scrub = $("machScrub");
    if (scrub) scrub.max = String(m.meta.cycle_count);
    machPaint();
  }

  function machSegAt(c) {
    const { segs } = machState;
    let lo = 0, hi = segs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (segs[mid].c0 <= c) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  function machPaint() {
    if (!machState) return;
    const st = machState;
    const c = Math.min(st.cycle, st.total);
    const i = machSegAt(c);
    const seg = st.segs[i];
    const span = Math.max(1, seg.c1 - seg.c0);
    const frac = Math.min(1, Math.max(0, (c - seg.c0) / span));
    const romSoFar = st.pre.rom[i] + seg.rom * frac;
    const macSoFar = st.pre.mac[i] + seg.macs * frac;
    const k = machData.energy_constants;
    const memJ = romSoFar * k.rom_read_bits * k.rom_pj_per_bit / 1e12;
    const macJ = macSoFar * k.mac_pj / 1e12;
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    const setW = (id, fracOfTotal) => { const el = $(id); if (el) el.style.width = `${Math.min(100, fracOfTotal * 100).toFixed(2)}%`; };
    setW("machMemBar", memJ / st.memTotal);
    setW("machMacBar", macJ / st.macTotal);
    set("machMemVal", machJ(memJ));
    set("machMacVal", machJ(macJ));
    const layerTxt = seg.layer > 0 ? ` · layer ${seg.layer + 1}/28` : "";
    const clock = machData.clock_hz || 500e6;
    set("machReadout", `${seg.stage}${layerTxt} · cycle ${c.toLocaleString("en-US")} · ${(c / clock * 1e3).toFixed(2)} ms @ 500 MHz`);
    set("machCycLab", `cycle ${c.toLocaleString("en-US")} / ${st.total.toLocaleString("en-US")}`);
    const scrub = $("machScrub");
    if (scrub && Number(scrub.value) !== c) scrub.value = String(c);
    const romRate = seg.rom / span;
    const macRate = seg.macs / span;
    const wire = $("machWire");
    if (wire) {
      wire.classList.toggle("is-live", romRate > 0);
      wire.classList.toggle("is-hot", romRate >= st.maxRomRate * 0.66);
      wire.style.setProperty("--frac", frac.toFixed(4));
    }
    const macBlk = $("machMacBlk");
    if (macBlk) {
      macBlk.classList.toggle("is-live", macRate > 0);
      macBlk.classList.toggle("is-hot", macRate >= st.maxMacRate * 0.66);
    }
    const macCap = $("machMacCap");
    const idleCap = (MACH_SCHEMATICS[st.key] || {}).macIdle || "idle";
    if (macCap) macCap.textContent = macRate > 0 ? `${Math.round(macRate).toLocaleString("en-US")} MAC/cyc` : idleCap;
    const play = $("machPlayBtn");
    if (play) play.textContent = st.playing ? "Pause" : "Play";
  }

  function machFrame(ts) {
    if (!machState || !machState.playing) return;
    const dt = machState.lastTs ? ts - machState.lastTs : 0;
    machState.lastTs = ts;
    machState.cycle += dt * (machState.total / MACH_PLAY_MS);
    if (machState.cycle >= machState.total) {
      machState.cycle = machState.total;
      machState.playing = false;
    }
    machPaint();
    if (machState.playing) machState.raf = requestAnimationFrame(machFrame);
  }

  function machStop() {
    if (machState && machState.raf) cancelAnimationFrame(machState.raf);
    if (machState) { machState.playing = false; machState.raf = 0; }
  }

  function machWirePlayer() {
    const play = $("machPlayBtn");
    if (play) play.addEventListener("click", () => {
      if (!machState) return;
      if (machState.playing) { machStop(); machPaint(); return; }
      if (machState.cycle >= machState.total) machState.cycle = 0;
      machState.playing = true;
      machState.lastTs = 0;
      machState.raf = requestAnimationFrame(machFrame);
      machPaint();
    });
    const scrub = $("machScrub");
    if (scrub) scrub.addEventListener("input", () => {
      if (!machState) return;
      machState.cycle = Number(scrub.value);
      machPaint();
    });
  }
  machWirePlayer();

  const ARRAY_SIZES = ["16×1", "16×16", "32×32", "64×64", "128×128"];

  async function loadArrayScaling() {
    const d = await deckEvidence("array_scaling");
    if (!d) return;
    // Labels and slider range come from the evidence, not a hardcoded list:
    // the series grew when the measured width sweep was added, and a fixed
    // max="4" silently hid the measured 64x1 and 128x1 points.
    const names = d.lab_series.arrays.map((a) => a.config.name
      .replace(/\s*@.*$/, "").replace(/\s*\(measured\)/, ""));
    const ticks = $("arrayTicks");
    if (ticks) {
      ticks.innerHTML = names.map((s) => `<span>${s}</span>`).join("");
    }
    const slider = $("arraySlider");
    if (slider) slider.max = String(Math.max(0, d.lab_series.arrays.length - 1));
    if (slider && !slider.dataset.bound) {
      slider.dataset.bound = "1";
      slider.addEventListener("input", () => paintArrayScaling(d, Number(slider.value)));
    }
    paintArrayScaling(d, Number((slider || {}).value || 0));
    const macs = d.lab_series.arrays[0].workload.total_macs;
    const macsEl = $("asMacs");
    if (macsEl) macsEl.textContent = fmtInt(macs);
  }

  function paintArrayScaling(d, idx) {
    const lab = d.lab_series.arrays[idx];
    const prod = d.production_series.arrays[idx];
    const fill = (id, r) => {
      const el = $(id);
      if (!el) return;
      const m = r.metrics;
      const prov = r.provenance.kind === "measured" ? "measured" : "analytical_estimate";
      el.innerHTML = `
        <tr><th>Array</th><td>${r.config.name}</td></tr>
        <tr><th>MAC lanes</th><td>${fmtInt(r.config.pes)}</td></tr>
        <tr data-provenance="${prov}"><th>tok/s</th><td>${fmtToks(m.throughput_toks)}</td></tr>
        <tr data-provenance="${prov}"><th>energy / token</th><td>${fmtMJ(m.total_energy_j)}</td></tr>
        <tr data-provenance="${prov}"><th>latency / token</th><td>${fmtLat(m.latency_s)}</td></tr>
        <tr><th>ground truth</th><td><span class="kp-tag ${r.provenance.kind === "measured" ? "kp-measured" : "kp-sim"}">${r.provenance.kind === "measured" ? "measured" : "simulated"}</span></td></tr>`;
    };
    fill("arrayLabRows", lab);
    fill("arrayProdRows", prod);
  }

  async function loadPrecisionScaling() {
    const d = await deckEvidence("precision_scaling");
    const tbody = $("precisionRows");
    if (!tbody || !d || !d.precisions) return;
    const order = ["bf16", "int8", "int4"];
    tbody.innerHTML = order.map((k) => {
      const p = d.precisions[k];
      return `<tr>
        <td class="kp-name">${p.precision}</td>
        <td>${p.bits_per_weight}</td>
        <td data-provenance="derived">${(p.transistors.rom / 1e9).toFixed(1)}B</td>
        <td data-provenance="derived">${p.transistors.total_billions.toFixed(1)}B</td>
        <td data-provenance="analytical_estimate">${fmtMJ(p.energy_per_token.total_j)}</td>
        <td data-provenance="published">${p.accuracy.degradation} — ${p.accuracy.reference}</td>
      </tr>`;
    }).join("");
  }

  async function loadMultiDieScaling() {
    const d = await deckEvidence("multi_die_scaling");
    const tbody = $("multiDieRows");
    if (!tbody || !d || !d.results) return;
    const seen = new Map();
    for (const r of d.results) {
      const name = r.config.name.replace(/ \(\d+b\)$/, "");
      if (!seen.has(name)) seen.set(name, {});
      seen.get(name)[r.config.precision_bits] = r;
    }
    const rows = [];
    for (const [name, byBits] of seen) {
      const r16 = byBits[16];
      if (!r16) continue;
      const dies = (b) => (byBits[b] ? byBits[b].dies.n_dies : "—");
      const moe = r16.moe;
      const expertsTxt = moe.total_experts > 1
        ? `${moe.active_experts} of ${moe.total_experts}`
        : "dense";
      rows.push(`<tr>
        <td class="kp-name">${name}</td>
        <td>${r16.config.total_params_billions >= 1000
              ? (r16.config.total_params_billions / 1000).toFixed(1) + " trillion"
              : r16.config.total_params_billions + " billion"}</td>
        <td>${expertsTxt}</td>
        <td data-provenance="analytical_estimate">${dies(16)}</td>
        <td data-provenance="analytical_estimate">${dies(8)}</td>
        <td data-provenance="analytical_estimate">${dies(4)}</td>
        <td data-provenance="analytical_estimate">${r16.interconnect.weight_bytes_crossing_dies}</td>
      </tr>`);
    }
    tbody.innerHTML = rows.join("");
  }


  const D2_N = 8;
  const D2_TILES = ["Q_PROJ", "K_PROJ", "V_PROJ", "O_PROJ", "GATE", "UP", "DOWN", "LM_HEAD"];
  const d2Anim = { t: 0, timer: null };

  function d2FillGrid(el) {
    if (!el || el.dataset.ready === "1") return;
    el.dataset.ready = "1";
    let html = "";
    for (let r = 0; r < D2_N; r += 1) {
      for (let c = 0; c < D2_N; c += 1) {
        html += `<div class="pe" data-pe="${r}-${c}" title="PE[${r},${c}] 1T ROM W + MAC"><span>W</span><em>×</em></div>`;
      }
    }
    el.innerHTML = html;
  }

  function buildD2Viz() {
    const lane = $("d2Lane1d");
    if (lane && lane.dataset.ready !== "1") {
      lane.dataset.ready = "1";
      lane.innerHTML = Array.from({ length: 16 }, (_, i) =>
        `<div class="pe" data-lane="${i}"><span>a${i}</span><em>×</em><span>w</span></div>`
      ).join("");
    }
    d2FillGrid($("d2GridMini"));
    d2FillGrid($("d2Grid"));
    const sram = $("d2Sram");
    if (sram && sram.dataset.ready !== "1") {
      sram.dataset.ready = "1";
      sram.innerHTML = Array.from({ length: 8 }, (_, i) =>
        `<div class="bank"><b>x${i}</b></div>`
      ).join("");
    }
    const tiles = $("d2Tiles");
    if (tiles && tiles.dataset.ready !== "1") {
      tiles.dataset.ready = "1";
      tiles.innerHTML = D2_TILES.map((n) => `<div class="tile" data-d2-tile="${n}">${n}</div>`).join("");
    }
    const layers = $("d2Layers");
    if (layers && layers.dataset.ready !== "1") {
      layers.dataset.ready = "1";
      layers.innerHTML = Array.from({ length: 28 }, (_, i) =>
        `<div class="layer" data-d2-layer="${i}">L${String(i).padStart(2, "0")}</div>`
      ).join("");
    }
  }

  function d2Mark(sel, on) {
    document.querySelectorAll(sel).forEach((el) => el.classList.toggle("active", !!on));
  }
  function d2Wire(name, on) {
    document.querySelectorAll(`[data-d2-wire="${name}"]`).forEach((el) => el.classList.toggle("live", !!on));
  }

  function d2PaintGrid(root, wave) {
    if (!root) return { live: 0, hitR: 0, hitC: 0 };
    let live = 0;
    let hitR = 0;
    let hitC = 0;
    root.querySelectorAll("[data-pe]").forEach((pe) => {
      const [r, c] = pe.getAttribute("data-pe").split("-").map(Number);
      const s = r + c;
      const hot = wave >= 0 && s === wave;
      pe.classList.toggle("hot", hot);
      pe.classList.toggle("done", wave > s);
      if (hot) { hitR = r; hitC = c; }
      if (hot || (wave > s && wave >= 0)) live += 1;
    });
    return { live, hitR, hitC };
  }

  function paintD2Anim() {
    if (!$("d2Chip")) return;
    const t = d2Anim.t;
    const layer = Math.floor(t / 24) % 28;
    const tileI = Math.floor(t / 20) % D2_TILES.length;
    const stage = D2_TILES[tileI];
    const phase = t % 20;
    const load = phase < 4;
    const wave = load ? -1 : phase - 4;
    const nlin = false;

    d2PaintGrid($("d2GridMini"), wave);
    const hit = d2PaintGrid($("d2Grid"), wave);

    const lane = $("d2Lane1d");
    if (lane) {
      const hot = t % 16;
      lane.querySelectorAll("[data-lane]").forEach((pe) => {
        const i = Number(pe.getAttribute("data-lane"));
        pe.classList.toggle("hot", i === hot);
        pe.classList.toggle("done", i < hot);
      });
    }

    const fsm = $("d2Fsm");
    if (fsm) fsm.textContent = load ? "S_LOAD_ACT" : `S_RUN · ${stage}`;
    d2Mark('[data-d2="host"]', true);
    d2Mark('[data-d2="fsm"]', true);
    d2Mark('[data-d2="sram"]', load || wave >= 0);
    d2Mark('[data-d2="mxu"]', !load);
    d2Mark('[data-d2="logits"]', !load && stage === "LM_HEAD");
    d2Mark('[data-d2="rmsnorm"]', false);
    d2Mark('[data-d2="softmax"]', false);
    d2Mark('[data-d2="rope"]', false);
    d2Mark('[data-d2="swiglu"]', false);
    d2Wire("cmd", true);
    d2Wire("fabric", !load);
    d2Wire("x", !load);
    d2Wire("nlin", nlin);
    document.querySelectorAll("[data-d2-layer]").forEach((el) => {
      el.classList.toggle("active", Number(el.getAttribute("data-d2-layer")) === layer);
    });
    document.querySelectorAll("[data-d2-tile]").forEach((el) => {
      el.classList.toggle("active", el.getAttribute("data-d2-tile") === stage);
    });
    const reduce = $("d2Reduce");
    if (reduce) {
      reduce.textContent = load
        ? "tile_sel holds · 1-transistor ROM already in every PE · load x from SRAM"
        : `x[${hit.hitR}] east · psum[${hit.hitC}] south · ${hit.live} PEs live · ${stage}`;
    }
    const logit = $("d2Logit");
    if (logit) {
      logit.textContent = stage === "LM_HEAD"
        ? "lm_head tiled on MXU · argmax in flight"
        : "151936 logits · tiled on the MXU";
    }
    const note = $("d2Note");
    if (note) {
      note.textContent = load
        ? `L${String(layer).padStart(2, "0")} ${stage}  S_LOAD_ACT  ·  W already burnt in`
        : `L${String(layer).padStart(2, "0")} ${stage}  S_RUN  beat ${wave}  ·  PE[${hit.hitR},${hit.hitC}]  ·  ${hit.live} PEs  ·  W_rom not on fabric`;
    }
    const laneCap = $("d2LaneCap");
    if (laneCap) {
      const j = Math.floor(t / 16) % 8;
      laneCap.textContent = `16 MACs/cycle · committing y[${j}] · W on the 256b bus · 8.8 tok/s measured`;
    }
  }

  function stopD2Anim() {
    if (d2Anim.timer) {
      clearInterval(d2Anim.timer);
      d2Anim.timer = null;
    }
  }

  function startD2Anim() {
    buildD2Viz();
    paintD2Anim();
    renderD2Math();
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (d2Anim.timer) return;
    d2Anim.timer = setInterval(() => {
      d2Anim.t += 1;
      paintD2Anim();
    }, 280);
  }

  // ---- MoE routing/weight-movement illustration --------------------------
  // Illustrative only (no MoE RTL exists in this repo) -- see the src-fold
  // on the MoE slide. 128 experts, top-8, matches Qwen3-30B-A3B / this
  // repo's lab/multi_die.py MOE_CONFIGS["moe_128x"].
  const MOE_TOTAL = 128;
  const MOE_ACTIVE = 8;
  const moeAnim = { t: 0, timer: null, playing: true };

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let x = Math.imul(a ^ (a >>> 15), 1 | a);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function moeActiveSet(t) {
    const rand = mulberry32(t + 1);
    const pool = Array.from({ length: MOE_TOTAL }, (_, i) => i);
    const chosen = [];
    for (let k = 0; k < MOE_ACTIVE; k += 1) {
      const j = Math.floor(rand() * pool.length);
      chosen.push(pool.splice(j, 1)[0]);
    }
    return new Set(chosen);
  }

  function buildMoeViz() {
    const hbm = $("moeGridHbm");
    const fused = $("moeGridFused");
    if (!hbm || !fused || hbm.childElementCount === MOE_TOTAL) return;
    for (const grid of [hbm, fused]) {
      grid.innerHTML = "";
      for (let i = 0; i < MOE_TOTAL; i += 1) {
        const cell = document.createElement("div");
        cell.className = "moe-cell";
        cell.setAttribute("data-idx", String(i));
        grid.appendChild(cell);
      }
    }
  }

  function paintMoeAnim() {
    buildMoeViz();
    const active = moeActiveSet(moeAnim.t);
    const hbmCells = document.querySelectorAll("#moeGridHbm .moe-cell");
    const fusedCells = document.querySelectorAll("#moeGridFused .moe-cell");
    hbmCells.forEach((cell, i) => {
      cell.classList.toggle("active", active.has(i));
      cell.classList.toggle("fetching", active.has(i));
    });
    setTimeout(() => {
      document.querySelectorAll("#moeGridHbm .moe-cell.fetching").forEach((cell) => {
        cell.classList.remove("fetching");
      });
    }, 260);
    fusedCells.forEach((cell, i) => cell.classList.toggle("active", active.has(i)));
    const tokenEl = $("moeTokenN");
    if (tokenEl) tokenEl.textContent = String(moeAnim.t);
  }

  function stopMoeAnim() {
    if (moeAnim.timer) {
      clearInterval(moeAnim.timer);
      moeAnim.timer = null;
    }
  }

  function startMoeAnim() {
    paintMoeAnim();
    const btn = $("moeToggle");
    if (btn && !btn.dataset.wired) {
      btn.dataset.wired = "1";
      btn.addEventListener("click", () => {
        moeAnim.playing = !moeAnim.playing;
        btn.textContent = moeAnim.playing ? "Pause" : "Play";
        if (moeAnim.playing) startMoeAnim(); else stopMoeAnim();
      });
    }
    if (!moeAnim.playing) return;
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (moeAnim.timer) return;
    moeAnim.timer = setInterval(() => {
      moeAnim.t += 1;
      paintMoeAnim();
    }, 800);
  }

  function renderGpuMath() {
    const root = document.querySelector(".slide-gpu");
    if (!root || root.dataset.math === "1" || typeof renderMathInElement !== "function") return;
    renderMathInElement(root, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
    root.dataset.math = "1";
  }

  function renderD2Math() {
    const root = document.querySelector(".slide-2d");
    if (!root || root.dataset.math === "1" || typeof renderMathInElement !== "function") return;
    renderMathInElement(root, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
    root.dataset.math = "1";
  }

  // ---- Slide 2 · GPU HBM→SM animation ---------------------------------------
  // The machine slide 1 eliminates: a tiny GPU replica whose weights stream
  // from HBM into the SMs on every token. Numbers render from
  // evidence/exec_models.json → platforms.h100 / platforms.b200.
  const GPU_MACH = {
    h100: {
      label: "NVIDIA H100 SXM",
      smCount: 132,
      smGrid: [12, 11],
      hbmStacks: 5,
      hbmName: "HBM3",
      hbmGb: 80,
      hbmIoBits: 5120,
      l2Bytes: 50 * 1024 * 1024,
      hbmBwTbs: 3.35,
      tdpW: 700,
      flops: "989 TFLOPS BF16 dense",
      flopsPerS: 989e12,
      note: "H100 SXM 80GB is five HBM3 stacks [P whitepaper]. Our RTL (gpu_hbm_bw_model) is one fetch port: stack_busy = {5{fetch_busy}}. We do not have NVIDIA's page map, so we never light stacks one-by-one. 1.19 GB of unique W per token misses the 50 MB L2."
    },
    b200: {
      label: "NVIDIA B200",
      smCount: 148,
      smGrid: [12, 13],
      hbmStacks: 8,
      hbmName: "HBM3e",
      hbmGb: 192,
      hbmIoBits: 8192,
      l2Bytes: 128 * 1024 * 1024,
      hbmBwTbs: 8.0,
      tdpW: 1000,
      flops: "2.25 PFLOPS BF16 dense",
      flopsPerS: 2.25e15,
      note: "B200: eight HBM3e stacks, same owned map — one port, all stacks busy on a sequential fetch. L2 still ≪ 1.19 GB. The stream is structural."
    }
  };

  // ---- Slide 2 · GPU cycle-accurate weight-streaming simulation -------------
  // Full-token cycle-accurate walk. Not one illustrative GEMM: this replays
  // every one of the real 395 per-stage/per-layer calls in
  // evidence/exec_models.json -> workload.stages_detail, in the model's
  // actual order (embedding -> 28 x [14 stages/layer] -> final_rmsnorm ->
  // lm_head) -- the same table the KPI table and every other slide use.
  // Per-call cycles use the exact formulas in
  // lab/exec_models.py::_simulate_stage_gpu (compute_s = 2*macs/flops_per_s,
  // weight_s = weight_bytes/hbm_bytes_per_s, same 1.5 GHz accounting clock),
  // decomposed into act/weight/compute sub-phases so each stage is watchable
  // instead of collapsed into one roofline max. Summed over all 395 calls
  // this lands within ~0.6% of the platform's published total_cycles.
  const GPU_STAGE_ORDER = [
    "input_rmsnorm", "qkv_proj", "qk_norm", "rope", "attn_scores", "softmax",
    "attn_value", "o_proj", "attn_residual", "post_rmsnorm",
    "gate_up_proj", "silu_swiglu", "down_proj", "mlp_residual"
  ];
  const GPU_STAGE_LABEL = {
    embedding: "embedding lookup",
    input_rmsnorm: "input RMSNorm",
    qkv_proj: "Q/K/V projection",
    qk_norm: "Q/K RMSNorm",
    rope: "RoPE",
    attn_scores: "attention scores",
    softmax: "softmax",
    attn_value: "attention × V",
    o_proj: "output projection",
    attn_residual: "attention residual",
    post_rmsnorm: "post RMSNorm",
    gate_up_proj: "gate/up projection",
    silu_swiglu: "SiLU · SwiGLU",
    down_proj: "down projection",
    mlp_residual: "MLP residual",
    final_rmsnorm: "final RMSNorm",
    lm_head: "LM head (151,936 logits)"
  };
  const GPU_STAGE_OPS = {
    embedding: ["embedding"],
    input_rmsnorm: ["input_norm"],
    qkv_proj: ["q_proj", "k_proj", "v_proj"],
    qk_norm: ["q_norm", "k_norm"],
    rope: ["q_rope", "k_rope"],
    attn_scores: ["attention_scores"],
    softmax: ["attention_softmax"],
    attn_value: ["causal_gqa"],
    o_proj: ["o_proj"],
    attn_residual: ["attention_residual"],
    post_rmsnorm: ["post_norm"],
    gate_up_proj: ["gate_proj", "up_proj"],
    silu_swiglu: ["silu", "swiglu"],
    down_proj: ["down_proj"],
    mlp_residual: ["output"],
    final_rmsnorm: ["final_norm"],
    lm_head: ["logits", "argmax"]
  };
  const GPU_PROMPT = "The capital of India is Delhi, and the capital of Japan is";
  const GPU_CLOCK_HZ = 1.5e9; // same accounting clock as lab/exec_models.py
  const GPU_STAGE_NOTES = {
    embedding: "u_hbm fetch of one E row, then u_mac. Host sent a token ID. L2 cannot hold E (151,936×1024).",
    input_rmsnorm: "u_norm on x in SM scratch. The RTL also fetches a tiny γ from u_hbm before the engine (NORM in gpu_stream_core). Not CUDA cores.",
    qkv_proj: "QKV_FETCH then QKV_MAC, the measured walk. src: u_hbm (all stacks = fetch_busy). dst: u_mac (qwen_bf16_mac_array). L2 miss.",
    qk_norm: "u_norm on Q/K already in SM scratch. u_hbm idle.",
    rope: "Later stage: u_rope rotates Q/K already produced by a prior MAC. Not between fetch and MAC. u_hbm idle.",
    attn_scores: "src: u_kv (on-die K), not u_hbm. dst: u_mac for QKᵀ. Zero weight bytes from HBM.",
    softmax: "u_softmax on the score row. Measured SOFTMAX stage. u_hbm idle.",
    attn_value: "src: u_kv (on-die V). dst: u_mac for P×V. Zero HBM.",
    o_proj: "u_hbm fetch of W_O, L2 miss, then u_mac. Same contract as QKV_FETCH → QKV_MAC.",
    attn_residual: "Add in SM scratch. No fetch, no MAC array.",
    post_rmsnorm: "u_norm. u_hbm idle except γ.",
    gate_up_proj: "Large u_hbm fetch (W_g, W_u), L2 miss, then u_mac. fetch_busy is long because the bytes are large, not because stacks take turns.",
    silu_swiglu: "Later stage: u_swiglu does SiLU(gate) ⊙ up on the vector the gate/up MAC just wrote. Not between that fetch and that MAC. u_hbm idle.",
    down_proj: "u_hbm fetch of W_d, L2 miss, u_mac. Same port as every other GEMM.",
    mlp_residual: "Add in SM scratch. Decoder block done.",
    final_rmsnorm: "u_norm after L27. Then the vocabulary fetch.",
    lm_head: "Longest u_hbm fetch (y Eᵀ, 1024×151,936). All stacks stay fetch_busy together because the request is huge. Then u_mac. Argmax after fetch+MAC, not a second HBM walk."
  };
  // Destinations are RTL modules, not NVIDIA product names.
  const GPU_STAGE_DEST = {
    embedding: "mac",
    input_rmsnorm: "engine",
    qkv_proj: "mac",
    qk_norm: "engine",
    rope: "engine",
    attn_scores: "mac",
    softmax: "engine",
    attn_value: "mac",
    o_proj: "mac",
    attn_residual: "engine",
    post_rmsnorm: "engine",
    gate_up_proj: "mac",
    silu_swiglu: "engine",
    down_proj: "mac",
    mlp_residual: "engine",
    final_rmsnorm: "engine",
    lm_head: "mac"
  };
  const GPU_STAGE_SRC = {
    embedding: "hbm",
    input_rmsnorm: "l1",
    qkv_proj: "hbm",
    qk_norm: "l1",
    rope: "l1",
    attn_scores: "kv",
    softmax: "l1",
    attn_value: "kv",
    o_proj: "hbm",
    attn_residual: "l1",
    post_rmsnorm: "l1",
    gate_up_proj: "hbm",
    silu_swiglu: "l1",
    down_proj: "hbm",
    mlp_residual: "l1",
    final_rmsnorm: "l1",
    lm_head: "hbm"
  };
  const GPU_STAGE_BLOCKS = {
    embedding: ["host", "fsm", "hbm", "l2", "l1", "mac"],
    input_rmsnorm: ["fsm", "l1", "rmsnorm"],
    qkv_proj: ["fsm", "hbm", "l2", "l1", "mac"],
    qk_norm: ["fsm", "l1", "rmsnorm"],
    rope: ["fsm", "l1", "rope"],
    attn_scores: ["fsm", "kv", "l1", "mac"],
    softmax: ["fsm", "l1", "softmax"],
    attn_value: ["fsm", "kv", "l1", "mac"],
    o_proj: ["fsm", "hbm", "l2", "l1", "mac"],
    attn_residual: ["fsm", "l1"],
    post_rmsnorm: ["fsm", "l1", "rmsnorm"],
    gate_up_proj: ["fsm", "hbm", "l2", "l1", "mac"],
    silu_swiglu: ["fsm", "l1", "swiglu"],
    down_proj: ["fsm", "hbm", "l2", "l1", "mac"],
    mlp_residual: ["fsm", "l1"],
    final_rmsnorm: ["fsm", "l1", "rmsnorm"],
    lm_head: ["fsm", "hbm", "l2", "l1", "mac", "logits"]
  };
  const GPU_FLOW = [
    { id: "embedding", where: "head", kind: "gemm", title: "embed",
      cells: [{ cls: "io", k: "host", v: "token ID" }, { cls: "rom", k: "u_hbm", v: "E[id]" }, { cls: "mac", k: "u_mac", v: "→ x" }] },
    { id: "input_rmsnorm", where: "layer", kind: "engine", title: "in RMSNorm",
      cells: [{ cls: "sram", k: "RF", v: "x" }, { cls: "eng", k: "u_norm", v: "RMS" }, { cls: "sram", k: "RF", v: "x̂" }] },
    { id: "qkv_proj", where: "layer", kind: "gemm", title: "QKV",
      cells: [{ cls: "rom", k: "u_hbm", v: "W" }, { cls: "sram", k: "RF", v: "x · W" }, { cls: "mac", k: "u_mac", v: "×16 → QKV" }] },
    { id: "qk_norm", where: "layer", kind: "engine", title: "Q/K RMS",
      cells: [{ cls: "sram", k: "RF", v: "Q K" }, { cls: "eng", k: "u_norm", v: "RMS" }, { cls: "sram", k: "RF", v: "Q̂ K̂" }] },
    { id: "rope", where: "layer", kind: "engine", title: "RoPE",
      cells: [{ cls: "sram", k: "RF", v: "Q̂ K̂" }, { cls: "eng", k: "u_rope", v: "rotate" }, { cls: "sram", k: "RF", v: "Q̃ K̃" }] },
    { id: "attn_scores", where: "layer", kind: "gemm", title: "QKᵀ",
      cells: [{ cls: "kv", k: "u_kv", v: "K" }, { cls: "sram", k: "RF", v: "Q" }, { cls: "mac", k: "u_mac", v: "×16 → S" }] },
    { id: "softmax", where: "layer", kind: "engine", title: "softmax",
      cells: [{ cls: "sram", k: "RF", v: "S" }, { cls: "eng", k: "u_softmax", v: "P" }, { cls: "sram", k: "RF", v: "P" }] },
    { id: "attn_value", where: "layer", kind: "gemm", title: "P×V",
      cells: [{ cls: "kv", k: "u_kv", v: "V" }, { cls: "sram", k: "RF", v: "P" }, { cls: "mac", k: "u_mac", v: "×16 → ctx" }] },
    { id: "o_proj", where: "layer", kind: "gemm", title: "O proj",
      cells: [{ cls: "rom", k: "u_hbm", v: "W_O" }, { cls: "sram", k: "RF", v: "ctx · W" }, { cls: "mac", k: "u_mac", v: "×16 → y" }] },
    { id: "attn_residual", where: "layer", kind: "add", title: "attn +",
      cells: [{ cls: "sram", k: "RF", v: "x" }, { cls: "sram", k: "add", v: "+" }, { cls: "sram", k: "RF", v: "x+y" }] },
    { id: "post_rmsnorm", where: "layer", kind: "engine", title: "post RMS",
      cells: [{ cls: "sram", k: "RF", v: "x" }, { cls: "eng", k: "u_norm", v: "RMS" }, { cls: "sram", k: "RF", v: "x̂" }] },
    { id: "gate_up_proj", where: "layer", kind: "gemm", title: "gate/up",
      cells: [{ cls: "rom", k: "u_hbm", v: "W_g W_u" }, { cls: "sram", k: "RF", v: "x · W" }, { cls: "mac", k: "u_mac", v: "×16 → y" }] },
    { id: "silu_swiglu", where: "layer", kind: "engine", title: "SwiGLU",
      cells: [{ cls: "sram", k: "RF", v: "gate up" }, { cls: "eng", k: "u_swiglu", v: "SiLU⊙" }, { cls: "sram", k: "RF", v: "h" }] },
    { id: "down_proj", where: "layer", kind: "gemm", title: "down",
      cells: [{ cls: "rom", k: "u_hbm", v: "W_d" }, { cls: "sram", k: "RF", v: "h · W" }, { cls: "mac", k: "u_mac", v: "×16 → y" }] },
    { id: "mlp_residual", where: "layer", kind: "add", title: "MLP +",
      cells: [{ cls: "sram", k: "RF", v: "x" }, { cls: "sram", k: "add", v: "+" }, { cls: "sram", k: "RF", v: "x+y" }] },
    { id: "final_rmsnorm", where: "tail", kind: "engine", title: "final RMS",
      cells: [{ cls: "sram", k: "RF", v: "x" }, { cls: "eng", k: "u_norm", v: "RMS" }, { cls: "sram", k: "RF", v: "y" }] },
    { id: "lm_head", where: "tail", kind: "gemm", title: "lm_head",
      cells: [{ cls: "rom", k: "u_hbm", v: "Eᵀ" }, { cls: "sram", k: "RF", v: "y · W" }, { cls: "mac", k: "u_mac", v: "argmax" }] }
  ];
  let gpuNextTok = "Tokyo";
  let gpuTokenIds = [];
  let gpuPassMetrics = null;

  function gpuStageCycles(m, s) {
    const hbmBytesPerS = m.hbmBwTbs * 1e12;
    const flops = 2 * s.macs_per_call;
    const computeS = flops > 0 ? flops / m.flopsPerS : 0;
    const actS = s.act_bytes_per_call > 0 ? s.act_bytes_per_call / hbmBytesPerS : 0;
    const weightS = s.weight_bytes_per_call > 0 ? s.weight_bytes_per_call / hbmBytesPerS : 0;
    const actCycles = actS > 0 ? Math.max(1, Math.round(actS * GPU_CLOCK_HZ)) : 0;
    const weightCycles = weightS > 0 ? Math.max(1, Math.round(weightS * GPU_CLOCK_HZ)) : 0;
    const computeCycles = Math.max(1, Math.round(computeS * GPU_CLOCK_HZ));
    return { actCycles, weightCycles, computeCycles, total: actCycles + weightCycles + computeCycles };
  }

  const gpuSeqCache = {};

  function gpuBuildStageSeq(mach, workload) {
    if (gpuSeqCache[mach]) return gpuSeqCache[mach];
    const m = GPU_MACH[mach];
    const byName = {};
    for (const s of workload.stages_detail) byName[s.stage] = s;
    const layers = byName.qkv_proj ? byName.qkv_proj.count : 28;
    const seq = [];
    let cursor = 0;
    function push(stageName, layer) {
      const s = byName[stageName];
      if (!s) return;
      const cyc = gpuStageCycles(m, s);
      seq.push({
        stage: stageName, layer,
        weightBytesPerCall: s.weight_bytes_per_call,
        actBytesPerCall: s.act_bytes_per_call,
        macsPerCall: s.macs_per_call,
        cyc, start: cursor
      });
      cursor += cyc.total;
    }
    push("embedding", null);
    for (let L = 0; L < layers; L++) {
      for (const name of GPU_STAGE_ORDER) push(name, L);
    }
    push("final_rmsnorm", null);
    push("lm_head", null);
    gpuSeqCache[mach] = { seq, total: cursor, layers };
    return gpuSeqCache[mach];
  }

  function gpuFindStage(seqObj, c) {
    const seq = seqObj.seq;
    let lo = 0, hi = seq.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (seq[mid].start <= c) lo = mid; else hi = mid - 1;
    }
    return seq[lo];
  }

  const gpuAnim = { mach: "h100", cycle: 0, playing: true, timer: null, raf: 0, rafLast: 0, dwellLeft: 0, simdFrame: 0, flow: false, story: true, introShown: false, speedIdx: 3, pendingJump: false };
  let gpuPaintPhaseKey = "";
  let gpuTileData = null; // evidence/gpu_tile_sim.json, see lab/gpu_tile_sim.py
  let gpuEnergySplit = null; // {machLabel, weightJ, computeJ, totalWeightBytes, totalMacs} from exec_models
  let gpuExperienceMounted = false;
  function mountGpuExperience() {
    const root = $("gpuExperience");
    const experience = window.GpuExperience;
    if (!root || !experience || gpuExperienceMounted) return;
    experience.mount(root);
    gpuExperienceMounted = true;
    document.querySelector(".slide-gpu").classList.add("has-experience");
    experience.onAction = (action) => {
      if (action.type === "details") { gpuSetStory(false); return; }
      if (action.type === "pause") { gpuTogglePlay(false); paintGpuCycle(); return; }
      if (action.type === "play") { gpuTogglePlay(); paintGpuCycle(); return; }
      gpuTogglePlay(false);
      if (action.type === "prev") gpuStepStage(-1);
      if (action.type === "next") gpuStepStage(1);
      if (action.type === "seekStage") {
        const seq = gpuSeqCache[gpuAnim.mach];
        const entry = seq && seq.seq.find((s) => s.stage === action.stage &&
          (action.layer == null || s.layer === action.layer));
        if (entry) {
          const offset = action.phase === "compute" ? entry.cyc.actCycles + entry.cyc.weightCycles
            : action.phase === "fetch" ? entry.cyc.actCycles : 0;
          gpuSetCycle(entry.start + offset);
        }
      }
    };
  }

  let gpuRtlLoaded = false;
  async function loadGpuRtlMeasured() {
    if (gpuRtlLoaded) return;
    gpuRtlLoaded = true;
    loadHwCompare();
    try {
      const res = await apiFetch("/api/walk?schedule=streaming");
      if (!res.ok) return;
      const d = await res.json();
      const stages = {};
      if (window.GpuExperience && window.GpuExperience.setRtl) window.GpuExperience.setRtl(d);
      for (const s of d.stages) stages[s.stage] = s.cycles;
      const line = $("gpuRtlLine");
      if (line && stages.QKV_FETCH && stages.QKV_MAC) {
        const ratio = stages.QKV_FETCH / stages.QKV_MAC;
        line.textContent =
          `One Q/K/V output column: ${stages.QKV_FETCH} cycles to stream its weights vs ` +
          `${stages.QKV_MAC} cycles to compute (${ratio.toFixed(1)}×) · ` +
          `KV cache round-trip: ${stages.KV_CACHE} cycles, zero HBM traffic · ` +
          `${d.total_cycles} total cycles`;
      }
    } catch (_) { /* optional */ }
  }


  // Same compute, different hardware: measured mask-ROM / GPU / TPU / LPU
  // comparison from lab.hw_verify runs (all bit-exact vs the Python oracle).
  let hwCompareLoaded = false;
  async function loadHwCompare() {
    if (hwCompareLoaded) return;
    hwCompareLoaded = true;
    const body = $("hwCompareBody");
    if (!body) return;
    try {
      const res = await apiFetch("/api/hw_comparison");
      if (!res.ok) { body.innerHTML = '<p class="muted small">No hw_verify runs yet - run <code>make verify-gate</code>.</p>'; return; }
      const d = await res.json();
      const runs = (d.runs || []).filter((r) => r.backends && r.backends.length);
      if (!runs.length) { body.innerHTML = '<p class="muted small">No hw_verify runs yet - run <code>make verify-gate</code>.</p>'; return; }
      const fmtMB = (b) => b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${(b / 1024).toFixed(0)} KB`;
      const fmtC = (c) => c >= 1e6 ? `${(c / 1e6).toFixed(2)}M` : c >= 1e3 ? `${(c / 1e3).toFixed(0)}K` : `${c}`;
      const fmtMJ = (j) => j >= 1e-3 ? `${(j * 1e3).toFixed(2)} mJ` : `${(j * 1e6).toFixed(0)} µJ`;
      let html = "";
      for (const run of runs) {
        const maxCycles = Math.max(...run.backends.map((b) => b.cycles || 1), 1);
        const maxHbm = Math.max(...run.backends.map((b) => b.hbm_weight_bytes || 1), 1);
        const extra = run.mask_rom_extra_macs;
        const computeNote = run.same_compute
          ? (extra ? `same MACs on the fused engines (${(run.engine_macs / 1e9).toFixed(2)}B); mask-ROM +${(extra / 1e6).toFixed(0)}M from unfused attention` : "same MAC count on all passing backends")
          : "MAC counts differ - investigate";
        html += `<p class="hw-run-h">${run.token_count}-token run · oracle argmax ${run.oracle_argmax} · ${computeNote}</p>`;
        html += `<table class="hw-compare-table"><thead><tr>` +
          `<th>hardware</th><th>weights live</th><th>bit-exact</th><th>cycles</th><th>weight traffic</th><th>weight energy</th><th>argmax</th>` +
          `</tr></thead><tbody>`;
        for (const b of run.backends) {
          const verdict = b.error ? `error` : (b.passed ? `pass` : `${b.mismatch_count} mismatches`);
          const cls = b.passed ? "hw-pass" : "hw-fail";
          const cycW = b.cycles ? Math.max(4, Math.round(100 * b.cycles / maxCycles)) : 0;
          const hbmW = Math.max(4, Math.round(100 * (b.hbm_weight_bytes || 0) / maxHbm));
          const mem = b.memory === "hbm" ? "HBM stream" : "on-die ROM";
          const wE = b.energy ? fmtMJ(b.energy.weight_energy_j) : "-";
          html += `<tr>` +
            `<td>${b.label || b.backend}</td>` +
            `<td>${mem}</td>` +
            `<td class="${cls}">${verdict}</td>` +
            `<td><div class="hw-bar"><div class="hw-bar-fill" style="width:${cycW}%"></div></div><span>${fmtC(b.cycles || 0)}</span></td>` +
            `<td><div class="hw-bar"><div class="hw-bar-fill" style="width:${hbmW}%"></div></div><span>${fmtMB(b.hbm_weight_bytes || 0)}</span></td>` +
            `<td>${wE}</td>` +
            `<td>${b.argmax ?? "-"}</td>` +
            `</tr>`;
        }
        html += `</tbody></table>`;
      }
      body.innerHTML = html;
    } catch (_) {
      body.innerHTML = '<p class="muted small">hw comparison unavailable.</p>';
    }
  }

  function paintGpuBusSpec(m) {
    const per = m.hbmBwTbs / m.hbmStacks;
    const bw = `${m.hbmBwTbs.toFixed(2)} TB/s`;
    const spec = `${m.hbmStacks} × 1024b ${m.hbmName} · ${m.hbmGb} GB · ${per.toFixed(2)} TB/s/stack`;
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    set("gpuBusBw", bw);
    set("gpuBusSpec", spec);
    set("gpuPathBw", bw);
    const wlab = $("gpuWBusLab");
    if (wlab) wlab.textContent = "W";
    const hop = document.querySelector("[data-gpu-hop='hbm']");
    if (hop && !$("gpuPathBw")) hop.textContent = `src · u_hbm · ${bw}`;
  }

  async function buildGpuViz() {
    paintGpuBoardPkg();
    const lanesEl = $("gpuMacLanes");
    if (lanesEl && lanesEl.dataset.ready !== "1") {
      lanesEl.innerHTML = Array.from({ length: 16 }, (_, i) =>
        `<div class="gpu-mac-lane" data-gpu-lane="${i}"><em>×</em><span>${i}</span></div>`).join("");
      lanesEl.dataset.ready = "1";
    }
    const grid = $("gpuSmGrid");
    if (!grid) return;
    if (grid.dataset.ready === "1" && grid.dataset.mach === gpuAnim.mach) {
      const d0 = await deckEvidence("exec_models");
      if (d0 && d0.workload) gpuBuildStageSeq(gpuAnim.mach, d0.workload);
      if (gpuAnim.pendingJump) gpuJumpDecode();
      return;
    }
    const m = GPU_MACH[gpuAnim.mach];
    grid.dataset.ready = "1";
    grid.dataset.mach = gpuAnim.mach;
    grid.style.gridTemplateColumns = `repeat(${m.smGrid[0]}, 1fr)`;
    grid.innerHTML = Array.from({ length: m.smCount }, (_, i) =>
      `<div class="gpu-sm-cell" data-gpu-sm="${i}"></div>`).join("");
    const stack = $("gpuHbmStack");
    if (stack) {
      stack.style.gridTemplateColumns = `repeat(${m.hbmStacks}, minmax(0, 1fr))`;
      const per = (m.hbmBwTbs / m.hbmStacks).toFixed(2);
      stack.innerHTML = Array.from({ length: m.hbmStacks }, (_, i) =>
        `<div class="gpu-hbm-die" data-gpu-hbm="${i}"><b>HBM${i}</b><span>${per} TB/s</span></div>`).join("");
    }
    paintGpuBusSpec(m);
    const cap = $("gpuSmCap");
    if (cap) cap.textContent = `16 lanes in lockstep · H100 ${m.smCount} SMs [P] are the package, not extra MAC cores`;
    const pkgLab = $("gpuPkgLab");
    if (pkgLab) pkgLab.textContent = `${m.smCount} SMs [P]`;
    const hcap = $("gpuHbmCap");
    if (hcap) hcap.textContent = `${m.hbmStacks} ${m.hbmName} stacks [P] · ${m.hbmBwTbs.toFixed(2)} TB/s · one u_hbm port`;
    const nLab = $("gpuPathStacks");
    if (nLab) nLab.textContent = String(m.hbmStacks);
    const why = $("gpuHbmWhy");
    if (why) {
      why.textContent = `one port · stack_busy = {${m.hbmStacks}{fetch_busy}}`;
    }
    const l2c = $("gpuL2Cap");
    if (l2c && m.l2Bytes) {
      l2c.textContent = `1.19 GB unique W ≫ ${(m.l2Bytes / (1024 * 1024)).toFixed(0)} MB L2`;
    }
    const d = await deckEvidence("exec_models");
    if (!d || !d.platforms || !d.platforms.h100 || !d.workload) return;
    gpuBuildStageSeq(gpuAnim.mach, d.workload);
    renderGpuStageCostTable(gpuAnim.mach, d.workload, d.platforms[gpuAnim.mach]);
    const tileData = await deckEvidence("gpu_tile_sim");
    if (tileData) gpuTileData = tileData;
    if (gpuAnim.pendingJump) gpuJumpDecode();
    else paintGpuCycle();
    const h = d.platforms.h100.metrics;
    const b = d.platforms.b200.metrics;
    gpuPassMetrics = gpuAnim.mach === "b200" ? b : h;
    gpuEnergySplit = {
      machLabel: gpuAnim.mach === "b200" ? "B200" : "H100",
      weightJ: h.weight_energy_j,
      computeJ: Math.max(0, h.total_energy_j - h.weight_energy_j),
      totalWeightBytes: h.total_weight_bytes,
      totalMacs: d.workload.total_macs
    };
    const accrualMach = $("gpuAccrualMach");
    const accrualTotal = $("gpuAccrualTotal");
    if (accrualMach) accrualMach.textContent = gpuEnergySplit.machLabel;
    if (accrualTotal) accrualTotal.textContent = fmtMJ(h.total_energy_j);
    const accrual = $("gpuAccrual");
    if (accrual) accrual.hidden = false;
    if (window.GpuExperience) window.GpuExperience.setData({
      workload: d.workload, platform: d.platforms[gpuAnim.mach], tileData: gpuTileData
    });
    const setGpuHot = (id, value) => {
      const el = $(id);
      if (el) el.textContent = value;
    };
    const heaviest = d.workload.stages_detail
      .filter((stage) => stage.weight_bytes_per_call > 0)
      .sort((a, b2) => b2.weight_bytes_per_call - a.weight_bytes_per_call)[0];
    setGpuHot("gpuHotMacs", fmtInt(d.workload.total_macs));
    setGpuHot("gpuHotWeightBytes", fmtBytes(h.total_weight_bytes));
    setGpuHot("gpuHotWeightEnergy", fmtMJ(h.weight_energy_j));
    setGpuHot("gpuStoryMacs", fmtInt(d.workload.total_macs));
    setGpuHot("gpuStoryWeightBytes", fmtBytes(h.total_weight_bytes));
    setGpuHot("gpuStoryWeightEnergy", fmtMJ(h.weight_energy_j));
    const mlpWeightBytes = d.workload.stages_detail
      .filter((stage) => ["gate_up_proj", "down_proj"].includes(stage.stage))
      .reduce((sum, stage) => sum + stage.weight_bytes_per_call * stage.count, 0);
    const lmHead = d.workload.stages_detail.find((stage) => stage.stage === "lm_head");
    setGpuHot("gpuStoryMlpTraffic", fmtBytes(mlpWeightBytes));
    if (lmHead) setGpuHot("gpuStoryLmTraffic", fmtBytes(lmHead.weight_bytes_per_call));
    if (heaviest) {
      setGpuHot("gpuHotStage", GPU_STAGE_LABEL[heaviest.stage] || heaviest.stage);
      setGpuHot("gpuHotStageNote", `${fmtBytes(heaviest.weight_bytes_per_call)} each call · the peak HBM transfer`);
    }
    const strip = $("gpuStrip");
    if (strip) {
      strip.innerHTML =
        `<div class="gpu-stat"><span class="gpu-stat-k">weights / token</span>` +
        `<span class="gpu-stat-n" data-provenance="analytical_estimate">${fmtBytes(h.total_weight_bytes)}</span>` +
        `<span class="gpu-stat-note">BF16 · re-read from HBM every token</span></div>` +
        `<div class="gpu-stat"><span class="gpu-stat-k">HBM energy / token</span>` +
        `<span class="gpu-stat-n" data-provenance="analytical_estimate">${fmtMJ(h.weight_energy_j)}</span>` +
        `<span class="gpu-stat-note">of ${fmtMJ(h.total_energy_j)} total · the rest is compute waiting</span></div>` +
        `<div class="gpu-stat"><span class="gpu-stat-k">HBM bandwidth</span>` +
        `<span class="gpu-stat-n" data-provenance="analytical_estimate">${m.hbmBwTbs.toFixed(2)} TB/s</span>` +
        `<span class="gpu-stat-note">${m.hbmStacks} stacks · ${m.tdpW} W TDP class</span></div>` +
        `<div class="gpu-stat"><span class="gpu-stat-k">utilization @ batch 1</span>` +
        `<span class="gpu-stat-n" data-provenance="analytical_estimate">${(h.avg_utilization * 100).toFixed(2)}%</span>` +
        `<span class="gpu-stat-note">cores idle waiting for W · datasheet model</span></div>` +
        `<div class="gpu-stat"><span class="gpu-stat-k">B200 · same pattern</span>` +
        `<span class="gpu-stat-n" data-provenance="analytical_estimate">${fmtMJ(b.total_energy_j)}</span>` +
        `<span class="gpu-stat-note">/token over 8 TB/s · streaming cost survives generations</span></div>`;
    }
    document.querySelectorAll("#gpuHbmStack .gpu-hbm-die span").forEach((el) => {
      el.textContent = `${(m.hbmBwTbs / m.hbmStacks).toFixed(2)} TB/s`;
    });
  }

  function setGpuMach(mach) {
    if (!GPU_MACH[mach] || gpuAnim.mach === mach) return;
    gpuAnim.mach = mach;
    gpuAnim.cycle = 0;
    gpuAnim.dwellLeft = 0;
    gpuAnim.simdFrame = 0;
    gpuPaintPhaseKey = "";
    const on = { h100: mach === "h100", b200: mach === "b200" };
    if ($("gpuMachH100")) $("gpuMachH100").setAttribute("aria-pressed", String(on.h100));
    if ($("gpuMachB200")) $("gpuMachB200").setAttribute("aria-pressed", String(on.b200));
    buildGpuViz();
    paintGpuCycle();
  }

  // Paints the die/bus/HBM/x-badge/MAC readout for the exact global cycle
  // gpuAnim.cycle, by locating which of the 395 real stage-instances it
  // falls in (gpuFindStage) and which of that instance's act/weight/compute
  // sub-phases it is in.
  // Live two-lane energy accrual for the bottleneck aside (details mode).
  // Weight-movement energy accrues during each stage's weight-fetch phase and
  // compute energy during its compute phase, scaled so the totals land exactly
  // on the exec_models platform split — no new constants invented here.
  function paintGpuAccrual(seqObj, entry, c) {
    const wrap = $("gpuAccrual");
    if (!wrap || wrap.hidden || !gpuEnergySplit) return;
    const es = gpuEnergySplit;
    const seq = seqObj.seq;
    const idx = seq.indexOf(entry);
    if (idx < 0) return;
    let wDone = 0;
    let mDone = 0;
    for (let i = 0; i < idx; i++) {
      wDone += seq[i].weightBytesPerCall || 0;
      mDone += seq[i].macsPerCall || 0;
    }
    const lc = c - entry.start;
    const wFrac = entry.cyc.weightCycles > 0
      ? Math.min(1, Math.max(0, (lc - entry.cyc.actCycles) / entry.cyc.weightCycles))
      : 1;
    const cStart = entry.cyc.actCycles + entry.cyc.weightCycles;
    const cFrac = lc >= cStart
      ? Math.min(1, (lc - cStart) / Math.max(1, entry.cyc.computeCycles))
      : 0;
    wDone += (entry.weightBytesPerCall || 0) * wFrac;
    mDone += (entry.macsPerCall || 0) * cFrac;
    const memJ = es.totalWeightBytes > 0 ? es.weightJ * (wDone / es.totalWeightBytes) : 0;
    const macJ = es.totalMacs > 0 ? es.computeJ * (mDone / es.totalMacs) : 0;
    const totalJ = es.weightJ + es.computeJ;
    const memBar = $("gpuAccrualMem");
    const macBar = $("gpuAccrualMac");
    if (memBar) memBar.style.width = `${Math.min(100, (memJ / totalJ) * 100).toFixed(2)}%`;
    if (macBar) macBar.style.width = `${Math.min(100, (macJ / totalJ) * 100).toFixed(2)}%`;
    const memVal = $("gpuAccrualMemVal");
    const macVal = $("gpuAccrualMacVal");
    if (memVal) memVal.textContent = fmtMJ(memJ);
    if (macVal) macVal.textContent = fmtMJ(macJ);
  }

  function paintGpuCycle() {
    const grid = $("gpuSmGrid");
    const m = GPU_MACH[gpuAnim.mach];
    const seqObj = gpuSeqCache[gpuAnim.mach];
    const label = $("gpuPlayLabel");
    const slider = $("gpuPlaySlider");
    const tag = $("gpuStageTag");
    if (!grid || grid.dataset.ready !== "1") {
      paintGpuDecode();
      return;
    }
    if (!seqObj) {
      paintGpuBoardPkg();
      if (label) label.textContent = "loading real per-stage cycle table…";
      if (tag) tag.textContent = "—";
      paintGpuDecode();
      return;
    }
    const c = ((gpuAnim.cycle % seqObj.total) + seqObj.total) % seqObj.total;
    gpuAnim.cycle = c;
    const entry = gpuFindStage(seqObj, c);
    const lc = c - entry.start;
    const streamStart = entry.cyc.actCycles;
    const streamEnd = entry.cyc.actCycles + entry.cyc.weightCycles;
    const loadingX = lc < streamStart;
    const streaming = lc >= streamStart && lc < streamEnd;
    const computing = lc >= streamEnd;
    const hasWeight = entry.weightBytesPerCall > 0;
    paintGpuAccrual(seqObj, entry, c);
    if (gpuExperienceMounted && gpuAnim.story) window.GpuExperience.update({
      cycle: c, total: seqObj.total, stage: entry.stage, layer: entry.layer,
      streaming, computing, loadingX, playing: gpuAnim.playing,
      weightBytes: entry.weightBytesPerCall, macs: entry.macsPerCall
    });
    const stageName = GPU_STAGE_LABEL[entry.stage] || entry.stage;
    const layerTxt = entry.layer == null ? "" : `layer ${entry.layer + 1}/${seqObj.layers} · `;

    const dest = GPU_STAGE_DEST[entry.stage] || "mac";
    const src = GPU_STAGE_SRC[entry.stage] || (hasWeight ? "hbm" : "l1");
    const macNow = computing && dest === "mac";
    const engineNow = computing && dest === "engine";
    const kvNow = src === "kv";
    const kvWrite = macNow && entry.stage === "qkv_proj";
    const kvUpdate = engineNow && (entry.stage === "qk_norm" || entry.stage === "rope");
    const kvActive = kvNow || kvWrite || kvUpdate;
    const l2Miss = streaming && entry.weightBytesPerCall > 256 * 1024;
    const phaseKey = `${entry.stage}|${entry.layer}|${+streaming}|${+macNow}|${+engineNow}|${+kvActive}`;
    if (phaseKey === gpuPaintPhaseKey) {
      if (tag) tag.textContent = `${layerTxt}${stageName}`;
      if (label) label.textContent = `cycle ${c.toLocaleString("en-US")} / ${(seqObj.total - 1).toLocaleString("en-US")}`;
      if (slider) {
        if (Number(slider.max) !== seqObj.total - 1) slider.max = String(seqObj.total - 1);
        if (Number(slider.value) !== c) slider.value = String(c);
      }
      paintGpuPass(seqObj, entry, c);
      paintGpuBoard(entry, { loadingX, streaming, computing, macNow, engineNow });
      paintGpuLegends(entry, { streaming, computing, macNow, engineNow, kvActive, embed: entry.stage === "embedding" });
      paintGpuDecode();
      return;
    }
    gpuPaintPhaseKey = phaseKey;
    paintGpuZoom(entry);

    if (tag) tag.textContent = `${layerTxt}${stageName}`;

    const xBadge = $("gpuXBadge");
    if (xBadge) xBadge.classList.toggle("loading", loadingX || streaming);

    const frac = streaming ? (lc - streamStart + 1) / entry.cyc.weightCycles : (computing ? 1 : 0);
    const bytesSoFar = Math.round(frac * entry.weightBytesPerCall);
    const pkt = $("gpuPkt");
    if (pkt) pkt.hidden = true;
    document.querySelectorAll("[data-gpu-hbm]").forEach((el) => {
      el.classList.toggle("hot", streaming);
    });
    const lanes = document.querySelectorAll("#gpuMacLanes [data-gpu-lane]");
    const reduce = $("gpuMacReduce");
    const smGrid = $("gpuSmGrid");
    if (smGrid) smGrid.classList.toggle("is-live", !!(macNow || engineNow));
    if (macNow) {
      gpuAnim.simdFrame = (gpuAnim.simdFrame || 0) + 1;
      lanes.forEach((el) => {
        el.classList.add("hot");
        el.classList.remove("lead");
      });
      if (reduce) {
        const tot = String(entry.cyc.computeCycles);
        const cc = String(lc - streamEnd + 1).padStart(tot.length, "0");
        reduce.textContent = `group ${cc}/${tot} · all 16 lanes × this cycle · acc[31:0] += Σ xᵢWᵢ`;
      }
    } else {
      gpuAnim.simdFrame = 0;
      lanes.forEach((el) => el.classList.remove("hot", "lead"));
      if (reduce) reduce.textContent = "all 16 lanes idle · valid_i waits on fetch_done";
    }
    document.querySelectorAll("#gpuGemmPipe [data-pipe]").forEach((el) => {
      const p = el.getAttribute("data-pipe");
      el.classList.toggle("on",
        (p === "fetch" && streaming) ||
        (p === "issue" && macNow) ||
        (p === "mac" && macNow) ||
        (p === "wb" && macNow));
    });
    const pkgFill = $("gpuPkgFill");
    const pkgUtil = $("gpuPkgUtil");
    const util = gpuPassMetrics && gpuPassMetrics.avg_utilization;
    if (pkgFill && util != null) pkgFill.style.width = `${Math.max(0.6, util * 100)}%`;
    if (pkgUtil && util != null) pkgUtil.textContent = `${(util * 100).toFixed(2)}% util @ batch 1`;
    const bus = $("gpuBus");
    if (bus) bus.classList.toggle("live", streaming);

    const hotBlocks = new Set(GPU_STAGE_BLOCKS[entry.stage] || ["fsm"]);
    if (streaming) {
      hotBlocks.add("hbm");
      hotBlocks.add("l1");
      hotBlocks.delete("mac");
    } else {
      hotBlocks.delete("hbm");
    }
    if (macNow) hotBlocks.add("mac");
    if (kvActive) hotBlocks.add("kv");
    if (computing && entry.stage === "lm_head") hotBlocks.add("logits");
    const engRow = document.querySelector("#gpuViz .sch-row.eng");
    if (engRow) engRow.classList.toggle("is-later", !engineNow);
    document.querySelectorAll("[data-gpu-blk]").forEach((el) => {
      const id = el.getAttribute("data-gpu-blk");
      el.classList.toggle("active", hotBlocks.has(id) && id !== "l2");
      el.classList.toggle("miss", id === "l2" && l2Miss);
    });
    document.querySelectorAll("[data-gpu-wire]").forEach((el) => {
      const need = el.getAttribute("data-gpu-wire");
      el.classList.toggle("live",
        (need === "hbm" && streaming) ||
        (need === "l2" && l2Miss) ||
        (need === "issue" && computing) ||
        (need === "mac" && macNow) ||
        (need === "eng" && engineNow) ||
        (need === "cmd" && entry.stage === "embedding"));
    });
    document.querySelectorAll("[data-gpu-hop]").forEach((el) => {
      const hop = el.getAttribute("data-gpu-hop");
      const on = (hop === "hbm" && streaming) ||
        (hop === "l2" && l2Miss) ||
        (hop === "l1" && (streaming || computing || kvActive)) ||
        (hop === "dst" && computing);
      el.classList.toggle("on", on);
    });
    const dstHop = $("gpuPathDstHop");
    if (dstHop) {
      dstHop.textContent = dest === "engine" ? "dst · engine" : (kvActive && !streaming ? "dst · u_mac ← u_kv" : "dst · u_mac");
    }
    const pathWhy = $("gpuPathWhy");
    if (pathWhy) {
      if (streaming) {
        pathWhy.textContent = l2Miss
          ? `u_hbm fetch_busy. stack_busy={${m.hbmStacks}{1}}. L2 miss (${fmtBytes(entry.weightBytesPerCall)} unique W ≫ ${(m.l2Bytes / (1024 * 1024)).toFixed(0)} MB). u_mac waits on fetch_done.`
          : `u_hbm fetch_busy for ${fmtBytes(entry.weightBytesPerCall)} (fits in L2; γ-scale). stack_busy={${m.hbmStacks}{1}}.`;
      } else if (macNow && src === "kv") {
        pathWhy.textContent = "src: u_kv (on-die). dst: u_mac. u_hbm idle — this op has zero weight bytes.";
      } else if (macNow && kvWrite) {
        pathWhy.textContent = "QKV_MAC done — K and V rows commit to u_kv (on-die). Next beats read them, not HBM.";
      } else if (macNow) {
        pathWhy.textContent = "fetch_done. dst: u_mac (qwen_bf16_mac_array, same RTL as the ASIC). u_hbm idle until the next GEMM.";
      } else if (engineNow && kvUpdate) {
        pathWhy.textContent = "u_norm / u_rope updates Q/K; K row in u_kv tracks the rotated head.";
      } else if (engineNow) {
        pathWhy.textContent = "u_hbm idle. dst: u_norm / u_softmax / u_rope / u_swiglu on x in scratch. These are the same engines as the mask-ROM chip, not u_mac.";
      } else if (kvActive) {
        pathWhy.textContent = "src: u_kv. Zero HBM. gpu_kv_cache.sv holds K/V for this layer.";
      } else {
        pathWhy.textContent = "Owned RTL: gpu_hbm_bw_model is one fetch port. H100 stacks are the package of that port [P], not a DRAM map we do not have.";
      }
    }
    const fsm = $("gpuFsmName");
    if (fsm) {
      fsm.textContent = streaming ? "QKV_FETCH" : (macNow ? "QKV_MAC" : (kvWrite ? "KV_WRITE" : (kvActive ? "KV_READ" : (engineNow ? "ENGINE" : "LOAD_ACT"))));
    }

    const bcap = $("gpuBusCap");
    if (bcap) {
      bcap.textContent = hasWeight
        ? (streaming
          ? `fetch_busy · ${fmtBytes(bytesSoFar)} / ${fmtBytes(entry.weightBytesPerCall)}`
          : `idle · next ${fmtBytes(entry.weightBytesPerCall)}`)
        : (kvActive ? (kvWrite ? "write K,V" : "read u_kv") : "idle · no W");
    }
    const l2n = $("gpuL2Name");
    if (l2n) l2n.textContent = l2Miss ? "MISS · unique W" : (streaming ? "small γ · could hit" : "idle · not in gpu_stream_core");
    const l1n = $("gpuL1Name");
    if (l1n) {
      l1n.textContent = streaming
        ? `x ready · W ${fmtBytes(bytesSoFar)} · y idle (MAC blocked)`
        : (macNow ? "issue x,W · 16 lanes × · y reducing" : (kvActive ? (kvWrite ? "K,V → u_kv" : "Q/K/V from u_kv") : (loadingX ? "x landing in scratch" : "x in scratch")));
    }
    const opXLab = $("gpuOpXLab");
    const opWLab = $("gpuOpWLab");
    const opYLab = $("gpuOpYLab");
    const opX = $("gpuOpX");
    const opW = $("gpuOpW");
    const opY = $("gpuOpY");
    const macGroup = macNow ? (lc - streamEnd + 1) : 0;
    const macDone = macNow && macGroup >= entry.cyc.computeCycles;
    if (opX) {
      opX.classList.toggle("on", !loadingX && (streaming || macNow || engineNow || kvActive));
      opX.classList.toggle("wait", loadingX);
    }
    if (opXLab) {
      opXLab.textContent = loadingX
        ? "loading"
        : (entry.stage === "embedding" ? "← E[token]" : "activation");
    }
    if (opW) {
      opW.classList.toggle("on", macNow && (hasWeight || kvActive));
      opW.classList.toggle("wait", streaming);
    }
    if (opWLab) {
      opWLab.textContent = kvActive
        ? (kvWrite ? "→ u_kv" : "← u_kv")
        : (streaming ? fmtBytes(bytesSoFar) : (macNow && hasWeight ? "in RF" : "idle"));
    }
    if (opY) {
      opY.classList.toggle("wait", macNow && !macDone);
      opY.classList.toggle("on", macDone || engineNow);
    }
    if (opYLab) {
      opYLab.textContent = macNow
        ? (macDone ? "← acc" : "reducing")
        : (engineNow ? "next op" : "idle");
    }
    const portAct = $("gpuPortAct");
    const portWgt = $("gpuPortWgt");
    const portAcc = $("gpuPortAcc");
    const portHost = (el) => (el && el.parentElement && el.parentElement.classList.contains("gpu-port") ? el.parentElement : el);
    if (portAct) {
      portHost(portAct).classList.toggle("on", macNow);
      portHost(portAct).classList.toggle("ready", streaming && hasWeight);
      portAct.textContent = "activation_i ← x";
    }
    if (portWgt) {
      portHost(portWgt).classList.toggle("on", macNow);
      portHost(portWgt).classList.toggle("blocked", streaming && hasWeight);
      portWgt.textContent = kvActive
        ? (kvWrite ? "K,V → u_kv" : "weight_i ← u_kv")
        : (streaming && hasWeight ? "weight_i held" : "weight_i ← W");
    }
    if (portAcc) {
      portHost(portAcc).classList.toggle("on", macNow);
      portAcc.textContent = macNow ? "acc += x·W" : "acc[31:0]";
    }
    const mac = $("gpuMac");
    if (mac) {
      if (macNow) {
        const tot = String(entry.cyc.computeCycles);
        const cc = String(lc - streamEnd + 1).padStart(tot.length, "0");
        mac.textContent = `valid_i · group ${cc}/${tot}: ${entry.macsPerCall.toLocaleString("en-US")} MACs · 16-lane BF16×BF16 → FP32`;
      } else if (streaming) {
        mac.textContent = `valid_i = 0 · x is ready, W ${fmtBytes(bytesSoFar)} / ${fmtBytes(entry.weightBytesPerCall)} still in u_hbm`;
      } else {
        mac.textContent = "idle — activation_i / weight_i not valid this stage";
      }
    }
    const kvCap = $("gpuKvCap");
    if (kvCap) {
      kvCap.textContent = kvWrite
        ? "active: QKV_MAC writes K and V into on-die cache · zero HBM"
        : (kvActive
          ? `active: ${stageName} reads K/V from u_kv · zero HBM`
          : "K/V land here after QKV_MAC; attn QKᵀ and P×V read back");
    }
    const xcap = $("gpuXCap");
    if (xcap) {
      if (streaming) xcap.textContent = "Step 1: W beats land in RF. Step 2–3 do not start. Engines stay idle.";
      else if (macNow && kvActive) xcap.textContent = "Issue Q and K/V to the 16 lanes. No HBM, no engine.";
      else if (macNow) xcap.textContent = "Steps 2–4: RF → activation_i / weight_i → 16-lane MAC → y. No engine on this path.";
      else if (engineNow) xcap.textContent = "A later decoder stage: engine on y/x in scratch. W fetch and MAC already finished.";
      else if (loadingX) xcap.textContent = "x (from the token IDs after embedding, or the residual) landing in RF.";
      else xcap.textContent = `x in scratch · ${fmtBytes(entry.actBytesPerCall)}`;
    }
    const note = $("gpuNote");
    if (note) {
      const pos = `cycle ${(c + 1).toLocaleString("en-US")}/${seqObj.total.toLocaleString("en-US")} (${((c + 1) / seqObj.total * 100).toFixed(1)}%)`;
      if (computing) {
        note.textContent = `${pos}: ${layerTxt}${stageName} — COMPUTE, ${entry.cyc.computeCycles} cycle${entry.cyc.computeCycles === 1 ? "" : "s"} · ${m.label}`;
      } else if (streaming) {
        note.textContent = `${pos}: ${layerTxt}${stageName} — FETCH, ${(frac * 100).toFixed(1)}% of ${fmtBytes(entry.weightBytesPerCall)} · ${m.label}`;
      } else {
        note.textContent = `${pos}: ${layerTxt}${stageName} — loading activation · ${m.label}`;
      }
    }
    paintGpuNotes(m, entry, seqObj, { loadingX, streaming, computing, frac, bytesSoFar, c, dest, src, macNow, engineNow, kvNow: kvActive, kvWrite, kvRead: kvNow });
    if (label) label.textContent = `cycle ${c.toLocaleString("en-US")} / ${(seqObj.total - 1).toLocaleString("en-US")}`;
    if (slider) {
      if (Number(slider.max) !== seqObj.total - 1) slider.max = String(seqObj.total - 1);
      if (Number(slider.value) !== c) slider.value = String(c);
    }
    paintGpuNet(entry);
    if (gpuAnim.flow) paintGpuFlow(entry, { loadingX, streaming, computing, macNow, engineNow });
    paintGpuBoard(entry, { loadingX, streaming, computing, macNow, engineNow });
    paintGpuLegends(entry, { streaming, computing, macNow, engineNow, kvActive, embed: entry.stage === "embedding" });
    paintGpuStageCostTable(entry);
    paintGpuPass(seqObj, entry, c);
    paintGpuDecode();
  }

  function paintGpuLegends(entry, phase) {
    const embed = !!(phase && phase.embed);
    const streaming = !!(phase && phase.streaming);
    const macNow = !!(phase && phase.macNow);
    const engineNow = !!(phase && phase.engineNow);
    const kvActive = !!(phase && phase.kvActive);
    const computing = entry && (phase && phase.computing);
    const prLive = {
      1: embed || gpuTokenIds.length > 0,
      2: embed,
      3: streaming,
      4: embed || streaming,
      5: macNow || engineNow || computing
    };
    document.querySelectorAll("#gpuPrLeg [data-leg]").forEach((el) => {
      el.classList.toggle("on", !!prLive[el.getAttribute("data-leg")]);
    });
    const rtlLive = {
      host: embed,
      hbm: streaming,
      kv: kvActive,
      l2: streaming,
      mac: macNow,
      eng: engineNow
    };
    document.querySelectorAll("#gpuRtlLeg [data-leg]").forEach((el) => {
      el.classList.toggle("on", !!rtlLive[el.getAttribute("data-leg")]);
    });
  }

  function gpuStageTableOrder() {
    return ["embedding"].concat(GPU_STAGE_ORDER, ["final_rmsnorm", "lm_head"]);
  }

  const GPU_STAGE_TBL_SHORT = {
    embedding: "embed",
    qkv_proj: "QKV",
    o_proj: "O proj",
    gate_up_proj: "gate/up",
    down_proj: "down",
    lm_head: "lm_head"
  };
  const GPU_STAGE_TBL_N = 6;

  function gpuUserMetrics(mach, metrics, seqTotal) {
    const cycles = (metrics && metrics.total_cycles) || seqTotal || 0;
    const lat = cycles / GPU_CLOCK_HZ;
    const tps = lat > 0 ? 1 / lat : 0;
    const wMj = (metrics.weight_energy_j || 0) * 1e3;
    const cMj = (metrics.compute_energy_j || 0) * 1e3;
    const aMj = (metrics.act_energy_j || 0) * 1e3;
    const totalMj = (metrics.total_energy_j || 0) * 1e3;
    const latMs = lat * 1e3;
    const latStr = latMs >= 1 ? latMs.toFixed(2) + " ms" : (latMs * 1e3).toFixed(0) + " µs";
    return {
      cycles,
      tps,
      totalMj,
      tpsEq: `1 ÷ (${fmtInt(cycles)} cyc ÷ 1.5 GHz) = ${latStr} → ${fmtToks(tps)} tok/s`,
      energyEq: `${wMj.toFixed(1)} mJ HBM + ${cMj.toFixed(1)} mJ compute + ${aMj.toFixed(2)} mJ act = ${totalMj.toFixed(1)} mJ`
    };
  }

  function renderGpuStageCostTable(mach, workload, platform) {
    const tbody = $("gpuStageCostRows");
    const machLab = $("gpuStageCostMach");
    const tpsEl = $("gpuStageTps");
    const tpsEq = $("gpuStageTpsEq");
    const energyEl = $("gpuStageEnergy");
    const energyEq = $("gpuStageEnergyEq");
    if (!tbody || !workload || !workload.stages_detail) return;
    const m = GPU_MACH[mach];
    if (!m) return;
    const metrics = platform && platform.metrics;
    if (machLab) {
      machLab.textContent = `${mach.toUpperCase()} · top GEMM stages · sorted by cycles, then W fetch`;
    }
    const byName = {};
    for (const s of workload.stages_detail) byName[s.stage] = s;
    const rows = [];
    for (const name of gpuStageTableOrder()) {
      const s = byName[name];
      if (!s || s.weight_bytes_per_call <= 0) continue;
      const cyc = gpuStageCycles(m, s);
      rows.push({
        name,
        s,
        total: cyc.total,
        wBytes: s.weight_bytes_per_call
      });
    }
    rows.sort((a, b) => b.total - a.total || b.wBytes - a.wBytes);
    const ranked = rows.slice(0, GPU_STAGE_TBL_N);
    const maxCyc = ranked.length ? ranked[0].total : 0;
    tbody.innerHTML = ranked.map((r) => {
      const label = GPU_STAGE_TBL_SHORT[r.name] || GPU_STAGE_LABEL[r.name] || r.name;
      const heavy = r.total === maxCyc ? " is-heavy" : "";
      return `<tr data-gpu-stage-cost="${r.name}" class="${heavy.trim()}">` +
        `<td>${label}</td>` +
        `<td>${fmtBytes(r.wBytes)}</td>` +
        `<td data-provenance="analytical_estimate">${fmtCyc(r.total)}</td></tr>`;
    }).join("");
    const seqObj = gpuSeqCache[mach];
    const tokenTotal = seqObj ? seqObj.total : 0;
    const user = gpuUserMetrics(mach, metrics, tokenTotal);
    if (tpsEl) tpsEl.textContent = fmtToks(user.tps);
    if (tpsEq) tpsEq.textContent = user.tpsEq;
    if (energyEl) energyEl.textContent = fmtMJ((metrics && metrics.total_energy_j) || user.totalMj / 1e3);
    if (energyEq) energyEq.textContent = user.energyEq;
  }

  function paintGpuStageCostTable(entry) {
    document.querySelectorAll("#gpuStageCostRows [data-gpu-stage-cost]").forEach((tr) => {
      tr.classList.toggle("is-now", tr.getAttribute("data-gpu-stage-cost") === entry.stage);
    });
  }

  function gpuStageEqText(stage) {
    const ops = GPU_STAGE_OPS[stage] || [];
    if (!ops.length) return "—";
    return ops.map((op) => OP_EQ[op] || op).join("  ·  ");
  }

  function gpuOpEl(op) {
    if (op === "embedding" || op === "final_norm" || op === "logits" || op === "argmax") {
      return $("g2g-" + op);
    }
    return $("g2-" + op);
  }

  function fillHbmRing(ids, n, name) {
    const top = $(ids.top);
    if (!top) return;
    if (top.dataset.mach === gpuAnim.mach && top.childElementCount) return;
    const nTop = 3;
    const nLeft = 1;
    const nRight = 1;
    const nBot = Math.max(0, n - 5);
    let i = 0;
    const mk = () => {
      const idx = i++;
      const pin = idx === 0 ? `<b class="gpu-num">3</b>` : "";
      return `<div class="gpu-pkg-stack" data-board-hbm="${idx}">${pin}<b>HBM${idx}</b><span>${name}</span></div>`;
    };
    const fill = (elId, count) => {
      const el = $(elId);
      if (!el) return;
      el.dataset.mach = gpuAnim.mach;
      el.innerHTML = Array.from({ length: count }, mk).join("");
    };
    fill(ids.top, nTop);
    fill(ids.left, nLeft);
    fill(ids.right, nRight);
    fill(ids.bot, nBot);
  }

  function paintGpuBoardPkg() {
    const m = GPU_MACH[gpuAnim.mach];
    if (!m) return;
    fillHbmRing({ top: "gpuFloorTop", left: "gpuFloorLeft", right: "gpuFloorRight", bot: "gpuFloorBot" }, m.hbmStacks, m.hbmName);
    const b200 = gpuAnim.mach === "b200";
    const paneK = $("gpuBoardPaneK");
    if (paneK) paneK.textContent = b200 ? "Package architecture · B200 schematic" : "Package architecture · H100 schematic";
    const dieK = $("gpuBoardDieK");
    if (dieK) dieK.textContent = b200 ? "Blackwell die [P]" : "Hopper die · 814 mm² [P]";
    const l2 = $("gpuDieL2");
    if (l2) l2.textContent = b200 ? "L2 128 MB · macro" : "L2 50 MB · macro";
    const sm = $("gpuDieSm");
    if (sm) sm.innerHTML = `<b class="gpu-num">5</b>${m.smCount} SMs · std-cell`;
    const call3 = $("gpuCall3t");
    if (call3) call3.textContent = b200 ? "HBM3e stacks" : "HBM3 stacks";
    paintGpuBoardBits();
  }

  function paintGpuBoardBits() {
    const last = gpuTokenIds.length ? gpuTokenIds[gpuTokenIds.length - 1] : null;
    const hex = last == null ? "—" : "0x" + last.toString(16).toUpperCase().padStart(5, "0");
    const idEl = $("gpuBitId");
    const hexEl = $("gpuBitHex");
    if (idEl) idEl.textContent = last == null ? "—" : String(last);
    if (hexEl) hexEl.textContent = hex;
    const host = $("gpuBoardHost");
    if (host) {
      host.textContent = last == null
        ? "token IDs"
        : `${gpuTokenIds.length} ids · last ${last}`;
    }
  }

  function paintGpuBoard(entry, phase) {
    const embed = !!(entry && entry.stage === "embedding");
    const streaming = !!(phase && phase.streaming);
    const macNow = !!(phase && phase.macNow);
    const engineNow = !!(phase && phase.engineNow);
    const computing = !!(phase && phase.computing);
    const host = document.querySelector("#gpuBoard [data-board='host']");
    const pcie = $("gpuBoardPcie");
    if (host) host.classList.toggle("active", embed);
    if (pcie) pcie.classList.toggle("live", embed);
    document.querySelectorAll("#gpuBoard [data-board-hbm]").forEach((el) => {
      el.classList.toggle("hot", streaming);
    });
    const dieOn = {
      phy: embed || streaming,
      l2: streaming,
      mc: streaming,
      sm: macNow || engineNow || computing
    };
    document.querySelectorAll("#gpuBoard [data-die]").forEach((el) => {
      el.classList.toggle("on", !!dieOn[el.getAttribute("data-die")]);
    });
    const rdl = $("gpuBoardPhy");
    if (rdl) rdl.classList.toggle("live", streaming);
    const addr = $("gpuBitAddr");
    if (addr) {
      addr.textContent = streaming && !embed ? "addr → this W tile" : "id → E[id]";
    }
  }

  function gpuHopHtml(h) {
    const [a, b, c] = h.cells;
    const b1 = h.kind === "gemm" ? "fetch" : (h.kind === "add" ? "+" : "x");
    const b2 = h.kind === "gemm" ? "issue" : "y";
    const cell = (m) =>
      `<span class="blk ${m.cls}"><span class="blk-k">${m.k}</span><strong>${m.v}</strong></span>`;
    return `<button type="button" class="gfs-hop ${h.kind}" data-gpu-flow="${h.id}">` +
      `<span class="gfs-spine"><i class="gfs-pkt" hidden>x</i></span>` +
      `<span class="gfs-row">` +
      `${cell(a)}<span class="bus h" data-sub="${h.kind === "gemm" ? "fetch" : "go"}"><span>${b1}</span></span>` +
      `${cell(b)}<span class="bus h" data-sub="${h.kind === "gemm" ? "issue" : "out"}"><span>${b2}</span></span>` +
      `${cell(c)}</span>` +
      `<span class="gfs-lab">${h.title}</span></button>`;
  }

  function buildGpuFlow() {
    const root = $("gpuFlow");
    const layers = $("gpuFlowLayers");
    if (!root) return;
    if (root.dataset.ready === "2") return;
    const head = GPU_FLOW.filter((h) => h.where === "head").map(gpuHopHtml).join("");
    const layer = GPU_FLOW.filter((h) => h.where === "layer").map(gpuHopHtml).join("");
    const tail = GPU_FLOW.filter((h) => h.where === "tail").map(gpuHopHtml).join("");
    root.innerHTML =
      `<div class="gfs-sec">${head}</div>` +
      `<div class="gfs-band">` +
      `<p class="gfs-band-k">decoder <span id="gpuFlowL">L··</span> · same circuit ×28 · new W each layer</p>` +
      `${layer}</div>` +
      `<div class="gfs-sec">${tail}</div>`;
    root.dataset.ready = "2";
    root.querySelectorAll("[data-gpu-flow]").forEach((el) => {
      el.addEventListener("click", () => gpuJumpFlow(el.getAttribute("data-gpu-flow")));
    });
    if (layers && layers.dataset.ready !== "1") {
      layers.innerHTML = Array.from({ length: 28 }, (_, i) =>
        `<button type="button" data-gpu-flow-layer="${i}">L${String(i).padStart(2, "0")}</button>`).join("");
      layers.dataset.ready = "1";
      layers.querySelectorAll("[data-gpu-flow-layer]").forEach((el) => {
        el.addEventListener("click", () => gpuJumpLayer(Number(el.getAttribute("data-gpu-flow-layer"))));
      });
    }
  }

  function gpuJumpFlow(stageId) {
    const seqObj = gpuSeqCache[gpuAnim.mach];
    if (!seqObj) return;
    gpuTogglePlay(false);
    gpuAnim.dwellLeft = 0;
    const hop = GPU_FLOW.find((h) => h.id === stageId);
    const cur = gpuFindStage(seqObj, gpuAnim.cycle);
    const layer = hop && hop.where === "layer" ? (cur.layer != null ? cur.layer : 0) : null;
    const hit = seqObj.seq.find((s) => s.stage === stageId && s.layer === layer);
    if (hit) gpuSetCycle(hit.start);
  }

  function gpuJumpLayer(L) {
    const seqObj = gpuSeqCache[gpuAnim.mach];
    if (!seqObj) return;
    gpuTogglePlay(false);
    gpuAnim.dwellLeft = 0;
    const cur = gpuFindStage(seqObj, gpuAnim.cycle);
    const stage = (cur.layer != null) ? cur.stage : "input_rmsnorm";
    const hit = seqObj.seq.find((s) => s.stage === stage && s.layer === L)
      || seqObj.seq.find((s) => s.layer === L);
    if (hit) gpuSetCycle(hit.start);
  }

  const GPU_LEFT_TABS = {
    board: { pane: "gpuBoardWrap", btn: "gpuTabBoardBtn" },
    net: { pane: "gpuNetPane", btn: "gpuTabNetBtn" },
    notes: { pane: "gpuNotesPane", btn: "gpuTabNotesBtn" },
  };
  function gpuSetLeftTab(tab) {
    if (!GPU_LEFT_TABS[tab]) return;
    Object.entries(GPU_LEFT_TABS).forEach(([key, ids]) => {
      const pane = $(ids.pane);
      const btn = $(ids.btn);
      const on = key === tab;
      if (pane) pane.hidden = !on;
      if (btn) btn.setAttribute("aria-pressed", String(on));
    });
  }

  function gpuSetStory(show) {
    const slide = document.querySelector(".slide-gpu");
    const story = $("gpuStory");
    const tri = $("gpuTri");
    const flow = $("gpuFlowPane");
    const btn = $("gpuFlowBtn");
    gpuAnim.story = show;
    gpuAnim.flow = !show;
    if (story) story.hidden = gpuExperienceMounted || !show;
    const experience = $("gpuExperience");
    if (experience) experience.hidden = !show;
    if (tri) tri.hidden = show;
    if (flow) flow.hidden = show;
    const bottleneck = document.querySelector(".gpu-bottleneck");
    if (bottleneck) bottleneck.hidden = show || !gpuEnergySplit;
    if (slide) {
      slide.classList.toggle("is-flow", !show);
      slide.classList.toggle("is-simple", show);
    }
    if (btn) {
      btn.setAttribute("aria-pressed", String(!show));
      btn.textContent = show ? "Inspect RTL evidence" : "Back to 3D chip";
    }
    if (!show) {
      buildGpuFlow();
      paintGpuCycle();
    }
  }

  function gpuToggleFlow(force) {
    gpuSetStory(force != null ? !force : !gpuAnim.story);
  }

  function paintGpuFlow(entry, phase) {
    const root = $("gpuFlow");
    if (!root || root.dataset.ready !== "2") return;
    const order = GPU_FLOW.map((h) => h.id);
    const idx = order.indexOf(entry.stage);
    const pktTxt = phase.streaming ? "W" : (phase.macNow ? "×" : (phase.engineNow ? "fn" : "x"));
    root.querySelectorAll("[data-gpu-flow]").forEach((el) => {
      const id = el.getAttribute("data-gpu-flow");
      const i = order.indexOf(id);
      const here = id === entry.stage;
      const hop = GPU_FLOW.find((h) => h.id === id);
      el.classList.toggle("now", here);
      el.classList.toggle("done", idx >= 0 && i >= 0 && i < idx);
      const pkt = el.querySelector(".gfs-pkt");
      if (pkt) {
        pkt.hidden = !here;
        if (here) pkt.textContent = pktTxt;
      }
      el.querySelectorAll("[data-sub]").forEach((sub) => {
        const p = sub.getAttribute("data-sub");
        const on = here && hop && (
          (hop.kind === "gemm" && p === "fetch" && phase.streaming) ||
          (hop.kind === "gemm" && p === "issue" && phase.macNow) ||
          (hop.kind !== "gemm" && here && (phase.engineNow || !phase.streaming))
        );
        sub.classList.toggle("live", !!on);
      });
    });
    const sub = $("gpuFlowSub");
    const hop = GPU_FLOW.find((h) => h.id === entry.stage);
    if (sub && hop) {
      const layerTxt = entry.layer == null ? "global" : `L${String(entry.layer).padStart(2, "0")}/28`;
      const beat = phase.streaming ? "fetch W into RF" : (phase.macNow ? "issue x,W · 16 lanes ×" : (phase.engineNow ? "engine on y in RF" : hop.title));
      sub.textContent = `${layerTxt} · ${hop.title} · ${beat}`;
    }
    const fl = $("gpuFlowL");
    if (fl) fl.textContent = entry.layer == null ? "—" : `L${String(entry.layer).padStart(2, "0")}`;
    document.querySelectorAll("[data-gpu-flow-layer]").forEach((el) => {
      el.classList.toggle("on", entry.layer != null && Number(el.getAttribute("data-gpu-flow-layer")) === entry.layer);
    });
    const key = `${entry.stage}:${entry.layer}`;
    if (gpuAnim.flowHop !== key) {
      gpuAnim.flowHop = key;
      // Do not scroll the document while the animation advances: it makes
      // the page jump past the title and the playback controls.
    }
  }

  function paintGpuNet(entry) {
    const root = $("gpuNet");
    if (!root || root.dataset.ready !== "4") return;
    const key = `${entry.stage}|${entry.layer}`;
    if (root.dataset.paintKey === key) return;
    root.dataset.paintKey = key;
    root.querySelectorAll(".gpu-net-chip.active, .gpu-circ.active, .gpu-net-ldot.active").forEach((el) => {
      el.classList.remove("active", "on-path");
    });
    root.querySelectorAll("[data-gpu-net-stage]").forEach((el) => {
      el.classList.toggle("active", el.getAttribute("data-gpu-net-stage") === entry.stage);
    });
    const ops = GPU_STAGE_OPS[entry.stage] || [];
    ops.forEach((op) => {
      const el = gpuOpEl(op);
      if (el) el.classList.add("active");
    });
    root.querySelectorAll("[data-gpu-layer]").forEach((el) => {
      el.classList.toggle("active", entry.layer != null && Number(el.getAttribute("data-gpu-layer")) === entry.layer);
    });
    const graph = $("gpuDecoderGraph");
    if (graph) graph.classList.toggle("current", entry.layer != null);
    const layerKey = entry.layer == null ? "global" : `L${String(entry.layer).padStart(2, "0")}`;
    const title = $("gpuDagTitle");
    if (title) {
      title.textContent = entry.layer == null
        ? "global ops"
        : `${layerKey} · decoder`;
    }
    const nowL = $("gpuNetNowLayer");
    const nowO = $("gpuNetNowOp");
    if (nowL) nowL.textContent = layerKey;
    if (nowO) nowO.textContent = GPU_STAGE_LABEL[entry.stage] || entry.stage;
    const eqEl = $("gpuNetEq");
    if (eqEl) {
      const eq = gpuStageEqText(entry.stage);
      eqEl.textContent = eq;
      eqEl.title = eq;
      eqEl.classList.toggle("is-live", eq !== "—");
    }
    const clsNow = OP_CLASS[(GPU_STAGE_OPS[entry.stage] || [])[0]] || "";
    const pane = root.closest(".gpu-pane-net");
    (pane || root).querySelectorAll(".gpu-net-legend [data-cls]").forEach((li) => {
      li.classList.toggle("is-now", li.getAttribute("data-cls") === clsNow);
    });
  }

  function paintGpuNotes(m, entry, seqObj, phase) {
    const stageName = GPU_STAGE_LABEL[entry.stage] || entry.stage;
    const now = $("gpuNotesNow");
    if (now) {
      now.textContent = entry.layer == null
        ? stageName
        : `Layer ${entry.layer + 1}/${seqObj.layers} · ${stageName}`;
    }
    const dest = GPU_STAGE_DEST[entry.stage] || "mac";
    const src = GPU_STAGE_SRC[entry.stage] || "hbm";
    const ph = $("gpuNotesPhase");
    if (ph) {
      if (phase.streaming) ph.textContent = "QKV_FETCH · u_hbm fetch_busy";
      else if (phase.macNow) ph.textContent = "QKV_MAC · u_mac";
      else if (phase.kvNow && phase.kvWrite) ph.textContent = "KV_WRITE · u_kv after QKV_MAC";
      else if (phase.kvNow) ph.textContent = "KV_READ · u_kv → u_mac, zero HBM";
      else if (phase.engineNow) ph.textContent = "ENGINE · u_norm / u_softmax / u_rope / u_swiglu";
      else ph.textContent = "LOAD_ACT · SM scratch";
    }
    const what = $("gpuNotesWhat");
    if (what) what.textContent = GPU_STAGE_NOTES[entry.stage] || stageName;
    const path = $("gpuNotesPath");
    if (path) {
      if (phase.streaming) {
        path.textContent = `1 fetch: W → RF. x already here. Nothing else runs — not RoPE, not SwiGLU, not MAC. u_mac.valid_i = 0 until fetch_done.`;
      } else if (phase.macNow && src === "kv") {
        path.textContent = `2–4: issue Q and K/V into the 16 lanes, all fire, y back to scratch. Engines idle.`;
      } else if (phase.macNow) {
        path.textContent = `2 issue x,W from RF → 3 all 16 lanes × in lockstep, reduce k in groups → 4 y into scratch. That is the whole GEMM. Engines are later graph nodes, not this path.`;
      } else if (phase.engineNow || dest === "engine") {
        path.textContent = `This is a later decoder stage on y already in scratch (u_norm / u_softmax / u_rope / u_swiglu). It is not inserted between weight fetch and MAC.`;
      } else if (src === "kv") {
        path.textContent = `src: u_kv. The MAC operands are Q and K/V already on-die. Zero HBM.`;
      } else {
        path.textContent = `Host sent token IDs. Embedding (or the last residual) is writing x into SM scratch. Next GEMM is fetch W, then issue, then 16-lane MAC.`;
      }
    }
    const cost = $("gpuNotesCost");
    if (cost) {
      const wc = entry.cyc.weightCycles;
      const cc = entry.cyc.computeCycles;
      if (entry.weightBytesPerCall > 0 && wc > 0) {
        const ratio = wc / Math.max(1, cc);
        const landed = phase.streaming
          ? `${(phase.frac * 100).toFixed(0)}% of ${fmtBytes(entry.weightBytesPerCall)} landed`
          : (phase.computing ? `${fmtBytes(entry.weightBytesPerCall)} already in L1` : `about to fetch ${fmtBytes(entry.weightBytesPerCall)}`);
        cost.textContent = `${wc.toLocaleString("en-US")} cycles to stream vs ${cc.toLocaleString("en-US")} to compute (${ratio.toFixed(1)}×). ${landed}.`;
      } else {
        cost.textContent = `No HBM weight walk. ${cc.toLocaleString("en-US")} compute cycle${cc === 1 ? "" : "s"} on the resident activation (${fmtBytes(entry.actBytesPerCall)}).`;
      }
    }
    const bill = $("gpuNotesBill");
    if (bill) bill.textContent = m.note;
  }

  function paintGpuPass(seqObj, entry, c) {
    const chart = $("gpuPassChart");
    const seq = seqObj.seq;
    let macs = 0;
    let wbytes = 0;
    const doneByStage = {};
    for (const s of seq) {
      if (s.start + s.cyc.total <= c + 1) {
        macs += s.macsPerCall;
        wbytes += s.weightBytesPerCall;
        doneByStage[s.stage] = (doneByStage[s.stage] || 0) + s.macsPerCall;
      } else if (s === entry) {
        const lc = c - entry.start;
        const frac = Math.min(1, (lc + 1) / Math.max(1, entry.cyc.total));
        macs += frac * s.macsPerCall;
        const streamStart = entry.cyc.actCycles;
        const streamEnd = entry.cyc.actCycles + entry.cyc.weightCycles;
        let wfrac = 0;
        if (lc >= streamEnd) wfrac = 1;
        else if (lc >= streamStart && entry.cyc.weightCycles) {
          wfrac = (lc - streamStart + 1) / entry.cyc.weightCycles;
        }
        wbytes += wfrac * s.weightBytesPerCall;
        doneByStage[s.stage] = (doneByStage[s.stage] || 0) + frac * s.macsPerCall;
        break;
      }
    }
    const m = gpuPassMetrics;
    const energy = m
      ? (wbytes / Math.max(1, m.total_weight_bytes)) * m.weight_energy_j
        + (macs / Math.max(1, m.total_macs)) * m.compute_energy_j
      : 0;
    const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    set("gpuPassCycles", (c + 1).toLocaleString("en-US"));
    set("gpuPassMacs", Math.round(macs).toLocaleString("en-US"));
    set("gpuPassBytes", fmtBytes(wbytes));
    set("gpuPassEnergy", fmtMJ(energy));
    if (!chart) return;
    const chartKey = `${entry.stage}|${entry.layer}`;
    if (chart.dataset.stageKey === chartKey) return;
    chart.dataset.stageKey = chartKey;
    const order = ["embedding"].concat(GPU_STAGE_ORDER, ["final_rmsnorm", "lm_head"]);
    const totals = {};
    for (const s of seq) totals[s.stage] = (totals[s.stage] || 0) + s.macsPerCall;
    chart.innerHTML = order.map((name) => {
      const tot = totals[name] || 0;
      const done = doneByStage[name] || 0;
      const pct = tot > 0 ? Math.min(100, 100 * done / tot) : (name === entry.stage ? 100 : 0);
      const hot = name === entry.stage ? " is-hot" : "";
      const label = GPU_STAGE_LABEL[name] || name;
      return `<div class="gpu-pass-row${hot}"><span>${label}</span>` +
        `<span class="gpu-pass-bar"><span style="width:${pct.toFixed(1)}%"></span></span>` +
        `<span>${pct.toFixed(0)}%</span></div>`;
    }).join("");
  }

  async function loadGpuPrompt() {
    const box = $("gpuTokens");
    if (!box || box.dataset.ready === "1") return;
    box.dataset.ready = "1";
    try {
      const res = await apiFetch("/api/replay?text=" + encodeURIComponent(GPU_PROMPT) + "&tier=T4");
      const body = await res.json();
      const ids = body.token_ids || (body.evidence && body.evidence.token_ids) || [];
      const texts = body.token_texts || [];
      gpuTokenIds = ids.map((id) => Number(id));
      const pred = body.python_argmax_text || body.rtl_argmax_text || body.predicted_text || "Tokyo";
      gpuNextTok = String(pred).replace(GPU_PROMPT, "").trim().replace(/^[\s"]+|[\s"]+$/g, "") || "Tokyo";
      if (ids.length) {
        box.innerHTML = ids.map((id, i) => {
          const piece = texts[i];
          const shown = piece == null || piece === "" ? String(id) : piece;
          return `<span class="tok" title="id ${id}">${escapeHtml(shown)}</span>`;
        }).join("");
      }
    } catch (_) { /* prompt line still shows the sentence */ }
    paintGpuBoardBits();
    paintGpuDecode();
    paintGpuCycle();
  }

  function paintGpuDecode() {
    const seqObj = gpuSeqCache[gpuAnim.mach];
    const lm = gpuLmHead(seqObj);
    const decoded = gpuLmHeadDecoded(seqObj, gpuAnim.cycle);
    const line = $("gpuPromptLine");
    if (line) line.classList.toggle("is-decoded", decoded);
    const em = $("gpuPredTok");
    if (em) {
      em.hidden = !decoded;
      em.textContent = decoded ? gpuNextTok : "";
    }
    const box = $("gpuTokens");
    if (box) {
      const chip = box.querySelector(".tok.is-next");
      if (decoded && !chip) {
        const span = document.createElement("span");
        span.className = "tok is-next";
        span.textContent = gpuNextTok;
        box.appendChild(span);
      } else if (!decoded && chip) {
        chip.remove();
      }
    }
    const arg = $("gpuArgmaxDetail");
    if (arg) {
      arg.textContent = decoded
        ? `argmax → ${gpuNextTok}`
        : (lm && gpuAnim.cycle >= lm.start)
          ? "streaming lm_head · argmax in flight"
          : "next token appears when lm_head finishes";
    }
  }

  // "Zoom into one tile": a microarchitecture-level drill-down into whatever
  // stage the macro walk (paintGpuCycle) is currently on. Uses
  // evidence/gpu_tile_sim.json (lab/gpu_tile_sim.py) -- the same tile
  // geometry the macro roofline model already assumes, costed as real
  // mma.sync.aligned.m16n8k16 issues bounded by the published (cited,
  // docs/references.md Sec.10) warp-occupancy cap. A compute-only stage
  // (no weights) has no GEMM tiling, so there is nothing to zoom into.
  const gpuZoom = { stage: null, wave: 0 };

  function gpuStepWave(delta) {
    const panel = $("gpuZoomPanel");
    if (!panel || panel.hidden || !gpuZoom.stage) return;
    const stageSim = gpuTileData && gpuTileData.stages.find((s) => s.stage === gpuZoom.stage);
    if (!stageSim || !stageSim.wave_trace.length) return;
    const n = stageSim.wave_trace.length;
    gpuZoom.wave = ((gpuZoom.wave + delta) % n + n) % n;
    paintGpuZoomWave(stageSim);
  }

  // Renders the CURRENT wave of the real per-wave execution trace
  // (lab/gpu_tile_sim.py::simulate_stage_waves) -- an actual simulated
  // step (tiles issued this wave, occupancy this wave, cycle span this
  // wave), not a single closed-form total.
  function paintGpuZoomWave(stageSim) {
    const c = gpuTileData.constants;
    const w = stageSim.wave_trace[gpuZoom.wave];
    const label = $("gpuWaveLabel");
    if (label) label.textContent = `${gpuZoom.wave + 1} / ${stageSim.wave_trace.length}`;
    const fill = $("gpuOccFill");
    if (fill) fill.style.width = `${(w.occupancy_frac * 100).toFixed(1)}%`;
    const occCap = $("gpuOccCap");
    if (occCap) {
      occCap.textContent =
        `${w.tiles_issued.toLocaleString("en-US")} tiles issued this wave ` +
        `(${(w.occupancy_frac * 100).toFixed(1)}% of ${c.concurrent_tiles_across_die.toLocaleString("en-US")} slots, ` +
        `${c.sm_count} SMs × ${c.max_warps_per_sm} warps/SM) · ` +
        `${w.tiles_remaining_after.toLocaleString("en-US")} tiles still queued · ` +
        `cycle ${w.cycle_start.toFixed(1)}–${w.cycle_end.toFixed(1)}`;
    }
    const latCap = $("gpuZoomLatCap");
    if (latCap) {
      latCap.textContent =
        `~${c.mma_latency_cycles} cycles/issue · ${stageSim.geometry.tiles_k} k-slices ` +
        `(one warp holds the fragment, ${c.warp_size} threads)`;
    }
  }

  function paintGpuZoom(entry) {
    const btn = $("gpuZoomBtn");
    const stageSim = gpuTileData && gpuTileData.stages.find((s) => s.stage === entry.stage);
    const hasTiles = !!(stageSim && stageSim.geometry.total_tiles > 0);
    if (btn) {
      btn.disabled = !hasTiles;
      if (!hasTiles) {
        btn.setAttribute("aria-pressed", "false");
        btn.setAttribute("aria-expanded", "false");
        const panel = $("gpuZoomPanel");
        if (panel) panel.hidden = true;
      }
    }
    const panel = $("gpuZoomPanel");
    if (!panel || panel.hidden || !hasTiles) return;
    if (gpuZoom.stage !== entry.stage) {
      gpuZoom.stage = entry.stage;
      gpuZoom.wave = 0;
    }
    const stageName = GPU_STAGE_LABEL[entry.stage] || entry.stage;

    const stageTag = $("gpuZoomStage");
    if (stageTag) stageTag.textContent = stageName;
    const eq = $("gpuZoomEq");
    if (eq) eq.textContent = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";

    paintGpuZoomWave(stageSim);

    const foot = $("gpuZoomFoot");
    if (foot) {
      const c = gpuTileData.constants;
      foot.textContent =
        `SM/tensor-core/warp counts are NVIDIA-published (Hopper whitepaper, ` +
        `Hopper Tuning Guide). The ${c.mma_latency_cycles}-cycle MMA latency is ` +
        `measured on Ampere A100 (arXiv:2206.02874) -- no Hopper-specific figure ` +
        `is published, so it is applied here as the closest analog. No wall-` +
        `clock time is shown at this level: H100 SXM5's SM clock is not ` +
        `officially published either (docs/references.md Sec.10).`;
    }
  }

  const GPU_MAC_DWELL_TICKS = 16;
  const GPU_SPEEDS = [1, 2, 4, 8, 16];

  function gpuSpeed() {
    return GPU_SPEEDS[gpuAnim.speedIdx] || 8;
  }

  function gpuTickMs() {
    const s = gpuSpeed();
    if (s >= 16) return 24;
    if (s >= 8) return 40;
    if (s >= 4) return 45;
    return 55;
  }

  function gpuDecodeHoldTicks() {
    return Math.max(24, Math.round(48 / Math.max(1, gpuSpeed() / 8)));
  }

  function gpuLmHead(seqObj) {
    return seqObj && seqObj.seq.find((s) => s.stage === "lm_head");
  }

  function gpuLmHeadEnd(seqObj) {
    const lm = gpuLmHead(seqObj);
    return lm ? lm.start + lm.cyc.total - 1 : 0;
  }

  function gpuLmHeadDecoded(seqObj, c) {
    const lm = gpuLmHead(seqObj);
    return !!(lm && c >= lm.start + lm.cyc.total - 1);
  }

  function gpuMacComputeCrossing(seqObj, from, to) {
    const cur = gpuFindStage(seqObj, from);
    const dest = GPU_STAGE_DEST[cur.stage] || "mac";
    const streamEnd = cur.start + cur.cyc.actCycles + cur.cyc.weightCycles;
    const stageEnd = cur.start + cur.cyc.total;
    if (dest === "mac" && from >= streamEnd && from < stageEnd) return null;
    for (const s of seqObj.seq) {
      if ((GPU_STAGE_DEST[s.stage] || "mac") !== "mac") continue;
      const c0 = s.start + s.cyc.actCycles + s.cyc.weightCycles;
      if (c0 > from && c0 <= to) return c0;
    }
    return null;
  }

  function gpuSetCycle(c) {
    const seqObj = gpuSeqCache[gpuAnim.mach];
    gpuAnim.cycle = seqObj ? ((c % seqObj.total) + seqObj.total) % seqObj.total : 0;
    paintGpuCycle();
  }
  function gpuStep(delta) {
    gpuSetCycle(gpuAnim.cycle + delta);
  }
  function gpuStepStage(delta) {
    const seqObj = gpuSeqCache[gpuAnim.mach];
    if (!seqObj) return;
    gpuAnim.dwellLeft = 0;
    const entry = gpuFindStage(seqObj, gpuAnim.cycle);
    const idx = seqObj.seq.indexOf(entry);
    const next = seqObj.seq[Math.max(0, Math.min(seqObj.seq.length - 1, idx + delta))];
    gpuSetCycle(next.start);
  }
  function gpuPlayTick() {
    const seqObj = gpuSeqCache[gpuAnim.mach];
    if (!seqObj) return;
    const speed = gpuSpeed();
    const last = seqObj.total - 1;
    const lm = gpuLmHead(seqObj);
    const lmEnd = lm ? gpuLmHeadEnd(seqObj) : last;

    if (gpuAnim.dwellLeft > 0) {
      gpuAnim.dwellLeft -= 1;
      if (gpuAnim.dwellLeft === 0 && gpuAnim.cycle >= last) {
        gpuSetCycle(0);
        return;
      }
      if (gpuAnim.cycle >= lmEnd) {
        paintGpuCycle();
        return;
      }
      const entry = gpuFindStage(seqObj, gpuAnim.cycle);
      const computeEnd = entry.start + entry.cyc.total - 1;
      if (gpuAnim.cycle < computeEnd) gpuSetCycle(gpuAnim.cycle + 1);
      else paintGpuCycle();
      return;
    }

    const frames = Math.max(16, Math.round(160 / speed));
    const step = Math.max(1, Math.round(seqObj.total / frames));
    const from = gpuAnim.cycle;

    if (from < lmEnd && from + step >= lmEnd) {
      gpuSetCycle(lmEnd);
      gpuAnim.dwellLeft = gpuDecodeHoldTicks();
      return;
    }
    if (from >= lmEnd && from + step >= seqObj.total) {
      gpuSetCycle(last);
      gpuAnim.dwellLeft = gpuDecodeHoldTicks();
      return;
    }
    if (from < lmEnd && from + step >= seqObj.total) {
      gpuSetCycle(lmEnd);
      gpuAnim.dwellLeft = gpuDecodeHoldTicks();
      return;
    }

    const macAt = gpuMacComputeCrossing(seqObj, from, from + step);
    if (macAt != null && macAt < lmEnd) {
      gpuSetCycle(macAt);
      gpuAnim.dwellLeft = speed < 4
        ? Math.max(1, Math.round(GPU_MAC_DWELL_TICKS / speed))
        : Math.max(1, Math.round(8 / speed));
      return;
    }
    gpuStep(step);
  }
  function gpuStartTimer() {
    if (gpuAnim.raf) return;
    gpuAnim.rafLast = 0;
    const loop = (t) => {
      if (!gpuAnim.playing) {
        gpuAnim.raf = 0;
        return;
      }
      gpuAnim.raf = requestAnimationFrame(loop);
      if (t - gpuAnim.rafLast < gpuTickMs()) return;
      gpuAnim.rafLast = t;
      gpuPlayTick();
    };
    gpuAnim.raf = requestAnimationFrame(loop);
  }
  function gpuStopTimer() {
    if (gpuAnim.raf) {
      cancelAnimationFrame(gpuAnim.raf);
      gpuAnim.raf = 0;
    }
    if (gpuAnim.timer) {
      clearInterval(gpuAnim.timer);
      gpuAnim.timer = null;
    }
  }
  function setGpuSpeed(idx) {
    gpuAnim.speedIdx = Math.max(0, Math.min(GPU_SPEEDS.length - 1, Number(idx)));
    const lab = $("gpuSpeedLab");
    if (lab) lab.textContent = `${gpuSpeed()}×`;
    const sl = $("gpuSpeed");
    if (sl && Number(sl.value) !== gpuAnim.speedIdx) sl.value = String(gpuAnim.speedIdx);
    if (gpuAnim.playing) {
      gpuStopTimer();
      gpuStartTimer();
    }
  }

  function gpuJumpDecode() {
    const seqObj = gpuSeqCache[gpuAnim.mach];
    if (!seqObj) {
      gpuAnim.pendingJump = true;
      return;
    }
    gpuAnim.pendingJump = false;
    gpuTogglePlay(false);
    gpuAnim.dwellLeft = 0;
    const lm = gpuLmHead(seqObj);
    gpuSetCycle(lm ? gpuLmHeadEnd(seqObj) : seqObj.total - 1);
  }

  function gpuTogglePlay(force) {
    gpuAnim.playing = force != null ? force : !gpuAnim.playing;
    const btn = $("gpuPlayBtn");
    if (btn) btn.textContent = gpuAnim.playing ? "Pause" : "Play";
    if (gpuAnim.playing) gpuStartTimer(); else gpuStopTimer();
  }

  function startGpuAnim() {
    mountGpuExperience();
    gpuSetStory(gpuAnim.story);
    renderGpuNetwork();
    loadGpuPrompt();
    buildGpuViz();
    buildGpuFlow();
    loadGpuRtlMeasured();
    setGpuSpeed(gpuAnim.speedIdx);
    paintGpuCycle();
    const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    gpuTogglePlay(!reduced);
  }

  function stopGpuAnim() {
    gpuTogglePlay(false);
  }

  if ($("gpuMachH100")) $("gpuMachH100").addEventListener("click", () => setGpuMach("h100"));
  if ($("gpuMachB200")) $("gpuMachB200").addEventListener("click", () => setGpuMach("b200"));
  if ($("gpuPlayBtn")) $("gpuPlayBtn").addEventListener("click", () => gpuTogglePlay());
  if ($("gpuStepBack")) $("gpuStepBack").addEventListener("click", () => { gpuTogglePlay(false); gpuStepStage(-1); });
  if ($("gpuStepFwd")) $("gpuStepFwd").addEventListener("click", () => { gpuTogglePlay(false); gpuStepStage(1); });
  if ($("gpuPlaySlider")) {
    $("gpuPlaySlider").addEventListener("input", (e) => {
      gpuTogglePlay(false);
      gpuAnim.dwellLeft = 0;
      gpuSetCycle(Number(e.target.value));
    });
  }
  if ($("gpuTabBoardBtn")) $("gpuTabBoardBtn").addEventListener("click", () => gpuSetLeftTab("board"));
  if ($("gpuTabNetBtn")) $("gpuTabNetBtn").addEventListener("click", () => gpuSetLeftTab("net"));
  if ($("gpuTabNotesBtn")) $("gpuTabNotesBtn").addEventListener("click", () => gpuSetLeftTab("notes"));
  if ($("gpuFlowBtn")) $("gpuFlowBtn").addEventListener("click", () => gpuToggleFlow());
  if ($("gpuJumpDecode")) $("gpuJumpDecode").addEventListener("click", () => gpuJumpDecode());
  if ($("gpuSpeed")) $("gpuSpeed").addEventListener("input", (e) => setGpuSpeed(Number(e.target.value)));
  if ($("gpuZoomBtn")) {
    $("gpuZoomBtn").addEventListener("click", () => {
      const btn = $("gpuZoomBtn");
      const panel = $("gpuZoomPanel");
      if (!btn || !panel || btn.disabled) return;
      const open = panel.hidden;
      panel.hidden = !open;
      btn.setAttribute("aria-pressed", String(open));
      btn.setAttribute("aria-expanded", String(open));
      if (open) {
        const seqObj = gpuSeqCache[gpuAnim.mach];
        if (seqObj) paintGpuZoom(gpuFindStage(seqObj, gpuAnim.cycle));
      }
    });
  }
  if ($("gpuWaveBack")) $("gpuWaveBack").addEventListener("click", () => gpuStepWave(-1));
  if ($("gpuWaveFwd")) $("gpuWaveFwd").addEventListener("click", () => gpuStepWave(1));

  const MXU_N = 8;
  const scaleAnim = { t: 0, timer: null, mach: "us" };
  const SCALE_MACH = {
    gpu: {
      cap: "GPU: same 2D MACs. Copy holding x in SRAM. Drop this: 1.19 GB of W walks HBM every token (~5 pJ/bit).",
      hostTitle: "kernels + HBM",
      hostSub: "weights are not on the die",
      fabric: "HBM PHY · 1.19 GB W / token · ~5 pJ/bit",
      hbm: "live · W every token · ~5 pJ/bit",
      sramK: "SRAM / L2 · x (copy this)",
      sramTitle: "holds x · not W",
      sramSub: "cheap 2 KB x tile · the bill is HBM W",
      mxuK: "tensor cores · 2D MMA",
      mxuStrong: "PE = HBM W tile + MMA · W not burnt-in",
      mxuSmall: "grid is 2D like ours · W is still a DRAM fetch",
      peW: "HBM W tile",
      peFoot: "W arrives on the fabric every token · that is the 197 mJ",
      peMark: "HBM",
      reduceLoad: "HBM → W tile  ·  1.19 GB  ·  ~5 pJ/bit",
      reduceRun: "MMA  ·  x from SRAM  ·  W from HBM"
    },
    tpu: {
      cap: "TPU: this grid — x pulses, W sits. Then the MXU reloads the next tile from HBM. Stationary is not burnt-in.",
      hostTitle: "XLA + HBM",
      hostSub: "each GEMM tile of W still walks HBM",
      fabric: "HBM → weight FIFO → MXU · reload between tiles",
      hbm: "live between tiles · W sits only until the next load",
      sramK: "VMEM · x",
      sramTitle: "holds x · W is in the MXU, then HBM again",
      sramSub: "copy the pulse · do not copy the reload",
      mxuK: "MXU · 128×128 class · 8×8 tile shown",
      mxuStrong: "PE = W register + MAC · reloads from HBM",
      mxuSmall: "systolic: x east, psum south · W sits, then FIFO refill",
      peW: "W reg (sits)",
      peFoot: "weight-stationary between HBM loads — not 6T ROM",
      peMark: "reg",
      reduceLoad: "HBM → FIFO → PE W  ·  next tile",
      reduceRun: "W sits  ·  x pulses  ·  then HBM reload"
    },
    lpu: {
      cap: "LPU: no HBM. W is already on-die — in SRAM, not ROM. 0.6B fits; 1T does not. 750 W TDP.",
      hostTitle: "token IDs",
      hostSub: "W already in SRAM · no HBM PHY",
      fabric: "no HBM · W already in on-die SRAM · 750 W class",
      hbm: "absent · LPU has no HBM",
      sramK: "weight SRAM · 1.19 GB W",
      sramTitle: "W lives here (writable SRAM)",
      sramSub: "0.6B fits · 1T does not · that is why we use ROM",
      mxuK: "static EXU · W from SRAM",
      mxuStrong: "PE = SRAM W + MAC · not 6T ROM",
      mxuSmall: "same on-die idea · different cell (SRAM, not burnt-in)",
      peW: "SRAM W[15:0]",
      peFoot: "on-die like us · writable SRAM, not fabricated ROM",
      peMark: "SRAM",
      reduceLoad: "W already in SRAM  ·  load x only",
      reduceRun: "x east  ·  W from SRAM  ·  no hbm_rd"
    },
    cim: {
      cap: "CIM: W is the cell. DAC drives x, Kirchhoff sums current, ADC reads y. Copy ‘no fetch.’ Drop 4 analog passes for native BF16.",
      hostTitle: "token IDs",
      hostSub: "W is conductance · no HBM W",
      fabric: "no HBM W · analog array · ADC/DAC periphery",
      hbm: "absent · analog W is not DRAM",
      sramK: "DAC row · x as voltages",
      sramTitle: "x into the array",
      sramSub: "one vector · the bill is ADC, not HBM",
      mxuK: "256×256 crossbar · G = W",
      mxuStrong: "PE = ROM cell conductance · current sum",
      mxuSmall: "not a digital BF16 MAC · 4 nibble passes at BF16",
      peW: "G = W nibble",
      peFoot: "weight IS the multiply · ADC ~5 pJ/conv is the bill",
      peMark: "G",
      reduceLoad: "DAC x  ·  cell already is W",
      reduceRun: "Kirchhoff  ·  ADC y  ·  4 passes at BF16"
    },
    pim: {
      cap: "PIM: ALU in the DRAM bank. Cheaper than HBM-to-GPU (~2 pJ/bit). W is still DRAM, still writable, still not ROM.",
      hostTitle: "HBM-PIM cmd",
      hostSub: "W lives in DRAM banks",
      fabric: "in-bank GEMV · no GPU HBM PHY walk · still DRAM",
      hbm: "DRAM banks · ~2 pJ/bit · not 6T ROM",
      sramK: "row buffer · x",
      sramTitle: "x next to the bank",
      sramSub: "near-memory, not in-cell analog",
      mxuK: "bank ALU · GEMV",
      mxuStrong: "PE = DRAM row + integer/BF16 ALU",
      mxuSmall: "move compute to W · do not burn W in",
      peW: "DRAM W row",
      peFoot: "closer than GPU HBM · still a DRAM fetch inside the stack",
      peMark: "DRAM",
      reduceLoad: "activate row  ·  W in DRAM",
      reduceRun: "bank GEMV  ·  ~2 pJ/bit"
    },
    us: {
      cap: "This die: each PE is 6T ROM + a BF16 MAC. No HBM PHY, no weight FIFO. Host: token IDs only.",
      hostTitle: "token[17:0]",
      hostSub: "pos[15:0] · TOKEN_W=18",
      fabric: "no HBM PHY · no weight FIFO · tile_sel pages on-die ROM · clk 2 GHz class",
      hbm: "~5 pJ/bit · not on this die",
      sramK: "scratch SRAM · u_sram",
      sramTitle: "8 banks · 256b · R/W",
      sramSub: "west-edge broadcast · 1 load / GEMM · x and KV only",
      mxuK: "ROM MXU · u_mxu · 8×8 tile shown",
      mxuStrong: "PE = 6T ROM W + BF16 MAC · qwen-ref-v2",
      mxuSmall: "systolic: x east, psum south · W local, never on the 256b fabric",
      peW: "6T ROM W[15:0]",
      peFoot: "W not on a bus · RNE · subnormals · qNaN · replica bit-exact",
      peMark: "ROM",
      reduceLoad: "tile_sel holds  ·  6T ROM already in every PE  ·  no weight FIFO",
      reduceRun: null
    }
  };

  function setScaleMach(mach) {
    if (!SCALE_MACH[mach]) return;
    scaleAnim.mach = mach;
    const chip = $("mxuChip");
    if (chip) chip.setAttribute("data-mach", mach);
    document.querySelectorAll("[data-scale-mach]").forEach((el) => {
      const on = el.getAttribute("data-scale-mach") === mach;
      if (el.tagName === "BUTTON") el.setAttribute("aria-selected", on ? "true" : "false");
      else el.setAttribute("aria-pressed", on ? "true" : "false");
    });
    const cap = $("mxuMachCap");
    if (cap) cap.textContent = SCALE_MACH[mach].cap;
    paintScaleAnim();
  }

  function bindScaleMach() {
    document.querySelectorAll("[data-scale-mach]").forEach((el) => {
      const go = () => setScaleMach(el.getAttribute("data-scale-mach"));
      el.addEventListener("click", go);
      el.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          go();
        }
      });
    });
  }

  function buildScaleViz() {
    const grid = $("mxuGrid");
    if (!grid) return;  // MXU viz lives on the 2D slide; absent is fine
    const sram = $("mxuSram");
    const layers = $("mxuLayers");
    if (!grid || grid.dataset.ready) return;
    grid.dataset.ready = "1";
    const n = MXU_N;
    let cells = "";
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        cells += `<div class="pe" data-pe="${r}-${c}" title="PE[${r},${c}] ROM W + MAC"><span>W</span><em>×</em></div>`;
      }
    }
    grid.innerHTML = cells;
    if (sram) {
      sram.innerHTML = Array.from({ length: 8 }, (_, i) =>
        `<div class="bank" data-mxu-bank="${i}"><b>x${i}</b></div>`
      ).join("");
    }
    const tiles = $("mxuTiles");
    if (tiles && !tiles.dataset.ready) {
      tiles.dataset.ready = "1";
      const names = ["Q_PROJ", "K_PROJ", "V_PROJ", "O_PROJ", "GATE", "UP", "DOWN", "LM_HEAD"];
      tiles.innerHTML = names.map((n) => `<div class="tile" data-mxu-tile="${n}">${n}</div>`).join("");
    }
    if (layers) {
      layers.innerHTML = Array.from({ length: 28 }, (_, i) =>
        `<div class="layer" data-mxu-layer="${i}">L${String(i).padStart(2, "0")}</div>`
      ).join("");
    }
  }

  function mxuOn(sel, on) {
    document.querySelectorAll(sel).forEach((el) => el.classList.toggle("active", !!on));
  }
  function mxuWire(name, on) {
    document.querySelectorAll(`[data-mxu-wire="${name}"]`).forEach((el) => el.classList.toggle("live", !!on));
  }

  function paintScaleAnim() {
    const grid = $("mxuGrid");
    if (!grid) return;
    const n = MXU_N;
    const t = scaleAnim.t;
    const layer = Math.floor(t / 24) % 28;
    const phase = t % 20;
    const load = phase < 4;
    const wave = load ? -1 : phase - 4;
    const fsm = $("mxuFsm");
    const host = $("mxuHost");
    const note = $("mxuNote");
    const reduce = $("mxuReduce");
    const logit = $("mxuLogit");
    const m = SCALE_MACH[scaleAnim.mach] || SCALE_MACH.us;
    const setTxt = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    setTxt("mxuHostTitle", m.hostTitle);
    setTxt("mxuHostSub", m.hostSub);
    setTxt("mxuFabric", m.fabric);
    setTxt("mxuHbmNote", m.hbm);
    setTxt("mxuSramK", m.sramK);
    setTxt("mxuSramTitle", m.sramTitle);
    setTxt("mxuSramSub", m.sramSub);
    setTxt("mxuMxuK", m.mxuK);
    setTxt("mxuMxuStrong", m.mxuStrong);
    setTxt("mxuMxuSmall", m.mxuSmall);
    setTxt("mxuPeW", m.peW);
    setTxt("mxuPeFoot", m.peFoot);
    if (fsm) fsm.textContent = load ? "S_LOAD_ACT" : "S_RUN";
    // Host input per machine: GPU/TPU/PIM stream weights from HBM/DRAM;
    // LPU and our die receive token IDs only; CIM receives token IDs too
    // (its DAC input comes from the activation row, not a weight stream).
    const hostTxt = { us: "token IDs only", lpu: "token IDs only", cim: "token IDs only",
                      gpu: "W from HBM", tpu: "W from HBM", pim: "W in DRAM banks" };
    if (host) host.textContent = hostTxt[scaleAnim.mach] || "W from HBM";
    mxuOn('[data-mxu="host"]', true);
    mxuOn('[data-mxu="fsm"]', true);
    mxuOn('[data-mxu="sram"]', load || wave >= 0);
    mxuOn('[data-mxu="mxu"]', !load);
    mxuOn('[data-mxu="hbm"]', scaleAnim.mach === "gpu" || scaleAnim.mach === "pim" || (scaleAnim.mach === "tpu" && load));
    mxuOn('[data-mxu="fifo"]', scaleAnim.mach === "tpu");
    mxuOn('[data-mxu="logits"]', !load && wave > 6);
    mxuWire("cmd", true);
    mxuWire("fabric", !load || scaleAnim.mach === "gpu");
    mxuWire("x", !load);
    mxuWire("hbm", scaleAnim.mach === "gpu" || scaleAnim.mach === "pim" || (scaleAnim.mach === "tpu" && load));
    mxuWire("nlin", false);
    document.querySelectorAll("[data-mxu-bank]").forEach((el, i) => {
      el.classList.toggle("active", load ? i === phase : i === (wave % 8));
    });
    document.querySelectorAll("[data-mxu-layer]").forEach((el) => {
      el.classList.toggle("active", Number(el.getAttribute("data-mxu-layer")) === layer);
    });
    const stages = ["Q_PROJ", "K_PROJ", "V_PROJ", "O_PROJ", "GATE", "UP", "DOWN", "LM_HEAD"];
    const stage = stages[Math.floor(t / 20) % stages.length];
    mxuOn('[data-mxu="pe"]', !load);
    mxuOn('[data-mxu="rmsnorm"]', false);
    document.querySelectorAll("[data-mxu-eq], [data-mxu-eq-note]").forEach((el) => {
      const key = el.getAttribute("data-mxu-eq") || el.getAttribute("data-mxu-eq-note");
      el.hidden = key !== stage;
    });
    document.querySelectorAll("[data-mxu-tile]").forEach((el) => {
      el.classList.toggle("active", el.getAttribute("data-mxu-tile") === stage);
    });
    document.querySelectorAll("[data-mxu-tag]").forEach((el) => {
      const tag = el.getAttribute("data-mxu-tag");
      let on = false;
      if (scaleAnim.mach === "lpu") on = tag === "ww" || (load && tag === "hidden");
      else if (load) on = tag === "hidden";
      else on = tag === "ws";
      el.classList.toggle("active", on);
    });
    document.querySelectorAll("[data-mxu-bank] b").forEach((el, i) => {
      el.textContent = scaleAnim.mach === "lpu" ? `W${i}` : `x${i}`;
    });
    let live = 0;
    let hitR = 0;
    let hitC = 0;
    grid.querySelectorAll("[data-pe]").forEach((pe) => {
      const [r, c] = pe.getAttribute("data-pe").split("-").map(Number);
      const s = r + c;
      const hot = s === wave;
      pe.classList.toggle("hot", hot);
      pe.classList.toggle("done", wave > s);
      const mark = pe.querySelector("span");
      if (mark) mark.textContent = m.peMark;
      if (hot) { hitR = r; hitC = c; }
      if (hot || (wave > s && wave >= 0)) live += 1;
    });
    const peTitle = $("mxuPeTitle");
    if (peTitle) {
      peTitle.textContent = load
        ? (scaleAnim.mach === "us" ? "PE idle · W already burnt in" : `PE idle · ${m.peW}`)
        : `PE[${hitR},${hitC}]  ${stage}  ·  ${m.peMark}`;
    }
    document.querySelectorAll("[data-pe-n]").forEach((el) => {
      const nme = el.getAttribute("data-pe-n");
      const on = !load && (nme === "w" || nme === "mul" || nme === "add" || nme === "x");
      el.classList.toggle("active", on);
    });
    if (reduce) {
      reduce.textContent = load
        ? m.reduceLoad
        : (m.reduceRun || `x[${hitR}] east · psum[${hitC}] south · pipe=2 · bf16_rne(acc) · ${Math.min(n, Math.max(0, wave + 1))} cols`);
    }
    if (logit) {
      logit.textContent = stage === "LM_HEAD"
        ? "write_enable=0  ·  argmax in flight  ·  logits never in SRAM"
        : "output row → SRAM workspace  ·  256b commit";
    }
    if (note) {
      const tailMap = {
        us: "W_rom not on fabric",
        gpu: "W from HBM every token",
        tpu: "W FIFO / HBM reload between tiles",
        lpu: "W from on-die SRAM",
        cim: "W is the cell · DAC x · ADC y",
        pim: "W from DRAM bank · in-bank ALU"
      };
      const tail = tailMap[scaleAnim.mach] || "W from SRAM";
      note.textContent = load
        ? `L${String(layer).padStart(2, "0")} ${stage}  S_LOAD_ACT  ·  ${tail}`
        : `L${String(layer).padStart(2, "0")} ${stage}  S_RUN  beat ${wave}  ·  PE[${hitR},${hitC}]  ·  ${live} PEs  ·  ${tail}`;
    }
  }

  function stopScaleAnim() {
    if (scaleAnim.timer) {
      clearInterval(scaleAnim.timer);
      scaleAnim.timer = null;
    }
  }

  function startScaleAnim() {
    buildScaleViz();
    paintScaleAnim();
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (scaleAnim.timer) return;
    scaleAnim.timer = setInterval(() => {
      scaleAnim.t += 1;
      paintScaleAnim();
    }, 720);
  }

  function renderScaleMath() {
    const root = document.querySelector(".slide-scale");
    if (!root || root.dataset.math === "1" || typeof renderMathInElement !== "function") return;
    renderMathInElement(root, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false,
      strict: false
    });
    root.dataset.math = "1";
  }

  function renderSpecMath() {
    const roots = document.querySelectorAll(".slide-spec");
    if (!roots.length || typeof renderMathInElement !== "function") return;
    roots.forEach((root) => {
      if (root.dataset.math === "1") return;
      renderMathInElement(root, {
        delimiters: [
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false,
        strict: false
      });
      root.dataset.math = "1";
    });
  }

  // ---- Slide 5 · Interactive energy advantage chart ----------------------------
  // SINGLE SOURCE OF TRUTH for the economics chart and its stat cards.
  // `energy` drives the plotted y-position, the point label and the stat card,
  // so they cannot disagree. NOTE: `energy` for 0.6B reads 62x here while the
  // title slide headlines 166x (197 mJ / 1.19 mJ). Those cannot both be right
  // - settle the constant, then edit only this table.
  const SPEC_MODELS = [
    { id: 0, label: "0.6B dense",    xLabel: "0.6B",      x: 100, energy: "62\u00d7",  dies: 1,   tx: "~18B", txNote: "0.2\u00d7 H100",             energyVal: "1.19 mJ (0.6B lab)", cost: "36\u00d7 lower", costNote: "0.2 \u00d7 (1/62) = 0.003",  tok: "8.8",     tokNote: "lab \u00b7 16 lanes" },
    { id: 1, label: "8B dense",      xLabel: "8B",        x: 180, energy: "132\u00d7", dies: 5,   tx: "0.3T", txNote: "4\u00d7 H100",               energyVal: "26 mJ",              cost: "33\u00d7 lower", costNote: "4 \u00d7 (1/132) = 0.031",   tok: "~2,000",  tokNote: "estimated" },
    { id: 2, label: "70B dense",     xLabel: "70B",       x: 260, energy: "199\u00d7", dies: 36,  tx: "1.8T", txNote: "11\u00d7 H100",              energyVal: "124 mJ",             cost: "18\u00d7 lower", costNote: "11 \u00d7 (1/199) = 0.055",  tok: "~15,000", tokNote: "estimated" },
    { id: 3, label: "500B MoE",      xLabel: "500B",      x: 340, energy: "211\u00d7", dies: 258, tx: "10.3T", txNote: "10\u00d7 H100 (1T/bit ROM)", energyVal: "124 mJ",             cost: "21\u00d7 lower", costNote: "10 \u00d7 (1/211) = 0.047",  tok: "40,000",  tokNote: "14\u00d7 faster than H100" },
    { id: 4, label: "1T params MoE", xLabel: "1T params", x: 420, energy: "211\u00d7", dies: 537, tx: "21.3T", txNote: "10\u00d7 H100 (1T/bit ROM)", energyVal: "125 mJ",             cost: "22\u00d7 lower", costNote: "10 \u00d7 (1/211) = 0.047",  tok: "40,000",  tokNote: "14\u00d7 faster than H100" }
  ];


  // Chart geometry. One source of truth: every y is derived from the point's
  // own `energy` value, so a label can never drift away from its position.
  const SPEC_GEO = { x0: 60, x1: 490, yTop: 50, yBot: 250 };
  const specVal = (m) => Number(String(m.energy).replace(/[^\d.]/g, "")) || 0;
  const specAxisMax = () =>
    Math.max(50, Math.ceil(Math.max(...SPEC_MODELS.map(specVal)) / 50) * 50);
  const specY = (v) =>
    SPEC_GEO.yBot - (v / specAxisMax()) * (SPEC_GEO.yBot - SPEC_GEO.yTop);

  function renderSpecChart() {
    const grid = $("specGrid");
    const series = $("specSeries");
    if (!grid || !series) return;
    const { x0, x1, yTop, yBot } = SPEC_GEO;
    const max = specAxisMax();
    const step = max / 5;

    let g = "";
    for (let v = 0; v <= max + 0.5; v += step) {
      const y = specY(v);
      const strong = v === 0;
      g += `<line x1="${x0}" y1="${y.toFixed(1)}" x2="${x1}" y2="${y.toFixed(1)}" stroke="${strong ? "#e2e8f0" : "#f1f5f9"}" stroke-width="1"/>`;
      g += `<text x="${x0 - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="spec-axis-label">${v === 0 ? "1×" : Math.round(v) + "×"}</text>`;
    }
    g += `<line x1="${x0}" y1="${yTop}" x2="${x0}" y2="${yBot}" stroke="#e2e8f0" stroke-width="1"/>`;
    // H100 baseline sits on the zero line; label goes inside the plot box so it
    // cannot be clipped by the viewBox edge
    g += `<line x1="${x0}" y1="${yBot}" x2="${x1}" y2="${yBot}" stroke="#64748b" stroke-width="2" stroke-dasharray="6,4"/>`;
    g += `<text x="${x1}" y="${yBot - 7}" text-anchor="end" class="spec-baseline-label">H100 baseline</text>`;
    SPEC_MODELS.forEach((m) => {
      g += `<text x="${m.x}" y="${yBot + 20}" text-anchor="middle" class="spec-axis-label">${m.xLabel || m.label}</text>`;
    });
    grid.innerHTML = g;

    const pts = SPEC_MODELS.map((m) => `${m.x},${specY(specVal(m)).toFixed(1)}`).join(" ");
    let s = `<polyline points="${pts}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`;
    SPEC_MODELS.forEach((m) => {
      const v = specVal(m);
      const y = specY(v);
      s += `<circle cx="${m.x}" cy="${y.toFixed(1)}" r="5.5" fill="#0f766e"/>`;
      // label above the point, flipped below if it would leave the plot box
      const above = y - 12 > yTop;
      s += `<text x="${m.x}" y="${(above ? y - 12 : y + 20).toFixed(1)}" text-anchor="middle" class="spec-point-label">${v}×</text>`;
    });
    series.innerHTML = s;
  }

  function paintSpecSlider(idx) {
    const m = SPEC_MODELS[idx] || SPEC_MODELS[3];
    renderSpecChart();
    const marker = $("specMarker");
    if (marker) {
      marker.setAttribute("transform", `translate(${m.x}, ${specY(specVal(m)).toFixed(1)})`);
      marker.innerHTML =
        '<circle r="12" fill="#0f766e" fill-opacity="0.2"/><circle r="8" fill="#0f766e"/>';
    }

    const setCard = (id, label, val, note, noteClass) => {
      const card = $(id);
      if (!card) return;
      const valEl = card.querySelector(".spec-stat-val");
      const noteEl = card.querySelector(".spec-stat-note");
      if (valEl) valEl.textContent = val;
      if (noteEl) {
        noteEl.textContent = note || "";
        noteEl.className = "spec-stat-note" + (noteClass ? " " + noteClass : "");
      }
    };

    setCard("specStatModel", "Model", m.label);
    setCard("specStatDies", "Dies required", String(m.dies));
    setCard("specStatTx", "Transistors", m.tx, m.txNote, "bad");
    setCard("specStatEnergy", "Energy/token", m.energyVal, m.energy + " less than H100", "good");
    setCard("specStatCost", "$/token vs H100", m.cost, m.costNote);
    setCard("specStatTok", "tok/s target", m.tok, m.tokNote, "good");
  }

  // ---- Scaling slide · array / precision / multi-die tabs ------------------
  // the layer graph on the Q = yW slide is collapsed by default (slide 3 owns
  // it); edges must be redrawn once it actually has a size
  function bindRmMechFold() {
    const fold = document.querySelector(".rm-mech-fold");
    if (!fold) return;
    fold.addEventListener("toggle", () => {
      if (fold.open) requestAnimationFrame(paintRoadmapBars);
    });
  }

  function bindNnMapFold() {
    const fold = document.getElementById("nnMapFold");
    if (!fold) return;
    fold.addEventListener("toggle", () => {
      if (fold.open) requestAnimationFrame(() => { drawNnDagEdges(); paintNnHeat(); });
    });
  }

  // Each .tabset is bound independently so several slides can use the same
  // tab component without cross-wiring each other's panels.
  function bindScalingTabs() {
    document.querySelectorAll(".tabset").forEach((set) => {
      const tabs = Array.from(set.querySelectorAll(".sc-tab"));
      if (!tabs.length) return;
      const panels = Array.from(set.querySelectorAll(".sc-panel"));
      const select = (key) => {
        tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.scTab === key)));
        panels.forEach((p) => { p.hidden = p.dataset.scPanel !== key; });
        if (key === "today" && typeof paintSpecSlider === "function") {
          paintSpecSlider(Number(($("specSlider") || {}).value || 3));
        }
        if (key === "roadmap") { paintRoadmapLive(); paintRoadmapBars(); }
      };
      tabs.forEach((t) => t.addEventListener("click", () => select(t.dataset.scTab)));
      select(tabs[0].dataset.scTab);
    });
  }

  function bindSpecSlider() {
    const slider = $("specSlider");
    if (!slider) return;
    slider.addEventListener("input", (ev) => {
      paintSpecSlider(Number(ev.target.value));
    });
    // Initialize with default value
    paintSpecSlider(Number(slider.value));
  }

  const UTIL_DIE_MM2 = 800;
  const UTIL_ROM_FRAC = 0.70;
  const UTIL_NODES = [
    { id: "n28", label: "N28", sub: "28 nm", romUm2: 0.109, romPj: 0.80, sramPj: 0.50, sramUm2: 0.12, clk: 0.5e9 },
    { id: "n16", label: "N16", sub: "~18 nm class", romUm2: 0.063, romPj: 0.40, sramPj: 0.25, sramUm2: 0.07, clk: 1.5e9 },
    { id: "n4", label: "N4", sub: "H100 node", romUm2: 0.018, romPj: 0.08, sramPj: 0.10, sramUm2: 0.021, clk: 2.0e9 },
    { id: "n2", label: "N2", sub: "2 nm", romUm2: 0.014, romPj: 0.05, sramPj: 0.06, sramUm2: 0.016, clk: 2.5e9 }
  ];
  const UTIL_PREC = [
    { id: "bf16", label: "BF16", bits: 16, prod: true, note: "lab / production" },
    { id: "int8", label: "INT8", bits: 8, note: "precision-aware trained (native FP8 class)" },
    { id: "int4", label: "INT4", bits: 4, note: "INT4 QAT — the mask is the checkpoint" },
    { id: "dsv4", label: "DS-V4 mix", bits: 4.12, note: "FP8-trained; FP4 experts + FP8 rest" },
    { id: "glm", label: "GLM-5.3 mix", bits: 4.20, note: "INT4 experts + INT8 rest" }
  ];
  const UTIL_MODELS = [
    { id: "q06", label: "0.6B dense", stored: 0.6e9, active: 0.6e9, layers: 28, kvHeads: 8, headDim: 128, hidden: 1024 },
    { id: "l8", label: "8B dense", stored: 8e9, active: 8e9, layers: 36, kvHeads: 8, headDim: 128, hidden: 4096 },
    { id: "l70", label: "70B dense", stored: 70e9, active: 70e9, layers: 80, kvHeads: 8, headDim: 128, hidden: 8192 },
    { id: "l405", label: "405B dense", stored: 405e9, active: 405e9, layers: 126, kvHeads: 8, headDim: 128, hidden: 16384 },
    { id: "l500", label: "500B MoE / 75B", stored: 500e9, active: 75e9, layers: 64, kvHeads: 8, headDim: 128, hidden: 8192, moe: true, prod: true },
    { id: "l1t", label: "1T MoE / 76B", stored: 1.0426e12, active: 76.2e9, layers: 64, kvHeads: 8, headDim: 128, hidden: 8192, moe: true, prod: true },
    { id: "dsv4", label: "V4-Flash 284B/13B", stored: 284e9, active: 13e9, layers: 61, kv: "mla", kvDim: 576, hidden: 7168 },
    { id: "glm", label: "GLM-Flash 320B/18B", stored: 320e9, active: 18e9, layers: 40, kvHeads: 8, headDim: 128, hidden: 4096 }
  ];
  const UTIL_TARGETS = ["l500", "l1t"];
  const UTIL_CTX_FLOOR = 131072;
  const UTIL_CTXS = [
    { id: 131072, label: "128k", prod: true },
    { id: 262144, label: "256k" },
    { id: 524288, label: "512k" },
    { id: 1048576, label: "1M" }
  ];
  const UTIL_HBM_PJ = 5;
  const UTIL_ROM_T = 1;  // mask ROM is a 1T/bit cell (0.018 um2 N4-class); 6T is SRAM-class
  const UTIL_SRAM_TX = 8e9;
  const UTIL_D2D_PJ = 1.0;
  const UTIL_D2D_BPS = 1e12;
  const UTIL_KV_AREA_FRAC = 0.20;
  const utilPick = { node: "n4", prec: "bf16", model: "l500", ctx: 131072, kvInt8: false, mla: false, batch: 1 };
  const UTIL_RIVALS = [
    { id: "gpu", k: "vs H100", short: "H100", name: "NVIDIA H100 SXM", e06: 0.197, tok: 2810, color: "#1d4ed8", txPkg: 80e9, memB: 80e9, mem: "80 GB HBM", where: "HBM ~5 pJ/bit", kvPj: 5 },
    { id: "tpu", k: "vs TPU v5e", short: "TPU v5e", name: "Google TPU v5e", e06: 0.124, tok: 1342, color: "#b45309", txPkg: 32e9, memB: 16e9, mem: "16 GB HBM", where: "HBM ~5 pJ/bit", kvPj: 5 },
    // LPU: full latency (compute + weight-SRAM stream) x 750 W x 0.6,
    // per lab/compute_metrics.py lpu_estimate — not compute-only.
    { id: "lpu", k: "vs LPU", short: "LPU", name: "Groq LPU", e06: 0.187, tok: 2460, color: "#6d28d9", txPkg: 26.8e9, memB: 1.2e9, mem: "SRAM ~1.19 GB", where: "SRAM ~0.5 pJ/bit", kvPj: 0.5, tokStay: true, capKv: true },
    // Analog CIM: lab/scale_1t cim_estimate at 0.6B BF16 (4 passes,
    // per-op ADC/DAC pricing). W never moves; ADC is the per-op bill.
    { id: "cim", k: "vs analog CIM", short: "CIM", name: "Analog CIM", e06: 0.00036, tok: 4000, color: "#be123c", txPkg: 26e9, memB: 13.2e9, mem: "analog array · 6.6B params/die", where: "W is the cell · ADC counted", kvPj: 0.5, tokStay: true, cim: true },
    // HBM-PIM: in-bank GEMV ~2 pJ/bit over the full weight stream.
    { id: "pim", k: "vs HBM-PIM", short: "PIM", name: "HBM-PIM", e06: 0.0195, tok: 3200, color: "#0e7490", txPkg: 20e9, memB: 24e9, mem: "HBM-PIM 24 GB · DRAM 1T counted", where: "DRAM bank ~2 pJ/bit", kvPj: 2 }
  ];

  function utilBitsPerDie(node) {
    return UTIL_DIE_MM2 * 1e6 * UTIL_ROM_FRAC / node.romUm2;
  }
  function utilMaxParams(node, prec) {
    return utilBitsPerDie(node) / prec.bits;
  }
  function utilFmtP(p) {
    if (p >= 1e12) return (p / 1e12).toFixed(2) + "T";
    if (p >= 1e9) return (p / 1e9).toFixed(p >= 10e9 ? 1 : 2) + "B";
    if (p >= 1e6) return (p / 1e6).toFixed(0) + "M";
    return String(Math.round(p));
  }
  function utilFmtE(j) {
    if (j >= 1) return j.toFixed(j >= 10 ? 0 : 2) + " J";
    if (j >= 1e-3) return (j * 1e3).toFixed(j * 1e3 >= 10 ? 1 : 2) + " mJ";
    return (j * 1e6).toFixed(1) + " µJ";
  }
  function utilFmtTok(t) {
    if (!Number.isFinite(t) || t <= 0) return "—";
    if (t >= 1000) return Math.round(t).toLocaleString("en-US");
    if (t >= 10) return String(Math.round(t));
    return t.toFixed(1);
  }
  function utilFmtX(x) {
    if (!Number.isFinite(x) || x <= 0) return "—";
    if (x >= 100) return String(Math.round(x));
    return x.toFixed(1);
  }
  function utilFmtTx(n) {
    const t = n / 1e12;
    if (t >= 1) return (t >= 10 ? t.toFixed(0) : t.toFixed(1)) + "T";
    const b = n / 1e9;
    if (b >= 100) return Math.round(b) + "B";
    if (Math.abs(b - Math.round(b)) < 0.05) return Math.round(b) + "B";
    return b.toFixed(1) + "B";
  }
  function utilFmtAct(p) {
    if (p >= 1e8 && p < 1e9) return (p / 1e9).toFixed(2) + "B";
    return utilFmtP(p);
  }
  function utilTxPhrase(ours, theirs) {
    const q = ours / theirs;
    if (!Number.isFinite(q) || q <= 0) return "— Tx";
    if (q < 0.97) return utilFmtX(1 / q) + "× fewer Tx";
    if (q > 1.03) return utilFmtX(q) + "× more Tx";
    return "about the same Tx";
  }
  function utilCostPhrase(x) {
    if (!Number.isFinite(x) || x <= 0) return "—";
    if (x < 1) return utilFmtX(1 / x) + "× more expensive · unviable";
    return utilFmtX(x) + "× cheaper";
  }
  function utilFmtBits(bits) {
    return Number.isInteger(bits) ? String(bits) : bits.toFixed(2);
  }
  function utilFmtGhz(hz) {
    const g = hz / 1e9;
    return (Math.abs(g - Math.round(g)) < 1e-9 ? String(Math.round(g)) : g.toFixed(1)) + " GHz";
  }
  function utilFmtCtx(n) {
    if (n >= 1048576) {
      const m = n / 1048576;
      return (Math.abs(m - Math.round(m)) < 1e-9 ? String(Math.round(m)) : m.toFixed(1)) + "M";
    }
    if (n >= 1024) return Math.round(n / 1024) + "k";
    return String(n);
  }
  function utilFmtBytes(b) {
    if (b >= 1e9) return (b / 1e9).toFixed(b >= 10e9 ? 0 : 2) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(0) + " MB";
    if (b >= 1e3) return (b / 1e3).toFixed(0) + " KB";
    return Math.round(b) + " B";
  }
  function utilKvBpp(model) {
    if (model.kv === "mla") return model.layers * model.kvDim * 2;
    return 2 * model.layers * model.kvHeads * model.headDim * 2;
  }
  function utilHidden(model) {
    return model.hidden || 1024;
  }
  function utilFmtUs(s) {
    if (!Number.isFinite(s) || s <= 0) return "—";
    if (s >= 1) return s.toFixed(s >= 10 ? 0 : 2) + " s";
    if (s >= 1e-3) return (s * 1e3).toFixed(s * 1e3 >= 10 ? 0 : 1) + " ms";
    return (s * 1e6).toFixed(s * 1e6 >= 10 ? 0 : 1) + " µs";
  }

  function utilSolve(node, prec, model, ctx) {
    ctx = ctx == null ? (utilPick.ctx || UTIL_CTX_FLOOR) : ctx;
    const bitsDie = utilBitsPerDie(node);
    const bytesDie = bitsDie / 8;
    const maxP = bitsDie / prec.bits;
    const storedB = model.stored * prec.bits / 8;
    const activeB = model.active * prec.bits / 8;
    const storedBf16B = model.stored * 2;
    const dies = Math.max(1, Math.ceil(storedB / bytesDie));
    const kvBpp = utilKvBpp(model);
    const kvB = kvBpp * ctx;
    const hidden = utilHidden(model);
    const kvLayerB = kvB / Math.max(model.layers, 1);
    const actB = hidden * 2;
    const d2dActB = actB * model.layers;
    const eD2d = d2dActB * 8 * UTIL_D2D_PJ / 1e12;
    const eKvRemote = kvB * 8 * UTIL_D2D_PJ / 1e12;
    const tD2dAct = d2dActB / UTIL_D2D_BPS;
    const tKvRemote = kvB / UTIL_D2D_BPS;
    const tokRemote = tKvRemote > 0 ? 1 / tKvRemote : 0;
    const kvLayerMm2 = kvLayerB * 8 * (node.sramUm2 || 0.021) / 1e6;
    const kvBudgetMm2 = UTIL_DIE_MM2 * UTIL_KV_AREA_FRAC;
    const kvOnDie = kvLayerMm2 <= kvBudgetMm2;
    // Cost levers (roadmap slide): KV precision, batch, MLA latent KV.
    const kvBp = utilPick.mla ? model.layers * 576 * 2 : kvBpp;
    const kvBEffFinal = kvBp * ctx * (utilPick.kvInt8 ? 0.5 : 1);
    const batch = utilPick.batch || 1;
    const eRom = activeB * 8 * node.romPj / 1e12;
    const eKv = kvBEffFinal * 8 * node.sramPj / 1e12;
    const eRomWalk = eRom;                 // per-token weight walk (batch 1)
    const eJ = eRom / batch + eKv;         // weights amortize across the batch; KV does not
    const scale = model.active / 0.6e9;
    const tok = 40000 * (node.clk / 2e9);
    const txRom = UTIL_ROM_T * model.stored * prec.bits;
    const txSram = UTIL_SRAM_TX * dies;
    const txKv = kvBEffFinal * 8 * UTIL_ROM_T;
    const tx = txRom + txSram + txKv;
    const txPerDie = (txRom + txSram) / dies;
    const txKvLayer = txKv / Math.max(model.layers, 1);
    const rivals = UTIL_RIVALS.map((r) => {
      const pass = r.cim ? Math.max(1, Math.ceil(prec.bits / 4)) / 4 : 1;
      const eW = r.e06 * scale * pass;
      const pj = r.kvPj == null ? UTIL_HBM_PJ : r.kvPj;
      const eKvR = kvB * 8 * pj / 1e12;
      const eRival = eW + eKvR;
      const nPkg = r.capKv
        ? Math.max(1, Math.ceil(Math.max(storedBf16B, kvB) / r.memB))
        : Math.max(1, Math.ceil(storedBf16B / r.memB));
      const txR = r.id === "pim"
        ? nPkg * r.txPkg + storedBf16B * 8
        : nPkg * r.txPkg;
      const tokR = r.tokStay
        ? r.tok
        : r.tok * (0.6e9 / model.active) * nPkg;
      return {
        id: r.id,
        k: r.k,
        short: r.short,
        name: r.name,
        where: r.where,
        mem: r.mem,
        tok06: r.tok,
        tok: tokR,
        color: r.color,
        e06: r.e06,
        eJ: eRival,
        eX: eRival / Math.max(eJ, 1e-18),
        tokX: tok / Math.max(tokR, 1e-9),
        nPkg,
        tx: txR,
        txPkg: r.txPkg,
        txPerDie: txR / nPkg,
        tokStay: !!r.tokStay,
        costX: (txR * eRival) / Math.max(tx * eJ, 1e-30)
      };
    });
    const h100J = rivals[0].eJ;
    const eX = rivals[0].eX;
    const tX = tok / Math.max(rivals[0].tok, 1e-9);
    const fits06 = 0.6e9 <= maxP;
    return {
      bitsDie, bytesDie, maxP, storedB, activeB, storedBf16B, dies, ctx, kvBpp, kvB,
      hidden, kvLayerB, actB, d2dActB, eD2d, eKvRemote, tD2dAct, tKvRemote, tokRemote,
      kvLayerMm2, kvBudgetMm2, kvOnDie, txKvLayer,
      eRom, eKv, eJ, h100J, tok, eX, tX, tx, txRom, txSram, txKv, txPerDie, fits06, scale, rivals
    };
  }

  function utilLogAt(v, vmin, vmax, a, span) {
    const t = (Math.log10(Math.max(v, vmin)) - Math.log10(vmin)) / (Math.log10(vmax) - Math.log10(vmin));
    return a + span * Math.max(0, Math.min(1, t));
  }

  function utilCrossover(node, prec, model, rivalIx) {
    rivalIx = rivalIx || 0;
    const xAt = (ctx) => utilSolve(node, prec, model, ctx).rivals[rivalIx].costX;
    const hiMax = 1 << 22;
    if (xAt(hiMax) >= 1) return null;
    if (xAt(1) < 1) return 1;
    let lo = 1, hi = hiMax;
    for (let i = 0; i < 44; i++) {
      const mid = Math.round((lo + hi) / 2);
      if (xAt(mid) < 1) hi = mid;
      else lo = mid;
    }
    return hi;
  }

  function utilPlotSvg(node, prec, model, ctxPick) {
    const W = 960, H = 400, L = 58, R = 22, T = 30, B = 48;
    const iw = W - L - R, ih = H - T - B;
    const xMin = 131072, xMax = 1048576;
    const ctxs = [];
    for (let c = xMin; c <= xMax; c *= 2) ctxs.push(c);
    [262144, 524288].forEach((c) => { if (!ctxs.includes(c)) ctxs.push(c); });
    ctxs.sort((a, b) => a - b);
    const rows = ctxs.map((ctx) => {
      const s = utilSolve(node, prec, model, ctx);
      return { ctx, gpu: s.rivals[0].costX, tpu: s.rivals[1].costX, lpu: s.rivals[2].costX };
    });
    const dead = utilCrossover(node, prec, model, 0);
    const yMin = 0.05;
    const yMax = Math.max(10, Math.max(...rows.flatMap((r) => [r.gpu, r.tpu, r.lpu])) * 1.35);
    const xAt = (c) => utilLogAt(c, xMin, xMax, L, iw);
    const yAt = (v) => T + ih * (1 - (Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)));
    const line = (key, color) => {
      let d = "";
      rows.forEach((r, i) => {
        d += (i ? "L" : "M") + xAt(r.ctx).toFixed(1) + "," + yAt(r[key]).toFixed(1);
      });
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
    };
    let g = "";
    const yMarks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000];
    yMarks.forEach((v) => {
      if (v < yMin * 0.99 || v > yMax) return;
      const y = yAt(v);
      g += `<line class="util-plot-grid" x1="${L}" y1="${y.toFixed(1)}" x2="${(L + iw).toFixed(1)}" y2="${y.toFixed(1)}"/>`;
      g += `<text class="util-plot-lbl" text-anchor="end" x="${L - 8}" y="${(y + 3.5).toFixed(1)}">${v >= 1 ? String(v) : String(v)}×</text>`;
    });
    [131072, 262144, 524288, 1048576].forEach((c) => {
      const x = xAt(c);
      g += `<line class="util-plot-grid" x1="${x.toFixed(1)}" y1="${T}" x2="${x.toFixed(1)}" y2="${(T + ih).toFixed(1)}"/>`;
      g += `<text class="util-plot-lbl" text-anchor="middle" x="${x.toFixed(1)}" y="${H - 20}">${utilFmtCtx(c)}</text>`;
    });
    const y1 = yAt(1);
    g += `<line class="util-plot-par" x1="${L}" y1="${y1.toFixed(1)}" x2="${(L + iw).toFixed(1)}" y2="${y1.toFixed(1)}"/>`;
    g += `<text class="util-plot-parl" x="${(L + 6).toFixed(1)}" y="${(y1 - 5).toFixed(1)}">1× parity · below is unviable</text>`;
    if (dead && dead <= xMax && dead >= xMin) {
      const x0 = xAt(Math.max(dead, xMin)), x1 = L + iw;
      g += `<rect class="util-plot-dead" x="${x0.toFixed(1)}" y="${T}" width="${Math.max(0, x1 - x0).toFixed(1)}" height="${ih}"/>`;
      g += `<text class="util-plot-deadl" x="${(x0 + 8).toFixed(1)}" y="${(T + 16).toFixed(1)}">unviable vs H100 after ${utilFmtCtx(dead)}</text>`;
    } else if (dead && dead < xMin) {
      g += `<rect class="util-plot-dead" x="${L}" y="${T}" width="${iw}" height="${ih}"/>`;
      g += `<text class="util-plot-deadl" x="${L + 8}" y="${T + 16}">unviable vs H100 in this ctx range</text>`;
    }
    g += line("gpu", "#1d4ed8");
    g += line("tpu", "#b45309");
    g += line("lpu", "#6d28d9");
    const here = rows.find((r) => r.ctx === ctxPick) || rows[0];
    g += `<circle cx="${xAt(here.ctx).toFixed(1)}" cy="${yAt(here.gpu).toFixed(1)}" r="7" fill="#1d4ed8" stroke="#0f172a" stroke-width="2"/>`;
    g += `<line class="util-plot-axis" x1="${L}" y1="${(T + ih).toFixed(1)}" x2="${(L + iw).toFixed(1)}" y2="${(T + ih).toFixed(1)}"/>`;
    g += `<line class="util-plot-axis" x1="${L}" y1="${T}" x2="${L}" y2="${(T + ih).toFixed(1)}"/>`;
    g += `<text class="util-plot-axk" x="${L}" y="${T - 12}">cost × cheaper · Tx × energy / token · log · down is worse</text>`;
    g += `<text class="util-plot-axk" text-anchor="middle" x="${L + iw / 2}" y="${H - 4}">context · 128k–1M · KV SRAM ∝ ctx · MoE energy uses 15% active</text>`;
    g += `<text class="util-plot-leg" x="${L + 8}" y="${T + ih - 10}" fill="#1d4ed8">vs H100</text>`;
    g += `<text class="util-plot-leg" x="${L + 78}" y="${T + ih - 10}" fill="#b45309">vs TPU</text>`;
    g += `<text class="util-plot-leg" x="${L + 140}" y="${T + ih - 10}" fill="#6d28d9">vs LPU</text>`;
    const note = dead && dead <= xMax
      ? `Cliff at ${utilFmtCtx(dead)} ctx for ${model.label}. ROM stores every expert; energy walks 15% active. KV SRAM grows with ctx. GPU KV is HBM, not die Tx.`
      : `No cliff by 1M ctx on this pick — ROM Tx still dominates KV even at 1M.`;
    return `<svg class="util-plot-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Cost versus context length from 128k to 1M">${g}</svg>` +
      `<p class="util-abs">${escapeHtml(note)}</p>`;
  }

  function utilTokPhrase(ours, theirs) {
    const q = ours / theirs;
    if (!Number.isFinite(q) || q <= 0) return "— tok/s";
    if (q < 0.97) return utilFmtX(1 / q) + "× slower";
    if (q > 1.03) return utilFmtX(q) + "× more tok/s";
    return "about the same tok/s";
  }

  function paintUtilExplorer() {
    if (!$("utilX")) return;
    if (!(utilPick.ctx >= UTIL_CTX_FLOOR)) utilPick.ctx = UTIL_CTX_FLOOR;
    const node = UTIL_NODES.find((n) => n.id === utilPick.node) || UTIL_NODES[2];
    const prec = UTIL_PREC.find((p) => p.id === utilPick.prec) || UTIL_PREC[0];
    const model = UTIL_MODELS.find((m) => m.id === utilPick.model) || UTIL_MODELS[4];
    const s = utilSolve(node, prec, model, utilPick.ctx);
    const gpu = s.rivals[0];
    const ctxLab = utilFmtCtx(s.ctx);
    const moePct = model.stored ? Math.round(100 * model.active / model.stored) : 100;
    const press = (id, items, key) => {
      const wrap = $(id);
      if (!wrap) return;
      if (!wrap.dataset.ready) {
        wrap.dataset.ready = "1";
        wrap.innerHTML = items.map((it) =>
          `<button type="button" data-util-${key}="${it.id}"${it.prod ? ' class="prod"' : ""}>${escapeHtml(it.label)}</button>`
        ).join("");
      }
      wrap.querySelectorAll("button").forEach((b) => {
        b.setAttribute("aria-pressed", String(b.getAttribute(`data-util-${key}`)) === String(utilPick[key]) ? "true" : "false");
      });
    };
    press("utilNode", UTIL_NODES, "node");
    press("utilPrec", UTIL_PREC, "prec");
    press("utilModel", UTIL_MODELS, "model");
    press("utilCtx", UTIL_CTXS, "ctx");
    // Cost-lever toggles (roadmap): int8 KV, MLA latent KV, serving batch.
    const leverBtn = (id, label, key, val, on) => {
      const wrap = $(id);
      if (!wrap) return;
      if (!wrap.dataset.ready) {
        wrap.dataset.ready = "1";
        wrap.innerHTML = `<button type="button" data-util-${key}="${val}">${escapeHtml(label)}</button>`;
      }
      const b = wrap.querySelector("button");
      if (b) b.setAttribute("aria-pressed", String(on));
    };
    leverBtn("utilKvInt8", "int8 KV", "kvint8", "1", !!utilPick.kvInt8);
    leverBtn("utilMla", "MLA latent KV", "mla", "1", !!utilPick.mla);
    leverBtn("utilBatch", `batch ${utilPick.batch || 1}`, "batch", String(utilPick.batch || 1), (utilPick.batch || 1) > 1);
    const claim = $("utilClaim");
    if (claim) {
      claim.innerHTML =
        `Fixed weights. The measure is <em>transistors × energy / token</em>.` +
        `<span><strong>${utilFmtTx(s.tx)}</strong> · ${s.dies} dies · ${ctxLab}` +
          `${model.active !== model.stored ? " · " + moePct + "% active" : ""} · ` +
          `<strong>${utilFmtX(gpu.eX)}× less energy</strong> · <strong>${utilCostPhrase(gpu.costX)}</strong> vs H100` +
          `${gpu.costX < 1 ? ". Tx × energy loses at this context." : "."}</span>`;
    }
    const kpis = $("utilKpis");
    if (kpis) {
      kpis.innerHTML =
        `<p class="util-x-kpi"><b>${s.dies}</b><span>dies · ${(s.storedB / 1e9).toFixed(s.storedB >= 1e11 ? 0 : 2)} GB ROM at ${escapeHtml(node.label)} ${escapeHtml(prec.label)}</span></p>` +
        `<p class="util-x-kpi"><b>${utilFmtTx(s.tx)}</b><span>transistors · ${utilFmtTx(s.txRom)} ROM + ${utilFmtTx(s.txSram)} workspace + ${utilFmtTx(s.txKv)} KV at ${ctxLab}</span></p>` +
        `<p class="util-x-kpi"><b>${utilFmtTx(s.txPerDie)}</b><span>transistors / die · ROM + workspace; KV SRAM sits on the attention die for that layer</span></p>` +
        `<p class="util-x-kpi"><b>${utilFmtBytes(s.kvB)}</b><span>KV at ${ctxLab} · ${utilFmtBytes(s.kvLayerB)} / layer on the attention die · not striped across ${s.dies} expert dies</span></p>` +
        `<p class="util-x-kpi"><b>${gpu.eX >= 100 ? "≥100×" : utilFmtX(gpu.eX) + "×"}</b><span>energy vs H100 · ${gpu.costX < 1 ? "Tx × energy unviable" : utilCostPhrase(gpu.costX)}</span></p>`;
    }
    const vs = $("utilVs");
    if (vs) {
      const us =
        `<article class="util-board-card us">` +
          `<p class="util-k">This · ROM shard</p>` +
          `<p class="util-board-e">${utilFmtE(s.eJ)}</p>` +
          `<p class="util-board-u">energy / token · ${ctxLab}</p>` +
          `<p class="util-board-t">${utilFmtTok(s.tok)} tok/s</p>` +
          `<p class="util-abs">${utilFmtGhz(node.clk)} · 2D ROM MXU · tok/s / user stays as dies add</p>` +
          `<p class="util-abs"><strong>${s.dies} dies</strong> · ${utilFmtTx(s.tx)} Tx · ${utilFmtTx(s.txPerDie)} ROM+SRAM / die · KV not in that split</p>` +
          `<p class="util-abs">ROM ${utilFmtE(s.eRom)} (active ${utilFmtAct(model.active)}) + KV ${utilFmtE(s.eKv)}</p>` +
        `</article>`;
      const them = s.rivals.map((r) =>
        `<article class="util-board-card ${r.id}">` +
          `<p class="util-k">${escapeHtml(r.short)}</p>` +
          `<p class="util-board-e">${utilFmtE(r.eJ)}</p>` +
          `<p class="util-board-u">energy / token · ${ctxLab}</p>` +
          `<p class="util-board-t">${utilFmtTok(r.tok)} tok/s</p>` +
          `<p class="util-abs">${escapeHtml(r.where)} · ${escapeHtml(r.mem)}</p>` +
          `<p class="util-abs"><strong>${r.nPkg} pkg</strong> · ${utilFmtTx(r.tx)} Tx · ${utilFmtTx(r.txPerDie)} / pkg</p>` +
          `<p class="util-abs"><strong>${utilFmtX(r.eX)}× less energy</strong> · ${utilTxPhrase(s.tx, r.tx)} · <strong>${utilCostPhrase(r.costX)}</strong></p>` +
          `<p class="util-abs"><strong>${utilTokPhrase(s.tok, r.tok)}</strong>${r.tokStay ? " · tok/s / user, not × packages" : ""}</p>` +
        `</article>`
      ).join("");
      vs.innerHTML = us + them;
    }
    const viable = $("utilViable");
    if (viable) {
      const ctxs = [UTIL_CTX_FLOOR, 1048576];
      const rows = UTIL_TARGETS.flatMap((id) => {
        const m = UTIL_MODELS.find((x) => x.id === id);
        return ctxs.map((ctx) => {
          const t = utilSolve(node, prec, m, ctx);
          const g = t.rivals[0];
          const cheap = t.rivals.every((r) => r.costX >= 1);
          const fast = t.rivals.every((r) => r.tokX >= 1);
          const ok = cheap && fast;
          const on = m.id === model.id && ctx === s.ctx;
          return `<tr class="${ok ? "ok" : "mid"}${on ? " on" : ""}">` +
            `<th>${escapeHtml(m.label)}</th>` +
            `<td>${utilFmtCtx(ctx)}</td>` +
            `<td>${t.dies}</td>` +
            `<td>${utilFmtTx(t.tx)}</td>` +
            `<td>${utilFmtTx(t.txPerDie)}</td>` +
            `<td>${utilFmtE(t.eJ)}</td>` +
            `<td>${utilFmtBytes(t.kvLayerB)}</td>` +
            `<td>${utilFmtE(g.eJ)} · ${utilFmtTok(g.tok)} tok/s</td>` +
            `<td>${utilFmtX(g.eX)}× E · ${utilCostPhrase(g.costX)} · ${utilFmtX(g.tokX)}× tok/s</td>` +
            `<td>${ok ? "yes · beats GPU/TPU/LPU/CIM/PIM" : (!cheap ? "no · Tx × energy" : "no · tok/s")}</td>` +
          `</tr>`;
        });
      }).join("");
      viable.innerHTML =
        `<table class="spec-gate util-tbl">` +
          `<thead><tr>` +
            `<th>MoE · 15% active</th><th>Ctx</th><th>Dies</th><th>Tx</th><th>Tx / die</th>` +
            `<th>This energy</th><th>KV / layer</th><th>H100 energy · tok/s</th><th>Gain vs H100</th><th></th>` +
          `</tr></thead><tbody>${rows}</tbody>` +
        `</table>` +
        `<p class="util-abs">${escapeHtml(node.label)} · ${escapeHtml(prec.label)}. ` +
          `Yes = cheaper on Tx × energy vs H100, TPU, LPU, analog CIM, and HBM-PIM, and faster per user than all five. ` +
          `ROM stores every expert; energy walks 15% active. KV SRAM is per attention layer, not striped across expert dies. ` +
          `GPU/TPU die Tx does not include HBM — we can have more counted Tx and still win on Tx × energy. Live pick is ${escapeHtml(model.label)} at ${ctxLab}.</p>`;
    }
    const kvBox = $("utilKv");
    if (kvBox) {
      const place = s.kvOnDie
        ? `fits on the attention die · ${s.kvLayerMm2.toFixed(0)} mm² of ${s.kvBudgetMm2.toFixed(0)} mm² SRAM budget`
        : `does not fit on-die · ${s.kvLayerMm2.toFixed(0)} mm² > ${s.kvBudgetMm2.toFixed(0)} mm² · 1:1 SRAM chiplet, still local`;
      kvBox.innerHTML =
        `<article class="util-board-card us">` +
          `<p class="util-k">Place · with the layer</p>` +
          `<p class="util-board-e">${utilFmtBytes(s.kvLayerB)}</p>` +
          `<p class="util-board-u">KV SRAM / layer · ${model.layers} layers · GQA ${model.kv === "mla" ? "MLA" : model.kvHeads + " KV heads"}</p>` +
          `<p class="util-abs">K and V never leave the die that ran W<sub>K</sub>, W<sub>V</sub> for that layer. Expert ROM dies hold FFN only — no cache.</p>` +
          `<p class="util-abs">${place}</p>` +
        `</article>` +
        `<article class="util-board-card">` +
          `<p class="util-k">Connect · D2D is x</p>` +
          `<p class="util-board-e">${utilFmtBytes(s.actB)}</p>` +
          `<p class="util-board-u">residual / hop · H=${s.hidden} BF16 · ${model.layers} hops = ${utilFmtBytes(s.d2dActB)}</p>` +
          `<p class="util-abs">${utilFmtUs(s.tD2dAct)} on a 1 TB/s package link · ${utilFmtE(s.eD2d)} · does not move the cache</p>` +
          `<p class="util-abs">Same NUMA idea as Jalapeño: compute + its memory, 1:1. Not one shared KV pool behind all dies.</p>` +
        `</article>` +
        `<article class="util-board-card lpu">` +
          `<p class="util-k">Wrong · KV over D2D</p>` +
          `<p class="util-board-e">${utilFmtTok(s.tokRemote)}</p>` +
          `<p class="util-board-u">tok/s if attention fetches ${utilFmtBytes(s.kvB)} of cache over the same 1 TB/s link</p>` +
          `<p class="util-abs">${utilFmtUs(s.tKvRemote)} / token · ${utilFmtE(s.eKvRemote)} D2D vs ${utilFmtE(s.eKv)} local SRAM. That is how 40k tok/s dies.</p>` +
          `<p class="util-abs">Do not stripe KV across the ${s.dies} weight dies. Most of those dies are experts and never run attention.</p>` +
        `</article>` +
        `<article class="util-board-card">` +
          `<p class="util-k">Counted in the economics</p>` +
          `<p class="util-board-e">${utilFmtBytes(s.kvB)}</p>` +
          `<p class="util-board-u">total KV at ${ctxLab} · ${utilFmtBytes(s.kvBpp)}/pos · BF16 even if ROM is not</p>` +
          `<p class="util-abs">${utilFmtTx(s.txKv)} Tx total · ${utilFmtTx(s.txKvLayer)} / attention die · ${utilFmtE(s.eKv)} / token already in the board above</p>` +
          `<p class="util-abs">${s.kvOnDie ? "On-die SRAM. Energy and Tx are in the live pick." : "Chiplet SRAM for KV only. W stays ROM. Energy still SRAM-class, not HBM for W."}</p>` +
        `</article>`;
    }
    const calc = $("utilCalc");
    if (calc) {
      const bits = utilFmtBits(prec.bits);
      const moe = model.active !== model.stored;
      calc.innerHTML =
          `<p>Energy is ROM walk of <em>active</em> W plus SRAM walk of KV at this context. 500B / 1T are MoE at 15% active — Tx uses stored experts; energy and GPU tok/s use the experts that fire. GPU/TPU tok/s = measured 0.6B × (0.6B / N_active) × packages. Ours is 40k × clock/2 GHz. KV stays BF16 and scales with ctx.${prec.id !== "bf16" ? " " + escapeHtml(prec.label) + " shrinks our ROM and our energy walk (precision-aware training: the mask is the checkpoint); rivals stay BF16." : ""}${moe ? " MoE: stored " + utilFmtAct(model.stored) + " · active " + utilFmtAct(model.active) + " (" + moePct + "%)." : ""}</p>` +
          `<ol>` +
            `<li><span>E = N<sub>active</sub> × bits × pJ<sub>ROM</sub> + KV(${ctxLab}) × 8 × pJ<sub>SRAM</sub></span>` +
              `<span>= ${utilFmtE(s.eRom)} + ${utilFmtE(s.eKv)} = <strong>${utilFmtE(s.eJ)}</strong></span></li>` +
            `<li><span>tok/s<sub>us</sub> = 40k × (f / 2 GHz) · GEMM-bound; KV is the energy bill of ctx</span>` +
              `<span>= 40,000 × ${utilFmtGhz(node.clk)} / 2 GHz = <strong>${utilFmtTok(s.tok)}</strong></span></li>` +
            `<li><span>tok/s<sub>rival</sub> · GPU/TPU/PIM = tok<sub>0.6B</sub> × (0.6B / N<sub>active</sub>) × packages. LPU/CIM = tok<sub>0.6B</sub> / user (more chips hold W)</span>` +
              `<span>` +
              s.rivals.map((r) =>
                r.tokStay
                  ? `${escapeHtml(r.short)} ${utilFmtTok(r.tok06)} / user · ${r.nPkg} pkg hold W = <strong>${utilFmtTok(r.tok)}</strong> → ${utilTokPhrase(s.tok, r.tok)}`
                  : `${escapeHtml(r.short)} ${utilFmtTok(r.tok06)} × ${(0.6e9 / model.active).toExponential(2)} × ${r.nPkg} = <strong>${utilFmtTok(r.tok)}</strong> → ${utilTokPhrase(s.tok, r.tok)}`
              ).join("<br>") +
              `</span></li>` +
            `<li><span>KV place · per attention layer, not striped across weight dies</span>` +
              `<span>${utilFmtBytes(s.kvB)} total · ${utilFmtBytes(s.kvLayerB)} / layer · ${s.kvOnDie ? "on-die SRAM" : "1:1 SRAM chiplet"} · D2D is x (${utilFmtBytes(s.actB)} × ${model.layers} = ${utilFmtBytes(s.d2dActB)}, ${utilFmtUs(s.tD2dAct)}). Fetching KV over D2D would be ${utilFmtTok(s.tokRemote)} tok/s.</span></li>` +
            `<li><span>Dies = ceil(stored bytes / bytes per die) · Tx/die = Tx / dies</span>` +
              `<span>${s.dies} dies · ${utilFmtTx(s.tx)} / ${s.dies} = <strong>${utilFmtTx(s.txPerDie)}</strong> · max ${utilFmtP(s.maxP)} params/die</span></li>` +
            `<li><span>E<sub>rival</sub> and Tx<sub>rival</sub> at ${ctxLab}</span>` +
              `<span>` +
              s.rivals.map((r) =>
                `${escapeHtml(r.short)} ${utilFmtE(r.eJ)} · ${r.nPkg} × ${utilFmtTx(r.txPkg)} = ${utilFmtTx(r.tx)} → <strong>${utilFmtX(r.eX)}× less energy</strong> · ${utilCostPhrase(r.costX)}`
              ).join("<br>") +
              `</span></li>` +
          `</ol>`;
    }
    const plot = $("utilPlot");
    if (plot) plot.innerHTML = utilPlotSvg(node, prec, model, utilPick.ctx);
    const heat = $("utilHeat");
    if (heat) {
      let h = `<span class="hh"></span>` + UTIL_PREC.map((p) => `<span class="hh">${escapeHtml(p.label)}</span>`).join("");
      UTIL_NODES.forEach((n) => {
        h += `<span class="hr">${escapeHtml(n.label)}</span>`;
        UTIL_PREC.forEach((p) => {
          const maxP = utilMaxParams(n, p);
          const on = n.id === utilPick.node && p.id === utilPick.prec;
          const no = maxP < 0.6e9;
          h += `<button type="button" class="${on ? "on" : ""} ${no ? "no" : ""}" data-heat-node="${n.id}" data-heat-prec="${p.id}">${utilFmtP(maxP)}</button>`;
        });
      });
      heat.innerHTML = h;
    }
  }


  // Roadmap mechanism bars: stacked weight-walk + KV-read per lever step.
  // Values from utilSolve's model at 500B MoE / 75B active / 128k / N4.
  const ROADMAP_STEPS = [
    { label: "Today\\nBF16 · b1", w: 96.0, k: 27.5 },
    { label: "FP8/INT8\\nQAT", w: 48.0, k: 27.5 },
    { label: "INT4\\nQAT", w: 24.0, k: 27.5 },
    { label: "+int8\\nKV", w: 24.0, k: 13.7 },
    { label: "+batch\\n16", w: 1.5, k: 13.7 },
    { label: "+MLA\\nlatent", w: 1.5, k: 3.9 }
  ];
  function paintRoadmapBars() {
    const host = $("roadmapBars");
    if (!host) return;
    const max = 124; // mJ, scale so the first bar fills the track
    host.innerHTML = ROADMAP_STEPS.map((s, i) => {
      const wH = (s.w / max * 100).toFixed(1);
      const kH = (s.k / max * 100).toFixed(1);
      const tot = (s.w + s.k).toFixed(1);
      const last = i === ROADMAP_STEPS.length - 1;
      return `<div class="rbar${last ? " end" : ""}">` +
        `<div class="rbar-val">${tot} mJ</div>` +
        `<div class="rbar-track">` +
          `<div class="rbar-w" style="height:${wH}%"></div>` +
          `<div class="rbar-k" style="height:${kH}%"></div>` +
        `</div>` +
        `<div class="rbar-lab">${s.label.replace(/\\n/g, "<br>")}</div>` +
      `</div>`;
    }).join("");
  }

  // Roadmap slide live lever strip: same utilSolve model, lever toggles.
  function paintRoadmapLive() {
    const endpoint = $("rmEndpoint");
    if (!endpoint) return;
    const node = UTIL_NODES.find((n) => n.id === "n4") || UTIL_NODES[2];
    const prec = UTIL_PREC.find((p) => p.id === utilPick.prec) || UTIL_PREC[0];
    const model = UTIL_MODELS.find((m) => m.id === "l500") || UTIL_MODELS[4];
    const s = utilSolve(node, prec, model, 131072);
    const gpu = s.rivals[0];
    endpoint.textContent =
      `${utilFmtE(s.eJ)} · ${utilFmtTx(s.tx)} Tx · ${utilFmtX(gpu.costX)}× vs H100`;
    const set = (id, on) => {
      const b = $(id);
      if (b) b.setAttribute("aria-pressed", String(on));
    };
    set("rmKvInt8", !!utilPick.kvInt8);
    set("rmMla", !!utilPick.mla);
    const bb = $("rmBatch");
    if (bb) {
      bb.textContent = `batch ${utilPick.batch || 1}`;
      bb.setAttribute("aria-pressed", String((utilPick.batch || 1) > 1));
    }
    paintRoadmapWalk(node, prec, model, s);
  }

  // Live equation walk + die diagram: which term each lever moves, where it happens.
  function paintRoadmapWalk(node, prec, model, s) {
    const wrap = $("rmWalk");
    if (!wrap) return;
    const ctx = 131072;
    const kvBpp = utilPick.mla ? model.layers * 576 * 2 : 2 * model.layers * model.kvHeads * model.headDim * 2;
    const kvDiv = utilPick.kvInt8 ? 2 : 1;
    const batch = utilPick.batch || 1;
    const kvB = kvBpp * ctx / kvDiv;
    const activeB = model.active * prec.bits / 8;
    const eRomAll = activeB * 8 * node.romPj / 1e12;          // J per token, batch 1
    const eRomTok = eRomAll / batch;                          // J per user-token
    const eKv = kvB * 8 * node.sramPj / 1e12;                 // J per token
    const kvLayerB = kvB / model.layers;
    const kvLayerMm2 = kvLayerB * 8 * node.sramUm2 / 1e6;
    const kvBudgetMm2 = UTIL_DIE_MM2 * UTIL_KV_AREA_FRAC;
    const kvOnDie = kvLayerMm2 <= kvBudgetMm2;
    const fmtM = (j) => (j * 1e3).toFixed(j * 1e3 < 10 ? 2 : 1);
    const strike = (on) => on ? " rm-hit" : "";
    wrap.innerHTML =
      `<div class="rm-walk">` +
        `<div class="rm-eq">` +
          `<p class="rm-eq-title">The only equation that matters</p>` +
          `<p class="rm-eq-line">E<sub>/token</sub> = <span class="rm-term rm-tw${strike(utilPick.prec !== "bf16")}">W-walk</span> + <span class="rm-term rm-tk${strike(utilPick.kvInt8 || utilPick.mla)}">KV-read</span></p>` +
          `<p class="rm-eq-calc">` +
            `<span class="rm-term rm-tw${strike(utilPick.prec !== "bf16")}">` +
              `${utilFmtBytes(activeB)} × 8 × ${node.romPj} pJ ÷ ${batch}` +
              ` = <b>${fmtM(eRomTok)} mJ</b></span>` +
            ` + ` +
            `<span class="rm-term rm-tk${strike(utilPick.kvInt8 || utilPick.mla)}">` +
              `${utilFmtBytes(kvB)} × 8 × ${node.sramPj} pJ` +
              ` = <b>${fmtM(eKv)} mJ</b></span>` +
            ` = <b class="rm-total">${fmtM(eRomTok + eKv)} mJ</b>` +
          `</p>` +
          `<ul class="rm-notes">` +
            `<li class="${utilPick.prec !== "bf16" ? "hit" : ""}"><b>QAT precision</b> shrinks W bytes: ${utilFmtBytes(model.active * 2)} → ${utilFmtBytes(activeB)} per token walk. The mask stores ${prec.bits} bits/weight — trained that way, burnt that way.</li>` +
            `<li class="${batch > 1 ? "hit" : ""}"><b>Batch ${batch}</b> divides the W walk only: ${batch} users share one ROM pass. KV does not divide — each user reads their own cache.</li>` +
            `<li class="${utilPick.kvInt8 ? "hit" : ""}"><b>int8 KV</b> halves the cache: ${utilFmtBytes(kvBpp * ctx)} → ${utilFmtBytes(kvB)} (÷${kvDiv}).</li>` +
            `<li class="${utilPick.mla ? "hit" : ""}"><b>MLA</b> compresses per-token KV to a 576-dim latent: ${utilFmtBytes(2 * model.layers * model.kvHeads * model.headDim * 2 * ctx / kvDiv)} → ${utilFmtBytes(kvB)}.</li>` +
          `</ul>` +
        `</div>` +
        `<div class="rm-die">` +
          `<p class="rm-eq-title">Where it happens on the die</p>` +
          `<svg viewBox="0 0 340 210" class="rm-die-svg" role="img" aria-label="Attention die: ROM weights and KV SRAM regions with the active term highlighted">` +
            `<rect x="8" y="8" width="324" height="194" rx="10" class="rm-die-outline"/>` +
            `<rect x="22" y="24" width="180" height="120" rx="6" class="rm-rom${utilPick.prec !== "bf16" ? " hit" : ""}"/>` +
            `<text x="112" y="76" text-anchor="middle" class="rm-die-t">ROM · W</text>` +
            `<text x="112" y="94" text-anchor="middle" class="rm-die-s">${prec.bits} b/weight · QAT mask</text>` +
            `<text x="112" y="112" text-anchor="middle" class="rm-die-s">walk ÷${batch} (batch)</text>` +
            `<rect x="216" y="24" width="100" height="120" rx="6" class="rm-kv${(utilPick.kvInt8 || utilPick.mla) ? " hit" : ""}"/>` +
            `<text x="266" y="64" text-anchor="middle" class="rm-die-t">KV SRAM</text>` +
            `<text x="266" y="82" text-anchor="middle" class="rm-die-s">${utilPick.mla ? "MLA 576-d" : "8×128 heads"}</text>` +
            `<text x="266" y="100" text-anchor="middle" class="rm-die-s">${utilPick.kvInt8 ? "int8" : "fp16"}</text>` +
            `<text x="266" y="118" text-anchor="middle" class="rm-die-s ${kvOnDie ? "" : "warn"}">${utilFmtBytes(kvLayerB)}/layer</text>` +
            `<text x="266" y="134" text-anchor="middle" class="rm-die-s ${kvOnDie ? "" : "warn"}">${kvOnDie ? "fits on-die" : "needs KV chiplet"}</text>` +
            `<rect x="22" y="158" width="294" height="30" rx="6" class="rm-mxu"/>` +
            `<text x="169" y="177" text-anchor="middle" class="rm-die-s">2D ROM MXU · 40k tok/s · D2D carries x, not KV</text>` +
          `</svg>` +
          `<p class="rm-die-cap">KV/layer ${utilFmtBytes(kvLayerB)} = ${kvLayerMm2.toFixed(0)} mm² of ${kvBudgetMm2.toFixed(0)} mm² budget ${kvOnDie ? "· stays local" : "· spills"}</p>` +
        `</div>` +
      `</div>`;
  }

  function bindRoadmapLive() {
    const toggle = (key) => {
      utilPick[key] = !utilPick[key];
      paintRoadmapLive();
    };
    const kv = $("rmKvInt8");
    if (kv) kv.addEventListener("click", () => toggle("kvInt8"));
    const mla = $("rmMla");
    if (mla) mla.addEventListener("click", () => toggle("mla"));
    const batch = $("rmBatch");
    if (batch) batch.addEventListener("click", () => {
      const seq = [1, 8, 16, 32];
      utilPick.batch = seq[(seq.indexOf(utilPick.batch || 1) + 1) % seq.length];
      paintRoadmapLive();
    });
    paintRoadmapLive();
    paintRoadmapBars();
  }

  function bindUtilExplorer() {    const go = (key, id) => {
      if (!id) return;
      utilPick[key] = id;
      paintUtilExplorer();
    };
    const root = $("utilX");
    if (!root) return;
    root.addEventListener("click", (ev) => {
      const dot = ev.target.closest("[data-plot-node]");
      if (dot) {
        utilPick.node = dot.getAttribute("data-plot-node");
        utilPick.prec = dot.getAttribute("data-plot-prec");
        const mid = dot.getAttribute("data-plot-model");
        if (mid) utilPick.model = mid;
        paintUtilExplorer();
        return;
      }
      const b = ev.target.closest("button");
      if (!b) return;
      if (b.hasAttribute("data-util-node")) go("node", b.getAttribute("data-util-node"));
      if (b.hasAttribute("data-util-prec")) go("prec", b.getAttribute("data-util-prec"));
      if (b.hasAttribute("data-util-model")) go("model", b.getAttribute("data-util-model"));
      if (b.hasAttribute("data-util-ctx")) {
        utilPick.ctx = Number(b.getAttribute("data-util-ctx"));
        paintUtilExplorer();
        return;
      }
      if (b.hasAttribute("data-util-kvint8")) {
        utilPick.kvInt8 = !utilPick.kvInt8;
        paintUtilExplorer();
        return;
      }
      if (b.hasAttribute("data-util-mla")) {
        utilPick.mla = !utilPick.mla;
        paintUtilExplorer();
        return;
      }
      if (b.hasAttribute("data-util-batch")) {
        // Cycle 1 -> 8 -> 16 -> 32 -> 1
        const seq = [1, 8, 16, 32];
        const cur = seq.indexOf(utilPick.batch || 1);
        utilPick.batch = seq[(cur + 1) % seq.length];
        paintUtilExplorer();
        return;
      }
      if (b.hasAttribute("data-heat-node")) {
        utilPick.node = b.getAttribute("data-heat-node");
        utilPick.prec = b.getAttribute("data-heat-prec");
        paintUtilExplorer();
      }
    });
    paintUtilExplorer();
  }

  function renderNnMath() {
    const root = document.querySelector(".slide-eqs");
    if (!root || root.dataset.math === "1" || typeof renderMathInElement !== "function") return;
    renderMathInElement(root, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false,
      strict: false
    });
    root.dataset.math = "1";
  }

  // Measured one-token RTL occupancy (evidence/calibration/*_stage_breakdown.json).
  const FUSED_STAGE_CYCLES = {
    EMBED: 5120, INPUT_NORM: 213220, Q_PROJ: 3734752, K_PROJ: 1869280, V_PROJ: 12873728,
    Q_NORM: 3555328, K_NORM: 1777664, ROPE: 688128, ATTN_SCORE: 29120, SOFTMAX: 6720,
    ATTN_VALUE: 516096, O_PROJ: 3707872, ATTN_RESIDUAL: 258048, POST_NORM: 213220,
    GATE_PROJ: 5600224, UP_PROJ: 5600224, SILU: 688128, DOWN_PROJ: 5546464,
    MLP_RESIDUAL: 258048, FINAL_NORM: 7615, LM_HEAD: 9875976, ARGMAX: 1
  };
  const UNFUSED_STAGE_CYCLES = {
    EMBED: 5120, INPUT_NORM: 13017088, Q_PROJ: 25747456, K_PROJ: 12873728, V_PROJ: 12873728,
    Q_NORM: 3555328, K_NORM: 1777664, ROPE: 688128, ATTN_SCORE: 29120, SOFTMAX: 6720,
    ATTN_VALUE: 516096, O_PROJ: 25718784, ATTN_RESIDUAL: 258048, POST_NORM: 13017088,
    GATE_PROJ: 38621184, UP_PROJ: 38621184, SILU: 688128, DOWN_PROJ: 38563840,
    MLP_RESIDUAL: 258048, FINAL_NORM: 464896, LM_HEAD: 68067328, ARGMAX: 1
  };
  function stageCycleMap(sched) {
    const sb = state.stageBreakdown && state.stageBreakdown.schedules && state.stageBreakdown.schedules[sched];
    const fallback = sched === "unfused" ? UNFUSED_STAGE_CYCLES : FUSED_STAGE_CYCLES;
    if (!sb) return fallback;
    const map = {};
    for (const r of sb.stages || []) map[r.stage] = r.cycles;
    return Object.keys(map).length ? map : fallback;
  }

  function fmtCyc(n) {
    const v = Number(n) || 0;
    if (v >= 1e6) return (v / 1e6).toFixed(v >= 10e6 ? 0 : 1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(v >= 10e3 ? 0 : 1) + "k";
    return String(Math.round(v));
  }

  function paintNnHeat() {
    const fused = stageCycleMap("fused");
    const unfused = stageCycleMap("unfused");
    const maxCyc = Math.max(...Object.values(unfused), 1);
    const clock = 500e6;
    document.querySelectorAll(".nn-pane [data-stage]").forEach((n) => {
      const pane = n.closest(".nn-pane");
      const sched = pane && pane.classList.contains("fused") ? "fused" : "unfused";
      const map = sched === "fused" ? fused : unfused;
      const stage = n.getAttribute("data-stage");
      const cyc = map[stage] || 0;
      const bar = n.querySelector(".nn-bar");
      const counts = n.querySelector(".nn-counts");
      if (bar) bar.style.setProperty("--w", (cyc / maxCyc).toFixed(3));
      if (counts) counts.textContent = fmtCyc(cyc);
      const ms = cyc / clock * 1e3;
      n.title = `${stage} · ${sched} ${Math.round(cyc).toLocaleString("en-US")} cycles · ${ms < 1 ? ms.toFixed(2) : ms.toFixed(1)} ms @ 500 MHz`;
    });
  }

  const NN_CHUNKS = 4;
  const NN_OUT = 6;
  const NN_LANES = 8;
  const NN_VB = { w: 660, h: 300 };
  const nnAnim = { t: 0, scene: "gemm", playing: false, timer: null };

  function nnXY(ux, uy) {
    return { left: (ux / NN_VB.w * 100).toFixed(3) + "%", top: (uy / NN_VB.h * 100).toFixed(3) + "%" };
  }
  const NN_POS = {
    sram: (k) => ({ x: 36 + k * 34, y: 66 }),
    act: (k) => ({ x: 224 + k * 34, y: 66 }),
    rom: (k) => ({ x: 224 + k * 34, y: 236 }),
    macX: { x: 430, y: 62 },
    macW: { x: 462, y: 108 },
    macOut: { x: 520, y: 90 },
    y: (j) => ({ x: 602, y: 50 + j * 20 }),
    sHold: { x: 258, y: 66 }
  };

  function nnCells(kind, x0, y0, n, lab) {
    return Array.from({ length: n }, (_, i) => {
      const x = x0 + i * 34;
      const id = `${kind}-${i}`;
      return `<g><rect class="nn-cell" data-cell="${id}" x="${x}" y="${y0}" width="28" height="28" rx="4"/>` +
        `<text class="nn-cell-t" data-ct="${id}" x="${x + 14}" y="${y0 + 15}">${lab(i)}</text></g>`;
    }).join("");
  }

  const NN_HOLD = { fused: 1, gpu: 1, tpu: 1, lpu: 1, cim: 1, pim: 1 };
  const NN_W_BANK = { gpu: 1, tpu: 1, lpu: 1, cim: 1, pim: 1 };
  const NN_LAB = {
    gpu: { arr: "nnArrG", fill: "#1d4ed8", sram: "HBM · x", act: "SRAM · x tile", rom: "HBM · W 1.19 GB", mac: "Tensor Core · MMA", y: "store y[j]", acc: "MMA acc  ·  store tile", wOnce: "once", wPj: "~3.5 pJ/bit", hold: true },
    tpu: { arr: "nnArrT", fill: "#b45309", sram: "HBM · x", act: "VMEM · x", rom: "HBM · W", mac: "MXU · 128×128 MAC grid", y: "store y[j]", acc: "W sits  ·  x pulses through", wOnce: "once", wPj: "~5 pJ/bit", hold: true },
    lpu: { arr: "nnArrL", fill: "#6d28d9", sram: "SRAM · x", act: "SRAM fabric", rom: "SRAM · W 1.19 GB", mac: "EXU · static", y: "store y[j]", acc: "deterministic  ·  no ISA", wOnce: "once", wPj: "~0.5 pJ/bit", hold: true },
    cim: { arr: "nnArrC", fill: "#be123c", sram: "DAC · x", act: "row voltages", rom: "G = W · analog cell", mac: "Kirchhoff + ADC", y: "ADC y[j]", acc: "current sum  ·  4 passes BF16", wOnce: "once", wPj: "ADC ~5 pJ", hold: true },
    pim: { arr: "nnArrP", fill: "#0e7490", sram: "row buf · x", act: "bank-local x", rom: "DRAM · W", mac: "bank GEMV", y: "store y[j]", acc: "in-bank  ·  ~2 pJ/bit", wOnce: "once", wPj: "~2 pJ/bit", hold: true },
    fused: { arr: "nnArrF", fill: "#0f766e", sram: "SRAM · x[1024]", act: "act_mem · resident x", rom: "ROM · burnt-in W", mac: "MAC · 16-lane BF16", y: "commit y[j]", acc: "acc f32  ·  bf16_rne at commit", wOnce: "once", wPj: "~0.8 pJ/bit", hold: true },
    unfused: { arr: "nnArrU", fill: "#c2410c", sram: "SRAM · x[1024]", act: "act_mem · unused", rom: "ROM · W[j, chunk]", mac: "MAC · 16-lane BF16", y: "commit y[j]", acc: "acc f32  ·  bf16_rne at commit", wOnce: "", wPj: "re-read every j", hold: false }
  };

  function buildNnStages() {
    document.querySelectorAll(".nn-stage").forEach((stage) => {
      const schem = stage.querySelector(".nn-schem");
      if (!schem || schem.dataset.ready) return;
      const side = stage.getAttribute("data-side");
      const lab = NN_LAB[side] || NN_LAB.unfused;
      const who = (NN_SIDE_META[side] || {}).name || side;
      if (!stage.querySelector(".nn-badge") && stage.id !== "nnMechStage") {
        const badge = document.createElement("p");
        badge.className = "nn-badge";
        badge.textContent = who;
        schem.before(badge);
      }
      const arrId = stage.id === "nnMechStage" ? lab.arr + "m" : lab.arr;
      schem.dataset.ready = "1";
      schem.innerHTML =
        `<svg class="nn-svg" viewBox="0 0 ${NN_VB.w} ${NN_VB.h}" role="img" aria-hidden="true">` +
        `<defs><marker id="${arrId}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">` +
        `<path d="M0 0 L10 5 L0 10 z" fill="${lab.fill}"/></marker></defs>` +
        `<path class="nn-wire" data-wire="sram-act" d="M160 70 H200"/>` +
        `<path class="nn-wire" data-wire="act-mac" d="M348 70 H388"/>` +
        `<path class="nn-wire bypass" data-wire="sram-mac" d="M86 126 C 86 170 300 170 388 110"/>` +
        `<path class="nn-wire" data-wire="rom-mac" d="M348 236 H462 V166"/>` +
        `<path class="nn-wire" data-wire="mac-y" d="M536 90 H556"/>` +
        `<text class="nn-wire-lab" data-wlab="sram-mac" x="200" y="162">${lab.hold ? "" : lab.wPj}</text>` +
        `<text class="nn-wire-lab" data-wlab="sram-act" x="164" y="62">${lab.wOnce}</text>` +
        `<text class="nn-wire-lab" data-wlab="rom-mac" x="400" y="252">${lab.hold ? lab.wPj : ""}</text>` +
        `<g class="nn-node" data-blk="sram"><rect class="nn-node-bg" x="12" y="16" width="148" height="110" rx="8"/>` +
        `<text class="nn-node-k" x="20" y="34">${lab.sram}</text>${nnCells("sram", 22, 52, NN_CHUNKS, (i) => "x" + i)}</g>` +
        `<g class="nn-node${lab.hold ? "" : " dim"}" data-blk="act"><rect class="nn-node-bg" x="200" y="16" width="148" height="110" rx="8"/>` +
        `<text class="nn-node-k" x="208" y="34">${lab.act}</text>` +
        `${nnCells("act", 210, 52, NN_CHUNKS, () => "·")}</g>` +
        `<g class="nn-node" data-blk="mac"><rect class="nn-node-bg" x="388" y="16" width="148" height="150" rx="8"/>` +
        `<text class="nn-node-k" x="396" y="34">${lab.mac}</text>` +
        `${Array.from({ length: NN_LANES }, (_, i) => {
          const x = 400 + (i % 4) * 32;
          const y = 44 + Math.floor(i / 4) * 22;
          return `<rect class="nn-lane" data-lane="${i}" x="${x}" y="${y}" width="26" height="18" rx="3"/>`;
        }).join("")}` +
        `<rect class="nn-accbar-bg" x="400" y="96" width="124" height="10" rx="2"/>` +
        `<rect class="nn-accbar" data-accbar x="400" y="96" width="0" height="10" rx="2"/>` +
        `<text class="nn-node-k" x="400" y="124">${lab.acc}</text></g>` +
        `<g class="nn-node" data-blk="y"><rect class="nn-node-bg" x="556" y="16" width="92" height="150" rx="8"/>` +
        `<text class="nn-node-k" x="564" y="34">${lab.y}</text>` +
        `${Array.from({ length: NN_OUT }, (_, i) => {
          const y = 42 + i * 20;
          return `<g><rect class="nn-cell" data-cell="y-${i}" x="566" y="${y}" width="72" height="16" rx="3"/>` +
            `<text class="nn-cell-t" data-ct="y-${i}" x="602" y="${y + 9}">·</text></g>`;
        }).join("")}</g>` +
        `<g class="nn-node" data-blk="rom"><rect class="nn-node-bg" x="200" y="186" width="148" height="100" rx="8"/>` +
        `<text class="nn-node-k" x="208" y="204">${lab.rom}</text>` +
        `${nnCells("rom", 210, 222, NN_CHUNKS, () => "W")}</g>` +
        `</svg>` +
        `<span class="nn-pkt" data-pkt="x" hidden>x0</span>` +
        `<span class="nn-pkt w" data-pkt="w" hidden>W</span>` +
        `<span class="nn-pkt y" data-pkt="y" hidden>y</span>`;
    });
  }

  function nnLed(stage, name, on) {
    const el = stage.querySelector(`[data-led="${name}"]`);
    if (el) el.classList.toggle("on", !!on);
  }
  function nnBlk(stage, name, on) {
    const el = stage.querySelector(`[data-blk="${name}"]`);
    if (el) el.classList.toggle("on", !!on);
  }
  function nnWire(stage, name, live, dim) {
    const el = stage.querySelector(`[data-wire="${name}"]`);
    if (!el) return;
    el.classList.toggle("live", !!live);
    el.classList.toggle("dim", !!dim && !live);
  }
  function nnMarkCells(stage, kind, n, fn) {
    for (let i = 0; i < n; i++) {
      const cell = stage.querySelector(`[data-cell="${kind}-${i}"]`);
      const ct = stage.querySelector(`[data-ct="${kind}-${i}"]`);
      const m = fn(i) || {};
      if (cell) {
        cell.classList.toggle("fill", !!m.fill);
        cell.classList.toggle("hot", !!m.hot);
        cell.classList.toggle("done", !!m.done);
      }
      if (ct) {
        if (m.text != null) ct.textContent = m.text;
        ct.style.fill = (m.hot || m.done) ? "#fff" : "";
      }
    }
  }
  function nnPlace(stage, name, pos, show, text) {
    const el = stage.querySelector(`[data-pkt="${name}"]`);
    if (!el) return;
    if (el._anim) {
      el._anim.cancel();
      el._anim = null;
    }
    if (!show) {
      el.hidden = true;
      return;
    }
    if (text != null) el.textContent = text;
    const xy = nnXY(pos.x, pos.y);
    el.hidden = false;
    el.style.left = xy.left;
    el.style.top = xy.top;
  }
  function nnFly(stage, name, from, to, text) {
    const el = stage.querySelector(`[data-pkt="${name}"]`);
    if (!el) return;
    if (text != null) el.textContent = text;
    const a = nnXY(from.x, from.y);
    const b = nnXY(to.x, to.y);
    if (el._anim) {
      el._anim.cancel();
      el._anim = null;
    }
    el.hidden = false;
    el.style.left = a.left;
    el.style.top = a.top;
    if (typeof el.animate !== "function") return;  // jsdom/test env: skip animation
    el._anim = el.animate(
      [
        { left: a.left, top: a.top },
        { left: b.left, top: b.top }
      ],
      { duration: 820, easing: "ease-in-out", fill: "forwards" }
    );
  }

  function paintNnAnim() {
    document.querySelectorAll(".nn-stage").forEach((stage) => {
      const side = stage.getAttribute("data-side");
      if (nnAnim.scene === "norm") paintNnNorm(stage, side, nnAnim.t);
      else paintNnGemm(stage, side, nnAnim.t);
    });
    const hint = $("nnAnimHint");
    if (hint) {
      hint.textContent = nnAnim.scene === "norm"
        ? "GPU/TPU/LPU already sum once. Unfused ASIC re-sums ||x||² for every y. RMSNorm is not the HBM bill."
        : "GPU/TPU: W flies HBM every beat. LPU: W already in SRAM. ASIC fused: same motion, W is ROM. Unfused: also re-reads x.";
    }
  }

  function paintNnWhere() {
    const gemm = nnAnim.scene !== "norm";
    const hot = gemm ? "q_proj" : "input_norm";
    const root = $("nnMap");
    if (root) {
      root.querySelectorAll(".op, .net-node").forEach((el) => {
        el.classList.toggle("active", el.getAttribute("data-op") === hot);
      });
      const svg = $("nnDagEdges");
      if (svg) {
        svg.querySelectorAll(".edge, .edge-lab").forEach((e) => e.classList.remove("live"));
        restoreEdgeLive(svg, $("nnDag"));
      }
    }
    document.querySelectorAll("#nnWhere [data-nn-note], #nnWhere [data-nn-eq]").forEach((el) => {
      const key = el.getAttribute("data-nn-note") || el.getAttribute("data-nn-eq");
      el.hidden = key !== (gemm ? "gemm" : "norm");
    });
  }

  function paintNnGemm(stage, side, t) {
    const hold = !!NN_HOLD[side];
    const wBank = !!NN_W_BANK[side];
    const tpu = side === "tpu";
    const now = stage.querySelector("[data-now]");
    const meta = stage.querySelector("[data-meta]");
    const accEl = stage.querySelector("[data-acc]");
    const bar = stage.querySelector("[data-accbar]");
    const actBlk = stage.querySelector("[data-blk=act]");
    if (actBlk) actBlk.classList.toggle("dim", !hold);
    let j = 0, k = 0, acc = 0, reading = false, loading = false, mac = false, commit = false;
    const committed = Array(NN_OUT).fill(false);
    if (!hold) {
      const per = NN_CHUNKS * 2;
      const tt = t % (NN_OUT * per);
      j = Math.floor(tt / per);
      const rem = tt % per;
      k = Math.floor(rem / 2);
      reading = rem % 2 === 0;
      mac = !reading;
      acc = k + (mac ? 1 : 0);
      for (let i = 0; i < j; i++) committed[i] = true;
      commit = mac && k === NN_CHUNKS - 1;
      if (commit) committed[j] = true;
      if (now) {
        now.textContent = reading
          ? `j=${j}: sram_req x[${k}] — same 1024-vector as j=0. act_mem is idle.`
          : `j=${j}: mac_valid  x[${k}]×W[${j},${k}]  → acc  (${acc}/${NN_CHUNKS})`;
      }
    } else {
      const load = NN_CHUNKS;
      const cycle = load + NN_OUT * NN_CHUNKS;
      const tt = t % cycle;
      if (tt < load) {
        loading = true;
        k = tt;
        if (now) {
          now.textContent = side === "gpu"
            ? `HBM→SRAM: load x[${k}]. 2 KB — cheap. The bill is still W.`
            : side === "tpu"
              ? `HBM→VMEM: load x[${k}]. MXU idle until W tiles arrive.`
              : side === "lpu"
                ? `SRAM x[${k}] onto the fabric. No HBM. 1.19 GB already on-die.`
                : `S_LOAD_ACT: sram_req x[${k}] → act_mem[${k}]. Only SRAM pass this GEMM.`;
        }
      } else {
        const t2 = tt - load;
        j = Math.floor(t2 / NN_CHUNKS);
        k = t2 % NN_CHUNKS;
        mac = true;
        acc = k + 1;
        for (let i = 0; i < j; i++) committed[i] = true;
        commit = k === NN_CHUNKS - 1;
        if (commit) committed[j] = true;
        if (now) {
          now.textContent = side === "gpu"
            ? `j=${j}: hbm_rd W[${j},${k}] ~3.5 pJ/bit off-chip. MMA over SRAM-resident x.`
            : side === "tpu"
              ? `j=${j}: x pulses across the MAC grid. hbm_rd W[${j},${k}] ~5 pJ/bit — that fetch is the bill.`
              : side === "lpu"
                ? `j=${j}: sram_rd W[${j},${k}] ~0.5 pJ/bit. Static EXU — on-die, not ROM-dense.`
                : `S_RUN j=${j}: SRAM idle. rom_req W[${j},${k}] over resident x → MAC.`;
        }
      }
    }
    nnBlk(stage, "sram", reading || loading);
    nnBlk(stage, "act", hold && (loading || mac));
    nnBlk(stage, "rom", mac || wBank);
    nnBlk(stage, "mac", mac);
    nnBlk(stage, "y", commit);
    nnWire(stage, "sram-act", hold && loading, !hold);
    nnWire(stage, "act-mac", hold && mac, !hold);
    nnWire(stage, "sram-mac", !hold && (reading || mac), hold);
    nnWire(stage, "rom-mac", mac, false);
    nnWire(stage, "mac-y", commit, false);
    nnLed(stage, "sram", reading || loading);
    nnLed(stage, "rom", mac);
    nnLed(stage, "mac", mac);
    nnMarkCells(stage, "sram", NN_CHUNKS, (i) => ({
      fill: hold ? (loading && i <= k) : ((reading || mac) && i === k),
      hot: (reading || loading) && i === k,
      text: "x" + i
    }));
    nnMarkCells(stage, "act", NN_CHUNKS, (i) => {
      if (!hold) return { fill: false, hot: false, text: "·" };
      const filled = loading ? i <= k : true;
      return { fill: filled, hot: (loading || mac) && i === k, text: filled ? "x" + i : "·" };
    });
    nnMarkCells(stage, "rom", NN_CHUNKS, (i) => ({
      fill: wBank ? true : (mac && i === k),
      hot: mac && i === k,
      text: mac && i === k ? "W" + j : "W"
    }));
    stage.querySelectorAll("[data-lane]").forEach((el, i) => {
      el.classList.toggle("fill", mac);
      el.classList.toggle("wave", tpu && mac && (i % 4) === (k % 4));
    });
    if (bar) bar.setAttribute("width", acc ? String(8 + 116 * (acc / NN_CHUNKS)) : "0");
    nnMarkCells(stage, "y", NN_OUT, (i) => ({
      done: committed[i],
      hot: commit && i === j,
      text: committed[i] ? "y" + i : "·"
    }));
    if (!hold && reading) {
      nnFly(stage, "x", NN_POS.sram(k), NN_POS.macX, "x" + k);
      nnPlace(stage, "w", NN_POS.rom(0), false);
    } else if (!hold && mac) {
      nnPlace(stage, "x", NN_POS.macX, true, "x" + k);
      nnFly(stage, "w", NN_POS.rom(k), NN_POS.macW, "W" + j);
    } else if (hold && loading) {
      nnFly(stage, "x", NN_POS.sram(k), NN_POS.act(k), "x" + k);
      nnPlace(stage, "w", NN_POS.rom(0), false);
    } else if (hold && mac) {
      nnFly(stage, "x", NN_POS.act(k), NN_POS.macX, "x" + k);
      nnFly(stage, "w", NN_POS.rom(k), NN_POS.macW, "W" + j);
    } else {
      nnPlace(stage, "x", NN_POS.sram(0), false);
      nnPlace(stage, "w", NN_POS.rom(0), false);
    }
    if (commit) nnFly(stage, "y", NN_POS.macOut, NN_POS.y(j), "y" + j);
    else nnPlace(stage, "y", NN_POS.y(0), false);
    if (accEl) accEl.textContent = acc ? acc + " / " + NN_CHUNKS + " chunks" : "0";
    if (meta) {
      const tag = side === "gpu" ? "H100 cuBLAS  ·  "
        : side === "tpu" ? "TPU v5e MXU  ·  "
        : side === "lpu" ? "Groq LPU  ·  "
        : hold ? "qwen3_stream_gemm  ·  " : "unfused reduce  ·  ";
      meta.textContent = tag + `j ${j}/${NN_OUT} · k ${k}/${NN_CHUNKS} · clk ${t}`;
    }
  }

  function paintNnNorm(stage, side, t) {
    const hold = !!NN_HOLD[side];
    const wBank = !!NN_W_BANK[side];
    const now = stage.querySelector("[data-now]");
    const meta = stage.querySelector("[data-meta]");
    const accEl = stage.querySelector("[data-acc]");
    const bar = stage.querySelector("[data-accbar]");
    const actBlk = stage.querySelector("[data-blk=act]");
    if (actBlk) actBlk.classList.toggle("dim", false);
    let j = 0, k = 0, acc = 0, summing = false, scaling = false, loading = false, commit = false;
    const committed = Array(NN_OUT).fill(false);
    if (!hold) {
      const per = NN_CHUNKS + 1;
      const tt = t % (NN_OUT * per);
      j = Math.floor(tt / per);
      const rem = tt % per;
      summing = rem < NN_CHUNKS;
      scaling = !summing;
      k = summing ? rem : NN_CHUNKS - 1;
      acc = summing ? k + 1 : NN_CHUNKS;
      for (let i = 0; i < j; i++) committed[i] = true;
      commit = scaling;
      if (commit) committed[j] = true;
      if (now) {
        now.textContent = summing
          ? `j=${j}: Σ x² chunk ${k} — same reduction as j=0. s is thrown away after this y.`
          : `j=${j}: scale y=γ x/√s  (s just recomputed)`;
      }
    } else {
      const load = NN_CHUNKS;
      const cycle = load + NN_OUT;
      const tt = t % cycle;
      if (tt < load) {
        loading = true;
        summing = true;
        k = tt;
        acc = k + 1;
        if (now) {
          now.textContent = side === "gpu"
            ? `Σ x² chunk ${k} in SRAM. GPU already sums once — RMSNorm is not the HBM bill.`
            : side === "tpu"
              ? `Σ x² in VMEM chunk ${k}. MXU is idle — RMSNorm is not the HBM bill.`
              : side === "lpu"
                ? `Σ x² chunk ${k} on SRAM. LPU already sums once.`
                : `Σ x² chunk ${k} → hold s in norm_sum_squares_q. One reduction.`;
        }
      } else {
        scaling = true;
        j = tt - load;
        k = NN_CHUNKS - 1;
        acc = NN_CHUNKS;
        commit = true;
        for (let i = 0; i <= j; i++) committed[i] = true;
        if (now) {
          now.textContent = side === "gpu"
            ? `j=${j}: scale γ·x/√s. γ is a 2 KB HBM read — nothing like W.`
            : side === "tpu"
              ? `j=${j}: scale γ from HBM (tiny). The expensive wire is still GEMM W.`
              : side === "lpu"
                ? `j=${j}: scale γ from SRAM. Capacity, not this kernel, is what breaks at 1T.`
                : `j=${j}: scale only. s held — C_ADVANCE, no re-reduce.`;
        }
      }
    }
    nnBlk(stage, "sram", summing || loading);
    nnBlk(stage, "act", hold && (loading || scaling));
    nnBlk(stage, "rom", scaling);
    nnBlk(stage, "mac", summing || loading);
    nnBlk(stage, "y", scaling);
    nnWire(stage, "sram-act", hold && loading, !hold);
    nnWire(stage, "act-mac", hold && scaling, !hold);
    nnWire(stage, "sram-mac", !hold && summing, hold);
    nnWire(stage, "rom-mac", scaling, false);
    nnWire(stage, "mac-y", scaling, false);
    nnLed(stage, "sram", summing || loading);
    nnLed(stage, "rom", scaling);
    nnLed(stage, "mac", summing || loading);
    nnMarkCells(stage, "sram", NN_CHUNKS, (i) => ({
      fill: (summing || loading) && i <= k,
      hot: (summing || loading) && i === k,
      text: "x" + i
    }));
    nnMarkCells(stage, "act", NN_CHUNKS, (i) => {
      if (!hold) return { fill: summing && i <= k, hot: summing && i === k, text: summing ? "Σ" : "·" };
      const has = loading ? i <= k : true;
      const holdS = has && !loading && i === 0;
      return { fill: has, hot: loading && i === k, text: holdS ? "s" : (has ? "x" + i : "·") };
    });
    nnMarkCells(stage, "rom", NN_CHUNKS, (i) => ({
      fill: wBank ? i === 0 : (scaling && i === 0),
      hot: scaling && i === 0,
      text: i === 0 ? "γ" : "·"
    }));
    stage.querySelectorAll("[data-lane]").forEach((el) => el.classList.toggle("fill", summing || loading));
    if (bar) bar.setAttribute("width", acc ? String(8 + 116 * (acc / NN_CHUNKS)) : "0");
    nnMarkCells(stage, "y", NN_OUT, (i) => ({
      done: committed[i],
      hot: commit && i === j,
      text: committed[i] ? "y" + i : "·"
    }));
    if (!hold && summing) {
      nnFly(stage, "x", NN_POS.sram(k), NN_POS.macX, "x" + k);
      nnPlace(stage, "w", NN_POS.rom(0), false);
    } else if (!hold && scaling) {
      nnPlace(stage, "x", NN_POS.macX, false);
      nnFly(stage, "w", NN_POS.rom(0), NN_POS.macW, "γ");
    } else if (hold && loading) {
      nnFly(stage, "x", NN_POS.sram(k), NN_POS.act(k), "x" + k);
      nnPlace(stage, "w", NN_POS.rom(0), false);
    } else if (hold && scaling) {
      nnPlace(stage, "x", NN_POS.sHold, true, "s");
      nnFly(stage, "w", NN_POS.rom(0), NN_POS.macW, "γ");
    } else {
      nnPlace(stage, "x", NN_POS.sram(0), false);
      nnPlace(stage, "w", NN_POS.rom(0), false);
    }
    if (commit) nnFly(stage, "y", NN_POS.macOut, NN_POS.y(j), "y" + j);
    else nnPlace(stage, "y", NN_POS.y(0), false);
    if (accEl) accEl.textContent = summing || loading ? `Σ ${k + 1}/${NN_CHUNKS}` : (hold ? "s held" : "s discarded");
    if (meta) {
      const tag = side === "gpu" ? "GPU RMSNorm  ·  "
        : side === "tpu" ? "TPU RMSNorm  ·  "
        : side === "lpu" ? "LPU RMSNorm  ·  "
        : hold ? "norm_sum_squares_q  ·  " : "unfused RMSNorm  ·  ";
      meta.textContent = tag + `j ${j}/${NN_OUT} · clk ${t}`;
    }
  }

  function stopNnAnim() {
    if (nnAnim.timer) {
      clearInterval(nnAnim.timer);
      nnAnim.timer = null;
    }
    nnAnim.playing = false;
    const btn = $("nnAnimPlay");
    if (btn) btn.textContent = "Play";
    const mbtn = $("nnMechPlay");
    if (mbtn) mbtn.textContent = "Play";
  }

  function startNnAnim() {
    buildNnStages();
    paintNnAnim();
    paintNnWhere();
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      stopNnAnim();
      return;
    }
    if (nnAnim.timer) return;
    nnAnim.playing = true;
    const btn = $("nnAnimPlay");
    if (btn) btn.textContent = "Pause";
    const mbtn = $("nnMechPlay");
    if (mbtn) mbtn.textContent = "Pause";
    nnAnim.timer = setInterval(() => {
      nnAnim.t += 1;
      paintNnAnim();
    }, 1100);
  }

  function bindNnAnim() {
    const play = $("nnAnimPlay");
    const step = $("nnAnimStep");
    const gemm = $("nnSceneGemm");
    const norm = $("nnSceneNorm");
    if (play) play.addEventListener("click", () => {
      if (nnAnim.timer) stopNnAnim();
      else startNnAnim();
    });
    if (step) step.addEventListener("click", () => {
      stopNnAnim();
      nnAnim.t += 1;
      paintNnAnim();
    });
    const setScene = (scene) => {
      nnAnim.scene = scene;
      nnAnim.t = 0;
      const pressed = (el, on) => { if (el) el.setAttribute("aria-pressed", on ? "true" : "false"); };
      pressed(gemm, scene === "gemm");
      pressed(norm, scene === "norm");
      pressed($("nnMechGemm"), scene === "gemm");
      pressed($("nnMechNorm"), scene === "norm");
      paintNnAnim();
      paintNnWhere();
    };
    if (gemm) gemm.addEventListener("click", () => setScene("gemm"));
    if (norm) norm.addEventListener("click", () => setScene("norm"));
    bindNnMechModal(setScene);
  }

  const NN_MECH = {
    gpu: {
      kind: "GPU",
      title: "NVIDIA H100 SXM",
      sub: "Stream \\(W\\) from HBM",
      cap: "cuBLAS holds \\(x\\) in SRAM. The bill is 1.19 GB of \\(W\\) off-chip every token.",
      more: "Die transistors <strong>80B</strong> (814 mm²). HBM stacks are not in that count. Energy is ~5 pJ/bit on 1.19 GB of \\(W\\) every token — that is why H100 is 197 mJ here. Holding \\(x\\) is the mechanic we copy; streaming \\(W\\) is the mechanic we drop.",
      leds: { sram: "hbm_x", rom: "hbm_W", mac: "mma" }
    },
    tpu: {
      kind: "TPU",
      title: "Google TPU v5e",
      sub: "Grid of MACs; \\(W\\) still from HBM",
      cap: "Systolic means a grid of multipliers that pass data like a heartbeat. Each cell keeps a weight, multiplies the \\(x\\) that just arrived, and hands \\(x\\) to the next cell. That pulse of \\(x\\) across the grid is the “wave.”",
      more: "<strong>What you are watching.</strong> The orange grid is the MXU (matrix multiply unit) — TPU v5e’s 128×128 MAC array, drawn small. Weights load into the cells and stay (weight-stationary). Activations enter from the left and step one cell per beat, so a diagonal of MACs lights as the pulse moves. That lockstep grid is all “systolic” means — named after the heartbeat. It is <em>not</em> a different math: still \\(Q = y W_Q\\). <strong>What it does not fix.</strong> Each GEMM tile of \\(W\\) still comes from HBM at ~5 pJ/bit. Die ~32B est (300–350 mm², unpublished) plus 16 GB HBM. Energy here is 124 mJ because of that HBM walk, not because of the grid.",
      leds: { sram: "hbm_x", rom: "hbm_W", mac: "mxu" }
    },
    lpu: {
      kind: "LPU",
      title: "Groq LPU",
      sub: "Keep \\(W\\) in SRAM",
      cap: "Closest cousin: deterministic, on-die \\(W\\). 1.19 GB fits; 1T does not. 750 W TDP is the other bill.",
      more: "Die transistors <strong>26.8B</strong> (~725 mm², 14 nm). No HBM — 1.19 GB of Qwen3-0.6B already sits in SRAM. Energy is 187 mJ: 750 W TDP × 0.6 utilization over the full 0.41 ms pass (compute + weight-SRAM stream), per lab/compute_metrics.py. The SRAM walk is cheap per bit (~0.5 pJ/bit) but the TDP dominates at batch 1. It does not scale to 1T on one die; we burn \\(W\\) in ROM so shards can.",
      leds: { sram: "sram_x", rom: "sram_W", mac: "exu" }
    },
    cim: {
      kind: "CIM",
      title: "Analog compute-in-memory",
      sub: "\\(W\\) is the cell conductance",
      cap: "Closest on where \\(W\\) lives. Kirchhoff sums current; ADC converts. Native BF16 is 4 analog passes. Attention still needs digital lanes.",
      more: "Energy here is <strong>0.36 mJ</strong> for Qwen3-0.6B BF16 from this repo’s 256×256 model (ADC ~5 pJ/conv, 4 nibble passes, per-op pricing) — the lowest on the board, because W never moves. So why does the lab path stay digital ROM? Not energy: cell precision and manufacturability. A multi-level analog cell must hold 16 conductance levels per BF16 nibble scheme with ADC noise margin; a 6T digital ROM bit is binary, bit-exact, and Verilator-verified end to end (qwen-ref-v2). CIM’s 0.36 mJ is an estimate; the ROM chip’s 1.19 mJ is measured cycles priced at the same constants. INT4 CIM (1 pass) is the explorer knob; BF16 CIM is the research target.",
      leds: { sram: "dac_x", rom: "G=W", mac: "adc" }
    },
    pim: {
      kind: "PIM",
      title: "HBM-PIM / DRAM-PIM",
      sub: "ALU next to the DRAM bank",
      cap: "Closer than HBM-to-GPU, still DRAM. \\(W\\) is writable and streamed from banks. Not burnt-in. 1T still needs more stacks.",
      more: "Published-class HBM-PIM: in-bank access ~2 pJ/bit vs ~5 pJ/bit off-chip HBM. Energy here is <strong>19.5 mJ</strong> for 0.6B — 10× better than GPU 197 mJ, ~16× worse than fused ROM 1.19 mJ, because W is still a DRAM walk every token (1.19 GB × 2 pJ/bit ≈ 19 mJ). Logic ~20B. For Tx × energy we count DRAM 1T1C of stored \\(W\\) (GPU’s 80B still excludes HBM stacks — that is why PIM is the honest DRAM compare). Scale-out is more stacks.",
      leds: { sram: "bank_x", rom: "dram_W", mac: "gemv" }
    },
    unfused: {
      kind: "ASIC · unfused",
      title: "This chip · unfused",
      sub: "Re-read \\(x\\) for every output",
      cap: "Naive ASIC schedule: \\(W\\) is already ROM, but \\(x\\) is re-issued from SRAM for every \\(j\\). This is worse than the GPU on the inner loop.",
      more: "Same <strong>~18B</strong> transistors as fused (9.5B ROM at 1T/bit + 8B SRAM). Fusion is a schedule, not more silicon. Unfused re-reads the 1024-vector of \\(x\\) for every output column — 295.4M cycles, ~30 mJ, 1.69 tok/s on this 16-lane lab die.",
      leds: { sram: "sram_req", rom: "rom_req", mac: "mac_valid" }
    },
    fused: {
      kind: "ASIC · fused",
      title: "This chip · fused",
      sub: "Burn \\(W\\) in ROM. Hold \\(x\\).",
      cap: "How the ASIC takes the HBM wire: \\(W\\) never leaves the die (ROM ~0.8 pJ/bit). <code>S_LOAD_ACT</code> then <code>S_RUN</code>. LPU does this in SRAM — we do it in ROM so 1T can still shard.",
      more: "Same <strong>~18B</strong> die (9.5B ROM at 1T/bit + 8B SRAM). Hold \\(x\\) after one SRAM load, stream \\(W\\) from ROM. Measured 57.0M cycles and 608.9M MACs (bit-exact full-statement compares); energy 1.19 mJ is those measured counters priced at N4-class constants (0.45 pJ/MAC, 0.08 pJ/bit). 8.77 tok/s, 5.18× the unfused schedule, 166× less energy than H100.",
      leds: { sram: "sram_req", rom: "rom_req", mac: "mac_valid" }
    }
  };

  function closeNnMech() {
    const modal = $("nnMechModal");
    if (modal) modal.hidden = true;
  }

  function openNnMech(side) {
    const info = NN_MECH[side];
    const modal = $("nnMechModal");
    const pane = $("nnMechPane");
    const stage = $("nnMechStage");
    if (!info || !modal || !pane || !stage) return;
    pane.className = "nn-pane " + side;
    stage.setAttribute("data-side", side);
    const kind = $("nnMechKind");
    const title = $("nnMechTitle");
    const sub = $("nnMechSub");
    const cap = $("nnMechCap");
    const more = $("nnMechMore");
    if (kind) {
      kind.textContent = info.kind;
      kind.setAttribute("data-side", side);
    }
    if (title) title.textContent = info.title;
    if (sub) sub.innerHTML = info.sub;
    if (cap) cap.innerHTML = info.cap;
    if (more) more.innerHTML = info.more;
    const srcK = document.querySelector(`[data-kpis="${side}"]`);
    const destK = $("nnMechKpis");
    if (srcK && destK) destK.innerHTML = srcK.innerHTML;
    const led = (name, text) => {
      const el = stage.querySelector(`[data-led="${name}"]`);
      if (el) el.textContent = text;
    };
    led("sram", info.leds.sram);
    led("rom", info.leds.rom);
    led("mac", info.leds.mac);
    const schem = stage.querySelector(".nn-schem");
    if (schem) {
      delete schem.dataset.ready;
      schem.innerHTML = "";
    }
    stage.querySelectorAll(".nn-badge").forEach((b) => b.remove());
    modal.hidden = false;
    buildNnStages();
    paintNnAnim();
    const gemm = $("nnMechGemm");
    const norm = $("nnMechNorm");
    if (gemm) gemm.setAttribute("aria-pressed", nnAnim.scene === "gemm" ? "true" : "false");
    if (norm) norm.setAttribute("aria-pressed", nnAnim.scene === "norm" ? "true" : "false");
    if (typeof renderMathInElement === "function") {
      renderMathInElement(modal, {
        delimiters: [
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }
        ],
        throwOnError: false,
        strict: false
      });
    }
    if (!nnAnim.timer) startNnAnim();
    const close = $("nnMechClose");
    if (close) close.focus();
  }

  function bindNnMechModal(setScene) {
    document.querySelectorAll(".nn-pane[data-mech]").forEach((pane) => {
      pane.addEventListener("click", (event) => {
        if (event.target.closest("details, a, button, summary")) return;
        openNnMech(pane.getAttribute("data-mech"));
      });
      pane.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        if (event.target.closest("details, a, button, summary")) return;
        event.preventDefault();
        openNnMech(pane.getAttribute("data-mech"));
      });
    });
    const close = $("nnMechClose");
    const modal = $("nnMechModal");
    if (close) close.addEventListener("click", closeNnMech);
    if (modal) modal.addEventListener("click", (event) => {
      if (event.target === modal) closeNnMech();
    });
    const play = $("nnMechPlay");
    const step = $("nnMechStep");
    const gemm = $("nnMechGemm");
    const norm = $("nnMechNorm");
    if (play) play.addEventListener("click", () => {
      if (nnAnim.timer) stopNnAnim();
      else startNnAnim();
    });
    if (step) step.addEventListener("click", () => {
      stopNnAnim();
      nnAnim.t += 1;
      paintNnAnim();
    });
    if (gemm) gemm.addEventListener("click", () => setScene("gemm"));
    if (norm) norm.addEventListener("click", () => setScene("norm"));
  }

  function bindFolds() {
    document.querySelectorAll(".slide-spec .fold, .slide-scale .fold").forEach((el) => {
      el.addEventListener("toggle", () => {
        if (!el.open) return;
        const root = el.closest("section.slide");
        if (!root) return;
        root.querySelectorAll("details.fold").forEach((other) => {
          if (other !== el) other.open = false;
        });
        if (typeof renderMathInElement === "function") {
          renderMathInElement(el, {
            delimiters: [
              { left: "\\[", right: "\\]", display: true },
              { left: "\\(", right: "\\)", display: false }
            ],
            throwOnError: false,
            strict: false
          });
        }
      });
    });
  }

  refreshReady();
  loadCachedReplay();
  loadPlatformCompare();
  bindScaleControls();
  bindScaleMach();
  buildD2Viz();
  bindUtilExplorer();
  bindRoadmapLive();
  bindFolds();
  bindArchDiagrams();
  loadArchCompare();
  loadScale1t();
  bindNnAnim();
  bindSpecSlider();
  bindScalingTabs();
  bindNnMapFold();
  bindRmMechFold();
  renderNnMath();
  paintNnHeat();
  loadNnKpis();
  const slideFromHash = () => {
    const raw = /^#slide-(.+)$/.exec(window.location.hash);
    if (!raw) return 0;
    const key = decodeURIComponent(raw[1]);
    if (/^\d+$/.test(key)) return Math.max(0, Number(key) - 1);
    const byId = slideIds().indexOf(key);
    return byId >= 0 ? byId : 0;
  };
  showSlide(slideFromHash());
  window.addEventListener("hashchange", () => showSlide(slideFromHash()));
})();
