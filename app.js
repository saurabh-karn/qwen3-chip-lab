(() => {
  "use strict";

  const SHARE_DEMO = document.documentElement.hasAttribute("data-share-demo");
  const tensorCache = new Map();

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
    attention_softmax: "P = softmax(S)",
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

  const $ = (id) => document.getElementById(id);
  const state = {
    jobId: null,
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
    live: false,
    viewLayer: 0,
    dagObs: null,
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

  function neuronInner(op, title) {
    const w = (OP_FEEDS[op] && OP_FEEDS[op].w) ? `<span class="io w">${OP_FEEDS[op].w}</span>` : "";
    return `<span class="soma" aria-hidden="true"></span>` +
      `<b>${title || OP_LABEL[op] || op}</b>` +
      `<span class="eq">${OP_EQ[op] || ""}</span>` +
      w +
      `<span class="shp">${OP_SHAPE[op] || ""}</span>` +
      `<span class="io out" data-samp>out —</span>`;
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
    const feeds = OP_FEEDS[op] || { in: [], w: null };
    (feeds.in || []).forEach((feed, i) => {
      const slot = el.querySelector(`[data-in="${i}"]`);
      if (slot) slot.textContent = "in  " + feedText(feed, rec, map);
    });
    const shp = el.querySelector(".shp");
    if (shp) shp.textContent = "out " + shapeText(rec, op);
    const samp = el.querySelector("[data-samp]");
    if (samp) samp.textContent = "out  " + sampleLine(rec.values, 6);
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
      const samp = port.querySelector("[data-samp]");
      if (samp) samp.textContent = "out  " + sampleLine(res.values, 6);
      const shp = port.querySelector(".shp");
      if (shp) shp.textContent = shapeText(res, L === 0 ? "embedding" : "output");
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
    if (name === "embedding" || name === "final_norm" || name === "logits") return name;
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

  function dagNode(op, title, extraClass) {
    const span = DAG_SPAN[op] || "1 / -1";
    return `<button type="button" class="op${extraClass ? " " + extraClass : ""}" id="dag-${op}" data-op="${op}" style="grid-column:${span}">${neuronInner(op, title)}</button>`;
  }

  function bindDagNodes(layer) {
    state.viewLayer = layer;
    const title = $("dagTitle");
    if (title) {
      title.textContent = `decoder ${String(layer).padStart(2, "0")} · 28 layers share this graph · arrows are tensors`;
    }
    const res = $("dag-residual");
    if (res) res.setAttribute("data-node", layer === 0 ? "g-embedding" : `L${layer - 1}-output`);
    DAG_OPS.forEach((op) => {
      const el = $("dag-" + op);
      if (el) el.setAttribute("data-node", `L${layer}-${op}`);
    });
  }

  function dagAnchor(el, dag, which) {
    const er = el.getBoundingClientRect();
    const dr = dag.getBoundingClientRect();
    const x = er.left - dr.left + er.width / 2;
    const left = er.left - dr.left;
    const right = er.right - dr.left;
    const top = er.top - dr.top;
    const bot = er.bottom - dr.top;
    const midY = top + er.height / 2;
    if (which === "left") return { x: left, y: midY };
    if (which === "right") return { x: right, y: midY };
    if (which === "top") return { x, y: top };
    return { x, y: bot };
  }

  function edgePath(a, b, kind, width) {
    const dy = Math.max(16, (b.y - a.y) * 0.32);
    if (kind === "skip-left") {
      const x = 8;
      return `M ${a.x} ${a.y} C ${x} ${a.y + 10}, ${x} ${b.y - 10}, ${b.x} ${b.y}`;
    }
    if (kind === "skip-right") {
      const x = Math.max(width - 8, a.x);
      return `M ${a.x} ${a.y} C ${x} ${a.y + 10}, ${x} ${b.y - 10}, ${b.x} ${b.y}`;
    }
    return `M ${a.x} ${a.y} C ${a.x} ${a.y + dy}, ${b.x} ${b.y - dy}, ${b.x} ${b.y}`;
  }

  function labelPoint(a, b, kind, width) {
    if (kind === "skip-left") return { x: 18, y: (a.y + b.y) / 2 };
    if (kind === "skip-right") return { x: width - 18, y: (a.y + b.y) / 2 };
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  function restoreEdgeLive() {
    const dag = $("dag");
    if (!dag) return;
    dag.querySelectorAll(".op.active").forEach((el) => {
      const op = el.getAttribute("data-op");
      document.querySelectorAll(`#dagEdges [data-to="${op}"]`).forEach((e) => e.classList.add("live"));
      document.querySelectorAll(`#dagEdges [data-from="${op}"]`).forEach((e) => e.classList.add("live-out"));
    });
  }

  function drawDagEdges() {
    const dag = $("dag");
    const svg = $("dagEdges");
    if (!dag || !svg) return;
    const w = Math.max(1, dag.clientWidth);
    const h = Math.max(1, dag.clientHeight);
    svg.setAttribute("width", String(w));
    svg.setAttribute("height", String(h));
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const defs = `<defs>
      <marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#7c8b99"/>
      </marker>
      <marker id="arrLive" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#0f766e"/>
      </marker>
    </defs>`;
    const paths = DAG_EDGES.map((edge) => {
      const src = $("dag-" + edge.from);
      const dst = $("dag-" + edge.to);
      if (!src || !dst) return "";
      const kind = edge.kind || "fwd";
      const a = dagAnchor(src, dag, kind === "skip-left" ? "left" : kind === "skip-right" ? "right" : "bottom");
      const b = dagAnchor(dst, dag, kind === "skip-left" ? "left" : kind === "skip-right" ? "right" : "top");
      const d = edgePath(a, b, kind, w);
      const p = labelPoint(a, b, kind, w);
      const cls = kind === "fwd" ? "edge" : "edge skip";
      const anchor = kind === "skip-left" ? "start" : kind === "skip-right" ? "end" : "middle";
      return `<path class="${cls}" data-from="${edge.from}" data-to="${edge.to}" d="${d}" marker-end="url(#arr)"/>` +
        `<text class="edge-lab" data-from="${edge.from}" data-to="${edge.to}" text-anchor="${anchor}" x="${p.x}" y="${p.y - 2}">${escapeHtml(edge.lab)}</text>`;
    }).join("");
    svg.innerHTML = defs + paths;
    restoreEdgeLive();
  }

  function renderNetwork() {
    const dagOps = [
      dagNode("residual", "x in", "port"),
      dagNode("input_norm"),
      dagNode("q_proj"), dagNode("k_proj"), dagNode("v_proj"),
      dagNode("q_norm"), dagNode("k_norm"),
      dagNode("q_rope"), dagNode("k_rope"),
      dagNode("attention_scores"), dagNode("attention_softmax"), dagNode("causal_gqa"),
      dagNode("o_proj"), dagNode("attention_residual"),
      dagNode("post_norm"),
      dagNode("gate_proj"), dagNode("up_proj"),
      dagNode("silu"), dagNode("swiglu"),
      dagNode("down_proj"), dagNode("output")
    ].join("");
    const rail = Array.from({ length: 28 }, (_, i) =>
      `<button type="button" class="layer-dot" data-layer="${i}" data-jump-layer="${i}">L${String(i).padStart(2, "0")}</button>`
    ).join("");
    $("network").innerHTML = `
      <button type="button" class="op net-node wide" id="g-embedding" data-node="g-embedding" data-op="embedding">${neuronInner("embedding", "embedding")}</button>
      <div class="flow-edge"><i>↓</i><span>h feeds decoder x</span></div>
      <div class="layer-rail" id="layerRail">${rail}</div>
      <article class="decoder-graph" id="decoderGraph">
        <p class="dag-title" id="dagTitle">decoder 00 · 28 layers share this graph · arrows are tensors</p>
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
    if (state.dagObs) state.dagObs.disconnect();
    const dag = $("dag");
    if (dag && typeof ResizeObserver !== "undefined") {
      state.dagObs = new ResizeObserver(() => drawDagEdges());
      state.dagObs.observe(dag);
    }
    requestAnimationFrame(drawDagEdges);
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
    if (!el) return;
    const { prompt, next, full } = predictedSentence();
    if (state.argmax == null) {
      el.innerHTML = `<span class="muted">Argmax is decoded after the forward pass.</span>`;
      if (sent) {
        sent.className = "pred-sentence muted";
        sent.textContent = "The completed sentence appears here after argmax.";
      }
      return;
    }
    const quoted = next == null ? "" : JSON.stringify(next);
    el.innerHTML = `<span class="token out">${state.argmax}<small>${escapeHtml(quoted)}</small></span>`;
    if (sent) {
      sent.className = "pred-sentence";
      sent.innerHTML = escapeHtml(prompt) + (next ? `<mark>${escapeHtml(next)}</mark>` : "");
    }
    if ($("argmaxDetail")) $("argmaxDetail").textContent = `argmax ${state.argmax}  ${quoted}`;
    const node = $("g-argmax");
    if (node) {
      const samp = node.querySelector("[data-samp]");
      if (samp) samp.textContent = `out  ${full}`;
    }
  }

  function clearActive() {
    document.querySelectorAll(".active, .current, .live, .live-out").forEach((el) => {
      el.classList.remove("active", "current", "live", "live-out");
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
    $("fsmName").textContent = stage || "IDLE";
    $("mStage").textContent = stage || "IDLE";
  }

  function activateDagOp(op) {
    const el = $("dag-" + op);
    if (el) el.classList.add("active");
    document.querySelectorAll(`#dagEdges [data-to="${op}"]`).forEach((e) => e.classList.add("live"));
    document.querySelectorAll(`#dagEdges [data-from="${op}"]`).forEach((e) => e.classList.add("live-out"));
  }

  function paintNetwork(event, layer) {
    const op = opKey(event);
    if (typeof layer === "number" && layer >= 0 && layer < 28) {
      if (layer !== state.viewLayer) {
        bindDagNodes(layer);
        fillDagLayer(layer);
      }
      const wrap = $("decoderGraph");
      if (wrap) wrap.classList.add("current");
      document.querySelectorAll(`[data-layer="${layer}"]`).forEach((el) => el.classList.add("active"));
    }
    if (["embedding", "final_norm", "logits", "argmax"].includes(op) || layer == null || layer < 0) {
      const gel = $("g-" + op);
      if (gel) gel.classList.add("active");
    } else {
      activateDagOp(op);
      if (op === "q_rope") activateDagOp("k_rope");
      if (op === "silu") activateDagOp("swiglu");
    }
  }

  function paintCursor({ event, layer, token, stage }) {
    const st = (stage && STAGE_BLOCK[stage]) ? stage : eventStage(event);
    const mismatched = event ? state.mismatch.has(event) : false;
    clearActive();
    markMismatches();
    paintNetwork(event, layer);
    paintChip(st, mismatched);
    $("mLayer").textContent = typeof layer === "number" && layer >= 0 ? String(layer) : "—";
    if (token !== undefined && token !== null && token !== -1) {
      paintTokens(state.tokenIds, token);
    }
  }

  function paintFromStage(stage, layer) {
    const spec = STAGE_OP[stage] || {};
    if (spec.global) {
      paintCursor({ event: spec.global, layer: -1, stage });
      if (typeof layer === "number" && layer >= 0) {
        const strip = document.querySelector(`.layer[data-layer="${layer}"]`);
        if (strip) strip.classList.add("active");
        $("mLayer").textContent = String(layer);
      }
      return;
    }
    paintCursor({ event: spec.op || "embedding", layer, stage });
    if (spec.extra && typeof layer === "number" && layer >= 0) {
      activateDagOp(spec.extra);
    }
  }

  function setTimeline(events) {
    state.timeline = Array.isArray(events) ? events : [];
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
    const url = `/api/tensor?event=${encodeURIComponent(rec.event)}&source=${source}` +
      `&offset=${offset}&limit=${limit}`;
    const res = await fetch(url);
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
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
    state.playing = false;
    $("playBtn").textContent = "Play";
  }

  function startPlay() {
    if (!state.timeline.length || state.live) return;
    state.playing = true;
    state.userPaused = false;
    $("playBtn").textContent = "Pause";
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = setInterval(() => {
      if (state.index >= state.timeline.length - 1) {
        stopPlay();
        return;
      }
      showIndex(state.index + 1);
    }, 55);
  }

  function applyMetrics(rtl, python) {
    const status = rtl || {};
    const py = python || {};
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
    if (status.argmax !== undefined && status.argmax !== null && (status.stage === "DONE" || status.stage === "ARGMAX")) {
      if (state.argmax == null) state.argmax = status.argmax;
      paintOutput();
    }
  }

  function applyEvidence(evidence) {
    if (!evidence) return;
    const comparison = evidence.comparison || {};
    const passed = evidence.passed;
    setBadge($("compareBadge"), passed ? "T4 match" : "mismatch, playing through", passed ? "ok" : "bad");
    const decoded = state.argmaxText ? JSON.stringify(state.argmaxText) : "";
    const sentence = predictedSentence().full;
    if (passed) {
      $("compareDetail").textContent =
        `T4 passed: ${evidence.token_count} token(s). Next token ${evidence.python_argmax} ${decoded}. ` +
        `Predicted: ${sentence}`;
      $("mismatchBox").hidden = true;
      return;
    }
    const first = comparison.first_mismatch;
    $("compareDetail").textContent = first
      ? `First mismatch at ${(Array.isArray(first.key) ? first.key[0] : first.event) || "checkpoint"}. ` +
        `Predicted: ${sentence}`
      : (evidence.message || "Verify did not pass.");
    $("mismatchBox").hidden = false;
    $("mismatchBox").textContent = JSON.stringify(first || comparison, null, 2);
  }

  function ingest(body, { autoplay } = {}) {
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
    if (body.evidence) applyEvidence(body.evidence);
    const rtl = body.rtl_status || {};
    const liveRtl = body.phase === "rtl" && rtl.stage && rtl.stage !== "IDLE" && rtl.stage !== "DONE";
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
      setBadge($("readyBadge"), "public demo · recorded T4", "ok");
      $("verifyBtn").disabled = true;
      $("verifyBtn").textContent = "Recorded run";
      $("verifyText").readOnly = true;
      $("formError").hidden = true;
      return;
    }
    try {
      const res = await fetch("/api/ready");
      const body = await res.json();
      if (body.ready) {
        setBadge($("readyBadge"), "ROM + tokenizer ready", "ok");
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
      const res = await fetch(`/api/verify/${jobId}`);
      const body = await res.json();
      state.tokenIds = body.token_ids || state.tokenIds;
      paintTokens(state.tokenIds, body.rtl_status && body.rtl_status.token_index);
      ingest(body, { autoplay: false });
      if (body.status === "running") {
        setBadge($("runBadge"), body.phase === "rtl" ? "RTL running" : "Python oracle", "run");
        $("verifyBtn").disabled = true;
        state.poll = setTimeout(() => pollJob(jobId), 400);
        return;
      }
      $("verifyBtn").disabled = false;
      stopPoll();
      $("formError").hidden = true;
      if (body.evidence || (body.timeline && body.timeline.length)) {
        setBadge($("runBadge"), body.evidence && body.evidence.passed ? "T4 complete" : "mismatch, playing through",
          body.evidence && body.evidence.passed ? "ok" : "bad");
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
    setBadge($("compareBadge"), "not compared");
    $("mismatchBox").hidden = true;
    stopPlay();
    state.userPaused = false;
    state.mismatch = new Set();
    state.prompt = text;
    state.predictedText = "";
    state.argmax = null;
    state.argmaxText = "";
    paintOutput();
    try {
      const tok = await fetch("/api/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const tokBody = await tok.json();
      if (!tok.ok) throw new Error(tokBody.detail || "tokenize failed");
      state.tokenIds = tokBody.token_ids;
      if (tokBody.token_texts) state.tokenTexts = tokBody.token_texts;
      paintTokens(state.tokenIds, 0);

      const res = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, tier: "T4" })
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
      state.tokenIds = body.token_ids || state.tokenIds;
      paintTokens(state.tokenIds, 0);
      if (body.status === "cached") {
        $("verifyBtn").disabled = false;
        setBadge($("runBadge"), body.evidence && body.evidence.passed ? "T4 complete" : "mismatch, playing through",
          body.evidence && body.evidence.passed ? "ok" : "bad");
        ingest(body, { autoplay: true });
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
      const res = await fetch(SHARE_DEMO ? "data/evidence.json" : "/api/evidence");
      const body = await res.json();
      if (!body.evidence && !(body.timeline && body.timeline.length)) return;
      state.tokenIds = body.token_ids || (body.evidence && body.evidence.token_ids) || [];
      if (body.evidence && body.evidence.text && (SHARE_DEMO || !$("verifyText").value)) {
        $("verifyText").value = body.evidence.text;
      }
      paintTokens(state.tokenIds, 0);
      setBadge($("runBadge"), body.evidence && body.evidence.passed ? "T4 complete" : "mismatch, playing through",
        body.evidence && body.evidence.passed ? "ok" : "bad");
      ingest(body, { autoplay: true });
    } catch (_) { /* no prior evidence */ }
  }

  renderNetwork();
  renderBanks();
  window.addEventListener("resize", () => requestAnimationFrame(drawDagEdges));
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
  $("compute").addEventListener("click", () => {
    if (state.timeline[state.index]) openTensorModal(state.timeline[state.index]);
  });
  $("modalClose").addEventListener("click", closeModal);
  $("tensorModal").addEventListener("click", (event) => {
    if (event.target === $("tensorModal")) closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("tensorModal").hidden) closeModal();
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
  refreshReady();
  loadLatest();
})();
