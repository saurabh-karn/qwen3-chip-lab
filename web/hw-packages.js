/* Physical package gallery: the same 3D material language as the GPU
 * experience, driven by a declarative spec per machine, so six very different
 * pieces of hardware can be compared at one camera angle and one scale.
 *
 * The whole argument of the deck is *where the weights physically live*, so
 * that is what each package is built to show:
 *   off-package (GPU, TPU) -> inside the memory stacks (PIM) -> on the die
 *   itself (LPU SRAM, analog CIM, mask ROM).
 *
 * Geometry is illustrative and deliberately not a floorplan claim.
 */
(function () {
  'use strict';

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  // Same 3D box primitive as gpu-experience.js: a top face plus four sides,
  // positioned in a preserve-3d scene.
  const box = (cls, x, y, w, h, z, depth, face = '') =>
    `<div class="gx-box ${cls}" style="--x:${x}px;--y:${y}px;--w:${w}px;--h:${h}px;--z:${z}px;--d:${depth}px">` +
    `<div class="gx-top">${face}</div><i class="gx-front"></i><i class="gx-back"></i>` +
    `<i class="gx-left"></i><i class="gx-right"></i></div>`;

  const cells = (n, cls = '') =>
    Array.from({ length: n }, () => `<i class="${cls}"></i>`).join('');

  // ---- die interiors -------------------------------------------------------
  function dieFace(die) {
    const title = `<div class="gx-die-title">${die.title}<span>${die.sub || ''}</span></div>`;
    let body = '';
    switch (die.kind) {
      case 'sm':       // GPU: many small streaming-multiprocessor tiles
        body = `<div class="gx-sm-grid" style="--cols:${Math.ceil(Math.sqrt((die.n || 96) * 1.1))}">${cells(die.n || 96)}</div>`;
        break;
      case 'mxu':      // TPU: one large systolic matrix unit
        body = `<div class="hp-mxu"><div class="hp-mxu-grid" style="--cols:${die.cols || 12}">${cells((die.cols || 12) * (die.rows || 10))}</div><span class="hp-block-k">SYSTOLIC MXU</span></div>`;
        break;
      case 'sram':     // Groq LPU: the die is mostly SRAM banks
        body = `<div class="hp-banks">${Array.from({ length: die.banks || 10 }, () => '<i></i>').join('')}<span class="hp-block-k">SRAM · W RESIDENT</span></div>`;
        break;
      case 'crossbar': // Analog CIM: W is the cell conductance, ADCs at the edge
        body = `<div class="hp-xbar"><div class="hp-xbar-grid" style="--cols:${die.cols || 8}">${cells((die.cols || 8) * (die.rows || 8), 'hp-xbar-tile')}</div><div class="hp-adc">ADC / DAC</div><span class="hp-block-k">W = CONDUCTANCE</span></div>`;
        break;
      case 'rom':      // this chip: nearly the whole die is mask-ROM array
        body = `<div class="hp-rom"><div class="hp-rom-grid" style="--cols:${die.cols || 22}">${cells((die.cols || 22) * (die.rows || 16), 'hp-rom-bit')}</div><div class="hp-mac-strip">MAC ARRAY</div><span class="hp-block-k">MASK ROM · W BURNT IN</span></div>`;
        break;
      case 'rom2d':    // production: ROM plane feeding a 2D systolic MXU
        body = `<div class="hp-rom2d">` +
          `<div class="hp-rom2d-plane" style="--cols:${die.cols || 20}">${cells((die.cols || 20) * (die.rows || 9), 'hp-rom-bit')}</div>` +
          `<div class="hp-rom2d-mxu" style="--cols:${die.mxu || 14}">${cells((die.mxu || 14) * 6, 'hp-mxu-pe')}</div>` +
          `<span class="hp-block-k">ROM PLANE \u2192 2D MXU \u00b7 W STATIONARY</span></div>`;
        break;
      case 'logic':    // PIM: thin controller die, compute has moved to memory
        body = `<div class="hp-logic"><span class="hp-block-k">CONTROL / SEQUENCER</span></div>`;
        break;
      default:
        body = '';
    }
    const cache = die.cache ? `<div class="gx-cache">${die.cache}</div>` : '';
    return title + body + cache;
  }

  // ---- package specs -------------------------------------------------------
  // stacks: memory towers on the interposer. kind 'hbm' = plain memory,
  // 'pim' = memory with MACs inside it.
  const PACKAGES = {
    h100: {
      factKey: 'h100',
      label: 'NVIDIA H100 SXM',
      kind: 'GPU',
      wLives: 'HBM, off-package',
      note: 'Weights cross the interposer for every GEMM, every token.',
      etch: 'GPU / ARCHITECTURE STUDY',
      accent: 'cost',
      die: { x: 123, y: 48, w: 214, h: 214, kind: 'sm', n: 96, title: 'COMPUTE DIE', sub: 'SM ARRAY', cache: 'ON-DIE SCRATCH / KV' },
      stacks: [[43, 39], [43, 126], [43, 213], [358, 72], [358, 179]],
      stackKind: 'hbm', stackLabel: 'HBM', stackLayers: 5,
      weightTraces: true
    },
    tpu: {
      factKey: 'tpu_v5e',
      label: 'Google TPU v5e',
      kind: 'TPU',
      wLives: 'HBM → MXU weight FIFO',
      note: 'A grid of multipliers, but W still walks in from HBM per tile.',
      etch: 'TPU / ARCHITECTURE STUDY',
      accent: 'cost',
      die: { x: 130, y: 55, w: 200, h: 200, kind: 'mxu', cols: 12, rows: 10, title: 'TPU DIE', sub: 'MXU + VPU', cache: 'VECTOR MEM' },
      stacks: [[46, 78], [46, 176], [356, 78], [356, 176]],
      stackKind: 'hbm', stackLabel: 'HBM', stackLayers: 4,
      weightTraces: true
    },
    lpu: {
      factKey: 'lpu',
      label: 'Groq LPU',
      kind: 'LPU',
      wLives: 'On-die SRAM',
      note: 'No HBM on the package at all — W is resident in SRAM.',
      etch: 'LPU / ARCHITECTURE STUDY',
      accent: 'ours',
      die: { x: 70, y: 40, w: 320, h: 230, kind: 'sram', banks: 11, title: 'LPU DIE', sub: 'SRAM RESIDENT', cache: 'STREAM REGISTERS' },
      stacks: [],
      weightTraces: false
    },
    cim: {
      factKey: 'cim',
      label: 'Analog compute-in-memory',
      kind: 'CIM',
      wLives: 'The memory cell itself',
      note: 'Kirchhoff sums the current; the weight never moves because it is the cell.',
      etch: 'CIM / ARCHITECTURE STUDY',
      accent: 'ours',
      die: { x: 92, y: 45, w: 276, h: 220, kind: 'crossbar', cols: 9, rows: 8, title: 'CIM DIE', sub: 'CROSSBAR', cache: 'DIGITAL LANES' },
      stacks: [],
      weightTraces: false
    },
    pim: {
      factKey: 'pim',
      label: 'HBM-PIM / DRAM-PIM',
      kind: 'PIM',
      wLives: 'DRAM banks, MACs beside the sense amps',
      note: 'Compute moves into the stacks, so W moves a much shorter distance.',
      etch: 'PIM / ARCHITECTURE STUDY',
      accent: 'mem',
      die: { x: 165, y: 110, w: 130, h: 92, kind: 'logic', title: 'LOGIC DIE', sub: 'HOST LINK' },
      stacks: [[40, 40], [40, 138], [40, 236], [352, 66], [352, 164], [352, 236]],
      stackKind: 'pim', stackLabel: 'PIM', stackLayers: 5,
      weightTraces: 'short'
    },
    b200: {
      factKey: 'b200',
      label: 'NVIDIA B200',
      kind: 'GPU',
      wLives: 'HBM, off-package',
      note: 'Two reticle-limited dies and eight stacks. More bandwidth, same weight walk.',
      etch: 'GPU / ARCHITECTURE STUDY',
      accent: 'cost',
      die: { x: 118, y: 46, w: 108, h: 218, kind: 'sm', n: 64, title: 'DIE 0', sub: 'SM ARRAY' },
      die2: { x: 234, y: 46, w: 108, h: 218, kind: 'sm', n: 64, title: 'DIE 1', sub: 'SM ARRAY', cache: 'NV-HBI' },
      stacks: [[30, 30], [30, 112], [30, 194], [30, 250], [370, 30], [370, 112], [370, 194], [370, 250]],
      stackKind: 'hbm', stackLabel: 'HBM', stackLayers: 5,
      weightTraces: true
    },
    maskrom: {
      factKey: 'rom',
      label: 'This chip · lab die (1D)',
      kind: 'ASIC',
      wLives: 'Burnt into the die at fabrication',
      note: 'What we actually measured: 16 sequential MAC lanes, 500 MHz. Nothing streams weights, ever.',
      etch: 'MASK-ROM LLM / QWEN3-0.6B',
      accent: 'ours',
      die: { x: 62, y: 34, w: 336, h: 242, kind: 'rom', cols: 24, rows: 17, title: 'MASK-ROM DIE', sub: '9.54B 1-T CELLS', cache: 'SCRATCH · x / KV' },
      stacks: [],
      host: true,
      weightTraces: false
    },
    rom2d: {
      facts: { energy: 'see roadmap', toks: '~40,000 tok/s', vs: 'projected', evidence: 'projected \u00b7 no RTL yet' },
      label: 'This chip \u00b7 production (2D)',
      kind: 'ASIC',
      wLives: 'Burnt into the die, stationary under a 2D array',
      note: 'Same burnt-in ROM, but x hops east and psums south across a 2D MXU. W never moves at all.',
      etch: 'MASK-ROM LLM / 2D ROM-STATIONARY',
      accent: 'ours',
      die: { x: 56, y: 30, w: 348, h: 250, kind: 'rom2d', cols: 22, rows: 9, mxu: 15, title: 'ROM-STATIONARY DIE', sub: '2D MXU', cache: 'SCRATCH \u00b7 x / KV' },
      stacks: [],
      host: true,
      weightTraces: false
    }
  };

  function traceSvg(spec) {
    if (!spec.weightTraces) {
      // only the activation / host path exists on these packages
      return `<svg class="gx-traces" viewBox="0 0 460 310" aria-hidden="true">` +
        `<path class="gx-x-trace" d="M230 296 V262"/></svg>`;
    }
    const short = spec.weightTraces === 'short';
    const paths = (spec.stacks || []).map(([x, y]) => {
      const sx = x < 230 ? x + 58 : x;
      const midX = x < 230 ? (short ? sx + 18 : 107) : (short ? sx - 18 : 353);
      return `<path class="gx-trace-base" d="M${sx} ${y + 27} H${midX} V155 H230"/>` +
             `<path class="gx-trace-flow" d="M${sx} ${y + 27} H${midX} V155 H230"/>`;
    }).join('');
    return `<svg class="gx-traces" viewBox="0 0 460 310" aria-hidden="true">${paths}` +
      `<path class="gx-x-trace" d="M230 296 V262"/></svg>`;
  }

  function chipHtml(spec) {
    let chip = box('gx-substrate', 0, 0, 460, 310, 0, 15,
      `<span class="gx-etched">${spec.etch}</span>`);
    chip += box('gx-interposer', 20, 22, 420, 266, 16, 6);
    chip += traceSvg(spec);
    chip += box('gx-die', spec.die.x, spec.die.y, spec.die.w, spec.die.h, 23, 10, dieFace(spec.die));
    if (spec.die2) {
      chip += box('gx-die', spec.die2.x, spec.die2.y, spec.die2.w, spec.die2.h, 23, 10, dieFace(spec.die2));
    }
    (spec.stacks || []).forEach(([x, y], i) => {
      chip += box('gx-hbm-base', x - 3, y - 3, 64, 60, 23, 5);
      const cls = spec.stackKind === 'pim' ? 'gx-hbm hp-stack-pim' : 'gx-hbm';
      for (let d = 0; d < (spec.stackLayers || 5); d++) {
        const top = d === (spec.stackLayers || 5) - 1;
        chip += box(cls, x, y, 58, 54, 29 + d * 5, 4,
          top ? `<span>${spec.stackLabel}</span><small>W / ${i + 1}</small>` : '');
      }
    });
    if (spec.host) {
      chip += box('hp-host', 196, 268, 68, 26, 23, 6, '<span class="hp-host-k">HOST · token IDs</span>');
    }
    for (let i = 0; i < 14; i++) chip += box('gx-passive', 86 + i * 22, 9, 10, 5, 16, 3);
    return chip;
  }

  // Facts are never authored here. They are pushed in from the deck's own
  // constants and live API numbers (see paintNnKpis in app.js) so the gallery
  // can never drift away from the tables and charts that use the same values.
  let FACTS = {};
  const mounted = [];

  function factsHtml(spec) {
    const f = spec.facts || FACTS[spec.factKey] || {};
    const row = (k, v) => v ? `<div><span>${k}</span><b>${v}</b></div>` : '';
    const body = row('Energy / token', f.energy) + row('Throughput', f.toks) +
                 row('vs this chip', f.vs) + row('Evidence', f.evidence);
    return body || '<div class="hp-facts-wait">figures load with the slide</div>';
  }

  function paintFacts() {
    mounted.forEach(({ hostEl, spec }) => {
      const slot = hostEl && hostEl.querySelector('.hp-facts');
      if (slot) slot.innerHTML = factsHtml(spec);
    });
  }

  function mountOne(hostEl, key, opts = {}) {
    const spec = PACKAGES[key];
    if (!hostEl || !spec) return null;
    const rot = { x: opts.rx ?? 55, z: opts.rz ?? -29 };

    hostEl.innerHTML =
      `<figure class="hp" data-accent="${spec.accent}" data-machine="${key}">
         <figcaption class="hp-cap">
           <span class="hp-kind">${spec.kind}</span>
           <b>${spec.label}</b>
           <span class="hp-wlives"><i></i>W lives: ${spec.wLives}</span>
         </figcaption>
         <div class="hp-orbit" tabindex="0" role="group"
              aria-label="${spec.label} package. Drag or use arrow keys to rotate.">
           <div class="hp-scene"><div class="hp-chip" aria-hidden="true">${chipHtml(spec)}</div></div>
         </div>
         <div class="hp-facts">${factsHtml(spec)}</div>
         <p class="hp-note">${spec.note}</p>
       </figure>`;

    const scene = hostEl.querySelector('.hp-scene');
    const orbit = hostEl.querySelector('.hp-orbit');
    const turn = () => {
      scene.style.setProperty('--rx', `${rot.x}deg`);
      scene.style.setProperty('--rz', `${rot.z}deg`);
    };
    const fit = () => {
      const w = orbit.clientWidth - 20;
      // 560 not 470: the rotated package's bounding box is wider than the
      // scene box, so it needs margin or it clips at the card edge
      scene.style.setProperty('--scale', String(clamp(w / 560, 0.24, opts.maxScale ?? 1.15)));
    };

    let drag = null;
    orbit.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      drag = { id: e.pointerId, x: e.clientX, y: e.clientY, rx: rot.x, rz: rot.z };
      orbit.setPointerCapture(e.pointerId);
    });
    orbit.addEventListener('pointermove', (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      rot.x = clamp(drag.rx - (e.clientY - drag.y) * 0.18, 25, 68);
      rot.z = clamp(drag.rz + (e.clientX - drag.x) * 0.18, -65, 25);
      turn();
    });
    const stop = () => { drag = null; };
    orbit.addEventListener('pointerup', stop);
    orbit.addEventListener('pointercancel', stop);
    orbit.addEventListener('lostpointercapture', stop);
    orbit.addEventListener('keydown', (e) => {
      const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home'];
      if (!keys.includes(e.key)) return;
      e.preventDefault();
      if (e.key === 'Home') { rot.x = 55; rot.z = -29; }
      else {
        rot.z = clamp(rot.z + (e.key === 'ArrowRight' ? 5 : e.key === 'ArrowLeft' ? -5 : 0), -65, 25);
        rot.x = clamp(rot.x + (e.key === 'ArrowDown' ? 5 : e.key === 'ArrowUp' ? -5 : 0), 25, 68);
      }
      turn();
    });
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(fit).observe(hostEl);
    fit(); turn();
    mounted.push({ hostEl, spec });
    return { key, spec };
  }

  window.HwPackages = {
    keys: () => Object.keys(PACKAGES),
    /* Push per-side facts in from the deck's single source of truth. */
    setFacts(next = {}) { FACTS = { ...FACTS, ...next }; paintFacts(); },
    spec: (k) => PACKAGES[k],
    mount: mountOne,
    /* Render every machine into a container as a comparison wall. */
    mountAll(containerEl, keys, opts = {}) {
      if (!containerEl) return [];
      const list = keys && keys.length ? keys : Object.keys(PACKAGES);
      containerEl.innerHTML = list.map((k) => `<div class="hp-slot" data-hp="${k}"></div>`).join('');
      return list.map((k) =>
        mountOne(containerEl.querySelector(`.hp-slot[data-hp="${k}"]`), k, opts));
    }
  };
})();
