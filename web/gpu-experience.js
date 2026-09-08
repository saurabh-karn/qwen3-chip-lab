/* Standalone GPU experience. No network requests except its sibling stylesheet.
 * Playback belongs to the host; all actions are requests, never a local clock.
 * layer is zero based. The tile model and the measured custom RTL stay distinct.
 */
(function () {
  'use strict';
  const cssUrl = new URL('gpu-experience.css', document.currentScript?.src || document.baseURI).href;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const num = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : null;
  const count = v => num(v) === null ? '—' : Math.round(v).toLocaleString('en-US');
  const bytes = v => num(v) === null ? '—' : v >= 1e9 ? `${(v / 1e9).toFixed(2)} GB` : v >= 1e6 ? `${(v / 1e6).toFixed(2)} MB` : v >= 1e3 ? `${(v / 1e3).toFixed(1)} kB` : `${v} B`;
  const energy = v => num(v) === null ? '—' : `${(v * 1000).toFixed(2)} mJ`;
  const stages = [
    ['embedding', 'Embedding', 'token → vector'],
    ['qkv_proj', 'Q · K · V', 'project'],
    ['attn_scores', 'Attention', 'mix context'],
    ['o_proj', 'O projection', 'combine'],
    ['gate_up_proj', 'MLP', 'expand → contract'],
    ['lm_head', 'LM head', 'vector → logits']
  ];
  const stories = [
    ['Enter', 'A token becomes a vector.', 'An embedding lookup turns a token ID into the activation vector x. That vector travels through every transformer layer.'],
    ['Fetch', 'The math starts with a journey.', 'Projection weights live beside the processor in HBM. They cross the interposer before the compute units can use them.'],
    ['Compute', 'Bring x and W together.', 'Multiply, accumulate, then pass the result onward. The custom RTL uses the same MAC arithmetic as the ASIC reference.'],
    ['Repeat', 'New layer. Another weight journey.', 'QKV, O projection, and the MLP each need their own weights. The pattern repeats through the transformer, then the LM head reads the output table.'],
    ['Rethink', 'What if the weights stayed on chip?', 'Fixed on-chip weights remove the external HBM weight journey. The math, activations, and KV work remain; local weight access still has a cost.']
  ];
  const box = (cls, x, y, w, h, z, depth, face = '') => `<div class="gx-box ${cls}" style="--x:${x}px;--y:${y}px;--w:${w}px;--h:${h}px;--z:${z}px;--d:${depth}px"><div class="gx-top">${face}</div><i class="gx-front"></i><i class="gx-back"></i><i class="gx-left"></i><i class="gx-right"></i></div>`;
  let root, host, observer, aborter, scene, refs = {}, cells = [];
  let data = {}, rtl = null, story = 0, pinnedStory = false, focusPart = '', lastKey = '';
  let state = { cycle: 0, total: 0, stage: 'embedding', layer: null, streaming: false, computing: false, loadingX: true, playing: false };
  let rotation = { x: 55, z: -29 }, drag = null;
  function emit(type, extra = {}) { if (typeof api.onAction === 'function') api.onAction({ type, ...extra }); }
  function set(name, value) { if (refs[name] && refs[name].textContent !== String(value)) refs[name].textContent = value; }
  function stageGroup(stage) {
    if (stage === 'embedding') return 0;
    if (['input_rmsnorm', 'qkv_proj', 'qk_norm', 'rope'].includes(stage)) return 1;
    if (['attn_scores', 'softmax', 'attn_value'].includes(stage)) return 2;
    if (['o_proj', 'attn_residual'].includes(stage)) return 3;
    if (['post_rmsnorm', 'gate_up_proj', 'silu_swiglu', 'down_proj', 'mlp_residual'].includes(stage)) return 4;
    return stage === 'lm_head' || stage === 'final_rmsnorm' ? 5 : -1;
  }
  function rows() { return Array.isArray(data.workload?.stages_detail) ? data.workload.stages_detail : []; }
  function row(name) { return rows().find(s => s.stage === name); }
  function turn() {
    if (!scene) return;
    scene.style.setProperty('--rx', `${rotation.x}deg`);
    scene.style.setProperty('--rz', `${rotation.z}deg`);
  }
  function renderChip() {
    const supplied = num(data.tileData?.constants?.sm_count);
    // A schematic grid is shown before evidence arrives; no default SM count is claimed.
    const n = supplied === null ? 96 : clamp(Math.round(supplied), 1, 512);
    const grid = `<div class="gx-sm-grid" style="--cols:${Math.ceil(Math.sqrt(n * 1.1))}">${Array.from({ length: n }, () => '<i></i>').join('')}</div>`;
    let chip = box('gx-substrate', 0, 0, 460, 310, 0, 15, '<span class="gx-etched">GPU / ARCHITECTURE STUDY</span>');
    chip += box('gx-interposer', 20, 22, 420, 266, 16, 6);
    chip += `<svg class="gx-traces" viewBox="0 0 460 310" aria-hidden="true">${[[80,65],[80,153],[80,240],[380,98],[380,205]].map(([x,y]) => `<path class="gx-trace-base" d="M${x} ${y} H${x < 230 ? 107 : 353} V155 H230"/><path class="gx-trace-flow" d="M${x} ${y} H${x < 230 ? 107 : 353} V155 H230"/>`).join('')}<path class="gx-x-trace" d="M230 290 V245 H230 V155"/></svg>`;
    chip += box('gx-die', 123, 48, 214, 214, 23, 10, `<div class="gx-die-title">COMPUTE DIE <span>SM ARRAY</span></div>${grid}<div class="gx-local-weights">FIXED WEIGHTS · LOCAL<span>CONCEPT</span></div><div class="gx-cache">ON-DIE SCRATCH / KV</div>`);
    [[43,39],[43,126],[43,213],[358,72],[358,179]].forEach(([x,y], i) => {
      chip += box('gx-hbm-base', x-3, y-3, 64, 60, 23, 5);
      for (let d = 0; d < 5; d++) chip += box('gx-hbm', x, y, 58, 54, 29 + d*5, 4, d === 4 ? `<span>HBM</span><small>W / ${i+1}</small>` : '');
    });
    // Tiny passive components make the package read as hardware, not a flat diagram.
    for (let i = 0; i < 14; i++) chip += box('gx-passive', 86+i*22, 9, 10, 5, 16, 3);
    refs.chip.innerHTML = chip;
    cells = [...refs.chip.querySelectorAll('.gx-sm-grid i')];
    set('smCount', supplied === null ? 'SM grid · illustrative' : `${count(supplied)} SMs · tile model`);
    lastKey = '';
  }
  function selectStory(index) {
    story = index; pinnedStory = index === 3 || index === 4; focusPart = '';
    if (index === 0) emit('seekStage', { stage: 'embedding' });
    if (index === 1 || index === 2) emit('seekStage', { stage: 'qkv_proj', phase: index === 1 ? 'fetch' : 'compute' });
    if (index === 3) emit('seekStage', { stage: 'gate_up_proj', phase: 'fetch' });
    if (index === 4) emit('pause');
    lastKey = ''; paint();
  }
  function mount(element) {
    if (!element || element.nodeType !== 1) throw new TypeError('GpuExperience.mount expects an element');
    observer?.disconnect(); aborter?.abort();
    if (host && host !== element) root?.replaceChildren();
    host = element; root = element.shadowRoot || element.attachShadow({ mode: 'open' });
    aborter = new AbortController();
    root.innerHTML = `<link rel="stylesheet" href="${cssUrl.replace(/&/g, '&amp;').replace(/"/g, '&quot;')}">
      <section class="gx" aria-label="GPU architecture experience">
        <header class="gx-header"><div><p class="gx-eyebrow">INSIDE THE GPU <span>01 / WEIGHT JOURNEY</span></p><h2>A fast chip.<br>A long journey for weights.</h2></div><span class="gx-model" data-ref="model">ARCHITECTURE</span></header>
        <div class="gx-viewport">
          <div class="gx-scene-status"><span class="gx-status-dot"></span><span data-ref="phase">Activation arriving</span><span class="gx-status-stage" data-ref="stageLabel">Embedding</span></div>
          <div class="gx-orbit" tabindex="0" role="group" aria-label="Rotate the 3D chip with arrow keys or drag. Home resets the view.">
            <div class="gx-scene"><div class="gx-chip" data-ref="chip" aria-hidden="true"></div></div>
          </div>
          <div class="gx-hardware-label gx-hardware-hbm"><span class="gx-label-line"></span><b>HBM stacks</b><span>weights live here</span></div>
          <div class="gx-hardware-label gx-hardware-die"><b>Silicon die</b><span data-ref="smCount">SM grid · illustrative</span></div>
          <div class="gx-view-tools"><span>Drag to orbit · arrow keys</span><button type="button" data-view="reset" aria-label="Reset chip view">↺ Reset view</button></div>
        </div>
        <div class="gx-part-controls" role="group" aria-label="Inspect the hardware"><button type="button" data-part="hbm" aria-pressed="false"><i class="gx-swatch gx-amber"></i>Weight traffic</button><button type="button" data-part="die" aria-pressed="false"><i class="gx-swatch gx-mint"></i>Compute</button><button type="button" data-part="kv" aria-pressed="false"><i class="gx-swatch gx-blue"></i>Activations / KV</button></div>
        <div class="gx-network" aria-label="Neural network execution path">
          <div class="gx-network-head"><span>THE NETWORK</span><span data-ref="layer">Transformer layers</span></div>
          <div class="gx-rail"><button type="button" data-stage="embedding"><b>Embedding</b><small>token → vector</small></button><div class="gx-transformer"><span class="gx-repeat" data-ref="repeat">REPEATED TRANSFORMER</span>${stages.slice(1,5).map(s => `<button type="button" data-stage="${s[0]}"><b>${s[1]}</b><small>${s[2]}</small></button>`).join('')}</div><button type="button" data-stage="lm_head"><b>LM head</b><small>vector → logits</small></button></div>
          <p class="gx-rail-note" data-ref="railNote">Norm, RoPE, and residual operations are included in the architectural schedule.</p>
        </div>
        <nav class="gx-story-steps" aria-label="Guided chip story">${stories.map((s,i) => `<button type="button" data-story="${i}" aria-pressed="${i===0}"><span>0${i+1}</span>${s[0]}</button>`).join('')}</nav>
        <div class="gx-story-card"><div class="gx-story-copy" aria-live="polite" aria-atomic="true"><p class="gx-eyebrow" data-ref="chapter">01 / ENTER</p><h3 data-ref="storyTitle"></h3><p data-ref="storyText"></p></div><div class="gx-story-fact"><strong data-ref="factValue">—</strong><span data-ref="factLabel">weight bytes / lookup</span></div></div>
        <div class="gx-transport"><div class="gx-play-controls"><button type="button" data-action="prev" aria-label="Previous architectural stage">←</button><button class="gx-play" type="button" data-action="play" aria-label="Play architectural schedule" aria-pressed="false">▶ Play</button><button type="button" data-action="next" aria-label="Next architectural stage">→</button></div><div class="gx-timeline"><div class="gx-timeline-caption"><span>Architectural schedule</span><span data-ref="cycle">Awaiting schedule</span></div><div class="gx-progress" role="progressbar" aria-label="Architectural schedule progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i></i></div></div></div>
        <details class="gx-evidence"><summary><span><i class="gx-evidence-dot"></i>Inside the real RTL</span><span data-ref="rtlSummary">Custom streaming core · inspect evidence</span><b aria-hidden="true">+</b></summary><div class="gx-evidence-body"><p>Measured custom <code>gpu_stream_core</code>, using the ASIC’s MAC and nonlinear engines. This is not an H100 implementation.</p><div class="gx-rtl-bars"><div><span>QKV_FETCH</span><i><em data-ref="fetchBar"></em></i><b data-ref="fetchCycles">—</b></div><div><span>QKV_MAC</span><i><em data-ref="macBar"></em></i><b data-ref="macCycles">—</b></div></div><p class="gx-rtl-note" data-ref="rtlNote">Waiting for /api/walk?schedule=streaming evidence.</p><p class="gx-evidence-foot">The package is illustrative. Playback uses an analytical architecture schedule; SM tiles use an occupancy model. Neither is an RTL waveform or a measured GPU utilization trace. Energy is modeled.</p><button type="button" class="gx-details-link" data-action="details">Open detailed simulation <span aria-hidden="true">↗</span></button></div></details>
        <footer class="gx-footer"><span>Illustrative physical layout · modeled schedule & energy</span><button type="button" data-action="details">Details ↗</button></footer>
      </section>`;
    refs = Object.fromEntries([...root.querySelectorAll('[data-ref]')].map(el => [el.dataset.ref, el]));
    refs.shell = root.querySelector('.gx'); refs.orbit = root.querySelector('.gx-orbit');
    refs.progress = root.querySelector('.gx-progress'); refs.play = root.querySelector('[data-action="play"]');
    scene = root.querySelector('.gx-scene');
    root.addEventListener('click', event => {
      const button = event.target.closest('button'); if (!button) return;
      if (button.dataset.story !== undefined) selectStory(Number(button.dataset.story));
      if (button.dataset.stage) { pinnedStory = false; story = 0; emit('seekStage', { stage: button.dataset.stage }); }
      if (button.dataset.action) { pinnedStory = false; if (story === 4) story = 0; emit(button.dataset.action); }
      if (button.dataset.view) { rotation = { x:55, z:-29 }; turn(); }
      if (button.dataset.part) { focusPart = focusPart === button.dataset.part ? '' : button.dataset.part; lastKey = ''; }
      paint();
    }, { signal: aborter.signal });
    refs.orbit.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'Home') rotation = { x:55, z:-29 };
      else { rotation.z = clamp(rotation.z + (event.key === 'ArrowRight' ? 5 : event.key === 'ArrowLeft' ? -5 : 0), -65, 25); rotation.x = clamp(rotation.x + (event.key === 'ArrowDown' ? 5 : event.key === 'ArrowUp' ? -5 : 0), 25, 68); }
      turn();
    }, { signal: aborter.signal });
    refs.orbit.addEventListener('pointerdown', event => { if (event.button !== 0) return; drag = { id:event.pointerId, x:event.clientX, y:event.clientY, rx:rotation.x, rz:rotation.z }; refs.orbit.setPointerCapture(event.pointerId); }, { signal: aborter.signal });
    refs.orbit.addEventListener('pointermove', event => { if (!drag || event.pointerId !== drag.id) return; rotation = { x:clamp(drag.rx-(event.clientY-drag.y)*.18,25,68), z:clamp(drag.rz+(event.clientX-drag.x)*.18,-65,25) }; turn(); }, { signal: aborter.signal });
    const stopDrag = () => { drag = null; };
    refs.orbit.addEventListener('lostpointercapture', stopDrag, { signal: aborter.signal });
    refs.orbit.addEventListener('pointerup', stopDrag, { signal: aborter.signal });
    refs.orbit.addEventListener('pointercancel', stopDrag, { signal: aborter.signal });
    const resize = () => scene.style.setProperty('--scale', String(Math.min(host.clientWidth < 850 ? .9 : 1.24, Math.max(.3, (refs.orbit.clientWidth - 36) / 570))));
    if (typeof ResizeObserver !== 'undefined') { observer = new ResizeObserver(resize); observer.observe(host); }
    resize(); turn(); renderChip(); paintRtl(); paint();
    return api;
  }
  function paintRtl() {
    if (!root) return;
    const list = Array.isArray(rtl?.stages) ? rtl.stages : [];
    const sumStage = name => { const matches = list.filter(s => s.stage === name && num(s.cycles) !== null); return matches.length ? matches.reduce((a,s) => a+s.cycles,0) : null; };
    const fetch = sumStage('QKV_FETCH'), mac = sumStage('QKV_MAC'), kv = sumStage('KV_CACHE');
    const max = Math.max(fetch || 0, mac || 0, 1);
    set('fetchCycles', fetch === null ? '—' : `${count(fetch)} cyc`); set('macCycles', mac === null ? '—' : `${count(mac)} cyc`);
    refs.fetchBar.style.width = `${(fetch || 0)/max*100}%`; refs.macBar.style.width = `${(mac || 0)/max*100}%`;
    const ratio = fetch !== null && mac > 0 ? `${(fetch/mac).toFixed(1)}× fetch / MAC cycles` : 'Custom streaming core · inspect evidence';
    set('rtlSummary', ratio);
    set('rtlNote', list.length ? `Measured walk: ${count(rtl.total_cycles)} total cycles. KV_CACHE: ${count(kv)} cycles. This local KV path has no HBM traffic.` : 'Waiting for /api/walk?schedule=streaming evidence.');
  }
  function paint() {
    if (!root) return;
    const progress = num(state.total) > 0 ? clamp((num(state.cycle) || 0)/state.total,0,1) : 0;
    refs.progress.firstElementChild.style.width = `${progress*100}%`;
    refs.progress.setAttribute('aria-valuenow', String(Math.round(progress*100)));
    set('cycle', num(state.total) > 0 ? `${count(state.cycle)} / ${count(state.total)} cyc` : 'Awaiting schedule');
    refs.play.textContent = state.playing ? 'Ⅱ Pause' : '▶ Play';
    refs.play.setAttribute('aria-label', `${state.playing ? 'Pause' : 'Play'} architectural schedule`);
    refs.play.setAttribute('aria-pressed', String(!!state.playing));
    if (!pinnedStory) story = state.stage === 'embedding' ? 0 : num(state.layer) > 0 ? 3 : state.streaming ? 1 : state.computing ? 2 : 1;
    const phase = state.loadingX ? 'load' : state.streaming ? 'fetch' : state.computing ? 'compute' : 'idle';
    const key = [state.stage,state.layer,phase,story,focusPart,state.playing].join('|');
    if (key === lastKey) return; lastKey = key;
    const group = stageGroup(state.stage), current = row(state.stage), metrics = data.platform?.metrics || {};
    const layers = row('qkv_proj')?.count;
    refs.shell.dataset.phase = phase; refs.shell.dataset.compare = String(story === 4);
    refs.shell.dataset.playing = String(!!state.playing); refs.shell.dataset.focus = focusPart;
    refs.shell.dataset.group = String(group);
    set('model', data.platform?.platform || data.platform?.label || 'ARCHITECTURE');
    set('phase', story === 4 ? 'Concept · weights stay on chip' : ({load:'Activation arriving',fetch:'Fetching weights',compute:'Computing',idle:'Ready'})[phase]);
    set('stageLabel', group >= 0 ? stages[group][1] : 'Schedule');
    set('repeat', layers ? `REPEAT × ${count(layers)}` : 'REPEATED TRANSFORMER');
    set('layer', state.layer != null && num(state.layer) !== null ? `Layer ${count(state.layer+1)}${layers ? ` / ${count(layers)}` : ''}` : group === 5 ? 'Final norm → output' : 'Token enters the network');
    root.querySelectorAll('[data-stage]').forEach(el => el.setAttribute('aria-current', String(stageGroup(el.dataset.stage) === group ? 'step' : 'false')));
    root.querySelectorAll('[data-story]').forEach(el => el.setAttribute('aria-pressed', String(Number(el.dataset.story) === story)));
    root.querySelectorAll('[data-part]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.part === focusPart)));
    let title = stories[story][1], description = stories[story][2];
    let value, label;
    if (story === 0) { value = bytes(row('embedding')?.weight_bytes_per_call); label = 'weight bytes / embedding lookup'; }
    if (story === 1) { value = bytes(current?.weight_bytes_per_call); label = 'weight bytes / current stage call'; }
    if (story === 2) { value = count(current?.macs_per_call); label = 'MACs / current stage call'; }
    if (story === 3) { value = bytes(metrics.total_weight_bytes ?? data.workload?.weight_bytes); label = 'weight traffic / supplied workload'; }
    if (story === 4) { value = energy(metrics.weight_energy_j); label = 'modeled external weight-energy term'; }
    if (focusPart === 'hbm') { title = 'Weights cross the package.'; description = 'Amber traces show HBM → die traffic. In gpu_stream_core, u_hbm feeds the QKV_FETCH state; the modeled architecture repeats weight transfers across projections.'; }
    if (focusPart === 'die') { title = 'A grid of compute resources.'; description = 'The package grid comes from the supplied SM tile model. The separate custom RTL executes a 16-lane MAC reduction in u_mac; it does not implement the full SM grid.'; }
    if (focusPart === 'kv') { title = 'Context has a different path.'; description = 'The custom RTL writes and reads K/V through local u_kv. Its KV_CACHE stage has no HBM traffic. This local-cache assumption is specific to the reference core.'; }
    set('chapter', `${String(story+1).padStart(2,'0')} / ${stories[story][0].toUpperCase()}${focusPart ? ' · INSPECT' : ''}`);
    set('storyTitle', title); set('storyText', description); set('factValue', value); set('factLabel', label);
    const tile = data.tileData?.stages?.find(s => s.stage === state.stage);
    // Do not turn occupancy fractions into a claimed measured SM utilization.
    cells.forEach((el,i) => el.classList.toggle('gx-active', phase === 'compute' && group !== 0 && (i%4 !== 3)));
    set('railNote', story === 4 ? 'Removing this HBM energy term alone is not a total-energy or speedup prediction.' : group === 2 ? 'Attention uses Q/K/V and the local cache in the custom RTL; it is not another projection-weight fetch.' : tile && ['qkv_proj','o_proj','gate_up_proj','down_proj','lm_head'].includes(state.stage) ? `${count(tile.geometry?.total_tiles)} tile operations · ${count(tile.waves)} modeled waves · chip lighting indicates phase, not utilization.` : 'Norm, RoPE, and residual operations are included in the architectural schedule.');
  }
  const api = {
    onAction: null,
    mount,
    setData(next = {}) { data = { ...data, ...next }; if (root) { renderChip(); paint(); } return api; },
    setRtl(next) { rtl = next; paintRtl(); return api; },
    update(next = {}) { state = { ...state, ...next }; paint(); return api; },
    destroy() { observer?.disconnect(); aborter?.abort(); root?.replaceChildren(); root = null; host = null; scene = null; refs = {}; cells = []; drag = null; }
  };
  window.GpuExperience = api;
})();
