// Can Eva learn to see and hear? Synthetic paired corpora, trained headless with the app's own model,
// trainer, optimizer and sense adapters, then evaluated on FRESH renders it never saw:
//   sense gap = caption loss with a WRONG sense − with the RIGHT one (nats; > 0 means it looks/listens)
//   picks     = for each new image/sound, is the right caption the likeliest of all captions?
//   captions  = greedy, no prompt — exactly what GENERATE does with a sense
// usage: TASK=shapes|vowels|balls node tests/senses-learn.js [steps] [seed]
//   env: DIM (48) LR (0.01) ALR (adapter LR multiplier, 0.1)
//   LEGACY_FILE=old.html: use that page's single-vector (v2) feature extractors instead, for comparison
const fs = require('fs'), path = require('path'), vm = require('vm');
const ctx = require('./load.js')();
const M = ctx.M;
let L = null;
if (process.env.LEGACY_FILE) { const c2 = { Math, Float32Array, Uint8Array, Array, Object }; vm.createContext(c2);
  const src = fs.readFileSync(process.env.LEGACY_FILE, 'utf8'), i = src.indexOf('const SenseFeatures = {'); vm.runInContext(src.slice(i, src.indexOf('\n};', i) + 3) + '\nthis.F = SenseFeatures;', c2); L = c2.F; }
const [STEPS = '1600', SEED = '1'] = process.argv.slice(2), TASK = process.env.TASK || 'shapes';
let seed = Number(SEED); const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
ctx.Math.random = Math.random = rnd;
// classes are chosen from an INDEPENDENT generator (mulberry32). Consecutive draws of the LCG above are
// correlated, so choosing the class and then drawing its pitch/size/position from the same stream let the
// "random" nuisance values encode the class: a linear probe read vowel ORDER at 100% from features that
// can't hear order (each order-feature differed by < 0.4 sd). Everything measured since uses pick().
const pick = (() => { let a = (Number(SEED) * 2654435761) | 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();

// ── renderers ──────────────────────────────────────────────────────────
const COLORS = { red: [220, 40, 40], green: [40, 200, 60], blue: [50, 80, 230], yellow: [230, 220, 40] };
const inside = { circle: (dx, dy, r) => dx * dx + dy * dy < r * r, square: (dx, dy, r) => Math.abs(dx) < r * 0.85 && Math.abs(dy) < r * 0.85,
  triangle: (dx, dy, r) => dy < r * 0.8 && dy > -r && Math.abs(dx) < (dy + r) * 0.6, cross: (dx, dy, r) => (Math.abs(dx) < r * 0.3 && Math.abs(dy) < r) || (Math.abs(dy) < r * 0.3 && Math.abs(dx) < r) };
function renderImage(cl) { const W = 48, H = 48, d = new Uint8Array(W * H * 4), col = COLORS[cl.c], bg = 30 + rnd() * 60;
  // SMALL=1: shapes about 1/8 of the frame (detail an 8×8 grid can't resolve)
  const r = (cl.shape ? (process.env.SMALL ? 4.5 : 11) : 8) + rnd() * (process.env.SMALL ? 2 : 5), cx = cl.s ? (cl.s === 'left' ? 0.27 : 0.73) * W + (rnd() - 0.5) * 6 : W * (0.35 + rnd() * 0.3), cy = H * (0.35 + rnd() * 0.3);
  const test = inside[cl.shape || 'circle'];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const p = (y * W + x) * 4, on = test(x - cx, y - cy, r);
    for (let k = 0; k < 3; k++) d[p + k] = Math.max(0, Math.min(255, (on ? col[k] : bg) + (rnd() - 0.5) * 30)); d[p + 3] = 255; }
  return { data: d, width: W, height: H }; }
const SR = 16000, VOWELS = { a: [730, 1090, 2440], i: [270, 2290, 3010], u: [300, 870, 2240] };
function renderVowels(seq) {           // formant synthesis: glottal pulse train → 3 resonators, per vowel
  // random silence before, between and after, so WHERE a vowel sits can't give the order away (the old
  // single-vector ears scored 100% on a fixed-layout version of this test by reading frame cut-offs at the end)
  const out = []; const f0 = 100 + rnd() * 50;
  for (let i = 0; i < SR * (0.03 + rnd() * 0.2); i++) out.push((rnd() - 0.5) * 0.01);
  for (const v of seq) { const n = Math.floor(SR * (0.25 + rnd() * 0.1)), F = VOWELS[v].map(f => f * (0.93 + rnd() * 0.14)), buf = new Float64Array(n);
    let ph = 0; for (let i = 0; i < n; i++) { ph += f0 / SR; buf[i] = ph >= 1 ? (ph -= 1, 1) : 0; }
    let x = buf; for (const fc of F) { const bw = 80 + fc * 0.05, r = Math.exp(-Math.PI * bw / SR), a1 = 2 * r * Math.cos(2 * Math.PI * fc / SR), a2 = -r * r, y = new Float64Array(n);
      for (let i = 0; i < n; i++) y[i] = x[i] + a1 * (y[i - 1] || 0) + a2 * (y[i - 2] || 0); x = y; }
    let mx = 1e-9; for (const s of x) mx = Math.max(mx, Math.abs(s));
    for (let i = 0; i < n; i++) { const env = Math.min(1, i / 400, (n - i) / 400); out.push(0.4 * env * x[i] / mx + (rnd() - 0.5) * 0.01); }
    for (let i = 0; i < SR * (0.04 + rnd() * 0.12); i++) out.push((rnd() - 0.5) * 0.01); }
  for (let i = 0; i < SR * rnd() * 0.15; i++) out.push((rnd() - 0.5) * 0.01);
  return Float64Array.from(out); }

// ── tasks ──────────────────────────────────────────────────────────────
let classes, modality, makeRaw;
if (TASK === 'shapes') { modality = 'vision'; classes = []; for (const c of Object.keys(COLORS)) for (const shape of Object.keys(inside)) classes.push({ c, shape, text: `a ${c} ${shape}.` }); makeRaw = renderImage; }
else if (TASK === 'balls') { modality = 'vision'; classes = []; for (const c of Object.keys(COLORS)) for (const s of ['left', 'right']) classes.push({ c, s, text: `a ${c} ball on the ${s}.` }); makeRaw = renderImage; }
else if (TASK === 'vowels') { modality = 'audio'; classes = []; for (const a of 'aiu') for (const b of 'aiu') if (a !== b) classes.push({ seq: [a, b], text: `${a} then ${b}.` }); makeRaw = cl => renderVowels(cl.seq); }
else throw new Error('TASK must be shapes, vowels or balls');

let filters = null;
const featurize = raw => {
  if (L) { const f = modality === 'vision' ? L.image(raw) : L.audio(Float32Array.from(raw), SR); return { T: 1, Fp: f.length, seq: f, pooled: f }; }
  return modality === 'vision' ? M.SenseFeatures.imageTokens(M.SenseFeatures.image32(raw), filters) : M.SenseFeatures.audioTokens(raw, SR);
};
if (modality === 'vision' && !L) {     // eyes learn their filters from unlabeled renders first, as the app does
  const imgs = []; for (let n = 0; n < 120; n++) imgs.push(M.SenseFeatures.image32(makeRaw(classes[Math.floor(pick() * classes.length)])));
  filters = M.SenseFeatures.learnEyeFilters(imgs);
}
const probe = featurize(makeRaw(classes[0]));
const spec = L ? { Fp: probe.Fp, Fq: probe.Fp, Tmax: 1, pos: false } : M.SENSE_SPEC[modality];
const enc = t => Array.from(new TextEncoder().encode(t)), dim = Number(process.env.DIM || 48), lr = Number(process.env.LR || 0.01);
const model = new M.LanguageModel({ vocab: 256, dim, layers: 2 }), adapter = new M.SenseAdapter(modality, spec, dim);
const opt = new M.AdamOptimizer(lr), tr = new M.Trainer(model, opt);

function evaluate() { let gap = 0, correct = 0;
  for (const [ci, cl] of classes.entries()) { const f = featurize(makeRaw(cl)), toks = enc(cl.text);
    const losses = classes.map(o => tr.captionLoss(adapter, f, enc(o.text))); let best = 0; losses.forEach((x, j) => { if (x < losses[best]) best = j; }); correct += best === ci;
    const wrong = classes[(ci + 1 + Math.floor(pick() * (classes.length - 1))) % classes.length]; gap += tr.captionLoss(adapter, featurize(makeRaw(wrong)), toks) - losses[ci]; }
  return { gap: gap / classes.length, acc: correct / classes.length }; }
(async () => {
  const t0 = Date.now(), tag = `${TASK}${L ? ' (legacy v2 senses)' : ''}`;
  const show = (s, r) => console.log(`${tag} step ${String(s).padStart(4)}: gap ${r.gap.toFixed(3)} nats, picks right caption ${(100 * r.acc).toFixed(0)}% (chance ${(100 / classes.length).toFixed(0)}%)  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  show(0, evaluate());
  for (let s = 1; s <= Number(STEPS); s++) {
    const cl = classes[Math.floor(pick() * classes.length)];
    tr.trainPaired(adapter, featurize(makeRaw(cl)), enc(cl.text));
    await opt.step(model); model.embStep(lr * 0.1); model.zeroGrad(); adapter.step(lr * Number(process.env.ALR || 0.1));
    if (s % Number(process.env.EVERY || 400) === 0) show(s, evaluate());
  }
  let exact = 0;
  for (const cl of classes) { const f = featurize(makeRaw(cl)), pre = adapter.prefix(f), cv = adapter.context(f); let st = model.getEmptyState(), r, out = [];
    for (const v of pre) { r = model.forwardVec(v, st); st = r.state; }
    for (let i = 0; i < 30; i++) { let b = 0; for (let j = 1; j < 256; j++) if (r.logits[j] > r.logits[b]) b = j; out.push(b); if (b === 46) break; r = model.forward(b, st, cv); st = r.state; }
    const said = new TextDecoder().decode(Uint8Array.from(out)); exact += said === cl.text;
    if (process.env.SHOW) console.log(`   ${JSON.stringify(cl.text).padEnd(22)} → ${JSON.stringify(said)}${said === cl.text ? '  ✓' : ''}`); }
  console.log(`${tag}: ${exact}/${classes.length} new ${modality === 'vision' ? 'images' : 'sounds'} captioned exactly`);
})().catch(e => { console.error('ERR', e); process.exit(1); });
