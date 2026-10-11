// Does Evangelion learn to USE its senses? Synthetic paired corpus, trained headless with the app's
// own model / trainer / optimizer / adapter. Evaluated on FRESH renders it never saw:
//   sense gap  = caption loss with a WRONG sense − with the RIGHT sense (nats; > 0 means it looks)
//   pick-the-caption accuracy = for each new image/sound, which caption does the model find likeliest?
// usage: node tests/senses-learn.js [steps=600] [seed=1]
const fs = require('fs');
const ctx = require('./load.js')();
const M = ctx.M; const [STEPS = '600', SEED = '1'] = process.argv.slice(2);
let seed = Number(SEED); const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
ctx.Math.random = Math.random = rnd;

const COLORS = { red: [220, 40, 40], green: [40, 200, 60], blue: [50, 80, 230], yellow: [230, 220, 40] };
const SIDES = ['left', 'right'];
const vClasses = []; for (const c of Object.keys(COLORS)) for (const s of SIDES) vClasses.push({ c, s, text: process.env.ORDER === 'first' ? `${c} ${s}: a ball.` : process.env.ORDER === 'sidefirst' ? `${s}: a ${c} ball.` : `a ${c} ball on the ${s}.` });
const renderImg = ({ c, s }) => { const W = 48, H = 36, d = new Uint8Array(W * H * 4), col = COLORS[c];
  const cx = (s === 'left' ? 0.25 : 0.75) * W + (rnd() - 0.5) * 8, cy = H * (0.35 + rnd() * 0.3), r = 7 + rnd() * 4, bg = 30 + rnd() * 50;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const p = (y * W + x) * 4, inside = (x - cx) ** 2 + (y - cy) ** 2 < r * r;
    for (let k = 0; k < 3; k++) d[p + k] = Math.max(0, Math.min(255, (inside ? col[k] : bg) + (rnd() - 0.5) * 40)); d[p + 3] = 255; }
  return M.SenseFeatures.image({ data: d, width: W, height: H }); };
const SR = 16000;
const aClasses = [{ p: 'low', r: 'hum', text: 'a low steady hum.' }, { p: 'low', r: 'beeps', text: 'low beeps, on and off.' },
                  { p: 'high', r: 'hum', text: 'a high steady whistle.' }, { p: 'high', r: 'beeps', text: 'high beeps, on and off.' }];
const renderAud = ({ p, r }) => { const f = (p === 'low' ? 180 : 2400) * (0.85 + rnd() * 0.3), n = SR, x = new Float32Array(n), rate = 3 + rnd() * 3;
  for (let i = 0; i < n; i++) { const env = r === 'hum' ? 1 : (Math.sin(2 * Math.PI * rate * i / SR) > 0 ? 1 : 0); x[i] = 0.4 * env * Math.sin(2 * Math.PI * f * i / SR) + (rnd() - 0.5) * 0.05; }
  return M.SenseFeatures.audio(x, SR); };

const enc = t => Array.from(new TextEncoder().encode(t));
const dim = Number(process.env.DIM || 48);
const model = new M.LanguageModel({ vocab: 256, dim, layers: 2 });
const adapters = { vision: new M.SenseAdapter('vision', 272, dim), audio: new M.SenseAdapter('audio', 64, dim) };
const lr = Number(process.env.LR || 3e-3);
const opt = new M.AdamOptimizer(lr); const tr = new M.Trainer(model, opt);
const mods = [{ name: 'vision', classes: vClasses, render: renderImg }, { name: 'audio', classes: aClasses, render: renderAud }].filter(md => !process.env.ONLY || md.name === process.env.ONLY);

function evaluate() {
  const res = {};
  for (const md of mods) { const ad = adapters[md.name]; let gap = 0, n = 0, correct = 0, total = 0;
    for (let rep = 0; rep < 3; rep++) md.classes.forEach((cl, ci) => {
      const f = md.render(cl), toks = enc(cl.text);
      const losses = md.classes.map(other => tr.captionLoss(ad, f, enc(other.text)));
      let best = 0; losses.forEach((L, j) => { if (L < losses[best]) best = j; }); correct += best === ci; total++;
      const wrong = md.classes[(ci + 1 + Math.floor(rnd() * (md.classes.length - 1))) % md.classes.length];
      gap += tr.captionLoss(ad, md.render(wrong), toks) - tr.captionLoss(ad, f, toks); n++;
    });
    res[md.name] = { gap: gap / n, acc: correct / total, chance: 1 / md.classes.length }; }
  return res;
}
(async () => {
  const t0 = Date.now(), show = (s, r) => console.log(`step ${String(s).padStart(4)}  ` + mods.map(md => `${md.name}: gap ${r[md.name].gap.toFixed(3)} nats, picks right caption ${(100 * r[md.name].acc).toFixed(0)}% (chance ${(100 * r[md.name].chance).toFixed(0)}%)`).join(' · ') + `  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  show(0, evaluate());
  let run = 0;
  for (let s = 1; s <= Number(STEPS); s++) {
    const md = mods[s % mods.length], cl = md.classes[Math.floor(rnd() * md.classes.length)];
    run = 0.98 * run + 0.02 * tr.trainPaired(adapters[md.name], md.render(cl), enc(cl.text));
    for (let k = 0; k < M.CONFIG.grad_accum_steps; k++) {}   // accumulation handled by the optimizer's divisor
    await opt.step(model); model.embStep(lr * 0.1); model.zeroGrad();
    for (const a of Object.values(adapters)) a.step(lr * Number(process.env.ALR || 1));
    if (s % 200 === 0) show(s, evaluate());
  }
  console.log(`train caption loss (EMA) ${run.toFixed(3)}`);
  // let it caption things it has never seen (greedy, no prompt — exactly what GENERATE does with a sense)
  for (const md of mods) for (const cl of md.classes.slice(0, md.name === 'vision' ? 8 : 4)) {
    const f = md.render(cl), pre = adapters[md.name].prefix(f), cv = adapters[md.name].context ? adapters[md.name].context(f) : null; let st = model.getEmptyState(), r, out = [];
    for (const v of pre) { r = model.forwardVec(v, st); st = r.state; }
    for (let i = 0; i < 30; i++) { let b = 0; for (let j = 1; j < 256; j++) if (r.logits[j] > r.logits[b]) b = j; out.push(b); if (b === 46) break; r = model.forward(b, st, cv); st = r.state; }
    const said = new TextDecoder().decode(Uint8Array.from(out));
    console.log(`  ${md.name} ${JSON.stringify(cl.text).padEnd(28)} → ${JSON.stringify(said)}${said === cl.text ? '  ✓' : ''}`);
  }
})().catch(e => { console.error("ERR", e); process.exit(1); });
