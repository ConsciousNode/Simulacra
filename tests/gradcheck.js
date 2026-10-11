// Finite-difference gradient check for SimulacraBlock / LanguageModel backward.
// Runs a 2-token sequence through a tiny 2-layer model so token-shift (BPTT)
// gradients are exercised, then compares every weight matrix's analytic
// gradient to central differences. ROSA predictions are frozen by restoring
// channel state before each evaluation (ROSA is stop-gradient by design).
// BitLinear.forward is swapped for a dense float matmul: the backward uses
// straight-through (STE) semantics, which is the gradient of the float model.
const ctx = require('./load.js')();
const M = ctx.M;
let seed = 12345;
ctx.Math.random = Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
M.BitLinear.prototype.forward = function (x) {
  const o = new Float32Array(this.outDim);
  for (let r = 0; r < this.outDim; r++) { let a = 0; for (let i = 0; i < this.inDim; i++) a += this.weight[r * this.inDim + i] * x[i]; o[r] = a; }
  return o;
};
const cfg = { vocab: 20, dim: 12, layers: 2 };
const model = new M.LanguageModel(cfg);
model._isFirstToken = false;
for (const b of model.blocks) {
  const H = []; for (let k = 0; k < 64; k++) H.push(k); for (let k = 0; k < 64; k++) H.push((k * 7) % 64);
  b.rosaChannel.history = H; b.rosaEmbed.scale.fill(1.0);
}
const snap = () => model.blocks.map(b => ({ h: b.rosaChannel.history.slice(), bm: b._bookmarkHistory.slice(), lp: b.rosaChannel.lastPrediction, tp: b.rosaChannel.totalPredictions, cp: b.rosaChannel.correctPredictions }));
const restore = s => model.blocks.forEach((b, i) => { b.rosaChannel.history = s[i].h.slice(); b._bookmarkHistory = s[i].bm.slice(); b.rosaChannel.lastPrediction = s[i].lp; b.rosaChannel.totalPredictions = s[i].tp; b.rosaChannel.correctPredictions = s[i].cp; });
const S0 = snap();
const seq = [5, 9], target = 7;
const ce = lg => { let mx = -1e9; for (const v of lg) mx = Math.max(mx, v); let s = 0; for (const v of lg) s += Math.exp(v - mx); return -(lg[target] - mx - Math.log(s)); };
const run = () => { restore(S0); let st = model.getEmptyState(); const outs = []; for (const t of seq) { const r = model.forward(t, st); st = r.state; outs.push(r); } return outs; };
const loss = () => ce(run().at(-1).logits);

model.zeroGrad();
const outs = run();
const p = Float32Array.from(outs[1].logits); let mx = -1e9; for (const v of p) mx = Math.max(mx, v); let s = 0;
for (let i = 0; i < p.length; i++) { p[i] = Math.exp(p[i] - mx); s += p[i]; } for (let i = 0; i < p.length; i++) p[i] /= s; p[target] -= 1;
const g2 = model.backward(p, outs[1].caches, null);
model.backward(new Float32Array(cfg.vocab), outs[0].caches, g2.gradStatesOut);

const eps = 1e-3; let worst = 1, fails = 0;
function check(label, W, G) {
  let dot = 0, na = 0, nn = 0;
  for (let i = 0; i < W.length; i++) {
    const o = W[i]; W[i] = o + eps; const a = loss(); W[i] = o - eps; const b = loss(); W[i] = o;
    const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num;
  }
  if (nn < 1e-14) { console.log(`${label.padEnd(16)} (no gradient in this configuration)`); return; }
  const cos = dot / Math.sqrt(na * nn), ratio = Math.sqrt(na / nn);
  const ok = cos > 0.99 && Math.abs(ratio - 1) < 0.05; if (!ok) fails++; worst = Math.min(worst, cos);
  console.log(`${label.padEnd(16)} cos=${cos.toFixed(4)}  |analytic|/|true|=${ratio.toFixed(4)}  ${ok ? 'ok' : 'FAIL'}`);
}
const names = ['W_x_r', 'W_x_k', 'W_x_v', 'W_o', 'W_g', 'W_rosa', 'W_x_rx2', 'W_x_kx2', 'Wr2', 'Wk2', 'Wv2'];
for (let L = cfg.layers - 1; L >= 0; L--) {
  for (const n of names) check(`L${L}.${n}`, model.blocks[L][n].weight, model.blocks[L][n].grad);
  check(`L${L}.ln1.w`, model.blocks[L].ln1.weight, model.blocks[L].ln1.grad_w);
  check(`L${L}.ln2.w`, model.blocks[L].ln2.weight, model.blocks[L].ln2.grad_w);
}
check('head', model.head.weight, model.head.grad);
check('embedding', model.embedding, model.embGrad);
console.log(fails ? `\n${fails} FAILED (worst cos ${worst.toFixed(4)})` : `\nALL OK (worst cos ${worst.toFixed(4)})`);
process.exit(fails ? 1 : 0);
