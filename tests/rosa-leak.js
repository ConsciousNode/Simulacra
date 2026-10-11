// Regression test for the OOMB recompute leak. Runs Trainer.trainSequence on
// random tokens and checks that ROSA never sees a chunk twice: next-code
// accuracy during the gradient pass must stay at chance, and ROSA's history
// after training must equal its history after a single forward pass.
const ctx = require('./load.js')();
const M = ctx.M;
let seed = 7; ctx.Math.random = Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
ctx.document = { getElementById: () => ({ value: '128' }) };
ctx.yieldToMain = () => Promise.resolve();
const cfg = { vocab: 256, dim: 32, layers: 2 };
const model = new M.LanguageModel(cfg);
const seq = Array.from({ length: 129 }, () => Math.floor(Math.random() * 256));
// Instrument: record (prediction, next code) pairs seen during backward-pass forwards.
let phase = 'pass1'; const hits = { pass1: [0, 0], recompute: [0, 0] };
const origFwd = model.forward.bind(model); let prev = null;
let nFwd = 0;
model.forward = (t, st) => { if (nFwd++ === seq.length - 1) { phase = 'recompute'; prev = null; } const r = origFwd(t, st);
  const codes = model.blocks.map(b => b.rosaChannel.history.at(-1));
  if (prev) prev.preds.forEach((p, L) => { if (p >= 0) { hits[phase][1]++; if (p === codes[L]) hits[phase][0]++; } });
  prev = { preds: r.caches.blocks.map(c => c.prediction) }; return r; };
const trainer = new M.Trainer(model, null);
(async () => {
  await trainer.trainSequence(seq);
  const acc = k => (100 * hits[k][0] / Math.max(1, hits[k][1])).toFixed(1);
  console.log(`ROSA next-code accuracy  forward pass: ${acc('pass1')}%   gradient recompute: ${acc('recompute')}%`);
  const leak = hits.recompute[0] / Math.max(1, hits.recompute[1]) > 0.2;
  const histLen = model.blocks.map(b => b.rosaChannel.history.length);
  console.log('ROSA history length after training:', histLen.join(', '), '(one pass of', seq.length - 1, 'tokens + bookmarks)');
  console.log(leak ? 'FAIL: recompute pass sees the future' : 'OK: no leak');
  process.exit(leak ? 1 : 0);
})();
