// End-to-end check: train a tiny model on a text file with the real Trainer
// (OOMB) and AdamOptimizer, then report HELD-OUT loss measured with a single
// forward pass from a fresh ROSA state (no recompute, so nothing can leak).
// usage: node tests/train-compare.js <file.txt> [steps] [seed]
const ctx = require('./load.js')(); const M = ctx.M;
let seed = Number(process.argv[4] || 1);
ctx.Math.random = Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const CTX = 32, STEPS = Number(process.argv[3] || 150);
ctx.document = { getElementById: () => ({ value: String(CTX) }) };
const text = require('fs').readFileSync(process.argv[2], 'utf8');
const toks = Array.from(new TextEncoder().encode(text)); const split = Math.floor(toks.length * 0.85);
const train = toks.slice(0, split), test = toks.slice(split);
const model = new M.LanguageModel({ vocab: 256, dim: 32, layers: 2 });
const opt = new M.AdamOptimizer(3e-3);
const trainer = new M.Trainer(model, opt);
const evalLoss = () => {
  const keep = trainer._snapshotRosa ? trainer._snapshotRosa() : null;
  model.resetContext(); let st = model.getEmptyState(), tot = 0, n = 0;
  for (let i = 0; i < Math.min(test.length - 1, 500); i++) {
    const r = model.forward(test[i], st); st = r.state;
    let mx = -1e9; for (const v of r.logits) mx = Math.max(mx, v); let s = 0; for (const v of r.logits) s += Math.exp(v - mx);
    tot += -(r.logits[test[i + 1]] - mx - Math.log(s)); n++;
  }
  if (keep) trainer._restoreRosa(keep);
  return tot / n;
};
(async () => {
  const t0 = Date.now(); let lastTrain = 0;
  console.log(`step   0  held-out ${evalLoss().toFixed(3)}`);
  for (let step = 1; step <= STEPS; step++) {
    let acc = 0;
    for (let m = 0; m < M.CONFIG.grad_accum_steps; m++) {
      const st = Math.floor(Math.random() * (train.length - CTX - 1));
      acc += await trainer.trainSequence(train.slice(st, st + CTX + 1));
    }
    await opt.step(model); model.embStep(3e-3 * 0.1 / M.CONFIG.grad_accum_steps); model.zeroGrad();
    lastTrain = acc / M.CONFIG.grad_accum_steps;
    if (step % 50 === 0) console.log(`step ${String(step).padStart(3)}  train ${lastTrain.toFixed(3)}  held-out ${evalLoss().toFixed(3)}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
})();
