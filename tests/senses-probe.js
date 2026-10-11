// Capability probe: can the sense FEATURES carry the information at all? A linear softmax classifier is
// trained directly on the (standardized) features of fresh renders, then tested on others: no language
// model involved. High test accuracy = the capability is there; learning to caption is then a training budget.
//   shapes: which of 4 shapes (colour, size, position, background, noise all random)
//   vowels: which of 6 ordered vowel pairs ("a then i" vs "i then a": same sounds, different ORDER)
// usage: node tests/senses-probe.js [nTrain=480] [nTest=240]      LEGACY_FILE=old.html: probe the v2 extractors
const fs = require('fs'), path = require('path');
process.env.TASK = 'shapes';
const [NTR = '480', NTE = '240'] = process.argv.slice(2);
// reuse the renderers and featurizers from senses-learn.js without training anything
const src = fs.readFileSync(path.join(__dirname, 'senses-learn.js'), 'utf8');
const head = src.slice(0, src.indexOf('const probe = featurize(')).replace("process.env.TASK || 'shapes'", 'TASKNAME');
const results = [];
for (const TASKNAME of (process.env.ONLY ? [process.env.ONLY] : ['shapes', 'vowels'])) {
  const fn = new Function('require', '__dirname', 'process', 'TASKNAME', head + `
    const label = cl => TASKNAME === 'shapes' ? Object.keys(inside).indexOf(cl.shape) : classes.indexOf(cl);
    const nCls = TASKNAME === 'shapes' ? 4 : classes.length;
    // fixed-length view: the summary + the stream resampled to 16 time slices (clips differ in length)
    // VIEW=pooled: only the position-free summary (for eyes: each filter's mean per quadrant + max anywhere)
    const vec = f => { if (f.T === 1) return Float64Array.from(f.seq); if (process.env.VIEW === 'pooled') return Float64Array.from(f.pooled); const o = [...f.pooled];
      for (let k = 0; k < 16; k++) { const t = Math.min(f.T - 1, Math.floor(k * f.T / 16)); for (let i = 0; i < f.Fp; i++) o.push(f.seq[t * f.Fp + i]); } return Float64Array.from(o); };
    return { make: () => { const cl = classes[Math.floor(pick() * classes.length)]; return { x: vec(featurize(makeRaw(cl))), y: label(cl) }; }, nCls };`);
  const { make, nCls } = fn(require, __dirname, process, TASKNAME);
  const tr = Array.from({ length: +NTR }, make), te = Array.from({ length: +NTE }, make), D = tr[0].x.length;
  const mu = new Float64Array(D), sd = new Float64Array(D);
  for (const s of tr) for (let i = 0; i < D; i++) mu[i] += s.x[i] / tr.length;
  for (const s of tr) for (let i = 0; i < D; i++) sd[i] += (s.x[i] - mu[i]) ** 2 / tr.length;
  const z = x => x.map((v, i) => (v - mu[i]) / Math.sqrt(sd[i] + 1e-6));
  const Xtr = tr.map(s => z(s.x)), Xte = te.map(s => z(s.x)), W = new Float64Array(nCls * D), b = new Float64Array(nCls);
  const logits = x => Array.from({ length: nCls }, (_, c) => { let s = b[c]; for (let i = 0; i < D; i++) s += W[c * D + i] * x[i]; return s; });
  for (let ep = 0; ep < 60; ep++) for (let n = 0; n < Xtr.length; n++) {      // SGD, L2, softmax cross-entropy
    const l = logits(Xtr[n]), mx = Math.max(...l), p = l.map(v => Math.exp(v - mx)), S = p.reduce((a, c) => a + c);
    for (let c = 0; c < nCls; c++) { const g = p[c] / S - (c === tr[n].y ? 1 : 0); b[c] -= 0.01 * g; for (let i = 0; i < D; i++) W[c * D + i] -= 0.01 * (g * Xtr[n][i] + 1e-3 * W[c * D + i]); }
  }
  const acc = X => X.reduce((a, x, n) => { const l = logits(x); return a + (l.indexOf(Math.max(...l)) === (X === Xtr ? tr : te)[n].y); }, 0) / X.length;
  results.push(`${(TASKNAME + (TASKNAME === 'shapes' && process.env.SMALL ? ' (small)' : '')).padEnd(7)} (${nCls}-way, chance ${(100 / nCls).toFixed(0)}%): test accuracy ${(100 * acc(Xte)).toFixed(0)}%  [train ${(100 * acc(Xtr)).toFixed(0)}%, ${D} features]`);
}
console.log((process.env.LEGACY_FILE ? 'legacy v2 senses\n' : 'senses v3\n') + results.join('\n'));
