// Senses v3 (Simulacra): feature extractors + exact gradients of the paired path [T sense tokens] + caption,
// through SenseAdapter (W, b, positions P, context C) and the whole ROSA-primary model. ROSA's in-block
// history is restored before every evaluation (stop-gradient by design). Float32 like tests/gradcheck.js,
// so the step is 1e-3 and the bar is cos > 0.99. `node tests/senses.js`
const ctx = require('./load.js')();
const M = ctx.M; let seed = 7;
ctx.Math.random = Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
M.BitLinear.prototype.forward = function (x) { const o = new Float32Array(this.outDim); for (let r = 0; r < this.outDim; r++) { let a = 0; for (let i = 0; i < this.inDim; i++) a += this.weight[r * this.inDim + i] * x[i]; o[r] = a; } return o; };
M.BitLinear.prototype.backward = function (g, x) { const gi = new Float64Array(this.inDim); for (let r = 0; r < this.outDim; r++) { if (!g[r]) continue; for (let i = 0; i < this.inDim; i++) { this.grad[r * this.inDim + i] += g[r] * x[i]; gi[i] += this.weight[r * this.inDim + i] * g[r]; } } return gi; };   // float STE pair
let fails = 0; const ok = (cond, msg) => { console.log(`${msg}  ${cond ? 'ok' : 'FAIL'}`); if (!cond) fails++; };

// ── 1. eigensolver ─────────────────────────────────────────────────────
{ const n = 6, A = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = i; j < n; j++) A[i * n + j] = A[j * n + i] = Math.sin(i * 3 + j * 7) + (i === j ? 3 : 0);
  const { vals, vecs } = M.symEig(A, n); let err = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < n; k++) s += vecs[i * n + k] * vals[k] * vecs[j * n + k]; err = Math.max(err, Math.abs(s - A[i * n + j])); }
  ok(err < 1e-4, `symEig reconstructs A (max err ${err.toExponential(1)})`); }

// ── 2. eyes ────────────────────────────────────────────────────────────
const draw = (fn, W = 64, H = 48) => { const d = new Uint8Array(W * H * 4); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const [r, g, b] = fn(x / W, y / H); const p = (y * W + x) * 4; d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255; } return { data: d, width: W, height: H }; };
const red = M.SenseFeatures.image32(draw(() => [255, 0, 0]));
ok(red.length === 32 * 32 * 3 && red[0] === 255 && red[1] === 0, 'image32: 32×32×3, colour preserved');
const imgs = []; for (let n = 0; n < 40; n++) { const ph = Math.random() * 6, fq = 3 + Math.random() * 10, ang = Math.random() * 3;
  imgs.push(M.SenseFeatures.image32(draw((x, y) => { const v = 127 + 120 * Math.sin(fq * (x * Math.cos(ang) + y * Math.sin(ang)) * 6 + ph); return [v, v * 0.8, 255 - v]; }))); }
const filt = M.SenseFeatures.learnEyeFilters(imgs, 16, 4000, 8);
ok(filt && filt.centroids.length === 16 * 48 && filt.centroids.every(Number.isFinite), 'learnEyeFilters: 16 finite filters from unlabeled images');
const tFlat = M.SenseFeatures.imageTokens(red, filt), tStripe = M.SenseFeatures.imageTokens(imgs[0], filt);
const tex = t => { let s = 0; for (let k = 0; k < t.T; k++) for (let i = 48; i < t.Fp; i++) s += t.seq[k * t.Fp + i]; return s; };
ok(tFlat.T === 16 && tFlat.Fp === M.SENSE_SPEC.vision.Fp && tFlat.pooled.length === M.SENSE_SPEC.vision.Fq && tex(tStripe) > tex(tFlat), `imageTokens: 16 tokens; striped image excites the learned texture filters more than a flat one (${tex(tStripe).toFixed(1)} vs ${tex(tFlat).toFixed(1)})`);

// ── 3. ears: the stream keeps ORDER ───────────────────────────────────
const sr = 16000, tone = (f, n) => Float64Array.from({ length: n }, (_, i) => 0.5 * Math.sin(2 * Math.PI * f * i / sr));
const cat = (a, b) => { const o = new Float64Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; };
const lohi = M.SenseFeatures.audioTokens(cat(tone(300, 8000), tone(3000, 8000)), sr), hilo = M.SenseFeatures.audioTokens(cat(tone(3000, 8000), tone(300, 8000)), sr);
const firstTok = (f) => Array.from(f.seq.slice(0, M.SENSE_SPEC.audio.Fp)); const peak = a => a.slice(0, 40).indexOf(Math.max(...a.slice(0, 40)));
let pd = 0; for (let i = 0; i < lohi.pooled.length; i++) pd = Math.max(pd, Math.abs(lohi.pooled[i] - hilo.pooled[i]));
ok(lohi.T === 24 && peak(firstTok(lohi)) < peak(firstTok(hilo)), `audioTokens: 1 s → ${lohi.T} tokens; "low→high" starts low, "high→low" starts high (bands ${peak(firstTok(lohi))} vs ${peak(firstTok(hilo))})`);
console.log(`   (their summaries now differ by up to ${pd.toFixed(2)} too: the context keeps coarse order)`);
const resamp = M.SenseFeatures.audioTokens(Float64Array.from({ length: 44100 }, (_, i) => 0.5 * Math.sin(2 * Math.PI * 440 * i / 44100)), 44100);
ok(resamp.T === 24 && resamp.seq.every(Number.isFinite), 'audioTokens: 44.1 kHz input resampled to the same 24 tokens per second');

// ── 4. gradient check of the paired path ──────────────────────────────
const cfg = { vocab: 20, dim: 12, layers: 2 };
const m = new M.LanguageModel(cfg);
m._isFirstToken = false;
for (const b of m.blocks) { const H = []; for (let k = 0; k < 64; k++) H.push(k); for (let k = 0; k < 64; k++) H.push((k * 7) % 64); b.rosaChannel.history = H; b.rosaEmbed.scale.fill(1.0); }
const spec = { Fp: 7, Fq: 5, Tmax: 3, pos: true };
const ad = new M.SenseAdapter('vision', spec, cfg.dim);
const mk = () => ({ T: 3, Fp: 7, seq: Float64Array.from({ length: 21 }, () => Math.random() * 2 - 1), pooled: Float64Array.from({ length: 5 }, () => Math.random() * 2 - 1) });
for (let n = 0; n < 30; n++) ad.observe(mk());
for (let i = 0; i < ad.C.length; i++) ad.C[i] = Math.sin(i * 0.37) * 0.2;      // non-zero so the context path is exercised
const feats = mk(), caption = [4, 11, 2, 17];
const tr = new M.Trainer(m, null);
const R0 = tr._snapshotRosa();
const ce = (lg, t) => { let mx = -1e9; for (const v of lg) mx = Math.max(mx, v); let s = 0; for (const v of lg) s += Math.exp(v - mx); return -(lg[t] - mx - Math.log(s)); };
const loss = () => { tr._restoreRosa(R0); const pre = ad.prefix(feats), cv = ad.context(feats); let st = m.getEmptyState(), r, L = 0;
  for (const v of pre) { r = m.forwardVec(v, st); st = r.state; }
  for (let t = 0; t < caption.length; t++) { L += ce(r.logits, caption[t]); if (t < caption.length - 1) { r = m.forward(caption[t], st, cv); st = r.state; } } return L; };
m.zeroGrad(); m.embGrad.fill(0); ad.zeroGrad();
tr._restoreRosa(R0); const obs = ad.observe; ad.observe = () => {};        // freeze the running stats so analytic and numeric see the same x̂
const Lan = tr.trainPaired(ad, feats, caption); ad.observe = obs;
const Lfd = loss() / caption.length;
ok(Math.abs(Lan - Lfd) < 1e-4, `trainPaired loss ${Lan.toFixed(6)} vs direct ${Lfd.toFixed(6)} (trainer adds 1e-8 inside its log)`);
const eps = 1e-3;
function check(label, W, G) {
  let dot = 0, na = 0, nn = 0;
  for (let i = 0; i < W.length; i++) { const o = W[i]; W[i] = o + eps; const a = loss(); W[i] = o - eps; const b = loss(); W[i] = o; const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num; }
  if (nn < 1e-14) { console.log(`${label.padEnd(22)} (unused)  FAIL`); fails++; return; }
  const cos = dot / Math.sqrt(na * nn), ratio = Math.sqrt(na / nn), good = cos > 0.99 && Math.abs(ratio - 1) < 0.05; if (!good) fails++;
  console.log(`${label.padEnd(22)} cos=${cos.toFixed(6)} ratio=${ratio.toFixed(6)} ${good ? 'ok' : 'FAIL'}`);
}
check('adapter.W', ad.W, ad.gW); check('adapter.b', ad.b, ad.gb); check('adapter.P (positions)', ad.P, ad.gP); check('adapter.C (context)', ad.C, ad.gC);
check('embedding', m.embedding, m.embGrad); check('block0.W_o', m.blocks[0].W_o.weight, m.blocks[0].W_o.grad); check('head', m.head.weight, m.head.grad);
console.log(fails ? `${fails} FAILED` : 'ALL OK — senses v3 path is exact');
process.exit(fails ? 1 : 0);
