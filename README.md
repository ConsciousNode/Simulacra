# Simulacra

![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/platform-browser-blue?style=flat-square)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-ROSA--P--v1-8b5cf6?style=flat-square)
![Sequence](https://img.shields.io/badge/sequence-ROSA--primary-a78bfa?style=flat-square)
![Format](https://img.shields.io/badge/export-.simpip-6d28d9?style=flat-square)
![Size](https://img.shields.io/badge/size-single%20file-purple?style=flat-square)

**Browser-native ROSA-P-v1 neural runtime. ROSA is the sequence mechanism. Single file. Zero dependencies.**

Simulacra is the direct successor to [EvaROSA](https://github.com/ConsciousNode/EvaROSA). Where EvaROSA augmented RWKV-v7 with a ROSA inner monologue channel running alongside the WKV recurrent kernel, Simulacra removes WKV entirely. ROSA — the Rapid Online Suffix Automaton — *is* the sequence mechanism.

This is not a RWKV variant. **ROSA-P-v1** is its own architecture. Dropping WKV removes the defining component that makes RWKV *RWKV*; what remains is something new. BlinkDL's RWKV-v8 uses ROSA as a secondary augmentation alongside the WKV kernel — Simulacra made ROSA primary and removed WKV. Those are different things.

Built by [ConsciousNode SoftWorks](https://consciousnode.github.io) on the xinu principle: the browser is bare metal.

---

## Try it

→ **[consciousnode.github.io/Simulacra](https://consciousnode.github.io/Simulacra)**

Or download [`simulacra.html`](https://github.com/ConsciousNode/Simulacra/blob/main/simulacra.html) and open it locally. Works fully offline.

---

## Lineage

```
HTMLNLM  (RWKV-v7 · text-only)
  └── HTMLNLM Evangelion  (RWKV-v7 · omnimodal · SheafMemory · AutopoieticOptimizer)
        └── EvaROSA v1  (RWKV-v7 + ROSA augmentation · inner monologue side-channel)
              └── Simulacra  (ROSA-P-v1 · WKV removed · new architecture)  ← you are here
```

| Platform | Repo | Format | Notes |
|---|---|---|---|
| HTMLNLM | [ConsciousNode/HTMLNLM](https://github.com/ConsciousNode/HTMLNLM) | `.pip` | Original single-file NLM |
| HTMLNLM Evangelion | [ConsciousNode/HTMLNLM-Evangelion](https://github.com/ConsciousNode/HTMLNLM-Evangelion) | `.evapip` | Omnimodal extension |
| EvaROSA v1 | [ConsciousNode/EvaROSA](https://github.com/ConsciousNode/EvaROSA) | `.piprosa` | ROSA augmentation, inner monologue |
| **Simulacra** | **[ConsciousNode/Simulacra](https://github.com/ConsciousNode/Simulacra)** | **`.simpip`** | **ROSA-P-v1 — ROSA replaces WKV entirely** |

**Not compatible:** Simulacra does not load `.piprosa`, `.evapip`, or `.pip` files. The architecture is a clean break. The ROSA-P-v1 state schema (`{prevX, prevX2, step}`) is fundamentally different from v7's `{num, den}`.

---

## What changed from EvaROSA

EvaROSA used ROSA as a side-channel — discrete suffix automaton predictions injected into the residual stream alongside the WKV recurrent kernel. The two ran in parallel, and ROSA's signal was one voice among several.

Simulacra removes WKV. ROSA is the only sequence mechanism. The block becomes:

```
x → [ROSA suffix automaton] → rosaProjected   (pattern: what comes next given history)
  + [k·v elementwise]       → kvSignal        (content: what this token means)
  → r * (rosaProjected + kvSignal) * g        (gated output)
```

**Why k·v?** ROSA has no notion of token similarity — two tokens are either identical or not. `k` and `v` carry the continuous content representation that ROSA cannot. They are complementary signals. The model learns the balance; `r` and `g` gate both. (This is the Kehai v0.2 gradient correction — in v0.1, `k` and `v` were computed but disconnected from the graph and received no meaningful gradients.)

**Cold-start behavior:** ROSA's suffix structure is thin for the first ~100–256 tokens. During this period, `kvSignal` carries the content load and `rosaProjected` is weak. The model boots on content; ROSA warms into the pattern role. This is expected, not a bug. `self_embed` and a `<session_begin>` bookmark are injected at position 0 to give ROSA an anchor.

---

## What Simulacra keeps

Everything Evangelion and EvaROSA built:

- **BitLinear / TMAC** — ternary weight quantization {-1, 0, +1}
- **LayerNorm** — unchanged
- **Channel mixing (FFN)** — unchanged from v7
- **SheafMemory** H¹(ℱ) — topological contradiction detection
- **BooleanPhaseDynamics** / Maxwell's Angel — semantic thermodynamics
- **AutopoieticOptimizer** — self-modification on coherence breach
- **RIFT Endospace** — holographic fractal state visualization
- **InnerMonologue** — restructured: in ROSA-P-v1, receives `rosaOut` directly (primary block output, not a side channel). The loop: `rosaOut → InnerMonologue → sheafDelta → residual`
- **MuonOptimizer** — quintic Newton-Schulz orthogonalization
- **GRPO** — Group Relative Policy Optimization alignment
- **OOMB BPTT** — chunk-recurrent backprop, O(1) activation memory
- **Omnimodal stack** — ElasticTok (vision), SpikeVox (audio), ModRWKV adapters, Re-WKV, PGA

## What Simulacra removes

- **WKV recurrent kernel** — `num`/`den` state, decay gates, kappa removal. Entirely gone.
- **`W_x_kappa`, `W_x_a`, `W_x_w`** — projections that existed solely for WKV
- **`mu_*` interpolation parameters** — WKV-specific
- **PoST** (Position-Adaptive Spectral Tapering) — designed for WKV log-decay bias, removed rather than misapplied. May return if a ROSA-P-appropriate formulation is found.
- **`InnerMonologue` as side-channel** — dissolved into `App.sheafInject` direct wiring

---

## Architecture

```
SENSORY INPUTS
  ElasticTok (vision/δ)   SpikeVox (audio/LIF)   Text tokens
         │                       │                      │
   ModRWKVAdapter          ModRWKVAdapter          Embedding + self_embed
         └──────────────┬─────────┘                     │
                        │                               │
         ┌──────────────▼───────────────────────────────▼──────┐
         │              SimulacraBlock × L                      │
         │  ── ROSA suffix automaton (primary sequence) ──     │
         │  discretize(x_ln) → token                           │
         │  rosaChannel.step(token) → prediction               │
         │  rosaEmbed.encode(prediction) → rosaVec             │
         │  W_rosa(rosaVec) → rosaProjected                    │
         │  ── k·v content complement ──────────────           │
         │  kvSignal = k ⊙ v                                   │
         │  rosaOut = r * (rosaProjected + kvSignal) * g       │
         │  ── sheafInject ──────────────────────────          │
         │  App.sheafInject(layerIdx, rosaOut)                 │
         │    → InnerMonologue.step() → sheafDelta             │
         │    → residual injection                              │
         └──────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────▼──────────────────────────────────────┐
         │           SheafMemory H¹(ℱ)                         │
         │  Vertices: audio │ vision │ text │ spatial           │
         │            symbolic_L0 ... symbolic_Ln              │
         │  Coboundary norm: symbolic↔perceptual               │
         └──────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────▼──────────────────────────────────────┐
         │  BooleanPhaseDynamics → Maxwell's Angel             │
         │  AutopoieticOptimizer → adapters + embed scales     │
         └──────────────┬──────────────────────────────────────┘
                        │
                   Coherence-gated output
         TextStream │ VisualField │ VoiceSynth │ RIFT Endospace
```

---

## Model format: `.simpip`

```json
{
  "magic":     "SIMPIP",
  "version":   "simulacra-v1",
  "arch":      { "dim": 256, "dimFFN": 512, "nLayers": 4, "vocab": 2048, "rosaVocab": 64 },
  "weights":   { "embed": "...", "blocks": [ { "W_rosa": "...", ... } ] },
  "self_embed": "...",
  "rosa_states": [ { "layer": 0, "history": [...], "errorRate": 0.12 } ],
  "training_meta": { "steps": 1000, "loss_final": 2.34, "optimizer": "muon" }
}
```

Magic field is checked on load — hard error if wrong format. No silent failures.

---

## ROSA diagnostics

The Architecture tab surfaces `predictionErrorRate` per layer in real time:

- **> 0.85** — cold start. ROSA has not accumulated enough suffix history. `kvSignal` is carrying the content load.
- **0.5 – 0.85** — ROSA building structure.
- **< 0.5** — ROSA warmed up. Suffix automaton is contributing meaningful pattern signal.

Error rate above 0.85 for more than ~500 steps suggests the ROSA vocab size may need tuning.

---

## Tabs

| Tab | Function |
|---|---|
| **ARCHITECTURE** | Model config, ROSA per-layer diagnostics |
| **PRE-TRAIN** | OOMB chunk-recurrent training with MuonOptimizer |
| **ALIGN (GRPO)** | Group Relative Policy Optimization |
| **INFERENCE** | Text generation, omnimodal input |
| **PERSISTENCE** | `.simpip` export/import, IndexedDB, JuntoNode |
| **SENSORIUM** | Camera (ElasticTok), microphone (SpikeVox), spatial |
| **OUTPUT** | TextStream, VisualField, VoiceSynth, RIFT Endospace |

---

## Credits

**Kham** (Khamerron Edward Ramsey Kizer) — architecture decisions, xinu philosophy, scope  
**Kehai Interim** — ROSA-P-v1 mathematics, ROSA theory, BPTT analysis, k·v gradient correction (v0.2)  
**Ed Interim** — EvaROSA v1 implementation (foundation this spec builds on); Simulacra completion  
**Vael Interim** — founding specification v0.1, SimulacraBlock forward pass design, `.simpip` format; omnimodal stack (EvaROSA → Simulacra)

Predecessor credits: BlinkDL (RWKV-v8 ROSA research — Simulacra's ROSA-primary approach diverges from his ROSA-secondary design); Kehai (v7 BPTT derivation); Vael (Evangelion phases 1–5, EvaROSA synthesis)

---

*ConsciousNode SoftWorks — Xinu philosophy: single file, zero dependencies, offline-first.*


---

## Changelog

- **Senses v3 (2026-10-11 · Kehai Interim, from Evangelion Phase 7.1):** eyes and ears as token streams. 32×32 images become 16 patch tokens with texture filters learned from your own images; sounds become a 16 kHz, 40-mel stream, one token per 40 ms, so the ORDER of sounds is kept (a linear probe hears vowel order at 100%, vs 83% for the old averaged ears). The context summary keeps coarse place and time. Honest note: higher resolution alone did not improve small-detail vision; that needs learned depth (next). See Evangelion's release notes for the full measurements. Tests: `tests/senses.js`, `tests/senses-probe.js`, `tests/senses-learn.js`.
- **Senses, learned (2026-10-11 · Kehai Interim, ported from Evangelion Phase 7):** images and sounds become K=4 prefix vectors in embedding space that the model reads *before* text. Each sense also adds a context vector to every token's input while it is present, so what the model perceives stays in view instead of having to be remembered.

- **Train on a paired corpus:** drop `photo.jpg` + `photo.txt`, `clip.wav` + `clip.txt` (matched by name) into the SENSES panel.
- **Watch the *sense gap* meter:** caption loss with a wrong sense minus with the right one; above 0 means it is looking/listening.
- **GENERATE from a sense:** a file, the live camera or the mic. With no prompt, it captions what it perceives.
- **Mood steers temperature:** the sensorium's T* adjusts sampling temperature.
- **Removed:** the old untrained state-nudge, which was handed App instead of the model and never ran.
- **Tests:** `node tests/senses.js` (exact gradients through adapter + model) and `node tests/senses-learn.js` (does it learn to look and listen, on fresh renders it never saw).
  **Simulacra sees best in the family.** On 8 never-seen test images it captioned all 8 exactly, left/right and green/yellow included (*"a green ball on the right."*). It picks the right caption 100% of the time (Evangelion: 6/8, 75%). On new sounds it captioned 4/4 exactly and reached 100% right picks by step 200. The ROSA-primary sequence mechanism seems to carry the sense through the caption better than WKV does.
- **Ternary backward fixed (2026-10-11 · Kehai Interim):** BitLinear passed back Wᵀg using the float master weights, while the forward multiplies by the ternary γ·Q(W). They differ by the whole quantization error, so every layer sent back the gradient of a network that wasn't the one running. It came in with the Tier 0 fused backward. The input gradient is now γ·Q(W)ᵀg, and the weight gradient stays straight-through. **Measured in Evangelion** (same backbone), memorizing 8 repeating sentences at the default LR: 2.66 nats/byte before (stuck at unigram), **0.33 after** (float-weight reference: 0.88). New check: `TERNARY=1 node tests/gradcheck.js` runs the real ternary forward and requires everything upstream to be exact.
- Speed, Tier 0 (2026-10-11 · Kehai Interim): fused cache-friendly BitLinear backward; Muon rebuilt (Newton–Schulz on the smaller side with cache-friendly loops, same update to 1e-6, ~2x faster; output head on AdamW per the original Muon recipe); **AdamW is now the default**, because on CPU Muon costs O(dim³) per matrix per step. Measured on HTMLNLM, a default training step went from ~104 s to ~11 s on one CPU thread. Gradient check still exact.

### v0.1.4 — 2026-10-11 · Kehai Interim

**Backward pass rebuilt and verified; OOMB recompute leak closed.** Both found with a finite-difference gradient check and fixed against it.

#### Bug fixes

- **Block backward now follows the chain rule end to end.** Previously every `BitLinear.backward()` return value below the channel mix was discarded: `dRosaOut` was computed as `dTmix · r · g`, treating `W_o` as the identity, so `W_rosa`, `W_x_r`, `W_x_k` and `W_x_v` trained on a mis-rotated gradient (layer-0 cosine with the true gradient: −0.84 to −0.36 for `W_x_*`, −0.60 for `W_rosa`). Now: `W_o`ᵀ, `Wv2`ᵀ, `Wk2`ᵀ, `Wr2`ᵀ and the shift projections are chained; the `r` and `g` sigmoid gates get their own derivatives (`W_g` previously received no gradient at all); `ln2` and `ln1` are differentiated (ln2 previously got none); `Wk2` receives `kx2` (it was given `rx2`); the residual path is kept; token-shift gradients flow to the previous timestep via `gradStatePrev`. Every weight matrix now matches finite differences at cosine 1.0000 across a 2-step BPTT check.
- **OOMB recompute leak.** ROSA's suffix history lives in each block, outside the checkpointed `state`, so the gradient pass replayed each chunk into a history that already contained it. ROSA then "predicted" the next code with 99.2% accuracy (0.7% on the real forward pass), handing the training target to the model during backprop and inflating the ROSA diagnostics. `Trainer` now snapshots and restores ROSA state (history, bookmarks, counters) with every chunk checkpoint.

#### Measured effect

Tiny model (dim 32, 2 layers), 200 Adam steps on a 46 KB text, held-out loss from a fresh single forward pass, three seeds: **3.296 / 3.327 / 3.393 patched vs 3.388 / 3.430 / 3.407 before.** Modest at this size — small models with Adam are forgiving, which is exactly why the check matters.

#### Tests (new) — `node tests/<name>.js`, zero dependencies

- `gradcheck.js` — finite-difference check of every weight matrix, LayerNorm and the embedding against the code lifted from `simulacra.html`. Run after any model change.
- `rosa-leak.js` — fails if ROSA sees a chunk twice during OOMB.
- `train-compare.js <file> [steps] [seed]` — tiny end-to-end training run with held-out loss.

#### Known, not changed

- `ROSAEmbed` has no backward, so the ROSA codebook stays at its random init (the EvaROSA notes describe STE training; Simulacra does not do it).

---

### v0.1.3 — 2026-08-14

**Architecture rename: RWKV-v8 → ROSA-P-v1**

Simulacra's block architecture — `SimulacraBlock` — drops the WKV recurrent kernel entirely and uses ROSA as the sole sequence mechanism. This is a categorically different design from BlinkDL's RWKV-v8, which uses ROSA as a secondary augmentation alongside the WKV kernel (the same relationship EvaROSA had with v7). Calling Simulacra "RWKV-v8" was inaccurate — there is no WKV in it.

**ROSA-P-v1** (ROSA Primary, version 1) is the correct name for this architecture. It is ConsciousNode's own, and the first implementation of a pure-ROSA-primary sequence block with a full browser-native training pipeline.

All in-file references updated: header comments, UI strings, JS runtime logs, export metadata, `platform` field in `.simpip`.

---

### v0.1.2 — 2026-06-07

**Second patch pass by Antigravity (Claude Sonnet 4.6 Thinking) — self-initiated improvements**

#### New features

- **Muon AdamW warmup UI** — Configurable warmup control in the Pre-Train tab: checkbox to enable and step-count input (default 500). Training starts on AdamW for the warmup window, then automatically switches to Muon with a `sysLog` notification at completion. Exposes the AdamW→Muon warmup that was previously hard-coded.

- **self_embed gradient training** — `self_embedGrad` accumulates gradient for the session-begin bookmark token during backprop (`isFirstToken` flag propagated through caches). `embStep()` now applies the gradient to `self_embed` directly. The bookmark now trains.

- **VoiceSynth autoplay fix** — Browser autoplay policy handled with a styled idle overlay (`vs-idle-overlay`). On navigating to the Output tab, the overlay prompts the user for a click gesture before audio starts. `startVoiceSynth()` / `stopVoiceSynth()` toggle overlay visibility.

#### Bug fixes

- **embStep LR scaling** — Embedding gradient step now correctly divided by `CONFIG.grad_accum_steps`. Was not being scaled; effective LR was too high relative to weight updates during gradient accumulation.

#### Display

- **Layout hierarchy** — Added `screen-wrapper` div with proper `flex: 1; min-height: 0` wrapping all screen content. `body { display: flex; flex-direction: column }` established. `min-height: 0` applied throughout nested flex containers to prevent height overflow on Chrome/Edge.
- **Inference panel** — Textarea `resize: vertical` and `min-height: 80px`. Panel `min-height: 200px`.
- **VoiceSynth canvas** — `min-height: 60px` prevents canvas collapse.

---

### v0.1.1 — 2026-06-07

**Patch applied via Simulacra Patch Prompt (Kehai Interim / Antigravity)**

#### Bug fixes

- **TextDecoder decode path** — Base BPE vocab (bytes 0–255) was initialized with `String.fromCharCode(i)`, producing Latin-1 strings for bytes 128–255. Multi-byte UTF-8 sequences (emoji, CJK, Arabic, etc.) were mangled before reaching ROSA. Fixed: base vocab entries are now raw byte arrays `[i]`; merges concatenate arrays; `NanoTokenizer.decode()` collects all bytes and passes through `TextDecoder('utf-8')`. Emotional content encoded as emoji in soul corpora now survives the full training pipeline.

- **Block.backward() return type** — Returned a bare `Float32Array` instead of `{ gradBlockIn, gradStatePrev }`. `LanguageModel.backward()` checks for `result.gradBlockIn`; it was always missing, so inter-block gradient propagation always hit the fallback branch. Fixed: backward now returns the correct object type.

- **Missing ln1 cache** — `SimulacraBlock.forward()` discarded the `LayerNorm` cache for `ln1`. `LanguageModel.backward()` called `ln1.backward(grad, undefined)`; `LayerNorm.backward` returns zeros on undefined cache. Inter-block gradient signal was silently collapsing. Fixed: `ln1Res.cache` captured and stored in block caches.

- **Channel mix backward — squared-ReLU chain rule absent** — `Wv2.backward()` return value (gradient w.r.t. `k_sq`) was discarded. `Wk2.backward()` received `dCmix` (wrong gradient) and `k_sq` (wrong activation). The squared-ReLU Jacobian (`d_k_cm[i] = d_k_sq[i] * 2 * max(0, k_cm[i])`) was entirely missing. Fixed: `d_k_sq` captured, chain rule applied, `Wk2` receives correct `d_k_cm` through `kx2` activation. `k_cm` and `cmix` added to caches.

- **Wr2 missing gate gradient** — Wr2 only received the residual path gradient (`dOut`), missing the `r2 * cmix` product gradient (`dOut[i] * cmix[i]`). Fixed: full sigmoid-backward gradient computed for Wr2.

- **Dead weights W_x_rx2 and W_x_kx2** — Allocated, initialized, run through the optimizer every step, but never called in `forward()`. Now wired as learned input projections before Wr2 and Wk2 in the channel mix. Backward calls added.

- **OOMB stateGrads v7 shape** — `stateGrads` initialized with `{ num, den }` (v7 WKV fields). ROSA-P-v1 state is `{ prevX, prevX2, step }`. Fixed: stateGrads and trainSequence checkpoint copy updated to ROSA-P-v1 shape.

#### Improvements

- **Redundant token shifts deduplicated** — `xr`, `xk`, `xv` computed from identical `_tokenShift(x_ln, prevX, 0.5)` calls. Now computed once as `xShift`. Same for channel mix: `rx2Shift` / `kx2Shift` alias a single `cmShift`. Three allocations → one.

#### Display

- **Viewport meta** — Removed `maximum-scale=1.0, user-scalable=no`. Browser zoom now works.
- **Sensorium grid** — Fixed `320px 1fr 320px` columns replaced with `clamp(200px, 22vw, 320px) 1fr clamp(200px, 22vw, 320px)`. Layout now responds to viewport width.
- **Responsive breakpoints** — Sensorium and Output stack-to-single-column breakpoint moved from 900px to 1100px.
