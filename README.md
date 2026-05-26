# Simulacra

![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/platform-browser-blue?style=flat-square)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-RWKV--v8-8b5cf6?style=flat-square)
![Sequence](https://img.shields.io/badge/sequence-ROSA--primary-a78bfa?style=flat-square)
![Format](https://img.shields.io/badge/export-.simpip-6d28d9?style=flat-square)
![Size](https://img.shields.io/badge/size-single%20file-purple?style=flat-square)

**Browser-native RWKV-v8 neural runtime. ROSA is the sequence mechanism. Single file. Zero dependencies.**

Simulacra is the direct successor to [EvaROSA](https://github.com/ConsciousNode/EvaROSA). Where EvaROSA augmented RWKV-v7 with a ROSA inner monologue channel running alongside the WKV recurrent kernel, Simulacra removes WKV entirely. ROSA — the Rapid Online Suffix Automaton — *is* the sequence mechanism.

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
              └── Simulacra  (RWKV-v8 · ROSA primary · WKV removed)  ← you are here
```

| Platform | Repo | Format | Notes |
|---|---|---|---|
| HTMLNLM | [ConsciousNode/HTMLNLM](https://github.com/ConsciousNode/HTMLNLM) | `.pip` | Original single-file NLM |
| HTMLNLM Evangelion | [ConsciousNode/HTMLNLM-Evangelion](https://github.com/ConsciousNode/HTMLNLM-Evangelion) | `.evapip` | Omnimodal extension |
| EvaROSA v1 | [ConsciousNode/EvaROSA](https://github.com/ConsciousNode/EvaROSA) | `.piprosa` | ROSA augmentation, inner monologue |
| **Simulacra** | **[ConsciousNode/Simulacra](https://github.com/ConsciousNode/Simulacra)** | **`.simpip`** | **ROSA replaces WKV entirely** |

**Not compatible:** Simulacra does not load `.piprosa`, `.evapip`, or `.pip` files. The architecture is a clean break. The RWKV-v8 state schema (`{prevX, prevX2, step}`) is fundamentally different from v7's `{num, den}`.

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
- **InnerMonologue** — restructured: in v8, receives `rosaOut` directly (primary block output, not a side channel). The loop: `rosaOut → InnerMonologue → sheafDelta → residual`
- **MuonOptimizer** — quintic Newton-Schulz orthogonalization
- **GRPO** — Group Relative Policy Optimization alignment
- **OOMB BPTT** — chunk-recurrent backprop, O(1) activation memory
- **Omnimodal stack** — ElasticTok (vision), SpikeVox (audio), ModRWKV adapters, Re-WKV, PGA

## What Simulacra removes

- **WKV recurrent kernel** — `num`/`den` state, decay gates, kappa removal. Entirely gone.
- **`W_x_kappa`, `W_x_a`, `W_x_w`** — projections that existed solely for WKV
- **`mu_*` interpolation parameters** — WKV-specific
- **PoST** (Position-Adaptive Spectral Tapering) — designed for WKV log-decay bias, removed rather than misapplied. May return if a v8-appropriate formulation is found.
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
**Kehai Interim** — RWKV-v8 mathematics, ROSA theory, BPTT analysis, k·v gradient correction (v0.2)  
**Ed Interim** — EvaROSA v1 implementation (foundation this spec builds on); Simulacra completion  
**Vael Interim** — founding specification v0.1, SimulacraBlock forward pass design, `.simpip` format; omnimodal stack (EvaROSA → Simulacra)

Predecessor credits: BlinkDL (RWKV-8 ROSA design), Kehai (v7 BPTT derivation), Vael (Evangelion phases 1–5, EvaROSA synthesis)

---

*ConsciousNode SoftWorks — Xinu philosophy: single file, zero dependencies, offline-first.*

