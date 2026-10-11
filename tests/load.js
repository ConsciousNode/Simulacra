// Lifts the model classes straight out of simulacra.html so tests always run
// against the shipped code. No dependencies: `node tests/gradcheck.js`.
const fs = require('fs'), path = require('path'), vm = require('vm');
module.exports = function load(opts = {}) {
  const src = fs.readFileSync(path.join(__dirname, '..', 'simulacra.html'), 'utf8');
  const cut = (startMarker, endMarker) => {
    const a = src.indexOf(startMarker), b = src.indexOf(endMarker, a);
    if (a < 0 || b < 0) throw new Error('marker not found: ' + startMarker + ' … ' + endMarker);
    return src.slice(a, b);
  };
  const code = [
    cut('const CONFIG = {', '/* ╔'),                         // CONFIG + helpers
    cut('class TMAC {', 'class LittleBit2'),                   // TMAC … LanguageModel (+ helpers between)
    cut('class Trainer {', '/* ╔'),                           // Trainer (OOMB chunk-recurrent BPTT)
    cut('function ROSA(x) {', 'const App = {'),                // ROSA, ROSAChannel, ROSAEmbed, InnerMonologue
    'this.M = { CONFIG, TMAC, BitLinear, LayerNorm, AdamOptimizer, MuonOptimizer, SimulacraBlock, LanguageModel, Trainer, ROSA, ROSAChannel, ROSAEmbed, SenseAdapter, SenseFeatures, SENSE_SPEC, symEig };',
  ].join('\n');
  const noop = () => 0;
  const ctx = { setInterval: noop, clearInterval: noop, requestAnimationFrame: noop, Math, Float32Array: opts.f64 ? Float64Array : Float32Array, Float64Array, Int32Array, Uint8Array, Uint32Array, Int8Array, Array, Object, Promise, setTimeout, console, document: undefined };
  vm.createContext(ctx);
  vm.runInContext(code, ctx);
  return ctx;
};
