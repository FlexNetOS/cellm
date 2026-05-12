// Node.js test harness for cellm WASM inference.
// Usage:
//   node --experimental-wasm-simd test-node.mjs [--webgpu] <model.cellm> <tokenizer.json>
//
// Requires Node >= 18 with --experimental-wasm-simd for SIMD acceleration.
// For WebGPU: Node >= 22 with --experimental-webgpu

import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Detect CLI args
const args = process.argv.slice(2);
const useWebGpu = args.includes("--webgpu");
const modelPath = args.find((a) => a.endsWith(".cellm"));
const tokPath = args.find((a) => a.endsWith(".json"));

if (!modelPath || !tokPath) {
  console.error(`
Usage:
  node --experimental-wasm-simd test-node.mjs [--webgpu] <model.cellm> <tokenizer.json>

Examples:
  node --experimental-wasm-simd test-node.mjs \\
    ../../models/nanowhale-100m.cellm \\
    ../../models/nanowhale-100m/tokenizer.json

  (With WebGPU):
  node --experimental-wasm-simd --experimental-webgpu test-node.mjs --webgpu \\
    ../../models/nanowhale-100m.cellm \\
    ../../models/nanowhale-100m/tokenizer.json
`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Load WASM module
// ---------------------------------------------------------------------------

const wasmPath = resolve(__dirname, "pkg/cellm_wasm.js");
console.log(`Loading WASM from: ${wasmPath}`);

// wasm-bindgen's generated JS uses Web APIs (fetch, Response) internally.
// We shim them for Node.js before importing.
globalThis.Response = class {
  constructor(body, init) {
    this._body = body;
    this._init = init;
  }
  arrayBuffer() {
    const buf =
      typeof this._body === "string"
        ? Buffer.from(this._body).buffer
        : this._body.buffer || this._body;
    return Promise.resolve(buf);
  }
};

const init = require(wasmPath);
const wasm = await init();

// ---------------------------------------------------------------------------
// Load model + tokenizer
// ---------------------------------------------------------------------------

const modelBytes = readFileSync(resolve(modelPath));
const tokenizerJson = readFileSync(resolve(tokPath), "utf-8");

console.log(`Model:  ${modelPath} (${(modelBytes.length / 1e6).toFixed(1)} MB)`);
console.log(
  `Tokens: ${tokPath} (${(tokenizerJson.length / 1e6).toFixed(1)} MB)`
);

// ---------------------------------------------------------------------------
// Create engine
// ---------------------------------------------------------------------------

const config = JSON.stringify({
  tokens_per_block: 16,
  total_blocks: 128,
  top_k: 40,
  temperature: 0.8,
  repeat_penalty: 1.05,
  repeat_window: 64,
  seed: 42,
  scheduling_policy: "Fair",
});

console.log("\nCreating engine...");
const t0 = performance.now();
const engine = new wasm.CellmEngine(
  new Uint8Array(modelBytes),
  config
);

// Optionally initialize WebGPU
if (useWebGpu) {
  console.log("Initializing WebGPU...");
  const gpuOk = await engine.try_init_webgpu();
  console.log(`WebGPU: ${gpuOk ? "ACTIVE" : "UNAVAILABLE (falling back to CPU)"}`);
}

const t1 = performance.now();
console.log(`Engine created in ${(t1 - t0).toFixed(1)}ms`);
console.log(`EOS token: ${engine.eos_token_id()}`);

// ---------------------------------------------------------------------------
// Set tokenizer
// ---------------------------------------------------------------------------

console.log("\nSetting tokenizer...");
engine.set_tokenizer(tokenizerJson);
console.log("Tokenizer ready");

// ---------------------------------------------------------------------------
// Tokenize prompt
// ---------------------------------------------------------------------------

const prompt = "What is the capital of France?";
console.log(`\nPrompt: "${prompt}"`);

const inputIds = engine.tokenize(prompt);
console.log(`Tokenized: [${inputIds.join(", ")}] (${inputIds.length} tokens)`);

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

const sid = engine.create_session();
console.log(`\nGenerating (max 50 tokens)...\n`);

const tGen0 = performance.now();
const result = engine.generate(sid, inputIds, 50);
const tGen1 = performance.now();

const genMs = (tGen1 - tGen0).toFixed(1);
const tokens = engine.pending_tokens(sid);

console.log(`Generated ${tokens} tokens in ${genMs}ms (${(tokens / (genMs / 1000)).toFixed(1)} tok/s)`);

// Decode output
const allTokens = [];
for (let i = 0; i < result.length; i++) {
  allTokens.push(result.get(i).getAt(1));
}

const output = engine.decode(new Uint32Array(allTokens));
console.log(`\nOutput:\n${output}\n`);

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

console.log("--- Stats ---");
console.log(`Total tokens generated: ${engine.total_tokens_generated()}`);
console.log(`Active sessions: ${engine.num_active_sessions()}`);
console.log(`Free blocks: ${engine.num_free_blocks()}`);
console.log(`GPU active: ${engine.has_gpu()}`);
