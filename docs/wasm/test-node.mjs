import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const useWebGpu = args.includes("--webgpu");
const modelPath = args.find(a => a.endsWith(".cellm"));
const tokPath = args.find(a => a.endsWith(".json"));
if (!modelPath || !tokPath) { console.error("Usage: node test-node.mjs <model.cellm> <tokenizer.json>"); process.exit(1); }

// Init WASM
const wasmUrl = resolve(__dirname, "pkg/cellm_wasm.js");
const wasmModule = await import(wasmUrl);
const wasmBin = readFileSync(resolve(__dirname, "pkg/cellm_wasm_bg.wasm"));
await wasmModule.default(new Uint8Array(wasmBin));
const wasm = wasmModule;

// Load model + tokenizer
const modelBytes = readFileSync(resolve(modelPath));
const tokenizerJson = readFileSync(resolve(tokPath), "utf-8");
console.log(`Model: ${(modelBytes.length/1e6).toFixed(1)}MB, Tokens: ${(tokenizerJson.length/1e6).toFixed(1)}MB`);

// Create engine
const config = JSON.stringify({tokens_per_block:16,total_blocks:128,top_k:40,temperature:0,repeat_penalty:1.05,repeat_window:64,seed:1,scheduling_policy:"Fair"});
const engine = new wasm.CellmEngine(new Uint8Array(modelBytes), config);
if (useWebGpu) { const ok = await engine.try_init_webgpu(); console.log(`WebGPU: ${ok?"YES":"no"}`); }
engine.set_tokenizer(tokenizerJson);

// Tokenize
const prompt = "What is the capital of France?";
const inputIds = engine.tokenize(prompt);
console.log(`Prompt: ${inputIds.length} tokens`);

// Decode loop (step_decode works, generate doesn't on WASM)
const sid = engine.create_session();
let tok = engine.submit_tokens(sid, inputIds);
let count = 0;
const t0 = performance.now();
while (count < 50) {
  count++;
  process.stdout.write(engine.decode(new Uint32Array([tok])));
  if (engine.is_stop_token(tok)) break;
  const r = engine.step_decode();
  if (!r) break;
  tok = r[1]; // [session_id, token_id]
}
const ms = (performance.now() - t0).toFixed(1);
console.log(`\n\n${count} tokens in ${ms}ms (${(count/(ms/1000)).toFixed(1)} tok/s)`);
console.log(`Total: ${engine.total_tokens_generated()}, Free: ${engine.num_free_blocks()}, GPU: ${engine.has_gpu()}`);
