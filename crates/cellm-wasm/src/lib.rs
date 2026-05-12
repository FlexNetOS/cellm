#![cfg(target_arch = "wasm32")]

// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! WebAssembly bindings for the cellm LLM inference engine.
//!
//! ## Usage (JavaScript)
//!
//! ```js
//! import init, { CellmEngine } from './cellm_wasm.js';
//!
//! await init();
//!
//! // Fetch your model .cellm file as an ArrayBuffer
//! const resp = await fetch('/models/my-model.cellm');
//! const modelBytes = new Uint8Array(await resp.arrayBuffer());
//!
//! const config = JSON.stringify({
//!   tokens_per_block: 16,
//!   total_blocks: 128,
//!   top_k: 40,
//!   temperature: 0.8,
//!   repeat_penalty: 1.05,
//!   repeat_window: 64,
//!   seed: 1,
//! });
//!
//! const engine = CellmEngine.new(modelBytes, config);
//! const tokenizerJson = '...'; // from tokenizer.json
//! engine.set_tokenizer(tokenizerJson);
//!
//! // Create a session and send tokens
//! let sid = engine.create_session();
//! let nextToken = engine.submit_tokens(sid, [1, 304, 11, 297, ...]);
//!
//! // Step decode
//! while (true) {
//!   let result = engine.step_decode();
//!   if (!result) break;
//!   // result = { sid, token }
//!   if (token === 2) break; // EOS
//! }
//! ```

use std::cell::RefCell;
use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use cellm_sdk::{Engine, EngineConfig, BackendKind, SessionId};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Panic hook
// ---------------------------------------------------------------------------

/// Initialise the WASM module. Must be called once from JavaScript before
/// any other function.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// CellmEngine
// ---------------------------------------------------------------------------

/// A cellm LLM inference engine instance, exposed to JavaScript via wasm-bindgen.
///
/// Owns a model, KV cache, tokenizer, and manages multiple inference sessions.
#[wasm_bindgen]
pub struct CellmEngine {
    engine: Mutex<Engine>,
    tokenizer: RefCell<Option<Tokenizer>>,
}

#[wasm_bindgen]
impl CellmEngine {
    /// Create a new engine from raw model bytes and a JSON config string.
    ///
    /// - `model_bytes`: the complete `.cellm` model file contents as a `Uint8Array`.
    /// - `config_json`: a JSON string matching `EngineConfig`:
    ///   ```json
    ///   {
    ///     "tokens_per_block": 16,
    ///     "total_blocks": 128,
    ///     "top_k": 40,
    ///     "temperature": 0.8,
    ///     "repeat_penalty": 1.05,
    ///     "repeat_window": 64,
    ///     "seed": 1,
    ///     "scheduling_policy": "Fair"
    ///   }
    ///   ```
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: Vec<u8>, config_json: &str) -> Result<CellmEngine, JsValue> {
        let cfg: EngineConfig = deserialize_config(config_json)?;
        let engine = Engine::from_vec(model_bytes, cfg)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::new: {e}")))?;
        Ok(CellmEngine {
            engine: Mutex::new(engine),
            tokenizer: RefCell::new(None),
        })
    }

    /// Set the tokenizer from a JSON string (contents of `tokenizer.json`).
    ///
    /// Must be called before `tokenize()` or `decode()`.
    pub fn set_tokenizer(&self, tokenizer_json: &str) -> Result<(), JsValue> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::set_tokenizer: {e}")))?;
        *self.tokenizer.borrow_mut() = Some(tokenizer);
        Ok(())
    }

    /// Check whether a tokenizer has been set.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.borrow().is_some()
    }

    /// Encode a prompt string into token IDs using the loaded tokenizer.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        let tok = self.tokenizer.borrow();
        let tokenizer = tok
            .as_ref()
            .ok_or_else(|| JsValue::from_str("CellmEngine: tokenizer not set"))?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::tokenize: {e}")))?;
        let mut ids = encoding.get_ids().to_vec();
        if let Some(bos) = self.engine.lock().unwrap().bos_token_id() {
            if ids.first().copied() != Some(bos) {
                ids.insert(0, bos);
            }
        }
        Ok(ids)
    }

    /// Decode a sequence of token IDs back to a string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, JsValue> {
        let tok = self.tokenizer.borrow();
        let tokenizer = tok
            .as_ref()
            .ok_or_else(|| JsValue::from_str("CellmEngine: tokenizer not set"))?;
        tokenizer
            .decode(tokens, false)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::decode: {e}")))
    }

    // -----------------------------------------------------------------------
    // Session management
    // -----------------------------------------------------------------------

    /// Create a new inference session. Returns a session ID.
    pub fn create_session(&self) -> SessionId {
        self.engine.lock().unwrap().create_session()
    }

    /// Submit pre-filled token IDs for a session.
    ///
    /// Returns the next predicted token ID (greedy sampling).
    pub fn submit_tokens(&self, session_id: SessionId, tokens: Vec<u32>) -> Result<u32, JsValue> {
        self.engine
            .lock()
            .unwrap()
            .submit_tokens(session_id, &tokens)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::submit_tokens: {e}")))
    }

    /// Run a single decode step for the next scheduled session.
    ///
    /// Returns `Some([session_id, token])` if a token was produced, or `None`
    /// if no sessions are ready to decode.
    pub fn step_decode(&self) -> Result<Option<js_sys::Array>, JsValue> {
        let result = self
            .engine
            .lock()
            .unwrap()
            .step_decode()
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::step_decode: {e}")))?;
        match result {
            Some((sid, token)) => {
                let arr = js_sys::Array::new();
                arr.push(&JsValue::from_f64(sid as f64));
                arr.push(&JsValue::from_f64(token as f64));
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }

    /// Convenience: submit tokens and run decode loop up to `max_tokens` steps.
    ///
    /// Returns an array of `[session_id, token_id]` pairs.
    pub fn generate(
        &self,
        session_id: SessionId,
        tokens: Vec<u32>,
        max_tokens: u32,
    ) -> Result<js_sys::Array, JsValue> {
        // Submit prefill
        let _next = self.submit_tokens(session_id, tokens)?;

        let results = js_sys::Array::new();

        for _ in 0..max_tokens {
            match self
                .engine
                .lock()
                .unwrap()
                .step_decode()
                .map_err(|e| JsValue::from_str(&format!("CellmEngine::generate: {e}")))?
            {
                Some((sid, token)) => {
                    let pair = js_sys::Array::new();
                    pair.push(&JsValue::from_f64(sid as f64));
                    pair.push(&JsValue::from_f64(token as f64));
                    results.push(&pair);
                }
                None => break,
            }
        }

        Ok(results)
    }

    /// Cancel a session and free its KV cache blocks.
    pub fn cancel_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .cancel_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::cancel_session: {e}")))
    }

    /// Reset a session's decode state while preserving the cached prefill.
    pub fn reset_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .reset_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::reset_session: {e}")))
    }

    /// Suspend a session (move to queued state, free no memory).
    pub fn suspend_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .suspend_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::suspend_session: {e}")))
    }

    /// Resume a previously suspended session.
    pub fn resume_session(&self, session_id: SessionId) -> Result<(), JsValue> {
        self.engine
            .lock()
            .unwrap()
            .resume_session(session_id)
            .map_err(|e| JsValue::from_str(&format!("CellmEngine::resume_session: {e}")))
    }

    /// Number of active (undecoded) tokens currently buffered for a session.
    pub fn pending_tokens(&self, session_id: SessionId) -> u32 {
        self.engine.lock().unwrap().pending_tokens(session_id) as u32
    }

    /// Total tokens generated so far across all sessions.
    pub fn total_tokens_generated(&self) -> f64 {
        self.engine.try_lock().map(|e| e.total_tokens_generated() as f64).unwrap_or(0.0)
    }

    /// Number of active (non-terminated) sessions.
    pub fn num_active_sessions(&self) -> u32 {
        self.engine.try_lock().map(|e| e.num_active_sessions() as u32).unwrap_or(0)
    }

    /// Number of free KV cache blocks remaining.
    pub fn num_free_blocks(&self) -> u32 {
        self.engine.try_lock().map(|e| e.num_free_blocks() as u32).unwrap_or(0)
    }

    /// Model EOS token id, or -1 when the model metadata does not provide one.
    pub fn eos_token_id(&self) -> i32 {
        self.engine
            .lock()
            .unwrap()
            .eos_token_id()
            .map(|id| id as i32)
            .unwrap_or(-1)
    }

    /// Model BOS token id, or -1 when the model metadata does not provide one.
    pub fn bos_token_id(&self) -> i32 {
        self.engine
            .lock()
            .unwrap()
            .bos_token_id()
            .map(|id| id as i32)
            .unwrap_or(-1)
    }

    /// Whether a token is the model's stop token.
    pub fn is_stop_token(&self, token: u32) -> bool {
        self.engine.lock().unwrap().is_stop_token(token)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn deserialize_config(json: &str) -> Result<EngineConfig, JsValue> {
    #[derive(serde::Deserialize)]
    struct Config {
        #[serde(default = "default_tokens_per_block")]
        tokens_per_block: usize,
        #[serde(default = "default_total_blocks")]
        total_blocks: usize,
        #[serde(default = "default_top_k")]
        top_k: usize,
        #[serde(default = "default_temperature")]
        temperature: f64,
        #[serde(default = "default_repeat_penalty")]
        repeat_penalty: f64,
        #[serde(default = "default_repeat_window")]
        repeat_window: usize,
        #[serde(default = "default_seed")]
        seed: u64,
        #[serde(default)]
        scheduling_policy: String,
    }

    fn default_tokens_per_block() -> usize { 16 }
    fn default_total_blocks() -> usize { 128 }
    fn default_top_k() -> usize { 40 }
    fn default_temperature() -> f64 { 0.8 }
    fn default_repeat_penalty() -> f64 { 1.05 }
    fn default_repeat_window() -> usize { 64 }
    fn default_seed() -> u64 { 1 }

    let c: Config = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("invalid config JSON: {e}")))?;

    let scheduling_policy = match c.scheduling_policy.as_str() {
        "" | "Fair" => cellm_scheduler::SchedulingPolicy::Fair,
        "LatencyFirst" => cellm_scheduler::SchedulingPolicy::LatencyFirst,
        "ThroughputFirst" => cellm_scheduler::SchedulingPolicy::ThroughputFirst,
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown scheduling_policy: {other} (expected Fair, LatencyFirst, or ThroughputFirst)"
            )));
        }
    };

    Ok(EngineConfig {
        tokens_per_block: c.tokens_per_block,
        total_blocks: c.total_blocks,
        top_k: c.top_k,
        temperature: c.temperature,
        repeat_penalty: c.repeat_penalty,
        repeat_window: c.repeat_window,
        seed: c.seed,
        backend: BackendKind::Cpu,
        kv_encoding: cellm_cache::KvEncodingKind::F16,
        turboq_int8_dot: false,
        turboq_qjl_corr: false,
        scheduling_policy,
    })
}
