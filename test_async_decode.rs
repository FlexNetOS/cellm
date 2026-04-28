#!/usr/bin/env rust-script
//! Test script to demonstrate async decode performance improvements

use std::time::Instant;
use cellm_model::{CellmFile, qwen::QwenRunner};
use cellm_cache::{PageTable, KVCache, KvEncodingKind};
use cellm_core::KvCacheLayout;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/to-huggingface/qwen2.5-0.5b-int8-v1/qwen2.5-0.5b-int8-v1.cellm";
    let tokenizer_path = "models/to-huggingface/qwen2.5-0.5b-int8-v1/tokenizer.json";
    
    println!("Loading model...");
    let file = CellmFile::load(model_path)?;
    let cfg = file.header.model_config();
    
    // Initialize runner
    let mut runner = QwenRunner::new(&file)?;
    runner.enable_metal_full_backend();
    
    // Initialize KV cache
    let layout = KvCacheLayout {
        total_blocks: 4,
        tokens_per_block: 16,
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
    };
    let mut kv_cache = KVCache::new_with_kind_and_encoding(layout, cellm_cache::KvStorageKind::Metal, KvEncodingKind::F16)?;
    let mut page_table = PageTable::new(layout.total_blocks, layout.tokens_per_block)?;
    
    // Setup tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    
    // Test prompt
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt, false)
        .map_err(|e| format!("Failed to encode prompt: {}", e))?
        .get_ids()
        .to_vec();
    
    println!("Prompt: \"{}\" -> {:?} tokens", prompt, tokens);
    
    // Prefill
    println!("Running prefill...");
    runner.prefill(&tokens, 0, &mut page_table, &mut kv_cache)?;
    
    let next_token = tokens[tokens.len() - 1];
    
    // Test sequential decode
    println!("\n=== Testing Sequential Decode ===");
    let t_seq = Instant::now();
    let mut cur = next_token;
    for i in 0..10 {
        let pos = tokens.len() + i;
        let cand = runner.step_topk(cur, pos, &mut page_table, &mut kv_cache, 5)?;
        let (next, _) = cand[0];
        cur = next;
        
        let text = tokenizer.decode(&[next], true)
            .unwrap_or_else(|_| format!("[{}]", next));
        print!("{}", text);
    }
    let seq_time = t_seq.elapsed();
    println!("\nSequential: 10 tokens in {:.2}s ({:.3}s/token)", 
             seq_time.as_secs_f64(), seq_time.as_secs_f64() / 10.0);
    
    // Test async decode (batch of 2)
    println!("\n=== Testing Async Decode (batch=2) ===");
    let t_async = Instant::now();
    let mut cur = next_token;
    for i in (0..10).step_by(2) {
        let pos = tokens.len() + i;
        let batch_tokens = vec![cur, cur]; // Simplified - using same token twice
        let batch_results = runner.step_topk_async(&batch_tokens, pos, &mut page_table, &mut kv_cache, 5)?;
        
        for (j, cand) in batch_results.into_iter().enumerate() {
            if i + j >= 10 { break; }
            let (next, _) = cand[0];
            cur = next;
            
            let text = tokenizer.decode(&[next], true)
                .unwrap_or_else(|_| format!("[{}]", next));
            print!("{}", text);
        }
    }
    let async_time = t_async.elapsed();
    println!("\nAsync (batch=2): 10 tokens in {:.2}s ({:.3}s/token)", 
             async_time.as_secs_f64(), async_time.as_secs_f64() / 10.0);
    
    // Calculate speedup
    let speedup = seq_time.as_secs_f64() / async_time.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);
    
    Ok(())
}
