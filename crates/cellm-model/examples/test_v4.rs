use cellm_model::deepseek_v4::DeepSeekV4Runner;
use cellm_cache::{KVCache, PageTable, CacheConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("models/nanowhale-100m.cellm");
    if !model_path.exists() {
        println!("Model not found at {:?}", model_path);
        return Ok(());
    }

    println!("Loading model...");
    let mut runner = DeepSeekV4Runner::load(model_path)?;
    println!("Model loaded.");

    let cache_cfg = CacheConfig {
        num_blocks: 128,
        block_size: 16,
        num_layers: 8,
        num_kv_heads: 1,
        head_dim: 96,
    };
    let mut kv_cache = KVCache::new(cache_cfg);
    let mut page_table = PageTable::new(128, 16);

    // Initial token (e.g. BOS = 0)
    let token = 0u32;
    let pos = 0;
    
    println!("Running forward pass for token {} at pos {}...", token, pos);
    let top_k = runner.step(token, pos, &mut page_table, &mut kv_cache, 10)?;
    
    println!("Top 10 logits:");
    for (i, (id, prob)) in top_k.iter().enumerate() {
        println!("{}: token_id={}, score={}", i + 1, id, prob);
    }

    Ok(())
}
