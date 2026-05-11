use cellm_model::deepseek_v4::DeepSeekV4Runner;
use cellm_cache::{KVCache, PageTable};
use cellm_core::KvCacheLayout;
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
    println!("Config: hidden_size={}, num_layers={}, num_heads={}, head_dim={}",
             runner.config().hidden_size,
             runner.config().num_hidden_layers,
             runner.config().num_attention_heads,
             runner.config().head_dim);
    println!("HC: hc_mult={}, sinkhorn_iters={}",
             runner.config().hc_mult.unwrap_or(0),
             runner.config().hc_sinkhorn_iters.unwrap_or(0));
    println!("MoE: n_routed_experts={}, experts_per_tok={}, moe_intermediate={}",
             runner.config().n_routed_experts.unwrap_or(0),
             runner.config().num_experts_per_tok.unwrap_or(0),
             runner.config().moe_intermediate_size.unwrap_or(0));

    let layout = KvCacheLayout {
        total_blocks: 128,
        tokens_per_block: 16,
        num_layers: 8,
        num_kv_heads: 1,
        head_dim: 96,
    };
    let mut kv_cache = KVCache::new(layout)?;
    let mut page_table = PageTable::new(1, 16)?;

    // Prompt tokens to process sequentially
    let prompt_tokens: Vec<u32> = vec![0, 128803, 9602, 734, 4615, 69, 3301, 6728, 33, 128804];
    println!("\n=== Processing {} prompt tokens sequentially ===", prompt_tokens.len());
    println!("Prompt: {:?}", prompt_tokens);

    // The first token at position 0
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        // Must append token to page table to allocate block space if needed
        if pos == page_table.token_count() {
            page_table.append_token(kv_cache.allocator_mut())?;
        }

        println!("\n--- Token {} at pos {} (token_id={}) ---", pos, pos, token);
        let top_k = runner.step_topk(token, pos, &mut page_table, &mut kv_cache, 100)?;

        println!("Top 10 logits:");
        for (i, (id, prob)) in top_k.iter().enumerate().take(10) {
            println!("  {}: token_id={}, score={}", i + 1, id, prob);
        }

        // Show what token would be selected (argmax)
        if let Some((best_id, best_score)) = top_k.first() {
            println!("  => Predicted next token: {} (score={})", best_id, best_score);
            // Check if this matches expected tokens
            if *best_id == 35 {
                println!("  => MATCHES expected token 35 = 'A'");
            } else if *best_id == 67 {
                println!("  => MATCHES expected token 67 = 'a' (lowercase)");
            }
        }

        // Print full logit vector for top 100 tokens
        println!("Full top-100:");
        for (i, (id, prob)) in top_k.iter().enumerate() {
            println!("  {}: token_id={}, score={}", i + 1, id, prob);
        }
    }

    Ok(())
}
