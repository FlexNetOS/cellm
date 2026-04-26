import Foundation

/// A thread-safe global cache that ensures only one heavy model engine is active in memory at a time.
/// This prevents OOM crashes on iOS when switching between different views or models.
final class GlobalEngineCache {
    static let shared = GlobalEngineCache()
    
    private let lock = NSLock()
    private var cachedLLMEngine: (key: String, engine: CellmEngine, tokenizer: CellmTokenizer)?
    private var cachedVLMEngine: (key: String, engine: CellmVLMEngine)?
    
    private init() {}
    
    /// Clears all cached engines to free up memory.
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        cachedLLMEngine = nil
        cachedVLMEngine = nil
    }
    
    /// Gets or creates a cached LLM engine.
    func getOrCreateLLM(
        key: String,
        factory: () throws -> (CellmEngine, CellmTokenizer)
    ) throws -> (CellmEngine, CellmTokenizer) {
        lock.lock()
        
        // If we have the exact same engine, reuse it.
        if let cached = cachedLLMEngine, cached.key == key {
            lock.unlock()
            return (cached.engine, cached.tokenizer)
        }
        
        // Otherwise, clear EVERYTHING to make room for the new heavy model.
        cachedLLMEngine = nil
        cachedVLMEngine = nil
        lock.unlock()
        
        let (newEngine, newTokenizer) = try factory()
        
        lock.lock()
        cachedLLMEngine = (key, newEngine, newTokenizer)
        lock.unlock()
        
        return (newEngine, newTokenizer)
    }
    
    /// Gets or creates a cached VLM engine.
    func getOrCreateVLM(
        key: String,
        factory: () throws -> CellmVLMEngine
    ) throws -> CellmVLMEngine {
        lock.lock()
        
        if let cached = cachedVLMEngine, cached.key == key {
            lock.unlock()
            return cached.engine
        }
        
        cachedLLMEngine = nil
        cachedVLMEngine = nil
        lock.unlock()
        
        let newEngine = try factory()
        
        lock.lock()
        cachedVLMEngine = (key, newEngine)
        lock.unlock()
        
        return newEngine
    }
}
