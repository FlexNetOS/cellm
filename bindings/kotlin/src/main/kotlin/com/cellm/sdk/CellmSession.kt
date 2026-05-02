package com.cellm.sdk

/**
 * Handle to an active inference session.
 * Owned by a [CellmEngine]; cancelling or closing the engine
 * invalidates all sessions.
 */
class CellmSession(
    internal val handle: Long,
    internal val engine: CellmEngine
) {
    fun cancel() {
        engine.cancelSession(this)
    }

    fun suspend() {
        engine.suspendSession(this)
    }

    fun resume() {
        engine.resumeSession(this)
    }

    fun reset() {
        engine.resetSession(this)
    }

    /**
     * Submit a text prompt and get the first generated token.
     * The tokenizer must be used separately to convert text to token IDs.
     */
    fun submitTokens(tokens: IntArray): Int {
        return engine.submitTokens(this, tokens)
    }

    /**
     * Submit tokens with cache reuse information.
     * Returns Pair(nextToken, cacheWasReused).
     */
    fun submitTokensCached(tokens: IntArray): Pair<Int, Boolean> {
        val cacheHit = IntArray(1)
        val nextToken = engine.submitTokensCached(this, tokens, cacheHit)
        return Pair(nextToken, cacheHit[0] == 1)
    }
}
