package com.cellm.sdk

/**
 * Kotlin wrapper around the cellm C FFI.
 *
 * Usage:
 * ```
 * val engine = CellmEngine.create("/data/local/tmp/model.cellm")
 * val session = engine.createSession()
 * engine.submitTokens(session, tokenIds)
 * while (true) {
 *     val token = engine.stepDecode()
 *     if (token == null) break
 *     // process token
 * }
 * engine.destroy()
 * ```
 */
class CellmEngine private constructor(
    private val handle: Long
) : AutoCloseable {

    enum class Backend(val value: Int) {
        CPU(0),
        METAL(1);
    }

    enum class KvEncoding(val value: Int) {
        F16(0),
        TURBOQUANT(1);
    }

    enum class SchedulingPolicy(val value: Int) {
        FAIR(0),
        LATENCY_FIRST(1),
        THROUGHPUT_FIRST(2);
    }

    data class KvStats(
        val usedBlocks: Int,
        val freeBlocks: Int
    )

    companion object {
        init {
            System.loadLibrary("cellm_sdk")
        }

        @JvmStatic
        private external fun nativeCreate(
            modelPath: String,
            tokensPerBlock: Int,
            totalBlocks: Int,
            topK: Int,
            temperature: Float,
            repeatPenalty: Float,
            repeatWindow: Int,
            seed: Long,
            backend: Int,
            kvEncoding: Int,
            turboqInt8Dot: Int,
            turboqQjlCorr: Int
        ): Long

        @JvmStatic
        private external fun nativeDestroy(handle: Long)

        @JvmStatic
        private external fun nativeSessionCreate(handle: Long): Long

        @JvmStatic
        private external fun nativeSubmitTokens(
            handle: Long,
            session: Long,
            tokens: IntArray
        ): Int

        @JvmStatic
        private external fun nativeSubmitTokensCached(
            handle: Long,
            session: Long,
            tokens: IntArray,
            cacheHit: IntArray
        ): Int

        @JvmStatic
        private external fun nativeStepDecode(
            handle: Long,
            outSession: LongArray,
            outToken: IntArray
        ): Int

        @JvmStatic
        private external fun nativeSessionCancel(handle: Long, session: Long): Int

        @JvmStatic
        private external fun nativeSessionSuspend(handle: Long, session: Long): Int

        @JvmStatic
        private external fun nativeSessionResume(handle: Long, session: Long): Int

        @JvmStatic
        private external fun nativeSessionReset(handle: Long, session: Long): Int

        @JvmStatic
        private external fun nativeSetThermalLevel(handle: Long, level: Int): Int

        @JvmStatic
        private external fun nativeKvStats(
            handle: Long,
            outUsed: IntArray,
            outFree: IntArray
        ): Int

        @JvmStatic
        private external fun nativeBackendName(handle: Long): String

        @JvmStatic
        private external fun nativeSetSchedulingPolicy(handle: Long, policy: Int): Int

        @JvmStatic
        private external fun nativeSchedulingPolicy(handle: Long): Int

        @JvmStatic
        private external fun nativeTotalTokens(handle: Long): Long

        @JvmStatic
        private external fun nativeTokPerSec(handle: Long, out: DoubleArray): Int

        @JvmStatic
        private external fun nativeResetStatsWindow(handle: Long): Int

        @JvmStatic
        private external fun nativeDescribeImage(handle: Long, session: Long, imageBytes: ByteArray, prompt: String): String

        /**
         * Create an engine with default settings (CPU backend, F16 KV encoding,
         * Fair scheduling, 256 blocks of 16 tokens).
         */
        @JvmStatic
        @JvmOverloads
        fun create(
            modelPath: String,
            tokensPerBlock: Int = 16,
            totalBlocks: Int = 256,
            topK: Int = 40,
            temperature: Float = 0.8f,
            repeatPenalty: Float = 1.05f,
            repeatWindow: Int = 64,
            seed: Long = 1L,
            backend: Backend = Backend.CPU,
            kvEncoding: KvEncoding = KvEncoding.F16,
            turboqInt8Dot: Boolean = true,
            turboqQjlCorr: Boolean = true
        ): CellmEngine {
            val handle = nativeCreate(
                modelPath,
                tokensPerBlock,
                totalBlocks,
                topK,
                temperature,
                repeatPenalty,
                repeatWindow,
                seed,
                backend.value,
                kvEncoding.value,
                if (turboqInt8Dot) 1 else 0,
                if (turboqQjlCorr) 1 else 0
            )
            if (handle == 0L) {
                throw RuntimeException("cellm_engine_create failed for $modelPath")
            }
            return CellmEngine(handle)
        }
    }

    fun createSession(): CellmSession {
        val sessionHandle = nativeSessionCreate(handle)
        if (sessionHandle == 0L) {
            throw RuntimeException("cellm_session_create failed")
        }
        return CellmSession(sessionHandle, this)
    }

    fun submitTokens(session: CellmSession, tokens: IntArray): Int {
        return nativeSubmitTokens(handle, session.handle, tokens)
    }

    /**
     * Submit tokens with prefill cache reuse detection.
     * Returns the next token ID, and sets cacheHit[0] to 1 if the prefill cache was reused.
     */
    fun submitTokensCached(
        session: CellmSession,
        tokens: IntArray,
        cacheHit: IntArray = IntArray(1)
    ): Int {
        return nativeSubmitTokensCached(handle, session.handle, tokens, cacheHit)
    }

    /**
     * Run one decode step.
     * Returns a Pair of (sessionHandle, tokenId) or null if nothing to decode.
     */
    fun stepDecode(): Pair<Long, Int>? {
        val outSession = LongArray(1)
        val outToken = IntArray(1)
        val result = nativeStepDecode(handle, outSession, outToken)
        return if (result == 1) Pair(outSession[0], outToken[0]) else null
    }

    fun cancelSession(session: CellmSession) {
        nativeSessionCancel(handle, session.handle)
    }

    fun suspendSession(session: CellmSession) {
        nativeSessionSuspend(handle, session.handle)
    }

    fun resumeSession(session: CellmSession) {
        nativeSessionResume(handle, session.handle)
    }

    fun resetSession(session: CellmSession) {
        nativeSessionReset(handle, session.handle)
    }

    fun setThermalLevel(level: Int): Boolean {
        return nativeSetThermalLevel(handle, level) == 0
    }

    fun getKvStats(): KvStats {
        val used = IntArray(1)
        val free = IntArray(1)
        nativeKvStats(handle, used, free)
        return KvStats(used[0], free[0])
    }

    fun getBackendName(): String {
        return nativeBackendName(handle)
    }

    fun setSchedulingPolicy(policy: SchedulingPolicy): Boolean {
        return nativeSetSchedulingPolicy(handle, policy.value) == 0
    }

    fun getSchedulingPolicy(): SchedulingPolicy {
        val value = nativeSchedulingPolicy(handle)
        return SchedulingPolicy.entries.firstOrNull { it.value == value } ?: SchedulingPolicy.FAIR
    }

    fun getTotalTokens(): Long {
        return nativeTotalTokens(handle)
    }

    fun getTokPerSec(): Double {
        val out = DoubleArray(1)
        nativeTokPerSec(handle, out)
        return out[0]
    }

    fun resetStatsWindow() {
        nativeResetStatsWindow(handle)
    }


    /**
     * Run VLM inference on an image.  The image must be JPEG-encoded bytes.
     * This call blocks until generation finishes; run on a background thread.
     */
    fun describeImage(session: CellmSession, imageBytes: ByteArray, prompt: String): String {
        return nativeDescribeImage(handle, session.handle, imageBytes, prompt)
    }
    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
        }
    }

}
