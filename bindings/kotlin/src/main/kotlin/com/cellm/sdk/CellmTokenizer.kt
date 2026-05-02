package com.cellm.sdk

/**
 * Tokenizer wrapper around the cellm C FFI tokenizer functions.
 *
 * Usage:
 * ```
 * val tokenizer = CellmTokenizer("/data/local/tmp/tokenizer.json")
 * val tokenIds = tokenizer.encode("Hello world")
 * val text = tokenizer.decode(tokenIds)
 * tokenizer.close()
 * ```
 */
class CellmTokenizer(
    private val handle: Long
) : AutoCloseable {

    companion object {
        init {
            System.loadLibrary("cellm_sdk")
        }

        @JvmStatic
        private external fun nativeTokenizerCreate(path: String): Long

        @JvmStatic
        private external fun nativeTokenizerDestroy(handle: Long)

        @JvmStatic
        private external fun nativeTokenizerEncode(
            handle: Long,
            text: String,
            outTokens: IntArray?,
            maxTokens: Int
        ): Int

        @JvmStatic
        private external fun nativeTokenizerDecode(
            handle: Long,
            tokens: IntArray,
            tokenCount: Int,
            outBuf: ByteArray?,
            bufLen: Int
        ): Int

        /**
         * Load a tokenizer from a HuggingFace tokenizer.json file.
         */
        @JvmStatic
        fun load(path: String): CellmTokenizer {
            val handle = nativeTokenizerCreate(path)
            if (handle == 0L) {
                throw RuntimeException("cellm_tokenizer_create failed for $path")
            }
            return CellmTokenizer(handle)
        }
    }

    /**
     * Encode text to token IDs.
     * Returns the token count. If outTokens is null, returns the required array size.
     */
    fun encode(text: String): IntArray {
        val count = nativeTokenizerEncode(handle, text, null, 0)
        if (count <= 0) return IntArray(0)
        val tokens = IntArray(count)
        nativeTokenizerEncode(handle, text, tokens, count)
        return tokens
    }

    /**
     * Encode text and write into a pre-allocated array.
     * Returns the number of tokens written.
     */
    fun encodeInto(text: String, outTokens: IntArray): Int {
        return nativeTokenizerEncode(handle, text, outTokens, outTokens.size)
    }

    /**
     * Decode token IDs to text.
     */
    fun decode(tokens: IntArray): String {
        val byteCount = nativeTokenizerDecode(handle, tokens, tokens.size, null, 0)
        if (byteCount <= 0) return ""
        val buf = ByteArray(byteCount)
        nativeTokenizerDecode(handle, tokens, tokens.size, buf, byteCount)
        return String(buf, 0, byteCount - 1) // strip null terminator
    }

    /**
     * Decode a single token to text.
     */
    fun decodeOne(token: Int): String {
        return decode(IntArray(1) { token })
    }

    override fun close() {
        if (handle != 0L) {
            nativeTokenizerDestroy(handle)
        }
    }
}
