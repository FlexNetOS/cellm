package com.cellm.sdk

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream

/**
 * Vision Language Model (VLM) inference for image-to-text.
 *
 * Wraps the cellm C FFI image description path.  Takes an image as
 * raw JPEG bytes, a text prompt, and returns generated text.
 *
 * Usage:
 * ```
 * val engine = CellmEngine.create(modelPath)
 * val session = engine.createSession()
 * val result = CellmVLM.describeImage(engine, session, jpegBytes, "Describe this image.")
 * ```
 */
object CellmVLM {

    /**
     * Encode a Bitmap to JPEG bytes suitable for VLM input.
     * Resizes images larger than 1024px on the longest side to keep
     * memory usage under control on mobile devices.
     */
    fun bitmapToJpegBytes(bitmap: Bitmap, quality: Int = 85): ByteArray {
        val maxDim = 1024
        val w = bitmap.width
        val h = bitmap.height

        val scaled: Bitmap = if (w > maxDim || h > maxDim) {
            val scale = minOf(maxDim.toFloat() / w, maxDim.toFloat() / h)
            Bitmap.createScaledBitmap(
                bitmap,
                (w * scale).toInt(),
                (h * scale).toInt(),
                true
            )
        } else {
            bitmap
        }

        val stream = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, quality, stream)

        if (scaled !== bitmap) {
            scaled.recycle()
        }

        return stream.toByteArray()
    }

    /**
     * Run VLM inference on an image with a text prompt.
     *
     * Returns the generated description text.  This call blocks until
     * generation completes; run it on a background coroutine.
     */
    fun describeImage(
        engine: CellmEngine,
        session: CellmSession,
        imageBytes: ByteArray,
        prompt: String
    ): String = engine.describeImage(session, imageBytes, prompt)
}
