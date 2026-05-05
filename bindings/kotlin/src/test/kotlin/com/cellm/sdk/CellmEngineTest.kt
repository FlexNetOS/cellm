package com.cellm.sdk

import org.junit.Test
import org.junit.Assert.*

class CellmEngineTest {
    @Test
    fun testBackendName() {
        println("java.library.path = " + System.getProperty("java.library.path"))
        // This tests that the Rust library loads and basic JNI string passing works.
        // nativeBackendName(0) should return "cpu" as an IntArray.
        System.load("/Users/jeff/Desktop/cellm/target/debug/libcellm_sdk.dylib")

        // Use reflection to access the private static method
        val method = Class.forName("com.cellm.sdk.CellmEngine")
            .getDeclaredMethod("nativeBackendName", Long::class.java)
        method.isAccessible = true

        val result = method.invoke(null, 0L) as IntArray?
        assertNotNull(result)
        val backendName = String(result!!.map { it.toChar() }.toCharArray())
        assertEquals("cpu", backendName)
        println("Backend name: $backendName")
    }

    @Test
    fun testStringConversionRoundTrip() {
        // Verify our IntArray string encoding is correct
        val path = "/data/local/tmp/test.cellm"
        val ints = path.toByteArray(Charsets.UTF_8).map { it.toInt() and 0xFF }.toIntArray()
        val decoded = String(ints.map { it.toChar() }.toCharArray())
        assertEquals(path, decoded)
        println("Round-trip OK: $decoded")
    }
}
