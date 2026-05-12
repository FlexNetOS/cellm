// Author: Jeffrey Asante (https://jeffasante.github.io/)
//! JNI wrappers using the raw JNIEnv function table pointer.
//! No dependency on the `jni` crate.

use std::path::Path;
use crate::{Engine, EngineConfig};

type jlong = i64;
type jint = i32;
type jfloat = f32;
type jsize = i32;
type jboolean = u8;

#[repr(C)]
struct JNIInvokeInterface {
    _reserved0: *mut std::ffi::c_void,
    _reserved1: *mut std::ffi::c_void,
    _reserved2: *mut std::ffi::c_void,
    _destroy_java_vm: *mut std::ffi::c_void,
    _attach_current_thread: *mut std::ffi::c_void,
    _detach_current_thread: *mut std::ffi::c_void,
    _get_env: *mut std::ffi::c_void,
    _attach_current_thread_as_daemon: *mut std::ffi::c_void,
}

#[repr(C)]
struct JNINativeInterface {
    _reserved0: *mut std::ffi::c_void,
    _reserved1: *mut std::ffi::c_void,
    _reserved2: *mut std::ffi::c_void,
    _reserved3: *mut std::ffi::c_void,
    get_version: *mut std::ffi::c_void,
    _define_class: *mut std::ffi::c_void,
    find_class: extern "system" fn(env: *mut JNIEnv, name: *const u8) -> jclass,
    _from_reflected_method: *mut std::ffi::c_void,
    _from_reflected_field: *mut std::ffi::c_void,
    _to_reflected_method: *mut std::ffi::c_void,
    _get_superclass: *mut std::ffi::c_void,
    _is_assignable_from: *mut std::ffi::c_void,
    _to_reflected_field: *mut std::ffi::c_void,
    throw: extern "system" fn(env: *mut JNIEnv, obj: jobject) -> jint,
    throw_new: extern "system" fn(env: *mut JNIEnv, clazz: jclass, msg: *const u8) -> jint,
    _fatal_error: *mut std::ffi::c_void,
    _push_local_frame: *mut std::ffi::c_void,
    _pop_local_frame: *mut std::ffi::c_void,
    _new_global_ref: *mut std::ffi::c_void,
    _delete_global_ref: *mut std::ffi::c_void,
    _delete_local_ref: *mut std::ffi::c_void,
    _is_same_object: *mut std::ffi::c_void,
    _new_local_ref: *mut std::ffi::c_void,
    _ensure_local_capacity: *mut std::ffi::c_void,
    _alloc_object: *mut std::ffi::c_void,
    _new_object: *mut std::ffi::c_void,
    _new_object_v: *mut std::ffi::c_void,
    _new_object_a: *mut std::ffi::c_void,
    _get_object_class: *mut std::ffi::c_void,
    _is_instance_of: *mut std::ffi::c_void,
    _get_method_id: *mut std::ffi::c_void,
    _call_object_method: *mut std::ffi::c_void,
    _call_object_method_v: *mut std::ffi::c_void,
    _call_object_method_a: *mut std::ffi::c_void,
    _call_boolean_method: *mut std::ffi::c_void,
    _call_boolean_method_v: *mut std::ffi::c_void,
    _call_boolean_method_a: *mut std::ffi::c_void,
    _call_byte_method: *mut std::ffi::c_void,
    _call_byte_method_v: *mut std::ffi::c_void,
    _call_byte_method_a: *mut std::ffi::c_void,
    _call_char_method: *mut std::ffi::c_void,
    _call_char_method_v: *mut std::ffi::c_void,
    _call_char_method_a: *mut std::ffi::c_void,
    _call_short_method: *mut std::ffi::c_void,
    _call_short_method_v: *mut std::ffi::c_void,
    _call_short_method_a: *mut std::ffi::c_void,
    _call_int_method: *mut std::ffi::c_void,
    _call_int_method_v: *mut std::ffi::c_void,
    _call_int_method_a: *mut std::ffi::c_void,
    _call_long_method: *mut std::ffi::c_void,
    _call_long_method_v: *mut std::ffi::c_void,
    _call_long_method_a: *mut std::ffi::c_void,
    _call_float_method: *mut std::ffi::c_void,
    _call_float_method_v: *mut std::ffi::c_void,
    _call_float_method_a: *mut std::ffi::c_void,
    _call_double_method: *mut std::ffi::c_void,
    _call_double_method_v: *mut std::ffi::c_void,
    _call_double_method_a: *mut std::ffi::c_void,
    _call_void_method: *mut std::ffi::c_void,
    _call_void_method_v: *mut std::ffi::c_void,
    _call_void_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_object_method: *mut std::ffi::c_void,
    _call_nonvirtual_object_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_object_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_boolean_method: *mut std::ffi::c_void,
    _call_nonvirtual_boolean_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_boolean_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_byte_method: *mut std::ffi::c_void,
    _call_nonvirtual_byte_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_byte_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_char_method: *mut std::ffi::c_void,
    _call_nonvirtual_char_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_char_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_short_method: *mut std::ffi::c_void,
    _call_nonvirtual_short_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_short_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_int_method: *mut std::ffi::c_void,
    _call_nonvirtual_int_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_int_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_long_method: *mut std::ffi::c_void,
    _call_nonvirtual_long_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_long_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_float_method: *mut std::ffi::c_void,
    _call_nonvirtual_float_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_float_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_double_method: *mut std::ffi::c_void,
    _call_nonvirtual_double_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_double_method_a: *mut std::ffi::c_void,
    _call_nonvirtual_void_method: *mut std::ffi::c_void,
    _call_nonvirtual_void_method_v: *mut std::ffi::c_void,
    _call_nonvirtual_void_method_a: *mut std::ffi::c_void,
    _get_field_id: *mut std::ffi::c_void,
    _get_object_field: *mut std::ffi::c_void,
    _get_boolean_field: *mut std::ffi::c_void,
    _get_byte_field: *mut std::ffi::c_void,
    _get_char_field: *mut std::ffi::c_void,
    _get_short_field: *mut std::ffi::c_void,
    _get_int_field: *mut std::ffi::c_void,
    _get_long_field: *mut std::ffi::c_void,
    _get_float_field: *mut std::ffi::c_void,
    _get_double_field: *mut std::ffi::c_void,
    _set_object_field: *mut std::ffi::c_void,
    _set_boolean_field: *mut std::ffi::c_void,
    _set_byte_field: *mut std::ffi::c_void,
    _set_char_field: *mut std::ffi::c_void,
    _set_short_field: *mut std::ffi::c_void,
    _set_int_field: *mut std::ffi::c_void,
    _set_long_field: *mut std::ffi::c_void,
    _set_float_field: *mut std::ffi::c_void,
    _set_double_field: *mut std::ffi::c_void,
    _get_static_method_id: *mut std::ffi::c_void,
    _call_static_object_method: *mut std::ffi::c_void,
    _call_static_object_method_v: *mut std::ffi::c_void,
    _call_static_object_method_a: *mut std::ffi::c_void,
    _call_static_boolean_method: *mut std::ffi::c_void,
    _call_static_boolean_method_v: *mut std::ffi::c_void,
    _call_static_boolean_method_a: *mut std::ffi::c_void,
    _call_static_byte_method: *mut std::ffi::c_void,
    _call_static_byte_method_v: *mut std::ffi::c_void,
    _call_static_byte_method_a: *mut std::ffi::c_void,
    _call_static_char_method: *mut std::ffi::c_void,
    _call_static_char_method_v: *mut std::ffi::c_void,
    _call_static_char_method_a: *mut std::ffi::c_void,
    _call_static_short_method: *mut std::ffi::c_void,
    _call_static_short_method_v: *mut std::ffi::c_void,
    _call_static_short_method_a: *mut std::ffi::c_void,
    _call_static_int_method: *mut std::ffi::c_void,
    _call_static_int_method_v: *mut std::ffi::c_void,
    _call_static_int_method_a: *mut std::ffi::c_void,
    _call_static_long_method: *mut std::ffi::c_void,
    _call_static_long_method_v: *mut std::ffi::c_void,
    _call_static_long_method_a: *mut std::ffi::c_void,
    _call_static_float_method: *mut std::ffi::c_void,
    _call_static_float_method_v: *mut std::ffi::c_void,
    _call_static_float_method_a: *mut std::ffi::c_void,
    _call_static_double_method: *mut std::ffi::c_void,
    _call_static_double_method_v: *mut std::ffi::c_void,
    _call_static_double_method_a: *mut std::ffi::c_void,
    _call_static_void_method: *mut std::ffi::c_void,
    _call_static_void_method_v: *mut std::ffi::c_void,
    _call_static_void_method_a: *mut std::ffi::c_void,
    _get_static_field_id: *mut std::ffi::c_void,
    _get_static_object_field: *mut std::ffi::c_void,
    _get_static_boolean_field: *mut std::ffi::c_void,
    _get_static_byte_field: *mut std::ffi::c_void,
    _get_static_char_field: *mut std::ffi::c_void,
    _get_static_short_field: *mut std::ffi::c_void,
    _get_static_int_field: *mut std::ffi::c_void,
    _get_static_long_field: *mut std::ffi::c_void,
    _get_static_float_field: *mut std::ffi::c_void,
    _get_static_double_field: *mut std::ffi::c_void,
    _set_static_object_field: *mut std::ffi::c_void,
    _set_static_boolean_field: *mut std::ffi::c_void,
    _set_static_byte_field: *mut std::ffi::c_void,
    _set_static_char_field: *mut std::ffi::c_void,
    _set_static_short_field: *mut std::ffi::c_void,
    _set_static_int_field: *mut std::ffi::c_void,
    _set_static_long_field: *mut std::ffi::c_void,
    _set_static_float_field: *mut std::ffi::c_void,
    _set_static_double_field: *mut std::ffi::c_void,
    new_string: extern "system" fn(env: *mut JNIEnv, chars: *const u16, len: jsize) -> jstring,
    get_string_length: *mut std::ffi::c_void,
    get_string_chars: *mut std::ffi::c_void,
    release_string_chars: *mut std::ffi::c_void,
    new_string_utf: extern "system" fn(env: *mut JNIEnv, bytes: *const u8) -> jstring,
    get_string_utf_chars: extern "system" fn(env: *mut JNIEnv, s: jstring, is_copy: *mut jboolean) -> *const u8,
    release_string_utf_chars: extern "system" fn(env: *mut JNIEnv, s: jstring, utf: *const u8),
    get_array_length: extern "system" fn(env: *mut JNIEnv, array: jarray) -> jsize,
    _new_object_array: *mut std::ffi::c_void,
    _get_object_array_element: *mut std::ffi::c_void,
    _set_object_array_element: *mut std::ffi::c_void,
    _new_boolean_array: *mut std::ffi::c_void,
    _new_byte_array: *mut std::ffi::c_void,
    _new_char_array: *mut std::ffi::c_void,
    _new_short_array: *mut std::ffi::c_void,
    new_int_array: *mut std::ffi::c_void,
    new_long_array: *mut std::ffi::c_void,
    _new_float_array: *mut std::ffi::c_void,
    _new_double_array: *mut std::ffi::c_void,
    _get_boolean_array_elements: *mut std::ffi::c_void,
    get_byte_array_elements: extern "system" fn(env: *mut JNIEnv, array: jbyteArray, is_copy: *mut jboolean) -> *mut u8,
    _get_char_array_elements: *mut std::ffi::c_void,
    _get_short_array_elements: *mut std::ffi::c_void,
    get_int_array_elements: extern "system" fn(env: *mut JNIEnv, array: jintArray, is_copy: *mut jboolean) -> *mut jint,
    _get_long_array_elements: *mut std::ffi::c_void,
    _get_float_array_elements: *mut std::ffi::c_void,
    _get_double_array_elements: *mut std::ffi::c_void,
    _release_boolean_array_elements: *mut std::ffi::c_void,
    release_byte_array_elements: extern "system" fn(env: *mut JNIEnv, array: jbyteArray, elems: *mut u8, mode: jint),
    _release_char_array_elements: *mut std::ffi::c_void,
    _release_short_array_elements: *mut std::ffi::c_void,
    release_int_array_elements: extern "system" fn(env: *mut JNIEnv, array: jintArray, elems: *mut jint, mode: jint),
    _release_long_array_elements: *mut std::ffi::c_void,
    _release_float_array_elements: *mut std::ffi::c_void,
    _release_double_array_elements: *mut std::ffi::c_void,
    _get_byte_array_region: *mut std::ffi::c_void,
    _set_byte_array_region: *mut std::ffi::c_void,
    _register_natives: *mut std::ffi::c_void,
    _unregister_natives: *mut std::ffi::c_void,
    _monitor_enter: *mut std::ffi::c_void,
    _monitor_exit: *mut std::ffi::c_void,
    _get_java_vm: *mut std::ffi::c_void,
    _get_int_array_region: *mut std::ffi::c_void,
    _set_int_array_region: *mut std::ffi::c_void,
    _get_long_array_region: *mut std::ffi::c_void,
    _set_long_array_region: *mut std::ffi::c_void,
    _get_float_array_region: *mut std::ffi::c_void,
    _set_float_array_region: *mut std::ffi::c_void,
    _get_double_array_region: *mut std::ffi::c_void,
    _set_double_array_region: *mut std::ffi::c_void,
    _new_direct_byte_buffer: *mut std::ffi::c_void,
    _get_direct_buffer_address: *mut std::ffi::c_void,
    _get_direct_buffer_capacity: *mut std::ffi::c_void,
    _get_object_ref_type: *mut std::ffi::c_void,
}

pub type jclass = *mut std::ffi::c_void;
pub type jobject = *mut std::ffi::c_void;
pub type jstring = jobject;
pub type jarray = jobject;
pub type jintArray = jobject;
pub type jbyteArray = jobject;
pub type jlongArray = jobject;

#[repr(C)]
pub struct JNIEnv {
    functions: *const JNINativeInterface,
}

impl JNIEnv {
    unsafe fn fns(&self) -> &JNINativeInterface { &*self.functions }

    unsafe fn throw_new(&self, msg: &str) {
        let cls = (self.fns().find_class)(self as *const JNIEnv as *mut JNIEnv, b"java/lang/RuntimeException\0".as_ptr());
        if !cls.is_null() {
            let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
            let _ = (self.fns().throw_new)(self as *const JNIEnv as *mut JNIEnv, cls, cmsg.as_ptr());
        }
    }

    unsafe fn get_string_utf_chars(&self, s: jstring) -> *const u8 {
        (self.fns().get_string_utf_chars)(self as *const JNIEnv as *mut JNIEnv, s, std::ptr::null_mut())
    }

    unsafe fn release_string_utf_chars(&self, s: jstring, utf: *const u8) {
        (self.fns().release_string_utf_chars)(self as *const JNIEnv as *mut JNIEnv, s, utf);
    }

    unsafe fn get_array_length(&self, array: jarray) -> jsize {
        (self.fns().get_array_length)(self as *const JNIEnv as *mut JNIEnv, array)
    }

    unsafe fn get_int_array_elements(&self, array: jintArray) -> *mut jint {
        (self.fns().get_int_array_elements)(self as *const JNIEnv as *mut JNIEnv, array, std::ptr::null_mut())
    }

    unsafe fn release_int_array_elements(&self, array: jintArray, elems: *mut jint) {
        (self.fns().release_int_array_elements)(self as *const JNIEnv as *mut JNIEnv, array, elems, 0);
    }

    unsafe fn get_byte_array_elements(&self, array: jbyteArray) -> *mut u8 {
        (self.fns().get_byte_array_elements)(self as *const JNIEnv as *mut JNIEnv, array, std::ptr::null_mut())
    }

    unsafe fn release_byte_array_elements(&self, array: jbyteArray, elems: *mut u8) {
        (self.fns().release_byte_array_elements)(self as *const JNIEnv as *mut JNIEnv, array, elems, 0);
    }

    unsafe fn new_string_utf(&self, bytes: *const u8) -> jstring {
        (self.fns().new_string_utf)(self as *const JNIEnv as *mut JNIEnv, bytes)
    }
}

fn jint_array_to_string(env: *const JNIEnv, arr: jintArray) -> String {
    if arr.is_null() { return String::new(); }
    unsafe {
        let len = (*env).get_array_length(arr) as usize;
        let elems = (*env).get_int_array_elements(arr);
        if elems.is_null() { return String::new(); }
        let ints = std::slice::from_raw_parts(elems, len);
        let bytes: Vec<u8> = ints.iter().map(|&i| (i as u32 & 0xFF) as u8).collect();
        let result = String::from_utf8_lossy(&bytes).into_owned();
        (*env).release_int_array_elements(arr, elems);
        result
    }
}

#[cfg(target_os = "android")]
fn android_log(tag: &str, msg: &str) {
    extern "C" {
        fn __android_log_write(prio: i32, tag: *const u8, text: *const u8) -> i32;
    }
    unsafe {
        let ctag = std::ffi::CString::new(tag).unwrap_or_default();
        let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
        __android_log_write(3, ctag.as_ptr() as *const u8, cmsg.as_ptr() as *const u8);
    }
}
#[cfg(not(target_os = "android"))]
fn android_log(_tag: &str, msg: &str) {
    eprintln!("{}", msg);
}

fn throw(_env: *const JNIEnv, _msg: &str) {
    unsafe { (*_env).throw_new(_msg); }
}

// Engine JNI 

#[no_mangle]
pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeCreate(
    env: *const JNIEnv, _class: jclass,
    model_path: jintArray,
    tokens_per_block: jint, total_blocks: jint, top_k: jint,
    temperature: jfloat, repeat_penalty: jfloat, repeat_window: jint,
    seed: jlong, backend: jint, kv_encoding: jint,
    turboq_int8_dot: jint, turboq_qjl_corr: jint,
) -> jlong {
    let path = jint_array_to_string(env, model_path);
    let cfg = EngineConfig {
        tokens_per_block: tokens_per_block as usize,
        total_blocks: total_blocks as usize,
        top_k: top_k as usize,
        temperature: temperature as f64,
        repeat_penalty: repeat_penalty as f64,
        repeat_window: repeat_window as usize,
        seed: seed as u64,
        backend: if backend == 0 { crate::BackendKind::Cpu } else { crate::BackendKind::Metal },
        kv_encoding: if kv_encoding == 0 { cellm_cache::KvEncodingKind::F16 } else { cellm_cache::KvEncodingKind::TurboQuant },
        turboq_int8_dot: turboq_int8_dot != 0,
        turboq_qjl_corr: turboq_qjl_corr != 0,
        scheduling_policy: cellm_scheduler::SchedulingPolicy::Fair,
    };
    android_log("cellm", &format!("Engine::new starting: path={path:?} backend={:?}", cfg.backend));
    let start = std::time::Instant::now();
    let res = Engine::new(Path::new(&path), cfg);
    let elapsed = start.elapsed();
    android_log("cellm", &format!("Engine::new finished in {elapsed:?}"));
    match res {
        Ok(e) => Box::into_raw(Box::new(e)) as jlong,
        Err(e) => {
            android_log("cellm", &format!("Engine::new error: {e}"));
            throw(env, &format!("{e}"));
            0
        }
    }
}

#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeDestroy(_: *const JNIEnv, _: jclass, h: jlong) { if h != 0 { unsafe { drop(Box::from_raw(h as *mut Engine)); } } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSessionCreate(_: *const JNIEnv, _: jclass, h: jlong) -> jlong { if h == 0 { 0 } else { unsafe { &mut *(h as *mut Engine) }.create_session() as jlong } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSubmitTokens(env: *const JNIEnv, _: jclass, h: jlong, sid: jlong, tokens: jintArray) -> jint {
    if h == 0 { return -1; }
    unsafe {
        let len = (*env).get_array_length(tokens) as usize;
        let elems = (*env).get_int_array_elements(tokens);
        let ids: Vec<u32> = std::slice::from_raw_parts(elems as *const u32, len).to_vec();
        (*env).release_int_array_elements(tokens, elems);
        (&mut *(h as *mut Engine)).submit_tokens(sid as u64, &ids).map(|n| n as jint).unwrap_or(-1)
    }
}
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeStepDecode(_: *const JNIEnv, _: jclass, h: jlong, _: jintArray, _: jintArray) -> jint {
    if h == 0 { -1 } else { match unsafe { &mut *(h as *mut Engine) }.step_decode() { Ok(Some(_)) => 1, Ok(None) => 0, Err(_) => -1 } }
}
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeBackendName(_env: *const JNIEnv, _: jclass, h: jlong) -> jintArray {
    // Return a small int array with ASCII values of backend name to avoid broken new_string_utf.
    let name = if h == 0 { "cpu" } else { unsafe { (&*(h as *const Engine)).backend_name() } };
    let bytes = name.as_bytes();
    unsafe {
        let arr = std::mem::transmute::<*mut std::ffi::c_void, extern "system" fn(*mut JNIEnv, jsize) -> jintArray>((*_env).fns().new_int_array);
        let jarr = arr(_env as *mut JNIEnv, bytes.len() as jsize);
        if !jarr.is_null() {
            let elems = (*_env).get_int_array_elements(jarr);
            for i in 0..bytes.len() { *elems.add(i) = bytes[i] as jint; }
            (*_env).release_int_array_elements(jarr, elems);
        }
        jarr
    }
}
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeTotalTokens(_: *const JNIEnv, _: jclass, h: jlong) -> jlong { if h == 0 { 0 } else { unsafe { &*(h as *const Engine) }.stats().total_tokens_generated as jlong } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSessionCancel(_: *const JNIEnv, _: jclass, h: jlong, s: jlong) -> jint { if h == 0 { -1 } else { unsafe { &mut *(h as *mut Engine) }.cancel_session(s as u64).map(|_|0).unwrap_or(-1) } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSessionSuspend(_: *const JNIEnv, _: jclass, h: jlong, s: jlong) -> jint { if h == 0 { -1 } else { unsafe { &mut *(h as *mut Engine) }.suspend_session(s as u64).map(|_|0).unwrap_or(-1) } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSessionResume(_: *const JNIEnv, _: jclass, h: jlong, s: jlong) -> jint { if h == 0 { -1 } else { unsafe { &mut *(h as *mut Engine) }.resume_session(s as u64).map(|_|0).unwrap_or(-1) } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSessionReset(_: *const JNIEnv, _: jclass, h: jlong, s: jlong) -> jint { if h == 0 { -1 } else { unsafe { &mut *(h as *mut Engine) }.reset_session(s as u64).map(|_|0).unwrap_or(-1) } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSetSchedulingPolicy(_: *const JNIEnv, _: jclass, h: jlong, p: jint) -> jint { if h == 0 { -1 } else { let sp = match p { 0 => cellm_scheduler::SchedulingPolicy::Fair, 1 => cellm_scheduler::SchedulingPolicy::LatencyFirst, 2 => cellm_scheduler::SchedulingPolicy::ThroughputFirst, _ => return -1 }; unsafe { &mut *(h as *mut Engine) }.set_scheduling_policy(sp); 0 } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeResetStatsWindow(_: *const JNIEnv, _: jclass, h: jlong) -> jint { if h == 0 { -1 } else { unsafe { &mut *(h as *mut Engine) }.reset_stats_window(); 0 } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSetThermalLevel(_: *const JNIEnv, _: jclass, h: jlong, l: jint) -> jint { if h == 0 { -1 } else { let tl = match l { 0 => cellm_scheduler::ThermalLevel::Nominal, 1 => cellm_scheduler::ThermalLevel::Elevated, 2 => cellm_scheduler::ThermalLevel::Critical, 3 => cellm_scheduler::ThermalLevel::Emergency, _ => return -1 }; unsafe { &mut *(h as *mut Engine) }.set_thermal_level(tl); 0 } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeKvStats(_: *const JNIEnv, _: jclass, _h: jlong, _: jintArray, _: jintArray) -> jint { 0 }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSchedulingPolicy(_: *const JNIEnv, _: jclass, h: jlong) -> jint { if h == 0 { 0 } else { match unsafe { &*(h as *const Engine) }.scheduling_policy() { cellm_scheduler::SchedulingPolicy::Fair => 0, cellm_scheduler::SchedulingPolicy::LatencyFirst => 1, cellm_scheduler::SchedulingPolicy::ThroughputFirst => 2 } } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeTokPerSec(_: *const JNIEnv, _: jclass, _h: jlong, _: jintArray) -> jint { 0 }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeDescribeImage(env: *const JNIEnv, _: jclass, _h: jlong, _s: jlong, _img: jbyteArray, _p: jstring) -> jstring { unsafe { (*env).new_string_utf(b"VLM not available\0".as_ptr()) } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmEngine_nativeSubmitTokensCached(env: *const JNIEnv, _: jclass, h: jlong, sid: jlong, tokens: jintArray, _: jintArray) -> jint {
    if h == 0 { return -1; }
    unsafe {
        let len = (*env).get_array_length(tokens) as usize;
        let elems = (*env).get_int_array_elements(tokens);
        let ids: Vec<u32> = std::slice::from_raw_parts(elems as *const u32, len).to_vec();
        (*env).release_int_array_elements(tokens, elems);
        (&mut *(h as *mut Engine)).submit_tokens_cached(sid as u64, &ids).map(|(n,_)| n as jint).unwrap_or(-1)
    }
}

// Tokenizer

#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmTokenizer_nativeTokenizerCreate(env: *const JNIEnv, _: jclass, path: jintArray) -> jlong {
    let p = jint_array_to_string(env, path);
    tokenizers::Tokenizer::from_file(Path::new(&p)).map(|t| Box::into_raw(Box::new(t)) as jlong).unwrap_or(0)
}
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmTokenizer_nativeTokenizerDestroy(_: *const JNIEnv, _: jclass, h: jlong) { if h != 0 { unsafe { drop(Box::from_raw(h as *mut tokenizers::Tokenizer)); } } }
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmTokenizer_nativeTokenizerEncode(env: *const JNIEnv, _: jclass, h: jlong, text: jintArray, out: jintArray, max_tokens: jint) -> jint {
    if h == 0 { return 0; }
    let s = jint_array_to_string(env, text);
    let tok = unsafe { &*(h as *const tokenizers::Tokenizer) };
    let ids = tok.encode(s.as_str(), false).map(|e| e.get_ids().to_vec()).unwrap_or_default();
    let n = ids.len().min(max_tokens as usize);
    unsafe {
        if !out.is_null() {
            let elems = (*env).get_int_array_elements(out);
            for i in 0..n { *elems.add(i) = ids[i] as jint; }
            (*env).release_int_array_elements(out, elems);
        }
    }
    n as jint
}
#[no_mangle] pub extern "system" fn Java_com_cellm_sdk_CellmTokenizer_nativeTokenizerDecode(env: *const JNIEnv, _: jclass, h: jlong, tokens: jintArray, count: jint, out: jbyteArray, buf_len: jint) -> jint {
    if h == 0 { return 0; }
    let tok = unsafe { &*(h as *const tokenizers::Tokenizer) };
    unsafe {
        let len = count as usize;
        let elems = (*env).get_int_array_elements(tokens);
        let ids: Vec<u32> = std::slice::from_raw_parts(elems as *const u32, len).to_vec();
        (*env).release_int_array_elements(tokens, elems);
        let decoded = tok.decode(&ids, true).unwrap_or_default();
        let bytes = decoded.as_bytes();
        let n = bytes.len().min(buf_len as usize - 1).max(0);
        if !out.is_null() && buf_len > 0 {
            let dst = (*env).get_int_array_elements(out) as *mut u8; // reusing int array as byte array
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, n);
            *dst.add(n) = 0;
            (*env).release_int_array_elements(out, dst as *mut jint);
        }
        (decoded.len() + 1) as jint
    }
}
