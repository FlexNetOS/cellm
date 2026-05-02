// Author: Jeffrey Asante (https://jeffasante.github.io/)
//!
//! Build script that compiles GLSL compute shaders to SPIR-V bytecode via
//! `glslangValidator`.  The resulting `.spv` files are placed under
//! `$OUT_DIR/shaders/` so they can be `include_bytes!`-ed at compile time.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: &[&str] = &[
    "matmul_f32",
    "attention_f32",
    "rms_norm_f32",
    "rope_f32",
    "silu_f32",
    "add_f32",
    "mul_f32",
    "softmax_f32",
];

fn main() {
    let validator = find_glslang_validator();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_src_dir = manifest_dir.join("src").join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_out_dir = out_dir.join("shaders");

    fs::create_dir_all(&shader_out_dir).expect("create shader output dir");

    for name in SHADERS {
        let glsl_path = shader_src_dir.join(format!("{name}.glsl"));
        let spv_path = shader_out_dir.join(format!("{name}.spv"));

        println!("cargo:rerun-if-changed={}", glsl_path.display());

        if let Some(ref validator_path) = validator {
            println!("cargo:warning=Compiling {name}.glsl");

            let status = Command::new(validator_path)
                .arg("-V")
                .arg("-S").arg("comp")
                .arg("--target-env").arg("vulkan1.1")
                .arg("-o").arg(&spv_path)
                .arg(&glsl_path)
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("cargo:warning=  ok {name}.spv");
                }
                Ok(s) => {
                    panic!(
                        "glslangValidator failed for {name}.glsl (exit {})",
                        s.code().unwrap_or(-1)
                    );
                }
                Err(e) => {
                    panic!("Failed to run glslangValidator for {name}.glsl: {e}");
                }
            }
        } else {
            println!("cargo:warning=glslangValidator not found, skipping SPIR-V for {name}");
            let _ = fs::write(&spv_path, SPIRV_EMPTY_MODULE);
        }
    }

    if validator.is_none() {
        println!("cargo:warning=Install Vulkan SDK for SPIR-V compilation.");
    }
}

fn find_glslang_validator() -> Option<PathBuf> {
    if let Ok(path) = env::var("GLSLANG_VALIDATOR") {
        let p = PathBuf::from(&path);
        if p.is_file() {
            return Some(p);
        }
    }

    let common: &[&str] = &[
        "/opt/homebrew/bin/glslangValidator",
        "/usr/local/bin/glslangValidator",
        "/usr/bin/glslangValidator",
    ];
    for path in common {
        if Path::new(path).is_file() {
            return Some(PathBuf::from(path));
        }
    }

    if let Ok(path_var) = env::var("PATH") {
        for dir in path_var.split(':') {
            let candidate = Path::new(dir).join("glslangValidator");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

const SPIRV_EMPTY_MODULE: &[u8] = &[
    0x03, 0x02, 0x23, 0x07,
    0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
];
