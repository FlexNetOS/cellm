fn main() {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    println!("neon enabled");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    println!("neon disabled");
}
