fn main() {
    println!("available_parallelism: {:?}", std::thread::available_parallelism());
    println!("rayon threads: {}", rayon::current_num_threads());
}
