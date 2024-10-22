use ndarray::{Array2, arr2};
use std::time::Instant;
extern crate blas_src;

fn main() {
    // Initialize OpenBLAS thread count (optional but recommended)
    println!("Running matrix multiplication benchmark...\n");
    println!("Format: (i x 768) * (768 x 4096)");
    println!("i\tTime (ms)");
    println!("-----------------");
    

    // Generate the second matrix (768 x 4096) once
    let b = Array2::<f64>::ones((768, 16384));

    // First benchmark with i = 1
    let i = 1;
    benchmark_size(i, &b);

    // Then benchmark for i = 16, 32, ..., 256
    for i in (16..=1024).step_by(64) {
        benchmark_size(i, &b);
    }
}
fn benchmark_size(i: usize, b: &Array2<f64>) {
    // Create matrix A with dimensions (i x 768)
    let a = Array2::<f64>::ones((i, 768));

    // Perform multiplication multiple times to get a more stable measurement
    const NUM_RUNS: u32 = 10;
    let mut total_duration = 0.0;

    for _ in 0..NUM_RUNS {
        let start = Instant::now();
        let _c = a.dot(b);
        let duration = start.elapsed();
        total_duration += duration.as_secs_f64() * 1000.0; // Convert to milliseconds
    }

    let avg_duration = total_duration / NUM_RUNS as f64;
    let time_per_row = avg_duration / i as f64;
    
    println!("{}\t{:.2}\t\t{:.4}", i, avg_duration, time_per_row);
}