// src/main.rs (example usage)
use gnbg_gpu::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    // Load problem
    let problem = load_gnbg_problem(1)?;
    println!("Loaded GNBG problem: {} dimensions, {} components", 
             problem.dimension, problem.comp_num);
    
    // Test CPU implementation
    let mut cpu_eval = CPUEvaluator::new(problem.clone());
    let test_solution: Vec<f64> = (0..problem.dimension)
        .map(|_| rand::random::<f64>() * (problem.max_coordinate - problem.min_coordinate) + problem.min_coordinate)
        .collect();
    
    let cpu_fitness = cpu_eval.fitness(&test_solution);
    println!("CPU fitness: {}", cpu_fitness);
    
    // Test GPU implementation
    let gpu_executor = GpuExecutor::new(&problem).await?;
    
    // Convert to f32 for GPU
    let test_solution_f32: Vec<f32> = test_solution.iter().map(|&x| x as f32).collect();
    let gpu_results = gpu_executor.evaluate_batch(&test_solution_f32).await?;
    
    println!("GPU fitness: {}", gpu_results[0]);
    println!("Difference: {}", (cpu_fitness as f32 - gpu_results[0]).abs());
    
    // Batch evaluation example
    let n_solutions = 1000;
    let mut batch_solutions = Vec::with_capacity(n_solutions * problem.dimension);
    for _ in 0..n_solutions {
        for _ in 0..problem.dimension {
            batch_solutions.push(
                rand::random::<f32>() * (problem.max_coordinate as f32 - problem.min_coordinate as f32) 
                + problem.min_coordinate as f32
            );
        }
    }
    
    let start = std::time::Instant::now();
    let batch_results = gpu_executor.evaluate_batch(&batch_solutions).await?;
    let gpu_time = start.elapsed();
    
    println!("\nBatch evaluation:");
    println!("Solutions: {}", n_solutions);
    println!("GPU time: {:?}", gpu_time);
    println!("Time per solution: {:?}", gpu_time / n_solutions as u32);
    
    Ok(())
}
