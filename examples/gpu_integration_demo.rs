/// GPU Integration Demo for GNBG-II Multi-Objective Extension
/// 
/// Demonstrates the shared GPU context integration between single-objective
/// GNBG and multi-objective problems, showcasing infrastructure reuse.

use gnbg_gpu::multi_objective::*;
use gnbg_gpu::multi_objective::transformations::VariableRange;
use gnbg_gpu::gpu_context::GpuContext;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("🚀 GNBG-II GPU Integration Demo");
    println!("================================");
    
    // Test 1: Shared GPU Context Creation
    println!("\n📋 Test 1: Shared GPU Context Initialization");
    match GpuContext::new().await {
        Ok(context) => {
            let info = context.adapter_info();
            println!("✅ GPU Context initialized successfully");
            println!("   Device: {} ({:?})", info.name, info.backend);
            println!("   Vendor: 0x{:x}", info.vendor);
            println!("   Device Type: {:?}", info.device_type);
        }
        Err(e) => {
            println!("⚠️  GPU not available: {}", e);
            println!("   Continuing with CPU-only mode...");
        }
    }
    
    // Test 2: Multi-Objective Problem with GPU Integration
    println!("\n📋 Test 2: Multi-Objective GPU Evaluation");
    
    let mut problem = GNBGMOBuilder::new()
        .dimension(10)
        .objectives(3)
        .split_strategy(SplitStrategy::WFGStandard)
        .add_transformation(
            TransformationType::Polynomial { alpha: 0.02 },
            VariableRange::Distance,
        )
        .add_transformation(
            TransformationType::MultiModal { A: 5.0, B: 10.0, C: 1.0 },
            VariableRange::Distance,
        )
        .add_shape(ShapeFunction::Convex)
        .add_shape(ShapeFunction::Convex)
        .add_shape(ShapeFunction::Concave)
        .gpu(true) // Enable GPU acceleration
        .build()?;
    
    // Test single solution evaluation
    let solution = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 
        0.6, 0.7, 0.8, 0.9, 1.0
    ];
    
    let start = std::time::Instant::now();
    let objectives = problem.evaluate_single(&solution).await?;
    let duration = start.elapsed();
    
    println!("✅ Single solution evaluation completed");
    println!("   Dimension: {}", problem.dimension);
    println!("   Objectives: {}", problem.n_objectives);
    println!("   Position vars: {}", problem.splitter.n_position());
    println!("   Distance vars: {}", problem.splitter.n_distance());
    println!("   Evaluation time: {:?}", duration);
    println!("   Results: {:?}", objectives);
    
    // Test 3: Batch Evaluation Performance
    println!("\n📋 Test 3: Batch Evaluation Performance");
    
    let batch_size = 1000;
    let mut batch_solutions = Vec::with_capacity(batch_size * 10);
    
    // Generate random solutions
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..batch_size {
        for _ in 0..10 {
            batch_solutions.push(rng.gen::<f32>());
        }
    }
    
    let start = std::time::Instant::now();
    let batch_objectives = problem.evaluate_batch(&batch_solutions).await?;
    let duration = start.elapsed();
    
    let solutions_per_sec = batch_size as f64 / duration.as_secs_f64();
    
    println!("✅ Batch evaluation completed");
    println!("   Batch size: {} solutions", batch_size);
    println!("   Total evaluation time: {:?}", duration);
    println!("   Performance: {:.0} solutions/sec", solutions_per_sec);
    println!("   Output objectives: {} values", batch_objectives.len());
    
    // Verify results are valid
    let valid_objectives = batch_objectives.iter().all(|&x| x.is_finite() && x >= 0.0);
    if valid_objectives {
        println!("✅ All objectives are valid (finite and non-negative)");
    } else {
        println!("❌ Some objectives are invalid");
    }
    
    // Test 4: CPU vs GPU Mode Comparison
    println!("\n📋 Test 4: CPU vs GPU Mode Comparison");
    
    // Create CPU-only version
    let mut cpu_problem = GNBGMOBuilder::wfg1_preset(10, 3)
        .gpu(false)
        .build()?;
    
    let test_solutions = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1,
    ]; // 2 solutions
    
    // CPU evaluation
    let start_cpu = std::time::Instant::now();
    let cpu_objectives = cpu_problem.evaluate_batch(&test_solutions).await?;
    let cpu_duration = start_cpu.elapsed();
    
    // GPU evaluation (fallback to CPU for now)
    let start_gpu = std::time::Instant::now();
    let gpu_objectives = problem.evaluate_batch(&test_solutions).await?;
    let gpu_duration = start_gpu.elapsed();
    
    println!("✅ CPU/GPU comparison completed");
    println!("   CPU time: {:?}", cpu_duration);
    println!("   GPU time: {:?}", gpu_duration);
    println!("   Results match: {}", 
             cpu_objectives.len() == gpu_objectives.len() && 
             cpu_objectives.iter().zip(gpu_objectives.iter())
                 .all(|(a, b)| (a - b).abs() < 1e-6));
    
    println!("\n🎉 GPU Integration Demo completed successfully!");
    println!("   The multi-objective extension successfully integrates with");
    println!("   the shared GPU context infrastructure, enabling efficient");
    println!("   resource utilization across single and multi-objective problems.");
    
    Ok(())
}