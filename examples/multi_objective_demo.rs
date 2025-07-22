/// Multi-Objective GNBG Demo (Rust-only, no Python dependencies)
/// 
/// This demonstrates the multi-objective engine without Python bindings,
/// so it can run in any environment without conda/Python conflicts.

use gnbg_gpu::multi_objective::*;
use gnbg_gpu::multi_objective::transformations::VariableRange;
use gnbg_gpu::multi_objective::position_distance::OptimizationTarget;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("🚀 GNBG-II Multi-Objective Engine Demo");
    println!("=====================================");
    
    // Demo 1: Basic Multi-Objective Problem Creation
    println!("\n📋 Demo 1: Problem Creation & Configuration");
    println!("{}", repeat_char('-', 45));
    
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
        .gpu(true)
        .build()?;
    
    println!("✅ Created multi-objective problem:");
    println!("   Total variables: {}", problem.dimension);
    println!("   Objectives: {}", problem.n_objectives);
    println!("   Position variables: {}", problem.splitter.n_position());
    println!("   Distance variables: {}", problem.splitter.n_distance());
    println!("   GPU enabled: {}", problem.use_gpu);
    
    // Demo 2: Single Solution Evaluation
    println!("\n📋 Demo 2: Single Solution Evaluation");
    println!("{}", repeat_char('-', 38));
    
    let solution = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 
        0.6, 0.7, 0.8, 0.9, 1.0
    ];
    
    println!("Evaluating solution: {:?}...", &solution[..5]);
    let start = std::time::Instant::now();
    let objectives = problem.evaluate_single(&solution).await?;
    let duration = start.elapsed();
    
    println!("✅ Single evaluation completed:");
    println!("   Time: {:?}", duration);
    println!("   Objectives: {:?}", objectives);
    println!("   All finite: {}", objectives.iter().all(|x| x.is_finite()));
    
    // Demo 3: Batch Performance Test
    println!("\n📋 Demo 3: Batch Performance Evaluation");
    println!("{}", repeat_char('-', 40));
    
    let batch_sizes = vec![100, 500, 1000, 5000];
    
    for &batch_size in &batch_sizes {
        println!("\nTesting batch size: {} solutions", batch_size);
        
        // Generate random solutions
        let mut batch_solutions = Vec::with_capacity(batch_size * 10);
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..batch_size {
            for _ in 0..10 {
                batch_solutions.push(rng.gen::<f32>() * 200.0 - 100.0); // [-100, 100] range
            }
        }
        
        let start = std::time::Instant::now();
        let batch_objectives = problem.evaluate_batch(&batch_solutions).await?;
        let duration = start.elapsed();
        
        let solutions_per_sec = batch_size as f64 / duration.as_secs_f64();
        let expected_output_len = batch_size * problem.n_objectives as usize;
        
        println!("   ⏱️  Time: {:?}", duration);
        println!("   📊 Throughput: {:.0} solutions/sec", solutions_per_sec);
        println!("   ✅ Output size: {} (expected: {})", 
                 batch_objectives.len(), expected_output_len);
        println!("   🎯 Validation: {}", 
                 if batch_objectives.len() == expected_output_len { "PASS" } else { "FAIL" });
    }
    
    // Demo 4: WFG Preset Configurations
    println!("\n📋 Demo 4: WFG Preset Configurations");
    println!("{}", repeat_char('-', 37));
    
    let wfg_presets = vec![
        ("WFG1", 1),
        ("WFG2", 2), 
        ("WFG3", 3),
    ];
    
    for (name, wfg_num) in wfg_presets {
        let mut preset_problem = match wfg_num {
            1 => GNBGMOBuilder::wfg1_preset(12, 4),
            2 => GNBGMOBuilder::wfg2_preset(12, 4),
            3 => GNBGMOBuilder::wfg3_preset(12, 4),
            _ => continue,
        }.build()?;
        
        // Test preset with small batch
        let test_solutions: Vec<f32> = (0..48) // 4 solutions * 12 variables
            .map(|i| (i as f32) * 0.1 - 2.0)
            .collect();
            
        let start = std::time::Instant::now();
        let preset_objectives = preset_problem.evaluate_batch(&test_solutions).await?;
        let duration = start.elapsed();
        
        println!("✅ {} preset (12D/4obj):", name);
        println!("   Evaluation time: {:?}", duration);
        println!("   Output objectives: {}", preset_objectives.len());
        println!("   Range: [{:.3}, {:.3}]", 
                 preset_objectives.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                 preset_objectives.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    }
    
    // Demo 5: Scalability Test
    println!("\n📋 Demo 5: Multi-Objective Scalability");
    println!("{}", repeat_char('-', 39));
    
    let scalability_tests = vec![
        (10, 2),   // Small: 10D, 2 objectives
        (20, 5),   // Medium: 20D, 5 objectives  
        (30, 10),  // Large: 30D, 10 objectives
        (50, 20),  // Very Large: 50D, 20 objectives
    ];
    
    for (dimensions, objectives) in scalability_tests {
        println!("\nTesting {}D problem with {} objectives:", dimensions, objectives);
        
        let scale_problem = GNBGMOBuilder::new()
            .dimension(dimensions)
            .objectives(objectives)
            .split_strategy(SplitStrategy::WFGStandard)
            .add_transformation(
                TransformationType::Polynomial { alpha: 0.02 },
                VariableRange::Distance,
            )
            .gpu(true)
            .build();
            
        match scale_problem {
            Ok(mut prob) => {
                // Test with 100 solutions
                let test_batch: Vec<f32> = (0..100 * dimensions as usize)
                    .map(|i| (i as f32 * 0.01) % 200.0 - 100.0)
                    .collect();
                    
                let start = std::time::Instant::now();
                match prob.evaluate_batch(&test_batch).await {
                    Ok(results) => {
                        let duration = start.elapsed();
                        let throughput = 100.0 / duration.as_secs_f64();
                        println!("   ✅ Success: {:.0} sol/sec, {} objectives computed", 
                                throughput, results.len());
                    }
                    Err(e) => println!("   ❌ Evaluation failed: {}", e),
                }
            }
            Err(e) => println!("   ❌ Problem creation failed: {}", e),
        }
    }
    
    // Demo 6: Position-Distance Variable Analysis
    println!("\n📋 Demo 6: Variable Splitting Analysis");
    println!("{}", repeat_char('-', 37));
    
    let splitting_strategies = vec![
        ("WFG Standard", SplitStrategy::WFGStandard),
        ("Adaptive Convergence", SplitStrategy::Adaptive { 
            min_k: 4, 
            max_k: 6, 
            optimization_target: OptimizationTarget::ConvergenceSpeed 
        }),
        ("Adaptive Diversity", SplitStrategy::Adaptive { 
            min_k: 3, 
            max_k: 7, 
            optimization_target: OptimizationTarget::FrontDiversity 
        }),
    ];
    
    for (name, strategy) in splitting_strategies {
        let split_problem = GNBGMOBuilder::new()
            .dimension(20)
            .objectives(5)
            .split_strategy(strategy)
            .build()?;
            
        println!("Strategy: {}:", name);
        println!("   Position vars: {}", split_problem.splitter.n_position());
        println!("   Distance vars: {}", split_problem.splitter.n_distance());
        println!("   Ratio: {:.2}", 
                 split_problem.splitter.n_position() as f32 / 20.0);
    }
    
    println!("\n🎉 Multi-Objective Demo completed successfully!");
    println!("   The GNBG-II multi-objective extension provides:");
    println!("   • High-performance GPU-accelerated evaluation");
    println!("   • Flexible WFG-style problem configuration");
    println!("   • Scalable architecture for many-objective optimization");
    println!("   • Position-distance variable paradigm support");
    println!("   • Ready for PyMOO algorithm integration");
    
    Ok(())
}

fn repeat_char(ch: char, count: usize) -> String {
    std::iter::repeat(ch).take(count).collect()
}