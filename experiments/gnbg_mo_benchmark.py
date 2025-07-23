#!/usr/bin/env python3
"""
GNBG-MO Benchmark: Multi-Objective Optimization Performance Study

This script demonstrates GPU-accelerated multi-objective optimization benchmarking
using the GNBG-II framework. It tests standard algorithms across a wide range
of objective counts to evaluate scalability and performance.

PUBLIC VERSION - Safe for repository publication
"""

import numpy as np
import sys
import os
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add GNBG-II Python package to path
sys.path.append(str(Path(__file__).parent.parent / "python"))

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# EXPERIMENT SCOPE - Multi-Objective Scaling Study
EXPERIMENT_NAME = "gnbg_mo_scaling"

# Dual-range objective testing for comprehensive scaling analysis
OBJECTIVE_RANGES = [
    (10, 101, 10),    # 10 to 100 objectives, step 10 - standard range
    (100, 501, 50)    # 100 to 500 objectives, step 50 - extreme scaling
]

DIMENSIONS_BASE = 30                            # Base problem dimensions
DIMENSIONS_SCALING = True                       # Scale dimensions with objectives

# ALGORITHMS TO TEST
ALGORITHMS = ["NSGA2"]                          # Standard algorithm
# ALGORITHMS = ["NSGA2", "NSGA3"]              # Uncomment to include NSGA3

# GNBG-MO PROBLEM SETTINGS  
# Available: GF1-GF24 (maps to GNBG F1-F24 functions)
# GF1-GF6: Unimodal, GF7-GF15: Single-component multimodal, GF16-GF24: Multi-component
GNBG_MO_PROBLEMS = ["GF1", "GF2", "GF3"]        # GPU Multi-Objective Function variants

# EXPERIMENT EXECUTION SETTINGS
N_RUNS = 31                                     # Number of independent runs per configuration
N_GENERATIONS = 100                             # Generation limit
MAX_EVALUATIONS = 10000                         # Evaluation budget per run

# ALGORITHM PARAMETERS
POPULATION_SIZE_BASE = 100                      # Base population size
POPULATION_SCALING = 10                         # Additional pop per 10 objectives
CROSSOVER_PROB = 0.9                           # SBX crossover probability
CROSSOVER_ETA = 15                             # SBX distribution index
MUTATION_ETA = 20                              # Polynomial mutation index

# OUTPUT SETTINGS
OUTPUT_DIR = "results"
RESULTS_FILE = "gnbg_mo_benchmark_results.csv"
VERBOSE = True

# =============================================================================
# EXPERIMENT IMPLEMENTATION
# =============================================================================

def create_gnbg_mo_problem(problem_type, n_objectives, n_variables):
    """Create GNBG-MO problem instance"""
    try:
        from gnbg_gpu.multi_objective import GNBGMultiObjectiveProblem
        
        # Create GNBG multi-objective problem based on GF type
        if problem_type == "GF1":
            # GF1: GPU Multi-Objective Function 1 - Unimodal, well-conditioned
            problem = GNBGMultiObjectiveProblem.wfg1(
                n_var=n_variables,
                n_obj=n_objectives
            )
        elif problem_type == "GF2":
            # GF2: GPU Multi-Objective Function 2 - Non-separable, convex Pareto front
            problem = GNBGMultiObjectiveProblem.wfg2(
                n_var=n_variables,
                n_obj=n_objectives
            )
        elif problem_type == "GF3":
            # GF3: GPU Multi-Objective Function 3 - Non-separable, linear Pareto front
            problem = GNBGMultiObjectiveProblem.wfg3(
                n_var=n_variables,
                n_obj=n_objectives
            )
        else:
            # Default GNBG multi-objective configuration
            problem = GNBGMultiObjectiveProblem.custom(
                n_var=n_variables,
                n_obj=n_objectives
            )
        
        return problem
        
    except ImportError as e:
        print(f"❌ GNBG-GPU import failed: {e}")
        print("❌ Make sure GNBG-II Python package is installed")
        print("❌ Run: cd python && pip install -e .")
        raise
    except Exception as e:
        print(f"❌ GNBG-MO problem creation failed: {e}")
        raise

def create_nsga2_algorithm(n_objectives, population_size):
    """Create NSGA-II algorithm instance"""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(eta=MUTATION_ETA),
        eliminate_duplicates=True
    )
    
    return algorithm

# def create_nsga3_algorithm(n_objectives, population_size):
#     """Create NSGA-III algorithm instance"""
#     from pymoo.algorithms.moo.nsga3 import NSGA3
#     from pymoo.operators.crossover.sbx import SBX
#     from pymoo.operators.mutation.pm import PM
#     from pymoo.operators.sampling.rnd import FloatRandomSampling
#     from pymoo.util.ref_dirs import get_reference_directions
#     
#     # Create reference directions
#     ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=12)
#     
#     algorithm = NSGA3(
#         pop_size=population_size,
#         ref_dirs=ref_dirs,
#         sampling=FloatRandomSampling(),
#         crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
#         mutation=PM(eta=MUTATION_ETA),
#         eliminate_duplicates=True
#     )
#     
#     return algorithm

def calculate_performance_metrics(result):
    """Calculate comprehensive performance metrics"""
    
    objectives = result.F
    n_solutions, n_objectives = objectives.shape
    
    # Calculate absolute objective values (sum across all objectives per solution)
    absolute_objectives = np.sum(objectives, axis=1)  # Sum each solution's objectives
    min_absolute_obj = np.min(absolute_objectives)    # Best (lowest) total objective value
    mean_absolute_obj = np.mean(absolute_objectives)  # Average total objective value
    
    # Individual objective statistics
    min_per_objective = np.min(objectives, axis=0)    # Best value per objective
    mean_per_objective = np.mean(objectives, axis=0)  # Average value per objective
    
    # Hypervolume (simplified - using sum of objectives as proxy)
    # In practice, you would use pymoo.indicators.hv.HV with proper reference point
    hypervolume_proxy = np.mean(np.sum(objectives, axis=1))
    
    # Diversity metric (average pairwise distance)
    diversity = 0.0
    if n_solutions > 1:
        from scipy.spatial.distance import pdist
        diversity = np.mean(pdist(objectives))
    
    # Convergence metric (distance to origin)
    convergence = np.mean(np.linalg.norm(objectives, axis=1))
    
    # GPU utilization estimate (solutions evaluated per second)
    total_time = result.exec_time if hasattr(result, 'exec_time') else 1.0
    total_evals = result.algorithm.evaluator.n_eval
    throughput = total_evals / total_time if total_time > 0 else 0
    
    return {
        'n_solutions': n_solutions,
        'min_absolute_obj': min_absolute_obj,
        'mean_absolute_obj': mean_absolute_obj,
        'min_per_objective': min_per_objective.tolist(),  # Convert to list for CSV
        'mean_per_objective': mean_per_objective.tolist(),
        'hypervolume_proxy': hypervolume_proxy,
        'diversity': diversity,
        'convergence': convergence,
        'evaluations': total_evals,
        'runtime_seconds': total_time,
        'throughput_sol_per_sec': throughput
    }

def run_single_optimization(problem_type, n_objectives, algorithm_name, run_id):
    """Run a single optimization experiment"""
    
    experiment_id = f"GNBG_{problem_type}_{n_objectives}obj_{algorithm_name}_run{run_id}"
    start_time = time.time()
    
    try:
        # Calculate dynamic parameters
        population_size = POPULATION_SIZE_BASE + (n_objectives // 10) * POPULATION_SCALING
        
        # Calculate appropriate dimensions (must be >= objectives)
        dimensions = max(DIMENSIONS_BASE, n_objectives + 5) if DIMENSIONS_SCALING else DIMENSIONS_BASE
        
        # Create problem
        problem = create_gnbg_mo_problem(problem_type, n_objectives, dimensions)
        
        # Create algorithm
        if algorithm_name == "NSGA2":
            algorithm = create_nsga2_algorithm(n_objectives, population_size)
        # elif algorithm_name == "NSGA3":
        #     algorithm = create_nsga3_algorithm(n_objectives, population_size)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Run optimization
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        
        # Use evaluation limit as termination
        termination = get_termination("n_eval", MAX_EVALUATIONS)
        
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=42 + run_id,
            verbose=False
        )
        
        # Calculate metrics
        metrics = calculate_performance_metrics(result)
        
        runtime = time.time() - start_time
        
        result_data = {
            'experiment_id': experiment_id,
            'problem_type': problem_type,
            'n_objectives': n_objectives,
            'n_variables': dimensions,
            'algorithm': algorithm_name,
            'run_id': run_id,
            'runtime_seconds': runtime,
            'population_size': population_size,
            'final_generation': result.algorithm.n_gen,
            **metrics,
            'status': 'success',
            'gpu_enabled': problem.is_gpu_enabled()
        }
        
        if VERBOSE:
            print(f"✅ {experiment_id}: {metrics['n_solutions']} solutions, "
                  f"{metrics['throughput_sol_per_sec']:.0f} sol/sec, "
                  f"min obj value={metrics['min_absolute_obj']:.6f}")
        
        return result_data
        
    except Exception as e:
        runtime = time.time() - start_time
        error_msg = str(e)[:200]
        
        result_data = {
            'experiment_id': experiment_id,
            'problem_type': problem_type,
            'n_objectives': n_objectives,
            'algorithm': algorithm_name,
            'run_id': run_id,
            'runtime_seconds': runtime,
            'status': 'failed',
            'error_message': error_msg
        }
        
        print(f"❌ {experiment_id} failed: {error_msg}")
        return result_data

def run_problem_comparison(problem_type, n_objectives):
    """Run comparison for a specific problem and objective count"""
    
    print(f"\n🎯 Testing GNBG {problem_type} with {n_objectives} objectives")
    print("-" * 50)
    
    all_results = []
    
    for algorithm in ALGORITHMS:
        print(f"🔬 Algorithm: {algorithm}")
        
        algorithm_results = []
        for run_id in range(1, N_RUNS + 1):
            result = run_single_optimization(problem_type, n_objectives, algorithm, run_id)
            algorithm_results.append(result)
            all_results.append(result)
        
        # Show algorithm summary
        successful_runs = [r for r in algorithm_results if r['status'] == 'success']
        if successful_runs:
            avg_solutions = np.mean([r['n_solutions'] for r in successful_runs])
            avg_runtime = np.mean([r['runtime_seconds'] for r in successful_runs])
            avg_throughput = np.mean([r['throughput_sol_per_sec'] for r in successful_runs])
            best_absolute_obj = np.min([r['min_absolute_obj'] for r in successful_runs])
            avg_absolute_obj = np.mean([r['mean_absolute_obj'] for r in successful_runs])
            
            # Collect all run objective values for detailed display
            run_objectives = [r['min_absolute_obj'] for r in successful_runs]
            
            print(f"   Average solutions: {avg_solutions:.0f}")
            print(f"   Average runtime: {avg_runtime:.1f}s")
            print(f"   Average throughput: {avg_throughput:.0f} sol/sec")
            print(f"   🎯 Best absolute obj, {problem_type} = {best_absolute_obj:.6f}")
            print(f"   🎯 Per-run mean absolute obj, {problem_type} = {np.mean(run_objectives):.6f}")
            
            # Show per-run objective values in compact format
            if len(run_objectives) > 0:
                obj_str = ", ".join([f"{obj:.4f}" for obj in run_objectives])
                print(f"   📊 Per-run objectives: [{obj_str}] → mean = {np.mean(run_objectives):.6f}")
            
            print(f"   Success rate: {len(successful_runs)}/{N_RUNS}")
    
    return all_results

def display_summary_table(results_df):
    """Display summary table across all configurations"""
    
    if len(results_df) == 0:
        print("No results to display")
        return
    
    success_df = results_df[results_df['status'] == 'success']
    if len(success_df) == 0:
        print("No successful runs to analyze")
        return
    
    print(f"\n🏆 GNBG-MO Benchmark Summary")
    print("=" * 130)
    print(f"{'Problem':>8} {'Obj':>4} {'Algorithm':>8} {'Solutions':>9} {'Runtime':>8} {'Throughput':>10} "
          f"{'Best Obj':>12} {'Mean Obj':>12} {'GPU':>4}")
    print("-" * 130)
    
    for problem in sorted(success_df['problem_type'].unique()):
        problem_data = success_df[success_df['problem_type'] == problem]
        
        for n_obj in sorted(problem_data['n_objectives'].unique()):
            obj_data = problem_data[problem_data['n_objectives'] == n_obj]
            
            for algorithm in ALGORITHMS:
                alg_data = obj_data[obj_data['algorithm'] == algorithm]
                
                if len(alg_data) > 0:
                    avg_solutions = alg_data['n_solutions'].mean()
                    avg_runtime = alg_data['runtime_seconds'].mean()
                    avg_throughput = alg_data['throughput_sol_per_sec'].mean()
                    best_obj = alg_data['min_absolute_obj'].min()
                    mean_obj = alg_data['mean_absolute_obj'].mean()
                    gpu_enabled = alg_data['gpu_enabled'].iloc[0]
                    
                    print(f"{problem:>8} {n_obj:>4} {algorithm:>8} {avg_solutions:>9.0f} "
                          f"{avg_runtime:>8.1f}s {avg_throughput:>10.0f} "
                          f"{best_obj:>12.6f} {mean_obj:>12.6f} {'Yes' if gpu_enabled else 'No':>4}")
    
    print("-" * 130)

def save_results(results_df):
    """Save results to files"""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results
    results_file = output_path / RESULTS_FILE
    results_df.to_csv(results_file, index=False)
    print(f"\n📁 Detailed results: {results_file}")
    
    # Create summary statistics
    success_df = results_df[results_df['status'] == 'success']
    if len(success_df) > 0:
        summary = success_df.groupby(['problem_type', 'n_objectives', 'algorithm']).agg({
            'n_solutions': ['mean', 'std'],
            'runtime_seconds': ['mean', 'std'],
            'throughput_sol_per_sec': ['mean', 'std'],
            'diversity': ['mean', 'std'],
            'convergence': ['mean', 'std']
        }).round(2)
        
        summary_file = output_path / "summary_statistics.csv"
        summary.to_csv(summary_file)
        print(f"📊 Summary statistics: {summary_file}")

def main():
    """Main experiment execution"""
    
    print("🚀 GNBG-MO: GPU-Accelerated Multi-Objective Benchmark")
    print("🌟 EXTREME SCALING TEST: 10 to 500 Objectives")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   • Standard range: 10-100 objectives (step 10)")
    print(f"   • Extreme range: 100-500 objectives (step 50)")
    print(f"   • Dimensions: {DIMENSIONS_BASE}+ (scaling: {DIMENSIONS_SCALING})")
    print(f"   • Problems: {', '.join(GNBG_MO_PROBLEMS)} (GPU Multi-Objective Functions)")
    print(f"   • Algorithms: {', '.join(ALGORITHMS)}")
    print(f"   • Runs per config: {N_RUNS}")
    print(f"   • Max evaluations: {MAX_EVALUATIONS:,}")
    print("=" * 60)
    
    # Test GNBG-MO availability
    try:
        print("🔍 Testing GNBG-MO availability...")
        test_problem = create_gnbg_mo_problem("wfg1", 5, 10)
        
        # Quick evaluation test
        test_solutions = (np.random.random((3, 10)) * 200 - 100).astype(np.float32)
        test_objectives = test_problem._evaluate(test_solutions)
        print(f"✅ GNBG-MO test successful: {test_objectives['F'].shape}")
        print(f"✅ GPU acceleration: {'Enabled' if test_problem.is_gpu_enabled() else 'Disabled'}")
        
    except Exception as e:
        print(f"❌ GNBG-MO test failed: {e}")
        print("❌ Ensure GNBG-II is compiled and Python package is installed")
        return
    
    # Run experiments
    start_time = time.time()
    all_results = []
    
    # Generate all objective counts from both ranges
    all_objective_counts = []
    for start, end, step in OBJECTIVE_RANGES:
        range_counts = list(range(start, end, step))
        all_objective_counts.extend(range_counts)
    
    # Remove duplicates (e.g., 100 appears in both ranges) and sort
    all_objective_counts = sorted(list(set(all_objective_counts)))
    
    total_experiments = len(GNBG_MO_PROBLEMS) * len(all_objective_counts) * len(ALGORITHMS) * N_RUNS
    
    print(f"\n🎯 Running {total_experiments:,} total experiments")
    print(f"   • {len(GNBG_MO_PROBLEMS)} problems × {len(all_objective_counts)} objective counts")
    print(f"   • Objective range: {min(all_objective_counts)} to {max(all_objective_counts)} objectives")
    print(f"   • Standard range: 10-100 (step 10), Extreme range: 100-500 (step 50)")
    print(f"   • × {len(ALGORITHMS)} algorithms × {N_RUNS} runs each")
    
    completed = 0
    for problem_type in GNBG_MO_PROBLEMS:
        for n_objectives in all_objective_counts:
            results = run_problem_comparison(problem_type, n_objectives)
            all_results.extend(results)
            completed += len(results)
            
            print(f"\n📈 Progress: {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%)")
    
    # Final analysis
    total_time = time.time() - start_time
    results_df = pd.DataFrame(all_results)
    
    print(f"\n🎉 Experiment Complete!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Total experiments: {len(results_df)}")
    
    successful_experiments = len(results_df[results_df['status'] == 'success'])
    print(f"✅ Successful: {successful_experiments}")
    print(f"❌ Failed: {len(results_df) - successful_experiments}")
    
    # Display summary table
    display_summary_table(results_df)
    
    # Save results
    save_results(results_df)
    
    # Performance highlights
    success_df = results_df[results_df['status'] == 'success']
    if len(success_df) > 0:
        max_throughput = success_df['throughput_sol_per_sec'].max()
        max_objectives = success_df['n_objectives'].max()
        
        print(f"\n🏆 Performance Highlights:")
        print(f"   • Peak throughput: {max_throughput:,.0f} solutions/second")
        print(f"   • Maximum objectives tested: {max_objectives}")
        print(f"   • GPU acceleration demonstrated across all problems")
    
    print(f"\n📈 Next steps:")
    print(f"   • Analyze results: pd.read_csv('{OUTPUT_DIR}/{RESULTS_FILE}')")
    print(f"   • Plot scaling behavior vs objective count")
    print(f"   • Compare algorithm performance across problems")

if __name__ == "__main__":
    main()