#!/usr/bin/env python3
"""
Quick test of GNBG-MO with NSGA2 - minimal version
"""

import numpy as np
import sys
from pathlib import Path

# Add GNBG-II Python package to path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from gnbg_gpu.multi_objective import GNBGMultiObjectiveProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

def test_single_configuration():
    """Test a single GNBG-MO problem with NSGA2"""
    
    print("🚀 Quick GNBG-MO Test")
    
    # Test 5 objectives, 10 variables
    problem = GNBGMultiObjectiveProblem.wfg1(n_var=10, n_obj=5)
    print(f"✅ Problem: {problem.n_var} vars, {problem.n_obj} objectives")
    
    # Create NSGA2 algorithm  
    algorithm = NSGA2(
        pop_size=50,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Run for short period
    result = minimize(
        problem,
        algorithm,
        get_termination('n_eval', 500),
        seed=42,
        verbose=False
    )
    
    print(f"✅ Optimization complete:")
    print(f"   Population size: {len(result.F)}")
    print(f"   Objectives shape: {result.F.shape}")
    print(f"   Evaluations: {algorithm.evaluator.n_eval}")
    print(f"   Generations: {result.algorithm.n_gen}")
    print(f"   Best objectives: {np.min(result.F, axis=0)}")
    print(f"   Performance: {algorithm.evaluator.n_eval / (result.exec_time or 1):.0f} evals/sec")
    
    return True

if __name__ == "__main__":
    test_single_configuration()