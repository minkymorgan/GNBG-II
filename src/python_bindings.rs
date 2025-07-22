/*
 * GNBG-II GPU-Accelerated Implementation - Python Bindings
 * Copyright (C) 2025 Andrew Morgan <minkymorgan@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use std::sync::{Arc, Mutex};

use crate::data_structures::GNBGProblem;
use crate::file_loader::load_gnbg_problem;
use crate::cpu_reference::CPUEvaluator;
use crate::gpu_executor::GpuExecutor;

#[pyclass]
pub struct GNBGGpu {
    problem: GNBGProblem,
    cpu_evaluator: Arc<Mutex<CPUEvaluator>>,
    gpu_executor: Option<Arc<Mutex<GpuExecutor>>>,
    use_gpu: bool,
}

#[pymethods]
impl GNBGGpu {
    #[new]
    #[pyo3(signature = (problem_index, use_gpu=true))]
    fn new(problem_index: usize, use_gpu: bool) -> PyResult<Self> {
        if problem_index < 1 || problem_index > 24 {
            return Err(PyRuntimeError::new_err(format!(
                "Problem index must be between 1 and 24, got {}",
                problem_index
            )));
        }

        let problem = load_gnbg_problem(problem_index)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load problem: {}", e)))?;
        
        let cpu_evaluator = Arc::new(Mutex::new(CPUEvaluator::new(problem.clone())));
        
        let gpu_executor = if use_gpu {
            match pollster::block_on(GpuExecutor::new(&problem)) {
                Ok(executor) => Some(Arc::new(Mutex::new(executor))),
                Err(e) => {
                    eprintln!("Failed to initialize GPU: {}. Falling back to CPU.", e);
                    None
                }
            }
        } else {
            None
        };

        let has_gpu = gpu_executor.is_some();
        
        Ok(GNBGGpu {
            problem,
            cpu_evaluator,
            gpu_executor,
            use_gpu: use_gpu && has_gpu,
        })
    }

    fn fitness(&mut self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let solutions = x.as_array();
        let (n_solutions, dim) = solutions.dim();
        
        if dim != self.problem.dimension {
            return Err(PyRuntimeError::new_err(format!(
                "Solution dimension {} does not match problem dimension {}",
                dim, self.problem.dimension
            )));
        }

        let mut results = vec![0.0f64; n_solutions];

        if self.use_gpu && n_solutions >= 64 {
            // Use GPU for large batches
            if let Some(gpu_executor) = &self.gpu_executor {
                let solutions_vec: Vec<f32> = solutions.iter()
                    .map(|&v| v as f32)
                    .collect();

                let gpu_results = pollster::block_on(async {
                    gpu_executor.lock().unwrap()
                        .evaluate_batch(&solutions_vec)
                        .await
                })
                .map_err(|e| PyRuntimeError::new_err(format!("GPU evaluation failed: {}", e)))?;

                results = gpu_results.into_iter().map(|v| v as f64).collect();
            }
        } else {
            // Use CPU for small batches or when GPU is not available
            let mut cpu_eval = self.cpu_evaluator.lock().unwrap();
            for (i, solution) in solutions.rows().into_iter().enumerate() {
                let solution_vec: Vec<f64> = solution.to_vec();
                results[i] = cpu_eval.fitness(&solution_vec);
            }
        }

        Ok(results.into_pyarray(py).to_owned())
    }

    fn fitness_single(&mut self, x: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let solution = x.as_array();
        
        if solution.len() != self.problem.dimension {
            return Err(PyRuntimeError::new_err(format!(
                "Solution dimension {} does not match problem dimension {}",
                solution.len(), self.problem.dimension
            )));
        }

        let mut cpu_eval = self.cpu_evaluator.lock().unwrap();
        Ok(cpu_eval.fitness(&solution.to_vec()))
    }

    #[getter]
    fn max_evals(&self) -> usize {
        self.problem.max_evals
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.problem.dimension
    }

    #[getter]
    fn acceptance_threshold(&self) -> f64 {
        self.problem.acceptance_threshold
    }

    #[getter]
    fn optimum_value(&self) -> f64 {
        self.problem.optimum_value
    }

    #[getter]
    fn optimum_position(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(self.problem.optimum_position.clone().into_pyarray(py).to_owned())
    }

    #[getter]
    fn min_coordinate(&self) -> f64 {
        self.problem.min_coordinate
    }

    #[getter]
    fn max_coordinate(&self) -> f64 {
        self.problem.max_coordinate
    }

    #[getter]
    fn fe_count(&self) -> usize {
        self.cpu_evaluator.lock().unwrap().fe_count
    }

    #[getter]
    fn best_found_result(&self) -> f64 {
        self.cpu_evaluator.lock().unwrap().best_found_result
    }

    #[getter]
    fn acceptance_reach_point(&self) -> usize {
        self.cpu_evaluator.lock().unwrap().acceptance_reach_point.unwrap_or(usize::MAX)
    }

    #[getter]
    fn fe_history(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        let cpu_eval = self.cpu_evaluator.lock().unwrap();
        let history = cpu_eval.fe_history.clone();
        Ok(history.into_pyarray(py).to_owned())
    }

    fn reset(&mut self) {
        let mut cpu_eval = self.cpu_evaluator.lock().unwrap();
        cpu_eval.reset();
    }

    #[getter]
    fn using_gpu(&self) -> bool {
        self.use_gpu
    }

    fn __repr__(&self) -> String {
        format!(
            "GNBGGpu(dimension={}, max_evals={}, using_gpu={})",
            self.problem.dimension,
            self.problem.max_evals,
            self.use_gpu
        )
    }
}

#[pymodule]
fn gnbg_gpu(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GNBGGpu>()?;
    Ok(())
}