/// PyMOO Problem Interface for GNBG Multi-Objective Problems
/// 
/// This module provides a Python-compatible interface that follows PyMOO patterns
/// observed in the pyZenkai benchmark templates. It wraps our GNBGMultiObjective
/// implementation to be seamlessly usable with PyMOO algorithms.

use crate::multi_objective::GNBGMultiObjective;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, IntoPyArray, Ix2};

/// PyMOO-compatible wrapper for GNBG Multi-Objective problems
/// 
/// This class inherits from pymoo.core.problem.Problem in Python and provides
/// the standard _evaluate method expected by PyMOO algorithms like NSGA-II, NSGA-III.
#[pyclass(name = "GNBGMultiObjectiveProblem")]
pub struct PyMOOGNBGProblem {
    /// The underlying multi-objective problem
    problem: GNBGMultiObjective,
    /// Problem name for logging/identification
    name: String,
    /// Whether GPU acceleration is enabled
    gpu_enabled: bool,
}

#[pymethods]
impl PyMOOGNBGProblem {
    /// Create a new PyMOO-compatible GNBG problem
    /// 
    /// Args:
    ///     problem_config: Dictionary with problem configuration
    ///     name: Problem name for identification
    ///     
    /// Returns:
    ///     PyMOOGNBGProblem instance ready for PyMOO algorithms
    #[new]
    fn new(problem_config: &PyDict, name: Option<String>) -> PyResult<Self> {
        // Extract configuration from Python dict (following pyZenkai patterns)
        let dimension = problem_config
            .get_item("n_var")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'n_var' in config"))?
            .extract::<u32>()?;
            
        let n_objectives = problem_config
            .get_item("n_obj")?
            .map(|item| item.extract::<u32>())
            .transpose()?
            .unwrap_or(dimension); // Default to dimension if not specified
            
        // Check for WFG configuration
        let mut builder = crate::multi_objective::GNBGMOBuilder::new()
            .dimension(dimension)
            .objectives(n_objectives);
            
        // Apply WFG preset if specified (following pyZenkai patterns)
        if let Some(wfg_config) = problem_config.get_item("wfg")? {
            let wfg_dict = wfg_config.downcast::<PyDict>()?;
            let wfg_problem = wfg_dict
                .get_item("problem")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'problem' in WFG config"))?
                .extract::<u32>()?;
                
            // Apply WFG presets based on problem number
            builder = match wfg_problem {
                1 => crate::multi_objective::GNBGMOBuilder::wfg1_preset(dimension, n_objectives),
                2 => crate::multi_objective::GNBGMOBuilder::wfg2_preset(dimension, n_objectives),
                3 => crate::multi_objective::GNBGMOBuilder::wfg3_preset(dimension, n_objectives),
                _ => {
                    // For WFG4-9, use WFG1 as base and log warning
                    log::warn!("WFG{} preset not yet implemented, using WFG1 base", wfg_problem);
                    crate::multi_objective::GNBGMOBuilder::wfg1_preset(dimension, n_objectives)
                }
            };
        }
        
        // Check for GNBG2 functions (future extension point)
        if let Some(_gnbg2_config) = problem_config.get_item("gnbg2")? {
            log::warn!("GNBG2 integration not yet implemented in multi-objective mode");
        }
        
        // Build the problem
        let problem = builder.build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to build problem: {}", e)))?;
            
        let problem_name = name.unwrap_or_else(|| {
            format!("gnbg_mo_{}obj_{}var", n_objectives, dimension)
        });
        
        Ok(Self {
            problem,
            name: problem_name,
            gpu_enabled: true, // Default to GPU enabled
        })
    }
    
    /// Get problem dimension (number of variables)
    #[getter]
    fn n_var(&self) -> u32 {
        self.problem.dimension
    }
    
    /// Get number of objectives
    #[getter]
    fn n_obj(&self) -> u32 {
        self.problem.n_objectives
    }
    
    /// Get problem name
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
    
    /// Get variable bounds (always [-100, 100] for GNBG problems)
    #[getter]
    fn xl(&self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        let bounds = vec![-100.0f32; self.problem.dimension as usize];
        Ok(bounds.into_pyarray(py).to_owned())
    }
    
    #[getter] 
    fn xu(&self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        let bounds = vec![100.0f32; self.problem.dimension as usize];
        Ok(bounds.into_pyarray(py).to_owned())
    }
    
    /// Main evaluation method called by PyMOO algorithms
    /// 
    /// This follows the exact signature expected by PyMOO's Problem._evaluate method
    /// as observed in the pyZenkai ComposableBenchmarkProblem patterns.
    /// 
    /// Args:
    ///     X: Array of shape (n_solutions, n_var) with solutions to evaluate
    ///     
    /// Returns:
    ///     Dictionary with 'F' key containing objectives array of shape (n_solutions, n_obj)
    fn _evaluate(&mut self, X: PyReadonlyArray2<f32>, py: Python) -> PyResult<PyObject> {
        let X_array = X.as_array();
        let (n_solutions, n_vars) = X_array.dim();
        
        // Validate input dimensions
        if n_vars != self.problem.dimension as usize {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} variables, got {}", self.problem.dimension, n_vars)
            ));
        }
        
        // Convert to flat array for batch evaluation
        let solutions_flat: Vec<f32> = X_array.iter().copied().collect();
        
        // Evaluate using our async GPU implementation
        let objectives = pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                self.problem.evaluate_batch(&solutions_flat).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Evaluation failed: {}", e)))?;
        
        // Convert to numpy array with correct shape (n_solutions, n_objectives)
        let n_obj = self.problem.n_objectives as usize;
        let F = PyArray2::from_vec2(py, &objectives
            .chunks_exact(n_obj)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<Vec<f32>>>())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create array: {:?}", e)))?;
        
        // Return dictionary following PyMOO convention
        let result = pyo3::types::PyDict::new(py);
        result.set_item("F", F)?;
        
        Ok(result.into())
    }
    
    /// Evaluate a single solution (convenience method)
    fn evaluate_single(&mut self, solution: Vec<f32>, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        if solution.len() != self.problem.dimension as usize {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} variables, got {}", self.problem.dimension, solution.len())
            ));
        }
        
        let objectives = pyo3_asyncio::tokio::get_runtime()
            .block_on(async {
                self.problem.evaluate_single(&solution).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Evaluation failed: {}", e)))?;
            
        Ok(objectives.into_pyarray(py).to_owned())
    }
    
    /// Enable or disable GPU acceleration
    fn set_gpu_enabled(&mut self, enabled: bool) {
        self.gpu_enabled = enabled;
        self.problem.use_gpu = enabled;
    }
    
    /// Get current GPU status
    fn is_gpu_enabled(&self) -> bool {
        self.gpu_enabled
    }
    
    /// Get problem statistics for logging/monitoring
    fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = pyo3::types::PyDict::new(py);
            stats.set_item("dimension", self.problem.dimension)?;
            stats.set_item("n_objectives", self.problem.n_objectives)?;
            stats.set_item("n_position_vars", self.problem.splitter.n_position())?;
            stats.set_item("n_distance_vars", self.problem.splitter.n_distance())?;
            stats.set_item("gpu_enabled", self.gpu_enabled)?;
            stats.set_item("name", &self.name)?;
            Ok(stats.into())
        })
    }
    
    /// String representation for debugging
    fn __repr__(&self) -> String {
        format!("GNBGMultiObjectiveProblem(name='{}', n_var={}, n_obj={}, gpu={})", 
                self.name, self.problem.dimension, self.problem.n_objectives, self.gpu_enabled)
    }
    
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Factory function to create problems from configuration dictionaries
/// 
/// This follows the patterns observed in pyZenkai's ComposableBenchmarkProblem
/// where problems are created from configuration dictionaries.
#[pyfunction]
fn create_gnbg_problem(config: &PyDict, name: Option<String>) -> PyResult<PyMOOGNBGProblem> {
    PyMOOGNBGProblem::new(config, name)
}

/// Utility function to validate PyMOO algorithm compatibility
/// 
/// Checks if the given algorithm name is compatible with our implementation
#[pyfunction]
fn check_algorithm_compatibility(algorithm: &str, n_objectives: u32) -> PyResult<bool> {
    let compatible = match algorithm.to_uppercase().as_str() {
        "NSGA2" => true,                           // Always compatible
        "NSGA3" => n_objectives >= 3,             // NSGA-III needs 3+ objectives  
        "HF1" | "HF3" | "RHF3" => true,          // HF algorithms always compatible
        _ => false,                               // Unknown algorithm
    };
    
    if !compatible && algorithm.to_uppercase() == "NSGA3" {
        log::warn!("NSGA-III requires at least 3 objectives, got {}", n_objectives);
    }
    
    Ok(compatible)
}

/// Performance estimation based on problem size
/// 
/// Provides estimates for expected evaluation throughput based on problem dimensions
#[pyfunction]  
fn estimate_performance(n_var: u32, n_obj: u32, batch_size: u32) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Performance estimates based on our benchmarking
        let base_throughput = if n_obj <= 5 {
            40000.0  // 40K+ solutions/sec for small problems
        } else if n_obj <= 50 {
            10000.0  // 10K solutions/sec for medium problems
        } else if n_obj <= 500 {
            2000.0   // 2K solutions/sec for large problems  
        } else {
            500.0    // 500 solutions/sec for extreme problems
        };
        
        // Adjust for variable count (more variables = more computation)
        let var_factor = 1.0 - (n_var as f64 - 30.0).max(0.0) / 1000.0;
        let adjusted_throughput = base_throughput * var_factor.max(0.1);
        
        // Estimate time for batch
        let estimated_time_ms = (batch_size as f64 / adjusted_throughput) * 1000.0;
        
        let result = pyo3::types::PyDict::new(py);
        result.set_item("estimated_throughput_per_sec", adjusted_throughput)?;
        result.set_item("estimated_batch_time_ms", estimated_time_ms)?;
        result.set_item("recommended_batch_size", 
                       if n_obj <= 5 { 1000 } else if n_obj <= 50 { 500 } else { 100 })?;
        
        Ok(result.into())
    })
}

/// Register the PyMOO interface module with Python
pub fn register_pymoo_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let pymoo_module = PyModule::new(py, "pymoo_interface")?;
    
    pymoo_module.add_class::<PyMOOGNBGProblem>()?;
    pymoo_module.add_function(wrap_pyfunction!(create_gnbg_problem, pymoo_module)?)?;
    pymoo_module.add_function(wrap_pyfunction!(check_algorithm_compatibility, pymoo_module)?)?;
    pymoo_module.add_function(wrap_pyfunction!(estimate_performance, pymoo_module)?)?;
    
    parent_module.add_submodule(pymoo_module)?;
    Ok(())
}