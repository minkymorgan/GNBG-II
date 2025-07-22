/// Shape functions for defining Pareto front geometry
/// 
/// These functions determine the shape of the Pareto front in objective space

pub mod convex;
pub mod concave;
pub mod linear;
pub mod mixed;
pub mod cache;

use crate::multi_objective::{GNBGMOError, Result};

/// Types of shape functions available
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum ShapeFunction {
    /// Linear Pareto front
    Linear,
    /// Convex Pareto front
    Convex,
    /// Concave Pareto front  
    Concave,
    /// Mixed convex/concave regions
    Mixed { transition_points: Vec<f32> },
    /// Disconnected Pareto front with gaps
    Disconnected { gaps: Vec<(f32, f32)> },
}

/// Shape function executor that can apply shape functions to objectives
pub struct ShapeFunctionExecutor {
    /// Shape functions for each objective
    pub shape_functions: Vec<ShapeFunction>,
    /// Number of objectives
    pub n_objectives: u32,
    /// Shape function cache
    pub cache: Option<cache::ShapeFunctionCache>,
}

impl ShapeFunctionExecutor {
    /// Create a new shape function executor
    pub fn new(shape_functions: Vec<ShapeFunction>) -> Self {
        Self {
            n_objectives: shape_functions.len() as u32,
            shape_functions,
            cache: None,
        }
    }
    
    /// Enable caching for shape functions
    pub fn with_cache(mut self /* device: &wgpu::Device */) -> Self {
        // TODO: Initialize cache with GPU device
        // self.cache = Some(cache::ShapeFunctionCache::new(device));
        self
    }
    
    /// Apply shape functions to position variables (CPU implementation)
    pub fn apply_cpu(
        &self,
        position_vars: &[f32],
        n_position: u32,
    ) -> Result<Vec<f32>> {
        let n_solutions = position_vars.len() / n_position as usize;
        let mut objectives = Vec::with_capacity(n_solutions * self.n_objectives as usize);
        
        for sol_idx in 0..n_solutions {
            let sol_start = sol_idx * n_position as usize;
            let sol_end = sol_start + n_position as usize;
            let position = &position_vars[sol_start..sol_end];
            
            // Apply each shape function
            for (obj_idx, shape_fn) in self.shape_functions.iter().enumerate() {
                let obj_value = self.compute_shape_value(shape_fn, position, obj_idx)?;
                objectives.push(obj_value);
            }
        }
        
        Ok(objectives)
    }
    
    /// Compute shape function value for a single objective
    fn compute_shape_value(
        &self,
        shape_fn: &ShapeFunction,
        position: &[f32],
        obj_idx: usize,
    ) -> Result<f32> {
        match shape_fn {
            ShapeFunction::Linear => {
                Ok(linear::shape_linear(position, obj_idx, self.n_objectives))
            },
            ShapeFunction::Convex => {
                Ok(convex::shape_convex(position, obj_idx, self.n_objectives))
            },
            ShapeFunction::Concave => {
                Ok(concave::shape_concave(position, obj_idx, self.n_objectives))
            },
            ShapeFunction::Mixed { transition_points } => {
                Ok(mixed::shape_mixed(position, obj_idx, self.n_objectives, transition_points))
            },
            ShapeFunction::Disconnected { gaps } => {
                Ok(mixed::shape_disconnected(position, obj_idx, self.n_objectives, gaps))
            },
        }
    }
    
    /// Apply shape functions using GPU (cached)
    /// TODO: Implement GPU execution
    pub async fn apply_gpu(
        &self,
        _position_vars: &[f32],
        _n_position: u32,
    ) -> Result<Vec<f32>> {
        Err(GNBGMOError::GpuExecutionError(
            "GPU shape functions not yet implemented".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shape_function_executor_creation() {
        let shapes = vec![
            ShapeFunction::Convex,
            ShapeFunction::Concave,
            ShapeFunction::Linear,
        ];
        
        let executor = ShapeFunctionExecutor::new(shapes);
        assert_eq!(executor.n_objectives, 3);
    }
    
    #[test]
    fn test_shape_function_executor_cpu() {
        let shapes = vec![
            ShapeFunction::Linear,
            ShapeFunction::Linear,
        ];
        
        let executor = ShapeFunctionExecutor::new(shapes);
        
        // Test with simple position variables
        let position_vars = vec![0.5, 0.5]; // One solution, 2 position variables
        let result = executor.apply_cpu(&position_vars, 2).unwrap();
        
        assert_eq!(result.len(), 2); // 2 objectives
        
        // All values should be in valid range
        for &val in &result {
            assert!(val >= 0.0);
        }
    }
}