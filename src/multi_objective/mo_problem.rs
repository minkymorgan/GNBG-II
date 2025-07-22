/// Multi-objective GNBG problem implementation
/// 
/// Main struct for evaluating multi-objective optimization problems

use crate::multi_objective::{
    GNBGMOError, Result,
    position_distance::PositionDistanceSplitter,
    transformations::TransformationPipeline,
    shapes::ShapeFunctionExecutor,
};
use crate::gpu_context::GpuContext;

/// Multi-objective GNBG problem
pub struct GNBGMultiObjective {
    /// Problem dimension (total number of variables)
    pub dimension: u32,
    /// Number of objectives
    pub n_objectives: u32,
    /// Position-distance variable splitter
    pub splitter: PositionDistanceSplitter,
    /// Transformation pipeline for distance variables
    pub transformation_pipeline: TransformationPipeline,
    /// Shape function executor for position variables
    pub shape_executor: ShapeFunctionExecutor,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Shared GPU context for acceleration
    pub gpu_context: Option<GpuContext>,
}

impl GNBGMultiObjective {
    /// Initialize GPU context for acceleration
    /// 
    /// This should be called once before any GPU operations.
    /// Reuses the shared GPU context infrastructure from the main GNBG executor.
    pub async fn initialize_gpu(&mut self) -> Result<()> {
        if self.use_gpu && self.gpu_context.is_none() {
            let context = GpuContext::new().await
                .map_err(|e| GNBGMOError::GpuExecutionError(
                    format!("Failed to initialize GPU context: {}", e)
                ))?;
            
            log::info!("Initialized GPU context for multi-objective evaluation: {}", 
                      context.adapter_info().name);
                      
            self.gpu_context = Some(context);
        }
        Ok(())
    }
    
    /// Get the GPU context, initializing if necessary
    pub async fn get_gpu_context(&mut self) -> Result<Option<&GpuContext>> {
        if self.use_gpu {
            if self.gpu_context.is_none() {
                self.initialize_gpu().await?;
            }
            Ok(self.gpu_context.as_ref())
        } else {
            Ok(None)
        }
    }
    /// Evaluate a batch of solutions
    /// 
    /// # Arguments
    /// * `solutions` - Flattened array of solutions [sol1_var1, sol1_var2, ..., sol2_var1, ...]
    /// 
    /// # Returns  
    /// * Flattened array of objectives [sol1_obj1, sol1_obj2, ..., sol2_obj1, ...]
    pub async fn evaluate_batch(&mut self, solutions: &[f32]) -> Result<Vec<f32>> {
        let n_solutions = solutions.len() / self.dimension as usize;
        
        if solutions.len() != (n_solutions * self.dimension as usize) {
            return Err(GNBGMOError::InvalidConfiguration(
                "Solution array size mismatch".to_string()
            ));
        }
        
        if self.use_gpu {
            self.evaluate_batch_gpu(solutions, n_solutions).await
        } else {
            self.evaluate_batch_cpu(solutions, n_solutions)
        }
    }
    
    /// GPU-accelerated batch evaluation
    async fn evaluate_batch_gpu(&mut self, solutions: &[f32], n_solutions: usize) -> Result<Vec<f32>> {
        // Ensure GPU context is initialized and get a reference
        self.initialize_gpu().await?;
        let gpu_context = self.gpu_context.as_ref()
            .ok_or_else(|| GNBGMOError::GpuExecutionError(
                "GPU context not available".to_string()
            ))?;
            
        // Step 1: Split position and distance variables
        let (position_vars, distance_vars) = self.splitter.split_variables(solutions)?;
        let n_position = self.splitter.n_position();
        
        // Step 2: Transform distance variables (GPU)
        let transformed_distances = if !distance_vars.is_empty() {
            self.transformation_pipeline
                .apply_gpu_batch(&distance_vars, n_position, gpu_context)
                .await?
        } else {
            // No distance variables, create dummy transformed distances
            vec![1.0; n_solutions] // Default scaling factor
        };
        
        // Step 3: Combine transformed distances (reduction operation)
        let combined_distances = self.combine_distance_variables(&transformed_distances, n_solutions)?;
        
        // Step 4: Apply shape functions to position variables (GPU)
        let raw_objectives = self.shape_executor
            .apply_gpu(&position_vars, n_position, gpu_context)
            .await?;
        
        // Step 5: Scale objectives by distance variables
        let scaled_objectives = self.scale_objectives(&raw_objectives, &combined_distances)?;
        
        Ok(scaled_objectives)
    }
    
    /// CPU batch evaluation (fallback/debug)
    fn evaluate_batch_cpu(&self, solutions: &[f32], n_solutions: usize) -> Result<Vec<f32>> {
        // Step 1: Split position and distance variables
        let (position_vars, distance_vars) = self.splitter.split_variables(solutions)?;
        
        // Step 2: Transform distance variables (CPU)
        let mut transformed_distances = distance_vars.clone();
        self.transformation_pipeline
            .apply_cpu(&mut transformed_distances, self.splitter.n_position() as usize)?;
        
        // Step 3: Combine transformed distances
        let combined_distances = self.combine_distance_variables(&transformed_distances, n_solutions)?;
        
        // Step 4: Apply shape functions to position variables (CPU)
        let raw_objectives = self.shape_executor
            .apply_cpu(&position_vars, self.splitter.n_position())?;
        
        // Step 5: Scale objectives by distance variables
        let scaled_objectives = self.scale_objectives(&raw_objectives, &combined_distances)?;
        
        Ok(scaled_objectives)
    }
    
    /// Evaluate a single solution
    pub async fn evaluate_single(&mut self, solution: &[f32]) -> Result<Vec<f32>> {
        if solution.len() != self.dimension as usize {
            return Err(GNBGMOError::InvalidConfiguration(
                "Solution dimension mismatch".to_string()
            ));
        }
        
        let batch_result = self.evaluate_batch(solution).await?;
        Ok(batch_result)
    }
    
    /// Combine transformed distance variables into scaling factors
    /// 
    /// This implements the WFG-style combination where distance variables
    /// influence the scaling of objectives
    fn combine_distance_variables(&self, distances: &[f32], n_solutions: usize) -> Result<Vec<f32>> {
        let n_distance = self.splitter.n_distance() as usize;
        
        if distances.len() != n_solutions * n_distance {
            return Err(GNBGMOError::InvalidConfiguration(
                "Distance variable array size mismatch".to_string()
            ));
        }
        
        let mut combined = Vec::with_capacity(n_solutions);
        
        for sol_idx in 0..n_solutions {
            let sol_start = sol_idx * n_distance;
            let sol_end = sol_start + n_distance;
            let sol_distances = &distances[sol_start..sol_end];
            
            // Simple averaging for now - could use more sophisticated combination
            let avg_distance = if n_distance > 0 {
                sol_distances.iter().sum::<f32>() / n_distance as f32
            } else {
                1.0
            };
            
            combined.push(1.0 + avg_distance); // Offset to avoid zero scaling
        }
        
        Ok(combined)
    }
    
    /// Scale raw objectives by distance-based scaling factors
    fn scale_objectives(&self, objectives: &[f32], scaling_factors: &[f32]) -> Result<Vec<f32>> {
        let n_solutions = scaling_factors.len();
        let expected_obj_len = n_solutions * self.n_objectives as usize;
        
        if objectives.len() != expected_obj_len {
            return Err(GNBGMOError::InvalidConfiguration(
                "Objectives array size mismatch".to_string()
            ));
        }
        
        let mut scaled = Vec::with_capacity(expected_obj_len);
        
        for sol_idx in 0..n_solutions {
            let scale_factor = scaling_factors[sol_idx];
            let obj_start = sol_idx * self.n_objectives as usize;
            
            for obj_idx in 0..self.n_objectives as usize {
                let raw_obj = objectives[obj_start + obj_idx];
                scaled.push(raw_obj * scale_factor);
            }
        }
        
        Ok(scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_objective::{
        builder::GNBGMOBuilder,
        position_distance::SplitStrategy,
        transformations::{TransformationType, VariableRange},
        shapes::ShapeFunction,
    };
    
    #[tokio::test]
    async fn test_mo_problem_evaluation() {
        let mut problem = GNBGMOBuilder::new()
            .dimension(5)
            .objectives(2)
            .split_strategy(SplitStrategy::WFGStandard)
            .add_transformation(
                TransformationType::Polynomial { alpha: 0.5 },
                VariableRange::Distance,
            )
            .add_shape(ShapeFunction::Convex)
            .add_shape(ShapeFunction::Convex)
            .gpu(false) // Use CPU for testing
            .build()
            .unwrap();
        
        // Test single solution
        let solution = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let objectives = problem.evaluate_single(&solution).await.unwrap();
        
        assert_eq!(objectives.len(), 2);
        assert!(objectives[0] >= 0.0);
        assert!(objectives[1] >= 0.0);
    }
    
    #[tokio::test]
    async fn test_mo_problem_batch_evaluation() {
        let mut problem = GNBGMOBuilder::new()
            .dimension(4)
            .objectives(3)
            .gpu(false) // Use CPU for testing
            .build()
            .unwrap();
        
        // Test batch of 2 solutions
        let solutions = vec![
            0.1, 0.2, 0.3, 0.4, // Solution 1
            0.5, 0.6, 0.7, 0.8, // Solution 2
        ];
        
        let objectives = problem.evaluate_batch(&solutions).await.unwrap();
        
        assert_eq!(objectives.len(), 6); // 2 solutions × 3 objectives
        
        // All objectives should be positive
        for &obj in &objectives {
            assert!(obj >= 0.0);
        }
    }
    
    #[test]
    fn test_distance_combination() {
        let problem = GNBGMOBuilder::new()
            .dimension(5)
            .objectives(2)
            .build()
            .unwrap();
        
        // Test with 2 solutions, each with 3 distance variables
        let distances = vec![
            0.1, 0.2, 0.3, // Solution 1
            0.4, 0.5, 0.6, // Solution 2
        ];
        
        let combined = problem.combine_distance_variables(&distances, 2).unwrap();
        
        assert_eq!(combined.len(), 2);
        
        // Should be 1.0 + average
        assert!((combined[0] - 1.2).abs() < 1e-6); // 1 + (0.1+0.2+0.3)/3 = 1.2
        assert!((combined[1] - 1.5).abs() < 1e-6); // 1 + (0.4+0.5+0.6)/3 = 1.5
    }
}