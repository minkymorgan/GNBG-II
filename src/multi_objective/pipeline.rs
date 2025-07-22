/// Multi-objective evaluation pipeline
/// 
/// Coordinates the complete evaluation process from solutions to objectives

use crate::multi_objective::{GNBGMOError, Result, GNBGMultiObjective};

/// Complete evaluation pipeline for multi-objective problems
pub struct EvaluationPipeline {
    /// The multi-objective problem
    problem: GNBGMultiObjective,
    /// Use asynchronous evaluation
    async_eval: bool,
}

impl EvaluationPipeline {
    /// Create a new evaluation pipeline
    pub fn new(problem: GNBGMultiObjective) -> Self {
        Self {
            problem,
            async_eval: true,
        }
    }
    
    /// Set asynchronous evaluation mode
    pub fn async_mode(mut self, enabled: bool) -> Self {
        self.async_eval = enabled;
        self
    }
    
    /// Evaluate solutions through the complete pipeline
    pub async fn evaluate(&mut self, solutions: &[f32]) -> Result<Vec<f32>> {
        self.problem.evaluate_batch(solutions).await
    }
    
    /// Get problem dimension
    pub fn dimension(&self) -> u32 {
        self.problem.dimension
    }
    
    /// Get number of objectives
    pub fn n_objectives(&self) -> u32 {
        self.problem.n_objectives
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_objective::builder::GNBGMOBuilder;
    
    #[tokio::test]
    async fn test_evaluation_pipeline() {
        let problem = GNBGMOBuilder::new()
            .dimension(5)
            .objectives(2)
            .gpu(false)
            .build()
            .unwrap();
            
        let mut pipeline = EvaluationPipeline::new(problem);
        
        assert_eq!(pipeline.dimension(), 5);
        assert_eq!(pipeline.n_objectives(), 2);
        
        let solutions = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let objectives = pipeline.evaluate(&solutions).await.unwrap();
        
        assert_eq!(objectives.len(), 2);
    }
}