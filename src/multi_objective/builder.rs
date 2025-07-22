/// Builder pattern for creating multi-objective GNBG problems
/// 
/// Provides a fluent interface for configuring complex multi-objective optimization problems

use crate::multi_objective::{
    GNBGMOError, Result, 
    position_distance::{PositionDistanceSplitter, SplitStrategy, OptimizationTarget},
    transformations::{TransformationPipeline, TransformationStage, TransformationType, VariableRange},
    shapes::{ShapeFunction, ShapeFunctionExecutor},
};

/// Builder for multi-objective GNBG problems
pub struct GNBGMOBuilder {
    /// Problem dimension
    dimension: Option<u32>,
    /// Number of objectives
    n_objectives: Option<u32>,
    /// Position-distance splitting strategy
    split_strategy: SplitStrategy,
    /// Transformation stages
    transformation_stages: Vec<TransformationStage>,
    /// Shape functions
    shape_functions: Vec<ShapeFunction>,
    /// Enable GPU acceleration
    use_gpu: bool,
    /// Enable shape function caching
    use_cache: bool,
}

impl GNBGMOBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            dimension: None,
            n_objectives: None,
            split_strategy: SplitStrategy::WFGStandard,
            transformation_stages: Vec::new(),
            shape_functions: Vec::new(),
            use_gpu: true,
            use_cache: true,
        }
    }
    
    /// Set problem dimension
    pub fn dimension(mut self, dimension: u32) -> Self {
        self.dimension = Some(dimension);
        self
    }
    
    /// Set number of objectives
    pub fn objectives(mut self, n_objectives: u32) -> Self {
        self.n_objectives = Some(n_objectives);
        self
    }
    
    /// Set position-distance splitting strategy
    pub fn split_strategy(mut self, strategy: SplitStrategy) -> Self {
        self.split_strategy = strategy;
        self
    }
    
    /// Add a transformation stage
    pub fn add_transformation(mut self, transform_type: TransformationType, apply_to: VariableRange) -> Self {
        self.transformation_stages.push(TransformationStage {
            transform_type,
            apply_to,
        });
        self
    }
    
    /// Add a shape function
    pub fn add_shape(mut self, shape: ShapeFunction) -> Self {
        self.shape_functions.push(shape);
        self
    }
    
    /// Enable/disable GPU acceleration
    pub fn gpu(mut self, enabled: bool) -> Self {
        self.use_gpu = enabled;
        self
    }
    
    /// Enable/disable shape function caching
    pub fn cache(mut self, enabled: bool) -> Self {
        self.use_cache = enabled;
        self
    }
    
    /// Create WFG1-style problem preset
    pub fn wfg1_preset(dimension: u32, n_objectives: u32) -> Self {
        let mut builder = Self::new()
            .dimension(dimension)
            .objectives(n_objectives)
            .split_strategy(SplitStrategy::WFGStandard)
            .add_transformation(
                TransformationType::Polynomial { alpha: 0.02 },
                VariableRange::Distance,
            )
            .add_transformation(
                TransformationType::MultiModal { A: 10.0, B: 0.35, C: 0.05 },
                VariableRange::Distance,
            );
        
        // Add convex shape functions for all objectives
        for _ in 0..n_objectives {
            builder = builder.add_shape(ShapeFunction::Convex);
        }
        
        builder
    }
    
    /// Create WFG2-style problem preset  
    pub fn wfg2_preset(dimension: u32, n_objectives: u32) -> Self {
        let mut builder = Self::new()
            .dimension(dimension)
            .objectives(n_objectives)
            .split_strategy(SplitStrategy::WFGStandard)
            .add_transformation(
                TransformationType::Polynomial { alpha: 0.02 },
                VariableRange::Distance,
            )
            .add_transformation(
                TransformationType::NonSeparable { A: 2 },
                VariableRange::Distance,
            );
        
        // Add convex shape functions for all objectives
        for _ in 0..n_objectives {
            builder = builder.add_shape(ShapeFunction::Convex);
        }
        
        builder
    }
    
    /// Create WFG3-style problem preset
    pub fn wfg3_preset(dimension: u32, n_objectives: u32) -> Self {
        let mut builder = Self::new()
            .dimension(dimension)
            .objectives(n_objectives)
            .split_strategy(SplitStrategy::WFGStandard)
            .add_transformation(
                TransformationType::Polynomial { alpha: 0.02 },
                VariableRange::Distance,
            )
            .add_transformation(
                TransformationType::NonSeparable { A: 2 },
                VariableRange::Distance,
            );
        
        // Add linear shape functions for all objectives
        for _ in 0..n_objectives {
            builder = builder.add_shape(ShapeFunction::Linear);
        }
        
        builder
    }
    
    /// Build the multi-objective problem
    pub fn build(self) -> Result<super::GNBGMultiObjective> {
        let dimension = self.dimension.ok_or_else(|| {
            GNBGMOError::InvalidConfiguration("Dimension must be specified".to_string())
        })?;
        
        let n_objectives = self.n_objectives.ok_or_else(|| {
            GNBGMOError::InvalidConfiguration("Number of objectives must be specified".to_string())
        })?;
        
        if n_objectives < 2 {
            return Err(GNBGMOError::InvalidConfiguration(
                "At least 2 objectives required".to_string()
            ));
        }
        
        if dimension < n_objectives {
            return Err(GNBGMOError::InvalidConfiguration(
                "Dimension must be >= number of objectives".to_string()
            ));
        }
        
        // Create splitter
        let splitter = PositionDistanceSplitter::new(dimension, n_objectives, self.split_strategy)?;
        
        // Create transformation pipeline
        let pipeline = if self.transformation_stages.is_empty() {
            // Default transformation if none specified
            TransformationPipeline::new(vec![
                TransformationStage {
                    transform_type: TransformationType::Polynomial { alpha: 0.02 },
                    apply_to: VariableRange::Distance,
                }
            ])?
        } else {
            TransformationPipeline::new(self.transformation_stages)?
        };
        
        // Create shape functions
        let shape_functions = if self.shape_functions.is_empty() {
            // Default to convex shapes
            vec![ShapeFunction::Convex; n_objectives as usize]
        } else if self.shape_functions.len() != n_objectives as usize {
            return Err(GNBGMOError::InvalidConfiguration(
                format!("Need exactly {} shape functions for {} objectives", 
                       n_objectives, n_objectives)
            ));
        } else {
            self.shape_functions
        };
        
        let shape_executor = if self.use_cache {
            ShapeFunctionExecutor::new(shape_functions).with_cache()
        } else {
            ShapeFunctionExecutor::new(shape_functions)
        };
        
        Ok(super::GNBGMultiObjective {
            dimension,
            n_objectives,
            splitter,
            transformation_pipeline: pipeline,
            shape_executor,
            use_gpu: self.use_gpu,
            gpu_context: None, // Will be initialized on first use
        })
    }
}

impl Default for GNBGMOBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_builder_basic() {
        let problem = GNBGMOBuilder::new()
            .dimension(10)
            .objectives(3)
            .build()
            .unwrap();
            
        assert_eq!(problem.dimension, 10);
        assert_eq!(problem.n_objectives, 3);
    }
    
    #[test]
    fn test_builder_validation() {
        // Test missing dimension
        let result = GNBGMOBuilder::new().objectives(3).build();
        assert!(result.is_err());
        
        // Test dimension < objectives
        let result = GNBGMOBuilder::new()
            .dimension(2)
            .objectives(3)
            .build();
        assert!(result.is_err());
        
        // Test < 2 objectives
        let result = GNBGMOBuilder::new()
            .dimension(5)
            .objectives(1)
            .build();
        assert!(result.is_err());
    }
    
    #[test]
    fn test_wfg_presets() {
        let wfg1 = GNBGMOBuilder::wfg1_preset(10, 3).build().unwrap();
        assert_eq!(wfg1.dimension, 10);
        assert_eq!(wfg1.n_objectives, 3);
        
        let wfg2 = GNBGMOBuilder::wfg2_preset(20, 5).build().unwrap();
        assert_eq!(wfg2.dimension, 20);
        assert_eq!(wfg2.n_objectives, 5);
    }
}