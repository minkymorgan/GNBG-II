/// WFG-style transformation functions
/// 
/// Implements the transformation pipeline that adds problem characteristics
/// such as bias, deception, multimodality, and non-separability.

pub mod bias;
pub mod deceptive; 
pub mod multimodal;
pub mod nonseparable;
pub mod fused_pipeline;

use crate::multi_objective::{GNBGMOError, Result};

/// Types of transformations available
#[derive(Debug, Clone)]
pub enum TransformationType {
    /// Bias transformation: y^alpha
    Bias { alpha: f32 },
    /// Deceptive transformation with multiple parameters
    Deceptive { a: f32, b: f32, c: f32 },
    /// Multi-modal transformation
    MultiModal { A: f32, B: f32, C: f32 },
    /// Non-separable reduction
    NonSeparable { A: u32 },
    /// Polynomial transformation
    Polynomial { alpha: f32 },
    /// Shift transformation
    Shift { shift: f32 },
}

/// Configuration for a transformation stage
#[derive(Debug, Clone)]
pub struct TransformationStage {
    /// Type of transformation to apply
    pub transform_type: TransformationType,
    /// Variable indices this transformation applies to
    pub apply_to: VariableRange,
}

/// Range of variables to apply transformation to
#[derive(Debug, Clone)]
pub enum VariableRange {
    /// Apply to all variables
    All,
    /// Apply to position variables only
    Position,
    /// Apply to distance variables only
    Distance,
    /// Apply to specific range [start, end)
    Range(usize, usize),
    /// Apply to specific indices
    Indices(Vec<usize>),
}

/// Transformation pipeline configuration
#[derive(Debug)]
pub struct TransformationPipeline {
    /// Ordered list of transformation stages
    pub stages: Vec<TransformationStage>,
    /// Maximum number of stages supported
    pub max_stages: usize,
}

impl TransformationPipeline {
    /// Create a new transformation pipeline
    pub fn new(stages: Vec<TransformationStage>) -> Result<Self> {
        const MAX_STAGES: usize = 16; // WGSL array limit
        
        if stages.len() > MAX_STAGES {
            return Err(GNBGMOError::InvalidConfiguration(
                format!("Too many transformation stages: {} > {}", stages.len(), MAX_STAGES)
            ));
        }
        
        Ok(Self {
            stages,
            max_stages: MAX_STAGES,
        })
    }
    
    /// Add a transformation stage
    pub fn add_stage(&mut self, stage: TransformationStage) -> Result<()> {
        if self.stages.len() >= self.max_stages {
            return Err(GNBGMOError::InvalidConfiguration(
                format!("Cannot add more stages: {} >= {}", self.stages.len(), self.max_stages)
            ));
        }
        
        self.stages.push(stage);
        Ok(())
    }
    
    /// Get the number of stages
    pub fn stage_count(&self) -> u32 {
        self.stages.len() as u32
    }
    
    /// Apply transformations to a single solution (CPU implementation)
    pub fn apply_cpu(&self, solution: &mut [f32], n_position: usize) -> Result<()> {
        for stage in &self.stages {
            let variable_indices = self.get_variable_indices(
                &stage.apply_to, 
                solution.len(), 
                n_position
            )?;
            
            for &idx in &variable_indices {
                if idx < solution.len() {
                    solution[idx] = self.apply_single_transform(
                        solution[idx], 
                        &stage.transform_type
                    )?;
                }
            }
        }
        Ok(())
    }
    
    /// Get indices for a variable range
    fn get_variable_indices(
        &self,
        range: &VariableRange,
        total_vars: usize,
        n_position: usize,
    ) -> Result<Vec<usize>> {
        match range {
            VariableRange::All => Ok((0..total_vars).collect()),
            VariableRange::Position => Ok((0..n_position).collect()),
            VariableRange::Distance => Ok((n_position..total_vars).collect()),
            VariableRange::Range(start, end) => {
                if *end > total_vars {
                    return Err(GNBGMOError::InvalidConfiguration(
                        format!("Range end {} exceeds total variables {}", end, total_vars)
                    ));
                }
                Ok((*start..*end).collect())
            },
            VariableRange::Indices(indices) => {
                for &idx in indices {
                    if idx >= total_vars {
                        return Err(GNBGMOError::InvalidConfiguration(
                            format!("Index {} exceeds total variables {}", idx, total_vars)
                        ));
                    }
                }
                Ok(indices.clone())
            }
        }
    }
    
    /// Apply a single transformation to a value
    fn apply_single_transform(
        &self,
        value: f32,
        transform_type: &TransformationType,
    ) -> Result<f32> {
        match transform_type {
            TransformationType::Bias { alpha } => {
                Ok(bias::bias_transform(value, *alpha))
            },
            TransformationType::Deceptive { a, b, c } => {
                Ok(deceptive::deceptive_transform(value, *a, *b, *c))
            },
            TransformationType::MultiModal { A, B, C } => {
                Ok(multimodal::multimodal_transform(value, *A, *B, *C))
            },
            TransformationType::Polynomial { alpha } => {
                Ok(value.powf(*alpha).clamp(0.0, 1.0))
            },
            TransformationType::Shift { shift } => {
                Ok((value + shift).clamp(0.0, 1.0))
            },
            TransformationType::NonSeparable { A: _ } => {
                // Non-separable transformations require group processing
                // This is handled separately in the GPU pipeline
                Ok(value)
            }
        }
    }
}

/// Helper function to correct values to [0,1] range
pub fn correct_to_01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_creation() {
        let stages = vec![
            TransformationStage {
                transform_type: TransformationType::Bias { alpha: 0.02 },
                apply_to: VariableRange::Position,
            },
            TransformationStage {
                transform_type: TransformationType::Shift { shift: 0.35 },
                apply_to: VariableRange::Distance,
            },
        ];
        
        let pipeline = TransformationPipeline::new(stages).unwrap();
        assert_eq!(pipeline.stage_count(), 2);
    }
    
    #[test]
    fn test_too_many_stages() {
        let stages = (0..20).map(|_| TransformationStage {
            transform_type: TransformationType::Bias { alpha: 1.0 },
            apply_to: VariableRange::All,
        }).collect();
        
        assert!(TransformationPipeline::new(stages).is_err());
    }
    
    #[test]
    fn test_cpu_transformation() {
        let stages = vec![
            TransformationStage {
                transform_type: TransformationType::Bias { alpha: 2.0 },
                apply_to: VariableRange::All,
            },
        ];
        
        let pipeline = TransformationPipeline::new(stages).unwrap();
        let mut solution = vec![0.5, 0.25, 0.75];
        
        pipeline.apply_cpu(&mut solution, 2).unwrap();
        
        // Values should be transformed by y^2.0
        assert!((solution[0] - 0.25).abs() < 1e-6); // 0.5^2
        assert!((solution[1] - 0.0625).abs() < 1e-6); // 0.25^2
        assert!((solution[2] - 0.5625).abs() < 1e-6); // 0.75^2
    }
}