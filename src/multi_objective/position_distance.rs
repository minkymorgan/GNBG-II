/// Position-Distance Variable Splitting
/// 
/// Implements WFG's position-distance paradigm with adaptive strategies
/// for optimal variable allocation based on optimization objectives.

use crate::multi_objective::{GNBGMOError, Result};

/// Strategies for splitting variables into position and distance components
#[derive(Debug, Clone)]
pub enum SplitStrategy {
    /// Standard WFG approach: k = 2*(M-1)
    WFGStandard,
    /// User-defined number of position variables
    Custom(u32),
    /// Proportional split: k = ratio * n_variables
    Proportional(f32),
    /// Adaptive strategy with optimization target
    Adaptive { 
        min_k: u32, 
        max_k: u32,
        optimization_target: OptimizationTarget 
    },
}

/// Optimization targets for adaptive splitting
#[derive(Debug, Clone, Copy)]
pub enum OptimizationTarget {
    /// Favor larger k for faster convergence
    ConvergenceSpeed,
    /// Favor smaller k for better diversity
    FrontDiversity,
    /// Auto-tune based on problem characteristics
    Balanced,
}

/// Position-Distance variable splitter
#[derive(Debug)]
pub struct PositionDistanceSplitter {
    /// Number of position variables (k)
    pub n_position: u32,
    /// Number of distance variables (l)
    pub n_distance: u32,
    /// Total number of variables
    pub n_total: u32,
    /// Splitting strategy used
    pub strategy: SplitStrategy,
}

impl PositionDistanceSplitter {
    /// Create a new splitter with the given strategy
    pub fn new(
        n_variables: u32, 
        n_objectives: u32, 
        strategy: SplitStrategy
    ) -> Result<Self> {
        let n_position = Self::compute_k(n_variables, n_objectives, &strategy)?;
        let n_distance = n_variables - n_position;
        
        // Validation
        if n_position >= n_variables {
            return Err(GNBGMOError::InvalidConfiguration(
                format!("Position variables ({}) must be less than total variables ({})", 
                       n_position, n_variables)
            ));
        }
        
        if n_distance == 0 {
            return Err(GNBGMOError::InvalidConfiguration(
                "At least one distance variable is required".to_string()
            ));
        }
        
        Ok(Self {
            n_position,
            n_distance,
            n_total: n_variables,
            strategy,
        })
    }
    
    /// Compute optimal k value based on strategy
    fn compute_k(
        n_variables: u32, 
        n_objectives: u32, 
        strategy: &SplitStrategy
    ) -> Result<u32> {
        let k = match strategy {
            SplitStrategy::WFGStandard => {
                2 * (n_objectives - 1)
            },
            SplitStrategy::Custom(k) => *k,
            SplitStrategy::Proportional(ratio) => {
                ((*ratio * n_variables as f32) as u32).max(1)
            },
            SplitStrategy::Adaptive { min_k, max_k, optimization_target } => {
                Self::auto_tune_k(n_objectives, n_variables, *optimization_target)
                    .clamp(*min_k, *max_k)
            },
        };
        
        // Ensure k is within valid bounds
        let k = k.clamp(1, n_variables - 1);
        Ok(k)
    }
    
    /// Auto-tune k based on optimization target
    fn auto_tune_k(
        n_objectives: u32, 
        n_variables: u32, 
        target: OptimizationTarget
    ) -> u32 {
        match target {
            OptimizationTarget::ConvergenceSpeed => {
                // Research suggests k ≈ 2.5*(M-1) for faster convergence
                ((2.5 * (n_objectives - 1) as f32) as u32).clamp(2, n_variables - 2)
            },
            OptimizationTarget::FrontDiversity => {
                // Smaller k maintains more diversity
                ((1.5 * (n_objectives - 1) as f32) as u32).max(2)
            },
            OptimizationTarget::Balanced => {
                // Classic WFG with safety bounds
                (2 * (n_objectives - 1)).clamp(4, n_variables / 2)
            }
        }
    }
    
    /// Split a solution vector into position and distance components
    pub fn split(&self, solution: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if solution.len() != self.n_total as usize {
            return Err(GNBGMOError::DimensionMismatch {
                expected: self.n_total as usize,
                actual: solution.len(),
            });
        }
        
        let position = solution[..self.n_position as usize].to_vec();
        let distance = solution[self.n_position as usize..].to_vec();
        
        Ok((position, distance))
    }
    
    /// Split multiple solutions into position and distance batches
    pub fn split_batch(&self, solutions: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let n_solutions = solutions.len() / self.n_total as usize;
        
        if solutions.len() != n_solutions * self.n_total as usize {
            return Err(GNBGMOError::DimensionMismatch {
                expected: n_solutions * self.n_total as usize,
                actual: solutions.len(),
            });
        }
        
        let mut position_batch = Vec::with_capacity(n_solutions * self.n_position as usize);
        let mut distance_batch = Vec::with_capacity(n_solutions * self.n_distance as usize);
        
        for sol_idx in 0..n_solutions {
            let sol_start = sol_idx * self.n_total as usize;
            let sol_end = sol_start + self.n_total as usize;
            let solution = &solutions[sol_start..sol_end];
            
            // Position variables
            position_batch.extend_from_slice(&solution[..self.n_position as usize]);
            
            // Distance variables
            distance_batch.extend_from_slice(&solution[self.n_position as usize..]);
        }
        
        Ok((position_batch, distance_batch))
    }
    
    /// Recombine position and distance variables back into full solutions
    pub fn recombine(&self, position: &[f32], distance: &[f32]) -> Result<Vec<f32>> {
        let n_solutions = position.len() / self.n_position as usize;
        
        if distance.len() / self.n_distance as usize != n_solutions {
            return Err(GNBGMOError::DimensionMismatch {
                expected: n_solutions * self.n_distance as usize,
                actual: distance.len(),
            });
        }
        
        let mut combined = Vec::with_capacity(n_solutions * self.n_total as usize);
        
        for sol_idx in 0..n_solutions {
            let pos_start = sol_idx * self.n_position as usize;
            let pos_end = pos_start + self.n_position as usize;
            
            let dist_start = sol_idx * self.n_distance as usize;
            let dist_end = dist_start + self.n_distance as usize;
            
            combined.extend_from_slice(&position[pos_start..pos_end]);
            combined.extend_from_slice(&distance[dist_start..dist_end]);
        }
        
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wfg_standard_splitting() {
        let splitter = PositionDistanceSplitter::new(
            30, 5, SplitStrategy::WFGStandard
        ).unwrap();
        
        assert_eq!(splitter.n_position, 8); // 2 * (5 - 1)
        assert_eq!(splitter.n_distance, 22);
    }
    
    #[test]
    fn test_adaptive_splitting() {
        let splitter = PositionDistanceSplitter::new(
            30, 5, 
            SplitStrategy::Adaptive { 
                min_k: 4, 
                max_k: 20, 
                optimization_target: OptimizationTarget::Balanced 
            }
        ).unwrap();
        
        assert!(splitter.n_position >= 4);
        assert!(splitter.n_position <= 20);
        assert_eq!(splitter.n_position + splitter.n_distance, 30);
    }
    
    #[test]
    fn test_solution_splitting() {
        let splitter = PositionDistanceSplitter::new(
            10, 3, SplitStrategy::WFGStandard
        ).unwrap();
        
        let solution = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (position, distance) = splitter.split(&solution).unwrap();
        
        assert_eq!(position.len(), splitter.n_position as usize);
        assert_eq!(distance.len(), splitter.n_distance as usize);
        assert_eq!(position, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(distance, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }
}