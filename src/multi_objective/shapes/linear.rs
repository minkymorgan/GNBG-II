/// Linear shape functions for WFG-style problems
/// 
/// Creates linear Pareto fronts in objective space

/// Linear shape function
/// Forms a linear (hyperplanar) Pareto front
pub fn shape_linear(position: &[f32], obj_idx: usize, n_objectives: u32) -> f32 {
    let m = n_objectives as usize;
    
    if obj_idx >= m {
        return 0.0;
    }
    
    // Ensure we have enough position variables
    let k = position.len();
    if k == 0 {
        return 0.0;
    }
    
    let mut result = 1.0;
    
    // WFG-style linear shape functions
    // For M objectives and K position variables (K = M-1 typically)
    
    if obj_idx == 0 {
        // First objective: product of x_i for i=1 to M-1
        for i in 0..(m - 1).min(k) {
            result *= position[i].clamp(0.0, 1.0);
        }
    } else if obj_idx < m - 1 {
        // Middle objectives: product of x_i for i=1 to M-obj_idx-1, then (1-x_{M-obj_idx})
        for i in 0..(m - obj_idx - 1).min(k) {
            result *= position[i].clamp(0.0, 1.0);
        }
        if (m - obj_idx - 1) < k {
            result *= 1.0 - position[m - obj_idx - 1].clamp(0.0, 1.0);
        }
    } else {
        // Last objective: 1 - x_1
        if k > 0 {
            result = 1.0 - position[0].clamp(0.0, 1.0);
        }
    }
    
    result.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_shape_2obj() {
        let position = vec![0.5];
        
        let obj1 = shape_linear(&position, 0, 2);
        let obj2 = shape_linear(&position, 1, 2);
        
        // Should form linear Pareto front
        assert!((obj1 - 0.5).abs() < 1e-6);
        assert!((obj2 - 0.5).abs() < 1e-6);
        assert!((obj1 + obj2 - 1.0).abs() < 1e-6); // Linear constraint
    }
    
    #[test]
    fn test_linear_shape_3obj() {
        let position = vec![0.5, 0.5];
        
        let obj1 = shape_linear(&position, 0, 3);
        let obj2 = shape_linear(&position, 1, 3);
        let obj3 = shape_linear(&position, 2, 3);
        
        // All should be positive
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);
        assert!(obj3 >= 0.0);
    }
}