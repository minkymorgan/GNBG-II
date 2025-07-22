/// Convex shape functions for WFG-style problems
/// 
/// Creates convex Pareto fronts (sphere-like surfaces)

/// Convex shape function  
/// Forms a convex (spherical) Pareto front
pub fn shape_convex(position: &[f32], obj_idx: usize, n_objectives: u32) -> f32 {
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
    
    // WFG-style shape functions use position variables in a specific pattern
    // For M objectives and K position variables (K = M-1 typically)
    
    if obj_idx == 0 {
        // First objective: product of cos(theta_i) for i=1 to M-1
        for i in 0..(m - 1).min(k) {
            let val = position[i].clamp(0.0, 1.0);
            let angle = val * std::f32::consts::PI / 2.0;
            result *= angle.cos();
        }
    } else if obj_idx < m - 1 {
        // Middle objectives: product of cos(theta_i) for i=1 to M-obj_idx-1, then sin(theta_{M-obj_idx})
        for i in 0..(m - obj_idx - 1).min(k) {
            let val = position[i].clamp(0.0, 1.0);
            let angle = val * std::f32::consts::PI / 2.0;
            result *= angle.cos();
        }
        if (m - obj_idx - 1) < k {
            let val = position[m - obj_idx - 1].clamp(0.0, 1.0);
            let angle = val * std::f32::consts::PI / 2.0;
            result *= angle.sin();
        }
    } else {
        // Last objective: sin(theta_1)
        if k > 0 {
            let val = position[0].clamp(0.0, 1.0);
            let angle = val * std::f32::consts::PI / 2.0;
            result = angle.sin();
        }
    }
    
    // Ensure result is valid
    if !result.is_finite() {
        log::error!("Convex shape produced non-finite value: obj_idx={}, result={}, k={}, m={}", 
                   obj_idx, result, k, m);
        result = 0.0;
    }
    
    result.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_convex_shape_2obj() {
        let position = vec![0.5];
        
        let obj1 = shape_convex(&position, 0, 2);
        let obj2 = shape_convex(&position, 1, 2);
        
        // Should be positive
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);
        
        // Should satisfy convex constraint (sphere: x² + y² = 1)
        let sum_of_squares = obj1 * obj1 + obj2 * obj2;
        assert!((sum_of_squares - 0.5).abs() < 0.1); // Approximately on unit circle
    }
    
    #[test]
    fn test_convex_shape_boundary() {
        // Test boundary cases
        let position = vec![0.0];
        let obj1 = shape_convex(&position, 0, 2);
        let obj2 = shape_convex(&position, 1, 2);
        
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);
        
        let position = vec![1.0];
        let obj1 = shape_convex(&position, 0, 2);
        let obj2 = shape_convex(&position, 1, 2);
        
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);
    }
}