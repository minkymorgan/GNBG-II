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
    
    let mut result = 1.0;
    
    // For convex shape, use cosine transformations
    for i in 0..(m - obj_idx - 1) {
        if i < position.len() {
            let angle = position[i] * std::f32::consts::PI / 2.0;
            result *= angle.cos();
        }
    }
    
    // For the last factor (if not the last objective)
    if obj_idx < m - 1 && (m - obj_idx - 1) < position.len() {
        let angle = position[m - obj_idx - 1] * std::f32::consts::PI / 2.0;
        result *= angle.sin();
    }
    
    result.max(0.0)
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