/// Concave shape functions for WFG-style problems
/// 
/// Creates concave Pareto fronts (inverted sphere-like surfaces)

/// Concave shape function
/// Forms a concave Pareto front (inverted sphere)
pub fn shape_concave(position: &[f32], obj_idx: usize, n_objectives: u32) -> f32 {
    let m = n_objectives as usize;
    
    if obj_idx >= m {
        return 0.0;
    }
    
    let mut result = 1.0;
    
    // For concave shape, use sine transformations
    for i in 0..(m - obj_idx - 1) {
        if i < position.len() {
            let angle = position[i] * std::f32::consts::PI / 2.0;
            result *= angle.sin();
        }
    }
    
    // For the last factor (if not the last objective)
    if obj_idx < m - 1 && (m - obj_idx - 1) < position.len() {
        let angle = position[m - obj_idx - 1] * std::f32::consts::PI / 2.0;
        result *= angle.cos();
    }
    
    result.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_concave_shape_2obj() {
        let position = vec![0.5];
        
        let obj1 = shape_concave(&position, 0, 2);
        let obj2 = shape_concave(&position, 1, 2);
        
        // Should be positive
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);
        
        // For concave shape with position=0.5, both should be non-zero
        assert!(obj1 > 0.0);
        assert!(obj2 > 0.0);
    }
    
    #[test]
    fn test_concave_shape_boundary() {
        // Test boundary cases
        let position = vec![0.0];
        let obj1 = shape_concave(&position, 0, 2);
        let obj2 = shape_concave(&position, 1, 2);
        
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);
        
        let position = vec![1.0];
        let obj1 = shape_concave(&position, 0, 2);
        let obj2 = shape_concave(&position, 1, 2);
        
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);
    }
    
    #[test]
    fn test_concave_shape_3obj() {
        let position = vec![0.3, 0.7];
        
        let obj1 = shape_concave(&position, 0, 3);
        let obj2 = shape_concave(&position, 1, 3);
        let obj3 = shape_concave(&position, 2, 3);
        
        // All should be positive for concave shape
        assert!(obj1 >= 0.0);
        assert!(obj2 >= 0.0);  
        assert!(obj3 >= 0.0);
    }
}