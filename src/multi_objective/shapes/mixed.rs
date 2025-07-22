/// Mixed and disconnected shape functions for WFG-style problems
/// 
/// Creates complex Pareto fronts with mixed convex/concave regions or discontinuities

use super::{convex, concave};

/// Mixed shape function with convex/concave transitions
/// Uses transition points to switch between convex and concave regions
pub fn shape_mixed(
    position: &[f32], 
    obj_idx: usize, 
    n_objectives: u32, 
    transition_points: &[f32]
) -> f32 {
    if transition_points.is_empty() || position.is_empty() {
        // Fallback to convex shape if no transitions defined
        return convex::shape_convex(position, obj_idx, n_objectives);
    }
    
    // Use first position variable to determine region
    let x = position[0];
    
    // Find which region we're in based on transition points
    let mut region = 0;
    for &transition in transition_points {
        if x <= transition {
            break;
        }
        region += 1;
    }
    
    // Alternate between convex and concave regions
    if region % 2 == 0 {
        convex::shape_convex(position, obj_idx, n_objectives)
    } else {
        concave::shape_concave(position, obj_idx, n_objectives)
    }
}

/// Disconnected shape function with gaps in the Pareto front
/// Creates discontinuities at specified gap regions
pub fn shape_disconnected(
    position: &[f32],
    obj_idx: usize,
    n_objectives: u32,
    gaps: &[(f32, f32)]
) -> f32 {
    if gaps.is_empty() || position.is_empty() {
        // Fallback to linear shape if no gaps defined
        return super::linear::shape_linear(position, obj_idx, n_objectives);
    }
    
    // Use first position variable to check for gaps
    let x = position[0];
    
    // Check if we're in a gap region
    for &(gap_start, gap_end) in gaps {
        if x >= gap_start && x <= gap_end {
            // In gap region - return very small value to create discontinuity
            return 0.001;
        }
    }
    
    // Not in gap - use normal convex shape
    convex::shape_convex(position, obj_idx, n_objectives)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mixed_shape_transitions() {
        let transition_points = vec![0.5];
        
        // Test first region (should be convex)
        let position1 = vec![0.3];
        let obj1_region1 = shape_mixed(&position1, 0, 2, &transition_points);
        let convex_result = convex::shape_convex(&position1, 0, 2);
        assert!((obj1_region1 - convex_result).abs() < 1e-6);
        
        // Test second region (should be concave)
        let position2 = vec![0.7];
        let obj1_region2 = shape_mixed(&position2, 0, 2, &transition_points);
        let concave_result = concave::shape_concave(&position2, 0, 2);
        assert!((obj1_region2 - concave_result).abs() < 1e-6);
    }
    
    #[test]
    fn test_disconnected_shape_gaps() {
        let gaps = vec![(0.3, 0.7)];
        
        // Test outside gap region
        let position1 = vec![0.1];
        let obj1_outside = shape_disconnected(&position1, 0, 2, &gaps);
        assert!(obj1_outside > 0.01); // Should be normal value
        
        // Test inside gap region
        let position2 = vec![0.5];
        let obj1_inside = shape_disconnected(&position2, 0, 2, &gaps);
        assert!((obj1_inside - 0.001).abs() < 1e-6); // Should be gap value
    }
    
    #[test]
    fn test_mixed_shape_no_transitions() {
        let transition_points = vec![];
        let position = vec![0.5];
        
        let result = shape_mixed(&position, 0, 2, &transition_points);
        let convex_result = convex::shape_convex(&position, 0, 2);
        
        // Should fall back to convex
        assert!((result - convex_result).abs() < 1e-6);
    }
    
    #[test]
    fn test_disconnected_shape_no_gaps() {
        let gaps = vec![];
        let position = vec![0.5];
        
        let result = shape_disconnected(&position, 0, 2, &gaps);
        let linear_result = crate::multi_objective::shapes::linear::shape_linear(&position, 0, 2);
        
        // Should fall back to linear
        assert!((result - linear_result).abs() < 1e-6);
    }
}