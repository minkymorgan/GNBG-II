/// Deceptive transformation function
/// 
/// Implements WFG's deceptive transformation that creates misleading gradients

use super::correct_to_01;

/// Apply deceptive transformation
/// 
/// This transformation creates deceptive landscapes where the gradient
/// information misleads optimization algorithms.
/// 
/// # Arguments
/// * `y` - Input value in [0,1]
/// * `a` - Deception parameter (typical: 0.35)
/// * `b` - Deception parameter (typical: 0.001) 
/// * `c` - Deception parameter (typical: 0.05)
/// 
/// # Returns
/// Transformed value in [0,1]
pub fn deceptive_transform(y: f32, a: f32, b: f32, c: f32) -> f32 {
    let tmp1 = (y - a + b).floor() * (1.0 - c + (a - b) / b) / (a - b);
    let tmp2 = (a + b - y).floor() * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b);
    
    correct_to_01(tmp1 + tmp2 + 1.0)
}

/// Shift-then-deceptive transformation
/// 
/// Combines shift with deceptive transformation for more complex landscapes
/// 
/// # Arguments
/// * `y` - Input value in [0,1]
/// * `a` - Primary deception parameter
/// * `b` - Secondary deception parameter  
/// * `c` - Scaling parameter
/// 
/// # Returns
/// Transformed value in [0,1]
pub fn shift_deceptive_transform(y: f32, a: f32, b: f32, c: f32) -> f32 {
    let abs_y_minus_a = (y - a).abs();
    let abs_y_minus_a_minus_b = (y - a - b).abs();
    let abs_y_minus_a_plus_b = (y - a + b).abs();
    
    let condition1 = y <= a;
    let condition2 = y <= a + b;
    
    let result = if condition1 {
        a - abs_y_minus_a
    } else if condition2 {
        a + abs_y_minus_a_minus_b  
    } else {
        a + b - abs_y_minus_a_plus_b
    };
    
    let scaled = result / (a + b);
    correct_to_01(c + (1.0 - c) * scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_deceptive_transform_typical() {
        // Test with typical WFG parameters
        let result = deceptive_transform(0.5, 0.35, 0.001, 0.05);
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_deceptive_transform_boundaries() {
        // Test boundary values
        let result1 = deceptive_transform(0.0, 0.35, 0.001, 0.05);
        let result2 = deceptive_transform(1.0, 0.35, 0.001, 0.05);
        
        assert!(result1 >= 0.0 && result1 <= 1.0);
        assert!(result2 >= 0.0 && result2 <= 1.0);
    }
    
    #[test]
    fn test_shift_deceptive_transform() {
        let result = shift_deceptive_transform(0.5, 0.35, 0.05, 0.02);
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_deceptive_creates_different_values() {
        // Deceptive transformation should create non-linear mapping
        let y1 = 0.3;
        let y2 = 0.7;
        let result1 = deceptive_transform(y1, 0.35, 0.001, 0.05);
        let result2 = deceptive_transform(y2, 0.35, 0.001, 0.05);
        
        // Results should be different (not just scaled)
        assert_ne!(result1, result2);
        assert!((result2 - result1).abs() > 0.01); // Some significant difference
    }
}