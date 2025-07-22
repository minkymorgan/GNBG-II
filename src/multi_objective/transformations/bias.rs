/// Bias transformation function
/// 
/// Implements y^alpha transformation to create bias in the fitness landscape

use super::correct_to_01;

/// Apply bias transformation: y^alpha
/// 
/// # Arguments
/// * `y` - Input value in [0,1]
/// * `alpha` - Bias parameter (alpha < 1 creates bias toward 1, alpha > 1 toward 0)
/// 
/// # Returns
/// Transformed value clamped to [0,1]
pub fn bias_transform(y: f32, alpha: f32) -> f32 {
    if alpha <= 0.0 {
        return y; // No transformation for invalid alpha
    }
    
    correct_to_01(y.powf(alpha))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bias_transform_identity() {
        // alpha = 1.0 should be identity transformation
        assert!((bias_transform(0.5, 1.0) - 0.5).abs() < 1e-6);
        assert!((bias_transform(0.25, 1.0) - 0.25).abs() < 1e-6);
    }
    
    #[test]
    fn test_bias_transform_toward_zero() {
        // alpha > 1.0 should bias toward 0
        let result = bias_transform(0.5, 2.0);
        assert!(result < 0.5);
        assert!((result - 0.25).abs() < 1e-6); // 0.5^2 = 0.25
    }
    
    #[test]
    fn test_bias_transform_toward_one() {
        // alpha < 1.0 should bias toward 1
        let result = bias_transform(0.25, 0.5);
        assert!(result > 0.25);
        assert!((result - 0.5).abs() < 1e-6); // 0.25^0.5 = 0.5
    }
    
    #[test]
    fn test_bias_transform_bounds() {
        // Test boundary values
        assert_eq!(bias_transform(0.0, 2.0), 0.0);
        assert_eq!(bias_transform(1.0, 2.0), 1.0);
        assert_eq!(bias_transform(0.0, 0.5), 0.0);
        assert_eq!(bias_transform(1.0, 0.5), 1.0);
    }
    
    #[test]
    fn test_bias_transform_invalid_alpha() {
        // Invalid alpha should return original value
        assert_eq!(bias_transform(0.5, 0.0), 0.5);
        assert_eq!(bias_transform(0.5, -1.0), 0.5);
    }
}