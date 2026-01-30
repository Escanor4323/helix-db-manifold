//! Numerical stability utilities for manifold operations.
//!
//! Based on manifold-spec.md Section 6.

/// General epsilon for clamping operations.
pub const EPSILON: f64 = 1e-7;

/// Minimum norm epsilon for very small values.
pub const MIN_NORM_EPSILON: f64 = 1e-15;

/// Safe arcosh that clamps input to avoid NaN.
/// arcosh(x) is only defined for x >= 1.
#[inline]
pub fn safe_arcosh(x: f64) -> f64 {
    (x.max(1.0 + EPSILON)).acosh()
}

/// Safe artanh that clamps input to (-1, 1) range.
/// artanh(x) is only defined for |x| < 1.
#[inline]
pub fn safe_artanh(x: f64) -> f64 {
    x.clamp(-1.0 + EPSILON, 1.0 - EPSILON).atanh()
}

/// Safe division that avoids division by zero.
#[inline]
pub fn safe_div(a: f64, b: f64) -> f64 {
    let denom = if b.abs() < MIN_NORM_EPSILON {
        b.signum() * MIN_NORM_EPSILON
    } else {
        b
    };
    a / denom
}

/// Safe sqrt that clamps input to non-negative.
#[inline]
pub fn safe_sqrt(x: f64) -> f64 {
    x.max(MIN_NORM_EPSILON).sqrt()
}

/// Softplus function: ln(1 + exp(x)).
/// Used for curvature constraints.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Derivative of softplus: sigmoid(x) = 1 / (1 + exp(-x)).
#[inline]
pub fn softplus_grad(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Euclidean dot product.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm squared.
#[inline]
pub fn norm_squared(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Euclidean norm.
#[inline]
pub fn norm(v: &[f64]) -> f64 {
    safe_sqrt(norm_squared(v))
}

/// Minkowski inner product for Lorentz model.
/// ⟨x, y⟩_L = -x₀y₀ + Σxᵢyᵢ
#[inline]
pub fn minkowski_dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert!(x.len() == y.len() && !x.is_empty());
    -x[0] * y[0] + x[1..].iter().zip(y[1..].iter()).map(|(a, b)| a * b).sum::<f64>()
}

/// Minkowski norm squared (can be negative for spacelike vectors).
#[inline]
pub fn minkowski_norm_squared(v: &[f64]) -> f64 {
    minkowski_dot(v, v)
}

/// Lorentzian norm for tangent vectors (which are spacelike, so positive).
#[inline]
pub fn lorentz_norm(v: &[f64]) -> f64 {
    safe_sqrt(minkowski_dot(v, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_arcosh_boundary() {
        assert!(!safe_arcosh(0.5).is_nan());
        assert!(!safe_arcosh(1.0).is_nan());
        assert!(safe_arcosh(2.0) > 0.0);
    }

    #[test]
    fn test_safe_artanh_clamping() {
        assert!(!safe_artanh(-1.5).is_nan());
        assert!(!safe_artanh(1.5).is_nan());
        assert!(safe_artanh(0.5).abs() < 1.0);
    }

    #[test]
    fn test_softplus_positive() {
        assert!(softplus(-10.0) > 0.0);
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(10.0) > 0.0);
    }

    #[test]
    fn test_minkowski_dot_signature() {
        let x = vec![2.0, 1.0, 0.0];
        let y = vec![2.0, 1.0, 0.0];
        assert!((minkowski_dot(&x, &y) - (-3.0)).abs() < EPSILON);
    }
}
