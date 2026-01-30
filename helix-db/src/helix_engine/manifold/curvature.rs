//! Learnable curvature for manifolds.
//!
//! Based on manifold-spec.md Section 7.

use super::numerics::{softplus, softplus_grad};

/// Sign constraint for curvature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurvatureSign {
    Negative,
    Positive,
}

/// Learnable curvature parameter with sign constraints.
///
/// Uses softplus to ensure curvature stays in valid range:
/// - Negative: κ = -softplus(θ) for hyperbolic space
/// - Positive: κ = +softplus(θ) for spherical space
#[derive(Debug, Clone)]
pub struct LearnableCurvature {
    theta: f64,
    sign: CurvatureSign,
}

impl LearnableCurvature {
    /// Create new learnable curvature with initial kappa value.
    pub fn new(initial_kappa: f64, sign: CurvatureSign) -> Self {
        let theta = match sign {
            CurvatureSign::Negative => {
                let k = initial_kappa.abs().max(1e-6);
                inverse_softplus(k)
            }
            CurvatureSign::Positive => {
                let k = initial_kappa.abs().max(1e-6);
                inverse_softplus(k)
            }
        };
        Self { theta, sign }
    }

    /// Get the constrained curvature value.
    #[inline]
    pub fn kappa(&self) -> f64 {
        match self.sign {
            CurvatureSign::Negative => -softplus(self.theta),
            CurvatureSign::Positive => softplus(self.theta),
        }
    }

    /// Get the unconstrained parameter.
    #[inline]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Set the unconstrained parameter.
    #[inline]
    pub fn set_theta(&mut self, t: f64) {
        self.theta = t;
    }

    /// Gradient scale for backpropagation.
    /// dκ/dθ = ±sigmoid(θ)
    #[inline]
    pub fn gradient_scale(&self) -> f64 {
        match self.sign {
            CurvatureSign::Negative => -softplus_grad(self.theta),
            CurvatureSign::Positive => softplus_grad(self.theta),
        }
    }

    /// Get the sign constraint.
    #[inline]
    pub fn sign(&self) -> CurvatureSign {
        self.sign
    }
}

/// Inverse of softplus: θ = ln(exp(k) - 1).
fn inverse_softplus(k: f64) -> f64 {
    if k > 20.0 {
        k
    } else {
        (k.exp() - 1.0).max(1e-10).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_curvature_negative() {
        let c = LearnableCurvature::new(-1.0, CurvatureSign::Negative);
        assert!(c.kappa() < 0.0);
    }

    #[test]
    fn test_spherical_curvature_positive() {
        let c = LearnableCurvature::new(1.0, CurvatureSign::Positive);
        assert!(c.kappa() > 0.0);
    }

    #[test]
    fn test_curvature_update() {
        let mut c = LearnableCurvature::new(-1.0, CurvatureSign::Negative);
        let old_k = c.kappa();
        c.set_theta(c.theta() + 0.5);
        assert!(c.kappa() != old_k);
        assert!(c.kappa() < 0.0);
    }
}
