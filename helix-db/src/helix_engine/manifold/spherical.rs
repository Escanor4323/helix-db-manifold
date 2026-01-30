//! Spherical manifold implementation.
//!
//! Based on manifold-spec.md Section 3.

use crate::helix_engine::types::ManifoldError;
use super::curvature::{LearnableCurvature, CurvatureSign};
use super::numerics::{EPSILON, dot, norm, norm_squared};
use super::traits::Manifold;

/// Spherical manifold S^d embedded in R^(d+1).
#[derive(Debug, Clone)]
pub struct SphericalManifold {
    dim: usize,
    curvature: LearnableCurvature,
}

impl SphericalManifold {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            curvature: LearnableCurvature::new(1.0, CurvatureSign::Positive),
        }
    }

    pub fn with_curvature(dim: usize, kappa: f64) -> Result<Self, ManifoldError> {
        if kappa <= 0.0 {
            return Err(ManifoldError::InvalidCurvature(
                "Spherical curvature must be positive".to_string()
            ));
        }
        Ok(Self {
            dim,
            curvature: LearnableCurvature::new(kappa, CurvatureSign::Positive),
        })
    }

    /// Get the antipodal point.
    pub fn antipodal(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| -xi).collect()
    }

    /// Maximum distance on sphere (between antipodal points).
    pub fn max_distance(&self) -> f64 {
        std::f64::consts::PI / self.curvature.kappa().sqrt()
    }
}

impl Manifold for SphericalManifold {
    fn intrinsic_dim(&self) -> usize {
        self.dim
    }

    fn ambient_dim(&self) -> usize {
        self.dim + 1
    }

    fn curvature(&self) -> f64 {
        self.curvature.kappa()
    }

    fn set_curvature(&mut self, kappa: f64) -> Result<(), ManifoldError> {
        if kappa <= 0.0 {
            return Err(ManifoldError::InvalidCurvature(
                "Spherical curvature must be positive".to_string()
            ));
        }
        let theta = inverse_softplus_for(kappa);
        self.curvature.set_theta(theta);
        Ok(())
    }

    /// d(x, y) = (1/√κ) · arccos(κ⟨x,y⟩)
    fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, ManifoldError> {
        if x.len() != self.dim + 1 || y.len() != self.dim + 1 {
            return Err(ManifoldError::DimensionMismatch);
        }
        let kappa = self.curvature.kappa();
        let inner = dot(x, y);
        let arg = (kappa * inner).clamp(-1.0, 1.0);
        // For identical points, arg ≈ 1.0 and acos(1) = 0
        if arg >= 1.0 - EPSILON {
            return Ok(0.0);
        }
        Ok(arg.acos() / kappa.sqrt())
    }

    /// exp_x(v) = cos(√κ‖v‖)·x + sin(√κ‖v‖)·(v/√κ‖v‖)
    fn exp_map(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim + 1 || v.len() != self.dim + 1 {
            return Err(ManifoldError::DimensionMismatch);
        }
        let kappa = self.curvature.kappa();
        let v_norm = norm(v);
        if v_norm < EPSILON {
            return Ok(x.to_vec());
        }
        let sqrt_k = kappa.sqrt();
        let arg = sqrt_k * v_norm;
        let cos_arg = arg.cos();
        let sin_arg = arg.sin();
        Ok(x.iter().zip(v.iter())
            .map(|(xi, vi)| cos_arg * xi + sin_arg * vi / (sqrt_k * v_norm))
            .collect())
    }

    /// log_x(y) = d(x,y)/‖y - κ⟨x,y⟩x‖ · (y - κ⟨x,y⟩x)
    fn log_map(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim + 1 || y.len() != self.dim + 1 {
            return Err(ManifoldError::DimensionMismatch);
        }
        let d = self.distance(x, y)?;
        if d < EPSILON {
            return Ok(vec![0.0; self.dim + 1]);
        }
        let kappa = self.curvature.kappa();
        let inner = dot(x, y);
        let direction: Vec<f64> = y.iter().zip(x.iter())
            .map(|(yi, xi)| yi - kappa * inner * xi)
            .collect();
        let dir_norm = norm(&direction);
        Ok(direction.iter().map(|di| d * di / dir_norm).collect())
    }

    /// Project onto sphere: normalize to radius 1/√κ.
    fn project(&self, v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if v.len() != self.dim + 1 {
            return Err(ManifoldError::DimensionMismatch);
        }
        let kappa = self.curvature.kappa();
        let target_norm = 1.0 / kappa.sqrt();
        let v_norm = norm(v);
        if v_norm < EPSILON {
            let mut result = vec![0.0; self.dim + 1];
            result[0] = target_norm;
            return Ok(result);
        }
        Ok(v.iter().map(|vi| vi * target_norm / v_norm).collect())
    }

    /// Project onto tangent space: v - κ⟨v,x⟩x
    fn project_tangent(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim + 1 || v.len() != self.dim + 1 {
            return Err(ManifoldError::DimensionMismatch);
        }
        let kappa = self.curvature.kappa();
        let inner = dot(v, x);
        Ok(v.iter().zip(x.iter())
            .map(|(vi, xi)| vi - kappa * inner * xi)
            .collect())
    }

    /// P_{x→y}(v) = v - κ⟨x+y,v⟩/(1+κ⟨x,y⟩)·(x+y)
    fn parallel_transport(&self, x: &[f64], y: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim + 1 || y.len() != self.dim + 1 || v.len() != self.dim + 1 {
            return Err(ManifoldError::DimensionMismatch);
        }
        let kappa = self.curvature.kappa();
        let xy_sum: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        let inner_sum_v = dot(&xy_sum, v);
        let inner_xy = dot(x, y);
        let coeff = kappa * inner_sum_v / (1.0 + kappa * inner_xy);
        Ok(v.iter().zip(xy_sum.iter())
            .map(|(vi, si)| vi - coeff * si)
            .collect())
    }

    /// North pole: (1/√κ, 0, 0, ..., 0)
    fn origin(&self) -> Vec<f64> {
        let kappa = self.curvature.kappa();
        let mut o = vec![1.0 / kappa.sqrt()];
        o.extend(vec![0.0; self.dim]);
        o
    }

    fn inner_product(&self, x: &[f64], u: &[f64], v: &[f64]) -> Result<f64, ManifoldError> {
        if x.len() != self.dim + 1 || u.len() != self.dim + 1 || v.len() != self.dim + 1 {
            return Err(ManifoldError::DimensionMismatch);
        }
        Ok(dot(u, v))
    }

    fn is_on_manifold(&self, x: &[f64], tol: f64) -> bool {
        if x.len() != self.dim + 1 {
            return false;
        }
        let kappa = self.curvature.kappa();
        let target = 1.0 / kappa;
        (norm_squared(x) - target).abs() < tol
    }
}

fn inverse_softplus_for(k: f64) -> f64 {
    if k > 20.0 { k } else { (k.exp() - 1.0).max(1e-10).ln() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_origin_on_manifold() {
        let m = SphericalManifold::new(2);
        let o = m.origin();
        assert!(m.is_on_manifold(&o, 1e-10));
    }

    #[test]
    fn test_spherical_antipodal_max_distance() {
        let m = SphericalManifold::new(2);
        let o = m.origin();
        let anti = m.antipodal(&o);
        let d = m.distance(&o, &anti).unwrap();
        assert!((d - m.max_distance()).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_exp_log_inverse() {
        let m = SphericalManifold::new(2);
        let x = m.origin();
        let v = vec![0.0, 0.3, 0.4];
        let v_tangent = m.project_tangent(&x, &v).unwrap();
        let y = m.exp_map(&x, &v_tangent).unwrap();
        let v2 = m.log_map(&x, &y).unwrap();
        for (a, b) in v_tangent.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-8, "values differ: {} vs {}", a, b);
        }
    }
}
