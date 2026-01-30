//! Euclidean manifold implementation.
//!
//! Based on manifold-spec.md Section 1.

use crate::helix_engine::types::ManifoldError;
use super::numerics::{dot, norm};
use super::traits::Manifold;

/// Euclidean space ℝ^d with flat geometry.
#[derive(Debug, Clone)]
pub struct EuclideanManifold {
    dim: usize,
}

impl EuclideanManifold {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Manifold for EuclideanManifold {
    fn intrinsic_dim(&self) -> usize {
        self.dim
    }

    fn ambient_dim(&self) -> usize {
        self.dim
    }

    fn curvature(&self) -> f64 {
        0.0
    }

    fn set_curvature(&mut self, kappa: f64) -> Result<(), ManifoldError> {
        if kappa != 0.0 {
            return Err(ManifoldError::InvalidCurvature(
                "Euclidean manifold has fixed curvature 0".to_string()
            ));
        }
        Ok(())
    }

    /// d(x, y) = ‖x - y‖₂
    fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, ManifoldError> {
        if x.len() != self.dim || y.len() != self.dim {
            return Err(ManifoldError::DimensionMismatch);
        }
        let diff: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        Ok(norm(&diff))
    }

    /// exp_x(v) = x + v
    fn exp_map(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim || v.len() != self.dim {
            return Err(ManifoldError::DimensionMismatch);
        }
        Ok(x.iter().zip(v.iter()).map(|(a, b)| a + b).collect())
    }

    /// log_x(y) = y - x
    fn log_map(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim || y.len() != self.dim {
            return Err(ManifoldError::DimensionMismatch);
        }
        Ok(y.iter().zip(x.iter()).map(|(a, b)| a - b).collect())
    }

    /// Identity projection (all points are valid).
    fn project(&self, v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if v.len() != self.dim {
            return Err(ManifoldError::DimensionMismatch);
        }
        Ok(v.to_vec())
    }

    /// Identity tangent projection.
    fn project_tangent(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim || v.len() != self.dim {
            return Err(ManifoldError::DimensionMismatch);
        }
        Ok(v.to_vec())
    }

    /// P_{x→y}(v) = v (identity transport in flat space).
    fn parallel_transport(&self, x: &[f64], y: &[f64], v: &[f64]) -> Result<Vec<f64>, ManifoldError> {
        if x.len() != self.dim || y.len() != self.dim || v.len() != self.dim {
            return Err(ManifoldError::DimensionMismatch);
        }
        Ok(v.to_vec())
    }

    fn origin(&self) -> Vec<f64> {
        vec![0.0; self.dim]
    }

    fn inner_product(&self, x: &[f64], u: &[f64], v: &[f64]) -> Result<f64, ManifoldError> {
        if x.len() != self.dim || u.len() != self.dim || v.len() != self.dim {
            return Err(ManifoldError::DimensionMismatch);
        }
        Ok(dot(u, v))
    }

    fn is_on_manifold(&self, x: &[f64], _tol: f64) -> bool {
        x.len() == self.dim && x.iter().all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let m = EuclideanManifold::new(3);
        let x = vec![0.0, 0.0, 0.0];
        let y = vec![3.0, 4.0, 0.0];
        assert!((m.distance(&x, &y).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_exp_log_inverse() {
        let m = EuclideanManifold::new(2);
        let x = vec![1.0, 2.0];
        let y = vec![4.0, 6.0];
        let v = m.log_map(&x, &y).unwrap();
        let y2 = m.exp_map(&x, &v).unwrap();
        for (a, b) in y.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
