//! Manifold Geometry Core Module
//!
//! Provides first-class manifold primitives for helix-db:
//! - Euclidean, Hyperbolic (Lorentz/Poincaré), Spherical spaces
//! - ProductManifold combinator for mixed-curvature embeddings
//! - Learnable curvature support for ML applications

pub mod traits;
pub mod numerics;
pub mod curvature;
pub mod euclidean;
pub mod hyperbolic;
pub mod spherical;
pub mod product;

pub use traits::Manifold;
pub use curvature::{LearnableCurvature, CurvatureSign};
pub use euclidean::EuclideanManifold;
pub use hyperbolic::{HyperbolicManifold, HyperbolicModel};
pub use spherical::SphericalManifold;
pub use product::ProductManifold;
