//! Acceleration structures for ray-geometry intersection.
//!
//! Ray tracing is inherently expensive: to find what a ray hits, you must in
//! principle test it against every triangle in the scene. For a scene with
//! millions of triangles that would be impossibly slow. Acceleration structures
//! solve this by organizing geometry into a hierarchy so that large chunks of
//! the scene can be ruled out with a single cheap bounding-box test.
//!
//! The acceleration structure built here is a **BVH** (Bounding Volume
//! Hierarchy). Once built from a [`Scene`](crate::scene::Scene), it is
//! uploaded to the GPU and used by the path tracer for all ray queries.

pub mod bvh;

pub use bvh::Bvh;
