//! N-Body Simulation Benchmark — Rust
//! =====================================
//! Direct pairwise gravitational force calculation, O(n²).
//! Tests FP throughput, auto-vectorization, and cache efficiency.

#[derive(Clone)]
pub struct Body {
    pub x: f64, pub y: f64, pub z: f64,
    pub vx: f64, pub vy: f64, pub vz: f64,
    pub mass: f64,
}

const SOFTENING: f64 = 1e-9;
const DT: f64 = 0.01;

/// Compute gravitational forces between all body pairs and update velocities.
pub fn compute_forces(bodies: &mut [Body]) {
    let n = bodies.len();
    for i in 0..n {
        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        let mut fz = 0.0_f64;

        for j in 0..n {
            if i == j { continue; }

            let dx = bodies[j].x - bodies[i].x;
            let dy = bodies[j].y - bodies[i].y;
            let dz = bodies[j].z - bodies[i].z;

            let dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
            let inv_dist = 1.0 / dist_sq.sqrt();
            let inv_dist3 = inv_dist * inv_dist * inv_dist;

            let force = bodies[j].mass * inv_dist3;
            fx += dx * force;
            fy += dy * force;
            fz += dz * force;
        }

        bodies[i].vx += DT * fx;
        bodies[i].vy += DT * fy;
        bodies[i].vz += DT * fz;
    }
}

/// Integrate positions using current velocities.
pub fn integrate_positions(bodies: &mut [Body]) {
    for b in bodies.iter_mut() {
        b.x += DT * b.vx;
        b.y += DT * b.vy;
        b.z += DT * b.vz;
    }
}

/// Generate n random bodies with deterministic seed.
pub fn generate_bodies(n: usize, seed: u64) -> Vec<Body> {
    let mut rng = seed;
    let mut next = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 11) as f64) / ((1u64 << 53) as f64)
    };

    (0..n).map(|_| Body {
        x: next() * 200.0 - 100.0,
        y: next() * 200.0 - 100.0,
        z: next() * 200.0 - 100.0,
        vx: next() * 2.0 - 1.0,
        vy: next() * 2.0 - 1.0,
        vz: next() * 2.0 - 1.0,
        mass: next() * 9.9 + 0.1,
    }).collect()
}
