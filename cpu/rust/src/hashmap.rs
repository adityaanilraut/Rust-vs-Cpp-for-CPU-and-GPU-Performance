//! HashMap Benchmark — Rust
//! =========================
//! Benchmarks std::collections::HashMap for insert and lookup.
//! Compares against C++ std::unordered_map.

use std::collections::HashMap;

/// Insert n integer key-value pairs into a HashMap.
pub fn hashmap_insert_int(keys: &[i64]) -> HashMap<i64, i64> {
    let mut map = HashMap::with_capacity(keys.len());
    for (i, &k) in keys.iter().enumerate() {
        map.insert(k, i as i64);
    }
    map
}

/// Look up queries in a pre-built HashMap, returning sum of found values.
pub fn hashmap_lookup_int(map: &HashMap<i64, i64>, queries: &[i64]) -> i64 {
    let mut sum: i64 = 0;
    for &q in queries {
        if let Some(&v) = map.get(&q) {
            sum += v;
        }
    }
    sum
}

/// Insert n string key-value pairs into a HashMap.
pub fn hashmap_insert_string(keys: &[String]) -> HashMap<String, i64> {
    let mut map = HashMap::with_capacity(keys.len());
    for (i, k) in keys.iter().enumerate() {
        map.insert(k.clone(), i as i64);
    }
    map
}

/// Generate random integer keys.
pub fn generate_int_keys(n: usize, seed: u64) -> Vec<i64> {
    let max_val = n as i64 * 10;
    let mut rng = seed;
    (0..n).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng as i64).abs() % max_val
    }).collect()
}

/// Generate random string keys (8-32 chars).
pub fn generate_string_keys(n: usize, seed: u64) -> Vec<String> {
    let mut rng = seed;
    let mut next = || -> u64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    };

    (0..n).map(|_| {
        let len = 8 + (next() % 25) as usize;
        (0..len).map(|_| (b'a' + (next() % 26) as u8) as char).collect()
    }).collect()
}

/// Generate lookup queries: alternating hits and probable misses.
pub fn generate_queries(keys: &[i64], n: usize, num_queries: usize, seed: u64) -> Vec<i64> {
    let max_val = n as i64 * 10;
    let mut rng = seed;
    let mut next = || -> u64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    };

    (0..num_queries).map(|i| {
        if i % 2 == 0 {
            let idx = next() as usize % keys.len();
            keys[idx]
        } else {
            (next() as i64).abs() % max_val
        }
    }).collect()
}
