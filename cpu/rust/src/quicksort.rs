//! QuickSort Benchmark — Rust
//! ===========================
//! Uses `slice::sort_unstable` (pattern-defeating quicksort).
//! This is Rust's recommended sorting for types without expensive equality.

/// Sort using Rust's built-in sort_unstable (pdqsort).
/// This is the idiomatic, safe Rust approach.
pub fn std_sort_unstable(data: &mut [i32]) {
    data.sort_unstable();
}

/// Manual quicksort with median-of-three pivot selection.
/// Matches the C++ manual implementation for kernel parity.
pub fn manual_quicksort(data: &mut [i32]) {
    if data.len() <= 1 {
        return;
    }
    quicksort_inner(data);
}

fn quicksort_inner(data: &mut [i32]) {
    if data.len() <= 16 {
        // Insertion sort for small slices
        insertion_sort(data);
        return;
    }

    // Median-of-three pivot
    let len = data.len();
    let mid = len / 2;
    let last = len - 1;

    if data[mid] < data[0] {
        data.swap(0, mid);
    }
    if data[last] < data[0] {
        data.swap(0, last);
    }
    if data[mid] < data[last] {
        data.swap(mid, last);
    }

    // Partition using the last element as pivot
    let pivot = data[last];
    let mut i = 0;
    for j in 0..last {
        if data[j] <= pivot {
            data.swap(i, j);
            i += 1;
        }
    }
    data.swap(i, last);

    // Recurse
    let (left, right) = data.split_at_mut(i);
    quicksort_inner(left);
    if right.len() > 1 {
        quicksort_inner(&mut right[1..]);
    }
}

fn insertion_sort(data: &mut [i32]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;
        while j > 0 && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_i32;

    #[test]
    fn test_std_sort() {
        let mut data = generate_random_i32(10_000, 42);
        std_sort_unstable(&mut data);
        assert!(data.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_manual_quicksort() {
        let mut data = generate_random_i32(10_000, 42);
        manual_quicksort(&mut data);
        assert!(data.windows(2).all(|w| w[0] <= w[1]));
    }
}
