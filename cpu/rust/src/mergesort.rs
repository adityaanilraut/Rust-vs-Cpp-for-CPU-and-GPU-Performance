//! MergeSort Benchmark — Rust
//! ===========================
//! Custom recursive merge sort matching the C++ implementation.
//! Ensures kernel parity for fair comparison.

/// Recursive merge sort with temporary buffer.
pub fn mergesort(data: &mut [i32]) {
    let len = data.len();
    if len <= 1 {
        return;
    }
    let mut tmp = vec![0i32; len];
    mergesort_impl(data, &mut tmp, 0, len);
}

fn mergesort_impl(data: &mut [i32], tmp: &mut [i32], left: usize, right: usize) {
    if right - left <= 1 {
        return;
    }

    // Switch to insertion sort for small ranges (matches C++ optimization)
    if right - left <= 32 {
        for i in (left + 1)..right {
            let key = data[i];
            let mut j = i;
            while j > left && data[j - 1] > key {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = key;
        }
        return;
    }

    let mid = left + (right - left) / 2;
    mergesort_impl(data, tmp, left, mid);
    mergesort_impl(data, tmp, mid, right);
    merge(data, tmp, left, mid, right);
}

fn merge(data: &mut [i32], tmp: &mut [i32], left: usize, mid: usize, right: usize) {
    let mut i = left;
    let mut j = mid;
    let mut k = left;

    while i < mid && j < right {
        if data[i] <= data[j] {
            tmp[k] = data[i];
            i += 1;
        } else {
            tmp[k] = data[j];
            j += 1;
        }
        k += 1;
    }

    while i < mid {
        tmp[k] = data[i];
        i += 1;
        k += 1;
    }

    while j < right {
        tmp[k] = data[j];
        j += 1;
        k += 1;
    }

    data[left..right].copy_from_slice(&tmp[left..right]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_i32;

    #[test]
    fn test_mergesort() {
        let mut data = generate_random_i32(10_000, 42);
        mergesort(&mut data);
        assert!(data.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_mergesort_small() {
        let mut data = vec![5, 3, 1, 4, 2];
        mergesort(&mut data);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_mergesort_empty() {
        let mut data: Vec<i32> = vec![];
        mergesort(&mut data);
        assert!(data.is_empty());
    }
}
