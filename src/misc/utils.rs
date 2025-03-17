use num_traits::{Num, NumCast, ToPrimitive};

/// Generates a vector of evenly spaced numbers over a specified interval.
///
/// This is equivalent to the NumPy [`linspace()`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) function.
///
/// # Arguments
/// * `start` - Start value of the sequence (supports `i32`, `i64`, `f32`, `f64`).
/// * `stop` - Stop value of the sequence.
/// * `samples` - Number of samples to be generated.
/// * `include_end` - Whether to include `stop` in the sequence.
///
/// # Returns
/// A `Vec<f64>` containing evenly spaced values.
///
/// # Examples
/// ```
/// use rusp::misc::utils::linspace;
///
/// let result = linspace(0, 10, 5, true);
/// assert_eq!(result, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
/// ```
pub fn linspace<T>(start: T, stop: T, samples: usize, include_end: bool) -> Vec<f64>
where
    T: NumCast + Copy + PartialOrd + ToPrimitive,
{
    if samples == 0 {
        return Vec::new();
    }

    let start_f = start.to_f64().unwrap();
    let stop_f = stop.to_f64().unwrap();
    let span = stop_f - start_f;

    let step = if include_end {
        span / (samples as f64 - 1.0)
    } else {
        span / (samples as f64)
    };

    let mut values = Vec::with_capacity(samples);
    let mut current = start_f;

    for _ in 0..samples {
        values.push(current);
        current += step;
    }

    if include_end {
        *values.last_mut().unwrap() = stop_f;
    }

    values
}

#[cfg(test)]
mod linspace_tests {
    use super::linspace;

    #[test]
    fn test_linspace_int_inclusive() {
        let result = linspace(0, 10, 5, true);
        assert_eq!(result, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_linspace_int_exclusive() {
        let result = linspace(0, 10, 5, false);
        assert_eq!(result, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_linspace_f64_inclusive() {
        let result = linspace(1.5, 3.0, 4, true);
        assert_eq!(result, vec![1.5, 2.0, 2.5, 3.0]);
    }

    #[test]
    fn test_linspace_f64_exclusive() {
        let result = linspace(1.5, 3.0, 4, false);
        assert_eq!(result, vec![1.5, 1.875, 2.25, 2.625]);
    }

    #[test]
    fn test_linspace_zero_samples() {
        let result = linspace(0, 10, 0, true);
        assert!(result.is_empty());
    }

    #[test]
    fn test_linspace_single_sample() {
        let result = linspace(5, 5, 1, true);
        assert_eq!(result, vec![5.0]);
    }

    #[test]
    fn test_linspace_large_samples() {
        let result = linspace(0.0, 1.0, 10, true);
        assert_eq!(result.len(), 10);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[9], 1.0);
    }
}

/// Generates a repeated sequence of evenly spaced numbers over a specified interval.
///
/// This is useful for periodic signal generation where the same sequence is needed multiple times.
///
/// # Arguments
/// * `start` - Start value of the sequence (supports `i32`, `i64`, `f32`, `f64`).
/// * `stop` - Stop value of the sequence.
/// * `samples` - Number of samples per sequence.
/// * `repeats` - Number of times the sequence should be concatenated.
///
/// # Returns
/// * `Vec<f64>` containing the repeated sequence.
///
/// # Example
/// ```
/// use rusp::misc::utils::linspace_repeated;
/// let result = linspace_repeated(0, 10, 5, 2);
/// assert_eq!(result, vec![0.0, 2.5, 5.0, 7.5, 10.0, 0.0, 2.5, 5.0, 7.5, 10.0]);
/// ```
pub fn linspace_repeated<T>(start: T, stop: T, samples: usize, repeats: usize) -> Vec<f64>
where
    T: NumCast + Copy + PartialOrd + ToPrimitive,
{
    if samples == 0 || repeats == 0 {
        return Vec::new();
    }

    let start_f = start.to_f64().unwrap();
    let stop_f = stop.to_f64().unwrap();
    let span = stop_f - start_f;

    let step = span / (samples as f64 - 1.0);
    let mut sequence = Vec::with_capacity(samples);

    let mut current = start_f;
    for _ in 0..samples {
        sequence.push(current);
        current += step;
    }

    let mut result = Vec::with_capacity(samples * repeats);
    for _ in 0..repeats {
        result.extend_from_slice(&sequence);
    }

    result
}

#[cfg(test)]
mod linspace_repeated_tests {
    use super::linspace_repeated;

    #[test]
    fn test_linspace_repeated_inclusive() {
        let result = linspace_repeated(0, 10, 5, 2);
        assert_eq!(
            result,
            vec![0.0, 2.5, 5.0, 7.5, 10.0, 0.0, 2.5, 5.0, 7.5, 10.0]
        );
    }

    #[test]
    fn test_linspace_repeated_single() {
        let result = linspace_repeated(0, 5, 6, 1);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_linspace_repeated_zero_repeats() {
        let result = linspace_repeated(0, 10, 5, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_linspace_repeated_zero_samples() {
        let result = linspace_repeated(0, 10, 0, 5);
        assert!(result.is_empty());
    }
}

/// Generates a sequence of evenly spaced numbers over a specified interval with a fixed step.
///
/// This is equivalent to the NumPy [`arange()`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) function.
///
/// # Arguments
/// * `start` - Start value of the sequence.
/// * `stop` - Stop value of the sequence.
/// * `step` - Spacing between elements (must be nonzero).
///
/// # Returns
/// * `Vec<i32> containing the evenly spaced values.`
///
/// # Panics
/// Panics if `step == 0` to prevent infinite loops.
///
/// # Examples
/// ```
/// use rusp::misc::utils::arange;
/// let result = arange(0,10,2);
/// assert_eq!(result, vec![0,2,4,6,8])
/// ```
pub fn arange(start: i32, stop: i32, step: i32) -> Vec<i32> {
    assert!(step != 0, "Step size cannot be zero");

    let mut values = Vec::new();
    let mut current = start;

    if step > 0 {
        while current < stop {
            values.push(current);
            current += step;
        }
    } else {
        while current > stop {
            values.push(current);
            current += step;
        }
    }

    values
}

#[cfg(test)]
mod arange_tests {
    use super::arange;

    #[test]
    fn test_arange_positive_step() {
        let result = arange(0, 10, 2);
        assert_eq!(result, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_arange_negative_step() {
        let result = arange(10, 0, -2);
        assert_eq!(result, vec![10, 8, 6, 4, 2]);
    }

    #[test]
    #[should_panic(expected = "Step size cannot be zero")]
    fn test_arange_zero_step() {
        arange(0, 10, 0);
    }
}

/// Reverses the order of elements in an array.
///
/// # Arguments
/// * `arr` - An array of elements (supports `i32`, `i64`, `f32`, `f64`).
///
/// # Returns
/// * `Vec<T>` - A new vector with elements in reverse order.
///
/// # Examples
/// ```
/// use num_complex::Complex;
/// use rusp::misc::utils::reverse;
/// let result = reverse(&[1, 2, 3, 4]);
/// assert_eq!(result, vec![4, 3, 2, 1]);
///
/// let result = reverse(&[1.5, 2.5, 3.5]);
/// assert_eq!(result, vec![3.5, 2.5, 1.5]);
///
/// let result = reverse(&[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
/// assert_eq!(result, vec![Complex::new(3.0, 4.0), Complex::new(1.0, 2.0)]);
/// ```
pub fn reverse<T>(arr: &[T]) -> Vec<T>
where
    T: Num + Copy,
{
    let mut reversed = arr.to_vec();
    reversed.reverse();
    reversed
}

#[cfg(test)]
mod reverse_tests {
    use super::reverse;
    use num_complex::Complex;

    #[test]
    fn test_reverse_int() {
        let result = reverse(&[1, 2, 3, 4]);
        assert_eq!(result, vec![4, 3, 2, 1]);
    }

    #[test]
    fn test_reverse_float() {
        let result = reverse(&[1.5, 2.5, 3.5]);
        assert_eq!(result, vec![3.5, 2.5, 1.5]);
    }

    #[test]
    fn test_reverse_complex() {
        let result = reverse(&[
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]);
        assert_eq!(
            result,
            vec![
                Complex::new(5.0, 6.0),
                Complex::new(3.0, 4.0),
                Complex::new(1.0, 2.0)
            ]
        );
    }

    #[test]
    fn test_reverse_unsorted_array() {
        let result = reverse(&[1, 4, 3]);
        assert_eq!(result, vec![3, 4, 1]);
    }

    #[test]
    fn test_reverse_empty() {
        let result: Vec<i32> = reverse(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_reverse_single_element() {
        let result = reverse(&[42]);
        assert_eq!(result, vec![42]);
    }
}
