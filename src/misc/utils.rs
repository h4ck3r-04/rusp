use num_traits::{NumCast, ToPrimitive};

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
