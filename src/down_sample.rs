use num::traits::{Float, FloatConst, NumAssign, Zero};

use rustfft::FftNum;

use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul},
};

use crate::windowed_fir::coeff;

pub struct DownSampler<U, T> {
    pub coeff_rev: Vec<T>,
    /// filter state
    pub initial_state: Vec<U>,
    down_sample_ratio: usize,
}

impl<U, T> DownSampler<U, T>
where
    T: Float + FloatConst + NumAssign + std::iter::Sum<T> + Debug + FftNum + Copy,
    U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default + Zero + Debug,
{
    pub fn new(tap: usize, down_sample_ratio: usize) -> Self {
        let c = coeff(tap, T::one() / T::from(down_sample_ratio).unwrap());
        let norm = c.iter().cloned().sum::<T>();
        let c: Vec<_> = c.iter().map(|&x| x / norm).collect();
        Self::from_coeffs(&c, down_sample_ratio)
    }

    pub fn from_coeffs(c: &[T], down_sample_ratio: usize) -> Self {
        let tap = c.len();
        let mut c = c.to_vec();
        c.reverse();
        Self {
            coeff_rev: c,
            initial_state: vec![U::zero(); tap - 1],
            down_sample_ratio,
        }
    }

    pub fn downsample(&mut self, input: &[U]) -> Vec<U> {
        self.initial_state.extend_from_slice(input);
        let tap = self.coeff_rev.len();
        let l = self.initial_state.len() - tap + 1;
        let noutput = l / self.down_sample_ratio;
        let new_init_state = self.initial_state[noutput * self.down_sample_ratio..].to_vec();
        let result = self
            .initial_state
            .windows(tap)
            .step_by(self.down_sample_ratio)
            .take(noutput)
            .map(|x| {
                x.iter()
                    .zip(self.coeff_rev.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<U>()
            })
            .collect();
        self.initial_state = new_init_state;
        result
    }
}

pub struct DirectDownSampler<U> {
    pub initial_state: Vec<U>,
    down_sample_ratio: usize,
}

impl<U> DirectDownSampler<U>
where
    U: Copy + Debug,
{
    pub fn new(down_sample_ratio: usize) -> Self {
        Self {
            initial_state: vec![],
            down_sample_ratio,
        }
    }

    pub fn down_sample(&mut self, input: &[U]) -> Vec<U> {
        self.initial_state.extend_from_slice(input);
        let l = self.initial_state.len();
        let noutput = l / self.down_sample_ratio;
        let new_init_state = self.initial_state[noutput * self.down_sample_ratio..].to_vec();
        let result = self
            .initial_state
            .iter()
            .step_by(self.down_sample_ratio)
            .cloned()
            .collect();
        self.initial_state = new_init_state;
        result
    }
}
