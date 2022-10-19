//! Containing critical sampling poly phase filter bank

#![allow(clippy::uninit_vec)]

use crate::batch_filter::{BatchFilter, BatchFilterFixed};
use ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView1, Axis, ScalarOperand};
use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use serde::{Deserialize, Serialize};

use rustfft::{Fft, FftNum, FftPlanner};
use std::{iter::Sum, ops::Mul, sync::Arc};

/// Analyze channelizer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analyzer<R, T>
where
    R: std::fmt::Debug,
    T: std::fmt::Debug,
{
    /// A vec of filters, one for each branch
    batch_filter: BatchFilter<R, T>,
    /// a buffer, ensurning that the input signal length need not to be nch*tap. The remaining elements will be stored and be concated with the input next time.
    buffer: Vec<R>,
}

impl<R, T> Analyzer<R, T>
where
    T: Copy
        + Float
        + FloatConst
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sync
        + Send
        + FftNum,
    R: Copy
        + Mul<T, Output = R>
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sum
        + Default
        + Sync
        + Send,
    Complex<T>: Copy + std::convert::From<R> + Default + ScalarOperand,
{
    /// constructor
    /// * `nch` - number of channels including both pos and neg ones
    /// * `coeff` - prototype low-pass filter, the tap of which should be nch times of the tap of each branch.
    /// * return value - `Analyzer`
    pub fn new(nch: usize, coeff: ArrayView1<T>) -> Analyzer<R, T> {
        let tap = coeff.len() / nch;
        assert!(nch * tap == coeff.len());
        let coeff = coeff
            .into_shape((tap, nch))
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();
        let coeff = coeff.slice(s![..;-1,..]);

        let batch_filter = BatchFilter::new(coeff);

        Analyzer {
            batch_filter,
            buffer: Vec::<R>::new(),
        }
    }

    /// return the number of channels
    /// return value - the number of channels
    pub fn nch(&self) -> usize {
        self.batch_filter.filters.len()
    }

    pub fn predict_output_length(&self, input_len: usize) -> usize {
        (self.buffer.len() + input_len) / self.batch_filter.filters.len()
    }

    pub fn buffer_input(&mut self, input_signal: &[R]) -> Array1<R> {
        let nch = self.batch_filter.filters.len();
        let batch = (self.buffer.len() + input_signal.len()) / nch;
        let signal = Array1::from_iter(
            self.buffer
                .iter()
                .chain(input_signal)
                .take(nch * batch)
                .cloned(),
        );
        self.buffer
            .reserve(input_signal.len() + self.buffer.len() - nch * batch);
        unsafe {
            self.buffer
                .set_len(input_signal.len() + self.buffer.len() - nch * batch)
        };
        let l = self.buffer.len();
        self.buffer
            .iter_mut()
            .zip(&input_signal[input_signal.len() - l..])
            .for_each(|(a, &b)| *a = b);

        signal
    }

    /// Channelize input signal
    /// * `input_signal` - a 1-d slice containing time domain input signal
    /// * return value - channelized data, with `nch` rows
    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch = self.batch_filter.filters.len();
        let batch = (self.buffer.len() + input_signal.len()) / nch;

        let signal = self.buffer_input(input_signal);

        let mut x1 = self.batch_filter.filter(signal.view());
        let mut result = unsafe { Array2::<Complex<T>>::uninit((nch, batch)).assume_init() };
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch);

        result
            .axis_iter_mut(Axis(1))
            .zip(x1.axis_iter_mut(Axis(0)))
            .for_each(|(mut r_col, mut x1_row)| {
                fft.process(x1_row.as_slice_mut().unwrap());
                r_col.assign(&x1_row.view());
            });
        result
    }

    /// A parallel version of [`Self::analyze`]
    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch = self.batch_filter.filters.len();
        let batch = (self.buffer.len() + input_signal.len()) / nch;

        let signal = self.buffer_input(input_signal);

        let mut x1 = self.batch_filter.filter_par(signal.view());
        let mut result = unsafe { Array2::<Complex<T>>::uninit((nch, batch)).assume_init() };
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch);

        result
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(x1.axis_iter_mut(Axis(0)).into_par_iter())
            .for_each(|(mut r_col, mut x1_row)| {
                fft.process(x1_row.as_slice_mut().unwrap());
                r_col.assign(&x1_row.view());
            });
        result
    }
}

/// Analyze channelizer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalyzerFixed<R, T>
where
    R: std::fmt::Debug,
    T: std::fmt::Debug,
{
    /// A vec of filters, one for each branch
    batch_filter: BatchFilterFixed<R, T>,
    buffer: Vec<R>,
}

impl<R, T> AnalyzerFixed<R, T>
where
    T: Copy
        + Float
        + FloatConst
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sync
        + Send
        + FftNum,
    R: Copy
        + Mul<T, Output = R>
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sum
        + Default
        + Sync
        + Send,
    Complex<T>: Copy + std::convert::From<R> + Default + ScalarOperand,
{
    /// constructor
    /// * `nch` - number of channels including both pos and neg ones
    /// * `coeff` - prototype low-pass filter, the tap of which should be nch times of the tap of each branch.
    /// * return value - `Analyzer`
    pub fn new(nch: usize, coeff: ArrayView1<T>) -> Self {
        let tap = coeff.len() / nch;
        assert!(nch * tap == coeff.len());
        let coeff = coeff
            .into_shape((tap, nch))
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();
        let coeff = coeff.slice(s![..;-1,..]);

        let batch_filter = BatchFilterFixed::new(coeff);
        Self {
            batch_filter,
            buffer: Vec::new(),
        }
    }

    /// return the number of channels
    /// return value - the number of channels
    pub fn nch(&self) -> usize {
        self.batch_filter.coeff[0].len()
    }

    pub fn predict_output_length(&self, input_len: usize) -> usize {
        (self.buffer.len() + input_len) / self.nch()
    }

    pub fn buffer_input(&mut self, input_signal: &[R]) -> Vec<R> {
        let nch = self.nch();
        let batch = (self.buffer.len() + input_signal.len()) / nch;
        let signal = self
            .buffer
            .iter()
            .chain(input_signal)
            .take(nch * batch)
            .cloned()
            .collect();
        self.buffer
            .reserve(input_signal.len() + self.buffer.len() - nch * batch);
        unsafe {
            self.buffer
                .set_len(input_signal.len() + self.buffer.len() - nch * batch)
        };
        let l = self.buffer.len();
        self.buffer
            .iter_mut()
            .zip(&input_signal[input_signal.len() - l..])
            .for_each(|(a, &b)| *a = b);

        signal
    }

    /// Channelize input signal
    /// * `input_signal` - a 1-d slice containing time domain input signal
    /// * return value - channelized data, with `nch` rows
    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch = self.nch();
        let batch = (self.buffer.len() + input_signal.len()) / nch;
        let signal = self.buffer_input(input_signal);
        let mut result = unsafe { Array2::<Complex<T>>::uninit((batch, nch)).assume_init() };

        result
            .axis_iter_mut(Axis(0))
            .zip(signal.chunks(nch))
            .for_each(|(mut result1, x)| {
                let y = Array1::from(self.batch_filter.filter(x)).map(|&r| Complex::<T>::from(r));
                result1.assign(&y);
            });
        let mut planner = FftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(nch);

        result.axis_iter_mut(Axis(0)).for_each(|mut x| {
            fft.process(x.as_slice_mut().unwrap());
        });

        result.t().as_standard_layout().to_owned()
    }

    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch = self.nch();
        let batch = (self.buffer.len() + input_signal.len()) / nch;
        let signal = self.buffer_input(input_signal);
        let mut result = unsafe { Array2::<Complex<T>>::uninit((batch, nch)).assume_init() };

        result
            .axis_iter_mut(Axis(0))
            .zip(signal.chunks(nch))
            .for_each(|(mut result1, x)| {
                let y = Array1::from(self.batch_filter.filter_par(x)).map(|&r| Complex::<T>::from(r));
                result1.assign(&y);
            });
        let mut planner = FftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(nch);

        result.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x| {
            fft.process(x.as_slice_mut().unwrap());
        });

        result.t().as_standard_layout().to_owned()
    }
}
