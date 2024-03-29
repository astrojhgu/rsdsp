//! oversampling poly phase filter bank

#![allow(clippy::uninit_vec)]
use crate::{batch_filter::BatchFilter, oscillator::HalfChShifter, utils::polyphase_decomp};
use ndarray::{parallel::prelude::*, s, Array1, Array2, Axis, ScalarOperand};
use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use serde::{Deserialize, Serialize};

use rustfft::{FftNum, FftPlanner};
use std::{iter::Sum, ops::Mul};

#[derive(Debug, Serialize, Deserialize)]
pub struct Analyzer<R, T>
where
    R: std::fmt::Debug,
    T: std::fmt::Debug + Float,
{
    /// filters for even channels
    filter_even: BatchFilter<R, T>,

    /// filters for odd channels
    filter_odd: BatchFilter<Complex<T>, T>,

    /// a buffer, ensurning that the input signal length need not to be nch*tap. The remaining elements will be stored and be concated with the input next time.
    buffer: Vec<R>,

    /// shifting input signal by half of the channel spacing
    shifter: HalfChShifter<T>,
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
        + Default
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sum
        + Sync
        + Send,
    Complex<T>: Copy + std::convert::From<R> + Sum + Default + ScalarOperand,
{
    /// constructor
    /// * `nch_total` - total number of channels, including even and odd, pos and neg channels
    /// * `coeff` - property low pass filter coefficients, the length of which should be equal to `nch_total`/2*`tap_per_ch`
    /// ```
    /// extern crate rsdsp;
    /// use num::complex::Complex;
    /// use rsdsp::{
    ///     windowed_fir
    ///     , ospfb::Analyzer
    /// };
    ///
    /// let nch=32;
    /// let tap_per_ch=16;
    /// let k=1.1;
    /// let coeff=windowed_fir::pfb_coeff::<f64>(nch/2, tap_per_ch, k);
    /// let mut pfb=Analyzer::<Complex<f64>, f64>::new(nch, coeff.view());
    /// ```
    pub fn new(nch_total: usize, coeff: &[T]) -> Self {
        let nch_each = nch_total / 2;
        let tap = coeff.len() / nch_each;
        assert!(nch_each * tap == coeff.len());
        let coeff=polyphase_decomp(coeff, nch_each);
        let filter_even = BatchFilter::new(coeff.view());
        let filter_odd = BatchFilter::<Complex<T>, T>::new(coeff.view());

        let shifter = HalfChShifter::<T>::new(nch_each, false);

        Self {
            filter_even,
            filter_odd,
            buffer: Vec::<R>::new(),
            shifter,
        }
    }

    pub fn nch_total(&self) -> usize {
        self.filter_even.coeff[0].len() * 2
    }

    pub fn predict_output_length(&self, input_len: usize) -> usize {
        (self.buffer.len() + input_len) / self.filter_even.coeff[0].len()
    }

    pub fn buffer_input(&mut self, input_signal: &[R]) -> Vec<R> {
        let nch_each = self.filter_even.coeff[0].len();

        let batch = (self.buffer.len() + input_signal.len()) / nch_each;

        let signal = self
            .buffer
            .iter()
            .chain(input_signal)
            .take(nch_each * batch)
            .cloned()
            .collect();
        //self.buffer =
        //    ArrayView1::from(&input_signal[nch_each * batch - self.buffer.len()..]).to_vec();
        self.buffer
            .reserve(input_signal.len() + self.buffer.len() - nch_each * batch);
        unsafe {
            self.buffer
                .set_len(input_signal.len() + self.buffer.len() - nch_each * batch)
        };
        let l = self.buffer.len();
        self.buffer
            .iter_mut()
            .zip(&input_signal[input_signal.len() - l..])
            .for_each(|(a, &b)| *a = b);
        signal
    }

    /// performing the channelizing
    /// * `input_signal` - input 1-d time series of the input signal
    /// * return value - channelized signal, with `nch_total` rows
    /// ```
    /// extern crate rsdsp;
    /// use num::complex::Complex;
    /// use rsdsp::{
    ///     windowed_fir
    ///     , ospfb::Analyzer
    ///     , oscillator::COscillator
    /// };
    /// use num::traits::{FloatConst};
    ///
    /// let nch=32;
    /// let tap_per_ch=16;
    /// let k=1.1;
    /// let coeff=windowed_fir::pfb_coeff::<f64>(nch/2, tap_per_ch, k);
    /// let mut pfb=Analyzer::<Complex<f64>, f64>::new(nch, coeff.view());
    /// let mut osc=COscillator::<f64>::new(0.0, f64::PI()/(nch/2) as f64*4.0);//some certain frequency
    /// let input_signal:Vec<_>=(0..256).map(|_| osc.get()).collect();
    /// let channelized_signal=pfb.analyze(&input_signal);
    /// assert_eq!(channelized_signal.nrows(), nch);
    /// ```
    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch_each = self.filter_even.coeff[0].len();

        let nch_total = nch_each * 2;

        let batch = (self.buffer.len() + input_signal.len()) / nch_each;
        assert_eq!(batch, self.predict_output_length(input_signal.len()));

        let signal = self.buffer_input(input_signal);
        let signal_shifted = signal
            .iter()
            .map(|&x| Complex::<T>::from(x) * self.shifter.get())
            .collect::<Vec<_>>();

        //let mut x1 = self.filter_even.filter(signal.view());
        //let mut x2 = self.filter_odd.filter(signal_shifted.view());

        let mut result_even =
            unsafe { Array2::<Complex<T>>::uninit((batch, nch_each)).assume_init() };

        let mut result_odd =
            unsafe { Array2::<Complex<T>>::uninit((batch, nch_each)).assume_init() };

        let mut result = unsafe { Array2::<Complex<T>>::uninit((batch, nch_total)).assume_init() };

        result_even
            .axis_iter_mut(Axis(0))
            .zip(signal.chunks(nch_each))
            .for_each(|(mut result1, x_even)| {
                let y_even =
                    Array1::from(self.filter_even.filter(x_even)).map(|&r| Complex::<T>::from(r));
                result1.assign(&y_even);
            });
        result_odd
            .axis_iter_mut(Axis(0))
            .zip(signal_shifted.chunks(nch_each))
            .for_each(|(mut result1, x_odd)| {
                let y_odd = Array1::from(self.filter_odd.filter(x_odd));
                result1.assign(&y_odd);
            });

        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch_each);
        result
            .axis_iter_mut(Axis(0))
            .zip(
                result_even
                    .axis_iter_mut(Axis(0))
                    .zip(result_odd.axis_iter_mut(Axis(0))),
            )
            .for_each(|(mut result1, (mut re, mut ro))| {
                fft.process(re.as_slice_mut().unwrap());
                fft.process(ro.as_slice_mut().unwrap());

                result1.slice_mut(s![0_usize..;2]).assign(&re);
                result1.slice_mut(s![1_usize..;2]).assign(&ro);
            });

        result.t().as_standard_layout().to_owned()
    }

    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch_each = self.filter_even.coeff[0].len();

        let nch_total = nch_each * 2;

        let batch = (self.buffer.len() + input_signal.len()) / nch_each;
        assert_eq!(batch, self.predict_output_length(input_signal.len()));

        let signal = self.buffer_input(input_signal);
        let signal_shifted = signal
            .iter()
            .map(|&x| Complex::<T>::from(x) * self.shifter.get())
            .collect::<Vec<_>>();

        //let mut x1 = self.filter_even.filter(signal.view());
        //let mut x2 = self.filter_odd.filter(signal_shifted.view());

        let mut result_even =
            unsafe { Array2::<Complex<T>>::uninit((batch, nch_each)).assume_init() };

        let mut result_odd =
            unsafe { Array2::<Complex<T>>::uninit((batch, nch_each)).assume_init() };

        let mut result = unsafe { Array2::<Complex<T>>::uninit((batch, nch_total)).assume_init() };

        result_even
            .axis_iter_mut(Axis(0))
            .zip(signal.chunks(nch_each))
            .for_each(|(mut result1, x_even)| {
                let y_even = Array1::from(self.filter_even.filter_par(x_even))
                    .map(|&r| Complex::<T>::from(r));
                result1.assign(&y_even);
            });
        result_odd
            .axis_iter_mut(Axis(0))
            .zip(signal_shifted.chunks(nch_each))
            .for_each(|(mut result1, x_odd)| {
                let y_odd = Array1::from(self.filter_odd.filter_par(x_odd));
                result1.assign(&y_odd);
            });

        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch_each);
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(
                result_even
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(result_odd.axis_iter_mut(Axis(0)).into_par_iter()),
            )
            .for_each(|(mut result1, (mut re, mut ro))| {
                fft.process(re.as_slice_mut().unwrap());
                fft.process(ro.as_slice_mut().unwrap());

                result1.slice_mut(s![0_usize..;2]).assign(&re);
                result1.slice_mut(s![1_usize..;2]).assign(&ro);
            });

        result.t().as_standard_layout().to_owned()
    }
}
