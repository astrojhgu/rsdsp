//! oversampling poly phase filter bank

#![allow(clippy::uninit_vec)]
use crate::cspfb;
use ndarray::{s, Array2, Axis, ScalarOperand};
use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use serde::{Deserialize, Serialize};

use rustfft::{FftNum};
use std::{iter::Sum, ops::Mul};

#[derive(Debug, Serialize, Deserialize)]
pub struct Analyzer<R, T>
where
    R: std::fmt::Debug,
    T: std::fmt::Debug + Float,
{
    /// filters for even channels
    pfb_even: cspfb::Analyzer<R, T>,

    /// filters for odd channels
    pfb_odd: cspfb::Analyzer<R, T>,
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
    pub fn new(nch: usize, coeff: &[T]) -> Self {
        let tap = coeff.len() / nch;
        assert!(nch * tap == coeff.len());

        let mut pfb_even = cspfb::Analyzer::new(nch, coeff);
        let pfb_odd = cspfb::Analyzer::new(nch, coeff);

        let x0 = vec![R::default(); nch / 2];
        pfb_even.feed(&x0);

        Self { pfb_even, pfb_odd }
    }

    pub fn nch_total(&self) -> usize {
        self.pfb_even.nch() * 2
    }

    pub fn analyze_raw(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let y_even = self.pfb_even.analyze_raw(input_signal);
        let mut y_odd = self.pfb_odd.analyze_raw(input_signal);

        y_odd
            .axis_iter_mut(Axis(1))
            .skip(1)
            .step_by(2)
            .for_each(|mut y| y.map_inplace(|x| *x *= -T::one()));

        //println!("{:?}", y_even.slice(s![..,1]).map(|x|x.re));
        //println!("{:?}", y_odd.slice(s![..,1]).map(|x|x.re));
        let batch = y_even.shape()[0] * 2;
        let nch = y_even.shape()[1];
        assert_eq!(y_even.shape(), y_odd.shape());
        let mut result = unsafe { Array2::<Complex<T>>::uninit((batch, nch)).assume_init() };
        result.slice_mut(s![0_usize..;2,..]).assign(&y_even);
        result.slice_mut(s![1_usize..;2,..]).assign(&y_odd);
        result
    }

    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        self.analyze_raw(input_signal).t().as_standard_layout().to_owned()
    }

    pub fn analyze_raw_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let y_even = self.pfb_even.analyze_raw_par(input_signal);
        let mut y_odd = self.pfb_odd.analyze_raw_par(input_signal);

        y_odd
            .axis_iter_mut(Axis(1))
            .skip(1)
            .step_by(2)
            .for_each(|mut y| y.map_inplace(|x| *x *= -T::one()));

        //println!("{:?}", y_even.slice(s![..,1]).map(|x|x.re));
        //println!("{:?}", y_odd.slice(s![..,1]).map(|x|x.re));
        let batch = y_even.shape()[0] * 2;
        let nch = y_even.shape()[1];
        assert_eq!(y_even.shape(), y_odd.shape());
        let mut result = unsafe { Array2::<Complex<T>>::uninit((batch, nch)).assume_init() };
        result.slice_mut(s![0_usize..;2,..]).assign(&y_even);
        result.slice_mut(s![1_usize..;2,..]).assign(&y_odd);
        result
    }

    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        self.analyze_raw_par(input_signal).t().as_standard_layout().to_owned()
    }
}
