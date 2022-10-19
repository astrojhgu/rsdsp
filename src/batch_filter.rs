//! A module containing FIR filter

use serde::{Deserialize, Serialize};

use std::{
    collections::VecDeque,
    iter::Sum,
    ops::{Add, Mul},
};

use num::complex::Complex;

use ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::filter;

/// FIR filter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchFilter<U, T>
where
    T: std::fmt::Debug,
    U: std::fmt::Debug,
{
    /// reversed coefficients, i.e., impulse respone
    pub filters: Vec<filter::Filter<U, T>>,
    //pub u: PhantomData<U>,
}

impl<U, T> BatchFilter<U, T>
where
    T: Copy + Sync + Send + std::fmt::Debug,
    U: Copy
        + Add<U, Output = U>
        + Mul<T, Output = U>
        + Sum
        + Default
        + Sync
        + Send
        + std::fmt::Debug,
    Complex<T>: std::convert::From<U> + std::fmt::Debug,
{
    /// construct a FIR with its coefficients    
    pub fn new(coeff: ArrayView2<T>) -> Self {
        //coeff.reverse();
        //let tap = coeff.shape()[0];
        let nch = coeff.shape()[0];
        let filters: Vec<_> = (0..nch)
            .map(|i| filter::Filter::<U, T>::new(coeff.slice(s![i, ..]).to_vec()))
            .collect();

        Self {
            filters,
            //u: PhantomData {},
        }
    }

    /// filter a time series signal
    /// return the filtered signal

    pub fn filter(&mut self, signal: ArrayView1<U>) -> Array2<Complex<T>> {
        let nch = self.filters.len();
        let batch = signal.len() / nch;
        /*
        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        */
        let mut x1: Array2<U> = signal
            .into_shape((batch, nch))
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();
        self.filters
            .iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });
        x1.t().as_standard_layout().map(|&x| Complex::<T>::from(x))
        //x1
    }

    pub fn feed(&mut self, signal: ArrayView1<U>) {
        let nch = self.filters.len();
        let batch = signal.len() / nch;
        /*
        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        */
        let x1: Array2<U> = signal
            .into_shape((batch, nch))
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();
        self.filters
            .iter_mut()
            .zip(x1.axis_iter(Axis(0)))
            .enumerate()
            .for_each(|(_i, (ft, x1_row))| ft.feed(x1_row.as_slice().unwrap()));
    }

    pub fn filter_par(&mut self, signal: ArrayView1<U>) -> Array2<Complex<T>> {
        let nch = self.filters.len();
        let batch = signal.len() / nch;
        /*
        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        */
        let mut x1: Array2<U> = signal
            .into_shape((batch, nch))
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();
        self.filters
            .par_iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });
        x1.t().as_standard_layout().map(|&x| Complex::<T>::from(x))
    }
}

///////////////////////////////////
/// FIR filter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchFilterFixed<U, T>
where
    T: std::fmt::Debug,
    U: std::fmt::Debug,
{
    /// reversed coefficients, i.e., impulse respone
    //pub filters: Vec<filter::Filter<U, T>>,
    pub coeff: Vec<Vec<T>>,
    pub state: VecDeque<Vec<U>>,
}

impl<U, T> BatchFilterFixed<U, T>
where
    T: Copy + Sync + Send + std::fmt::Debug,
    U: Copy
        + Add<U, Output = U>
        + Mul<T, Output = U>
        + Sum
        + Default
        + Sync
        + Send
        + std::fmt::Debug,
    Complex<T>: std::convert::From<U> + std::fmt::Debug,
{
    /// construct a FIR with its coefficients    
    pub fn new(coeff: ArrayView2<T>) -> Self {
        //coeff.reverse();
        //let tap = coeff.shape()[0];
        let nch = coeff.shape()[0];
        let tap = coeff.shape()[1];

        let coeff = coeff
            .axis_iter(Axis(1))
            .rev()
            .map(|x| {
                let x = x.to_owned();
                x.to_vec()
            })
            .collect::<Vec<_>>();
        let state = (0..tap - 1)
            .map(|_| vec![<U as Default>::default(); nch])
            .collect();

        Self { coeff, state }
    }

    /// filter a time series signal
    /// return the filtered signal
    pub fn filter(&mut self, signal: &[U]) -> Vec<U> {
        let nch = self.coeff[0].len();
        assert_eq!(nch, signal.len());

        self.state.push_back(signal.to_vec());

        let result = self
            .state
            .iter()
            .zip(self.coeff.iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(&a1, &b1)| a1 * b1)
                    .collect::<Vec<_>>()
            })
            .reduce(|a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|(&a1, &b1)| a1 + b1)
                    .collect::<Vec<_>>()
            })
            .unwrap();

        self.state.pop_front();
        assert_eq!(result.len(), nch);
        result
    }

    pub fn feed(&mut self, signal: &[U]) {
        let nch = self.coeff[0].len();
        assert_eq!(nch, signal.len());
        self.state.push_front(signal.to_vec());
        self.state.pop_back();
    }

    pub fn filter_par(&mut self, signal: &[U]) -> Vec<U> {
        let nch = self.coeff[0].len();
        assert_eq!(nch, signal.len());

        self.state.push_back(signal.to_vec());

        let result = self
            .state
            .par_iter()
            .zip(self.coeff.par_iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(&a1, &b1)| a1 * b1)
                    .collect::<Vec<_>>()
            })
            .reduce(
                || vec![<U as Default>::default(); nch],
                |a, b| {
                    a.iter()
                        .zip(b.iter())
                        .map(|(&a1, &b1)| a1 + b1)
                        .collect::<Vec<_>>()
                },
            );

        self.state.pop_front();
        assert_eq!(result.len(), nch);
        result
    }
}
