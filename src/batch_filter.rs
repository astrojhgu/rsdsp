//! A module containing FIR filter

use serde::{Deserialize, Serialize};

use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use num::complex::Complex;

use ndarray::{parallel::prelude::*, ArrayView2, Axis};

///////////////////////////////////
/// FIR filter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchFilter<U, T>
where
    T: std::fmt::Debug,
    U: std::fmt::Debug,
{
    /// reversed coefficients, i.e., impulse respone
    //pub filters: Vec<filter::Filter<U, T>>,
    pub coeff: Vec<Vec<T>>,
    pub state: Vec<Vec<U>>,
    pub head: usize,
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
        let tap = coeff.shape()[1];

        let coeff = coeff
            .axis_iter(Axis(1))
            .rev()
            .map(|x| {
                let x = x.to_owned();
                x.to_vec()
            })
            .collect::<Vec<_>>();
        let state = (0..tap)
            .map(|_| vec![<U as Default>::default(); nch])
            .collect();

        Self {
            coeff,
            state,
            head: 0,
        }
    }

    /// filter a time series signal
    /// return the filtered signal
    pub fn filter(&mut self, signal: &[U]) -> Vec<U> {
        let nch = self.coeff[0].len();
        let tap = self.coeff.len();
        assert_eq!(nch, signal.len());

        self.state[(self.head + tap - 1) % tap].clone_from_slice(signal);

        let result = self
            .state
            .iter()
            .cycle()
            .skip(self.head)
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

        self.head = (self.head + 1) % tap;
        assert_eq!(result.len(), nch);
        result
    }

    pub fn feed(&mut self, signal: &[U]) {
        let nch = self.coeff[0].len();
        let tap = self.coeff.len();
        assert_eq!(nch, signal.len());
        self.state[(self.head + tap - 1) % tap].clone_from_slice(signal);
        self.head = (self.head + 1) % tap;
    }

    pub fn filter_par(&mut self, signal: &[U]) -> Vec<U> {
        let nch = self.coeff[0].len();
        let tap = self.coeff.len();
        assert_eq!(nch, signal.len());

        self.state[(self.head + tap - 1) % tap].clone_from_slice(signal);

        let result = (0..tap)
            .into_par_iter()
            .zip(self.coeff.par_iter())
            .map(|(i, b)| {
                self.state[(i + self.head) % tap]
                    .iter()
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

        self.head = (self.head + 1) % tap;
        assert_eq!(result.len(), nch);
        result
    }
}
