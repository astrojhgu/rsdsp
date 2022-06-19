//! digital oscillator for single frequency complex signal
use num::{
    complex::Complex,
    traits::{Float, FloatConst},
};

use serde::{Deserialize, Serialize};

/// Complex oscillator
#[derive(Debug, Serialize, Deserialize)]
pub struct COscillator<T>
where
    T: Float,
{
    /// current phase
    pub phi: T,
    /// phase difference between points
    pub dphi_dpt: T,
}

impl<T> COscillator<T>
where
    T: Float,
{
    /// constructor
    pub fn new(phi: T, dphi_dpt: T) -> COscillator<T> {
        COscillator { phi, dphi_dpt }
    }

    /// get the next value
    pub fn get(&mut self) -> Complex<T> {
        let y = (Complex::<T>::new(T::zero(), T::one()) * self.phi).exp();
        self.phi = self.phi + self.dphi_dpt;
        y
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CFreqScanner<T> {
    pub phi: T,
    pub dphi_dpt: T,
    pub ddphi_dpt2: T,
}

impl<T> CFreqScanner<T>
where
    T: Float + FloatConst,
{
    pub fn new(phi: T, dphi_dpt: T, ddphi_dpt2: T) -> Self {
        CFreqScanner {
            phi,
            dphi_dpt,
            ddphi_dpt2,
        }
    }

    pub fn get(&mut self) -> Complex<T> {
        let y = (Complex::<T>::new(T::zero(), T::one()) * self.phi).exp();
        self.phi = self.phi + self.dphi_dpt;
        self.dphi_dpt = self.dphi_dpt + self.ddphi_dpt2;
        if self.dphi_dpt > T::PI() {
            self.dphi_dpt = self.dphi_dpt - T::from(2).unwrap() * T::PI();
        }
        if self.dphi_dpt < -T::PI() {
            self.dphi_dpt = self.dphi_dpt + T::from(2).unwrap() * T::PI();
        }
        y
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ROscillator<T> {
    pub phi: T,
    pub dphi_dpt: T,
}

impl<T> ROscillator<T>
where
    T: Float,
{
    pub fn new(phi: T, dphi_dpt: T) -> Self {
        ROscillator { phi, dphi_dpt }
    }

    pub fn get(&mut self) -> T {
        let y = self.phi.cos();
        self.phi = self.phi + self.dphi_dpt;
        y
    }
}

/// Shifting signal by half of the channel spacing
#[derive(Debug, Serialize, Deserialize)]
pub struct HalfChShifter<T>
where
    T: Float,
{
    /// number of channels
    nch: usize,
    /// buffered factor, so that they need not to be computed repeatly
    factor: Vec<Complex<T>>,
    idx: usize,
}

impl<T> HalfChShifter<T>
where
    T: Float + FloatConst + std::fmt::Debug,
{
    /// constructor from number of channels and the direction
    /// * `nch` -  number of channels
    /// * `upshift` - shifting the frequency upward (`true`) or downward (`false`)
    pub fn new(nch: usize, upshift: bool) -> HalfChShifter<T> {
        let mut osc = COscillator::<T>::new(
            T::zero(),
            if upshift {
                T::PI() / T::from(nch).unwrap()
            } else {
                -T::PI() / T::from(nch).unwrap()
            },
        );
        let mut factor = Vec::new();
        for _ in 0..nch * 2 {
            factor.push(osc.get());
        }
        HalfChShifter {
            nch,
            factor,
            idx: 0,
        }
    }

    /// get the next factor
    pub fn get(&mut self) -> Complex<T> {
        let x = self.factor[self.idx];
        self.idx = (self.idx + 1) % (2 * self.nch);
        x
    }
}
