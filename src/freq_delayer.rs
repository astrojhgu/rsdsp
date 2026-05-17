use ndarray::{Array1, ArrayViewMut2, Axis, ScalarOperand};
use num::{
    Complex, One,
    traits::{Float, FloatConst, NumAssign},
};

use crate::frac_delayer::{FracDelayer, ToDelayValue};

#[derive(Clone, Debug)]
pub struct FreqDelayer<T>
where
    T: std::fmt::Debug,
{
    frac_delayers: Vec<FracDelayer<T, Complex<T>>>,
    wgt: Vec<Complex<T>>,
}

impl<T> FreqDelayer<T>
where
    T: Copy
        + Float
        + FloatConst
        + std::ops::MulAssign<T>
        + ScalarOperand
        + NumAssign
        + std::iter::Sum
        + std::fmt::Debug
        + Sync
        + Send,
    Complex<T>: ScalarOperand,
{
    pub fn new(nch: usize, max_delay: usize, half_tap: usize) -> FreqDelayer<T> {
        FreqDelayer {
            frac_delayers: vec![FracDelayer::new(max_delay, half_tap); nch],
            wgt: vec![Complex::<T>::one(); nch],
        }
    }

    pub fn update_delay_value(&mut self, dv: T) {
        let dv1 = dv.to_delay_value();
        let d = (T::from(dv1.i).unwrap() + dv1.f) / T::from(self.frac_delayers.len()).unwrap()
            * T::from(2).unwrap();
        println!("{:?}", d);
        self.frac_delayers.iter_mut().for_each(|fd| {
            fd.update_delay_value(d);
            //fd.update_delay_value(T::zero());
        });

        let nch = self.frac_delayers.len();
        let wgt: Vec<Complex<T>> = (0..nch)
            .map(|k| {
                let k_signed = if k <= nch / 2 {
                    T::from(k).unwrap()
                } else {
                    T::from(k as isize - nch as isize).unwrap()
                };

                let angle = -T::from(2).unwrap() * T::PI() * k_signed * dv / T::from(nch).unwrap();

                Complex::<T>::from_polar(T::one(), angle)
            })
            .collect();
        self.wgt = wgt;
    }

    pub fn delay(&mut self, mut x: ArrayViewMut2<Complex<T>>) {
        x.axis_iter_mut(Axis(0))
            .zip(self.frac_delayers.iter_mut())
            .zip(self.wgt.iter())
            .for_each(|((mut x1, d1), &w)| {
                let mut y1 = Array1::from(d1.delay(x1.as_slice().unwrap()));
                //println!("{:?}", y1);
                y1 *= w;
                x1.assign(&y1);
            });
    }
}
