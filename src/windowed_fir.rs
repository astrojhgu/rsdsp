#![allow(clippy::many_single_char_names)]
use crate::{
    utils::{fftshift, ifft},
    window_funcs::apply_blackman_window,
};
use num::{
    complex::{
        Complex
    }
    , traits::{
        Float
        , FloatConst
        , NumAssign
        , Zero
    }
};

use rustfft::FftNum;


pub fn lp_coeff<T>(tap: usize, k: T) -> Vec<T>
where
    T: Float,
{
    (0..tap)
        .map(|i| {
            let x = T::from(if i < tap / 2 {
                i
            } else {
                tap - i
            })
            .unwrap();
            let y = T::from(tap / 2).unwrap() * k;
            if x >= y {
                T::zero()
            } else if x < y.floor() {
                T::one()
            } else {
                y - y.floor()
            }
        })
        .collect()
}

pub fn symmetrize<T: Copy + Zero>(workpiece: &mut [T]) {
    let n1 = workpiece.len();

    for n in 0..=(n1 / 2 - 2) {
        workpiece[n1 - n - 1] = workpiece[n + 1];
    }
    //workpiece[n1 / 2] = T::zero();
}

pub fn to_time_domain<T>(input: &[T]) -> Vec<T>
where
    T: Float + FloatConst + NumAssign + std::iter::Sum<T> + std::fmt::Debug + FftNum,
{
    let a: Vec<_> = input.iter().map(Complex::<T>::from).collect();
    let b: Vec<_> = ifft(&a[..]).iter().map(|x| x.re).collect();
    let s = b.iter().cloned().sum();
    let b: Vec<_> = b.iter().map(|&x| x / s).collect();
    fftshift(&b)
}

pub fn coeff<T>(tap: usize, k: T) -> Vec<T>
where
    T: Float + FloatConst + NumAssign + std::iter::Sum<T> + std::fmt::Debug + FftNum,
{
    let mut a = lp_coeff(tap, k);
    symmetrize(&mut a);
    let mut b = to_time_domain(&a);
    //apply_hamming_window(&mut b);
    apply_blackman_window(&mut b);
    //Array1::from(b).into_shape((l, m)).unwrap().t().to_owned()
    b
}

pub fn shift_coeff<T>(input: &[T], w: T)->Vec<Complex<T>>
where T: Float + FloatConst + NumAssign + std::fmt::Debug ,{
    let tap=input.len() as isize;
    input.iter().enumerate().map(|(i, &x)|{
        let f=(Complex::new(T::zero(), T::one())*w*T::from(i as isize-tap as isize/2).unwrap()).exp();
        f*x
    }).collect()
}

pub fn shift_coeff_real<T>(input: &[T], w: T)->Vec<T>
where T: Float + FloatConst + NumAssign + std::fmt::Debug ,{
    let tap=input.len() as isize;
    input.iter().enumerate().map(|(i, &x)|{
        let f=(w*T::from(i as isize-tap as isize/2).unwrap()).cos();
        f*x
    }).collect()
}
