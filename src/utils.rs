#![allow(non_snake_case)]
use std::{
    ops::{
        Index
        , Add
        , Mul
    }
    , iter::{
        Sum
    }, fmt::Debug
};
/*
use fftw::{
    plan::{C2CPlan, C2CPlan32, C2CPlan64},
    types::{Flag, Sign},
};*/
use rustfft::{FftNum, FftPlanner};

use ndarray::{s, Array1, Array2, ArrayView2};

use num::{
    complex::Complex
    , traits::{
        Float
        , FloatConst
        , Num
        , NumAssign
        , NumCast
    }
};



pub fn fft<T>(in_data: &[Complex<T>]) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    //let mut fft = CFft1D::<T>::with_len(in_data.len());
    let mut output = Vec::from(in_data);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(in_data.len());
    fft.process(&mut output);
    output
}

pub fn ifft<T>(in_data: &[Complex<T>]) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    let mut output = Vec::from(in_data);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(in_data.len());
    fft.process(&mut output);
    let norm = T::from(in_data.len()).unwrap();
    output.iter_mut().for_each(|x| *x /= norm);
    output
}

pub fn ifft0<T>(in_data: &[Complex<T>]) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    let mut output = Vec::from(in_data);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(in_data.len());
    fft.process(&mut output);
    output
}


pub fn fftshift<T>(in_data: &[T]) -> Vec<T>
where
    T: Copy,
{
    let mut result = Vec::with_capacity(in_data.len());
    let n = (in_data.len() + 1) as usize / 2;
    //for i in n..in_data.len() {
    for item in in_data.iter().skip(n).cloned() {
        //result[i-n-1]=in_data[i];
        result.push(item);
    }

    for item in in_data.iter().take(n).cloned() {
        //result[i+n]=in_data[i];
        result.push(item);
    }
    result
}

pub fn fftshift2<T>(in_data: ArrayView2<T>) -> Array2<T>
where
    T: Copy,
{
    assert!(in_data.shape()[0] % 2 == 0);
    let n2 = in_data.shape()[0] / 2;
    let mut result =
        unsafe { Array2::uninit((in_data.shape()[0], in_data.shape()[1])).assume_init() };

    for i in 0..n2 {
        result
            .slice_mut(s![i, ..])
            .assign(&in_data.slice(s![i + n2, ..]));
        result
            .slice_mut(s![i + n2, ..])
            .assign(&in_data.slice(s![i, ..]));
    }
    result
}


pub fn fftfreq<T>(n: usize) -> Vec<T>
where
    T: Float,
{
    let n = n as isize;
    let result = (0..=((n - 1) / 2))
        .chain(-n / 2..=-1)
        .map(|x| T::from(x).unwrap() / T::from(n).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(result.len(), n as usize);
    result
}


pub struct ConcatedSlice<'a, 'b, T> {
    pub old: &'a [T],
    pub appended: &'b [T],
}

impl<'a, 'b, T> ConcatedSlice<'a, 'b, T> {
    pub fn new(a: &'a [T], b: &'b [T]) -> ConcatedSlice<'a, 'b, T> {
        ConcatedSlice {
            old: a,
            appended: b,
        }
    }

    pub fn len(&self) -> usize {
        self.old.len() + self.appended.len()
    }

    pub fn is_empty(&self) -> bool {
        self.old.is_empty() && self.appended.is_empty()
    }
}

impl<'a, 'b, T> Index<usize> for ConcatedSlice<'a, 'b, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        let b = self.old.len();
        if idx < b {
            &self.old[idx]
        } else {
            &self.appended[idx - b]
        }
    }
}

pub fn add_neg_freq_no_dc<T>(input: Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: Num + Copy + std::ops::Neg<Output = T>,
{
    let mut result = Array2::<Complex<T>>::zeros((input.shape()[0] * 2, input.shape()[1]));

    result
        .slice_mut(s![1..input.shape()[0], ..])
        .assign(&input.slice(s![..input.shape()[0] - 1, ..]));
    result.slice_mut(s![input.shape()[0] + 1.., ..]).assign(
        &input
            .slice(s![..input.shape()[0]-1;-1,..])
            .map(|x| x.conj()),
    );
    result
}

pub fn apply_delay<T>(x: &mut Array2<Complex<T>>, d: T)
where
    T: Float + Copy + FloatConst + std::fmt::Debug,
{
    let two = T::one() + T::one();
    let freqs = fftfreq::<T>(x.shape()[0]);

    for (r, k) in x.rows_mut().into_iter().zip(
        freqs
            .into_iter()
            .map(|f| Complex::<T>::new(T::zero(), two * T::PI() * f * d).exp()),
    ) {
        for x1 in r {
            *x1 = *x1 * k;
        }
    }
}

pub fn corr<T>(x: &[T], y: &[T], fold_len: usize) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    let result: Array1<Complex<T>> = x
        .chunks(fold_len)
        .zip(y.chunks(fold_len))
        .map(|(x1, y1)| {
            let x1: Vec<_> = x1
                .iter()
                .map(|&x11| Complex::<T>::new(x11, T::zero()))
                .collect();
            let y1: Vec<_> = y1
                .iter()
                .map(|&y11| Complex::<T>::new(y11, T::zero()))
                .collect();
            let X1 = fft(&x1[..]);
            let Y1 = fft(&y1[..]);
            Array1::from(
                X1.iter()
                    .zip(Y1.iter())
                    .map(|(&x1, &y1)| x1 * y1.conj())
                    .collect::<Vec<_>>(),
            )
        })
        .fold(Array1::<Complex<T>>::zeros(fold_len), |x, y| {
            if y.len() == fold_len {
                x + y
            } else {
                x
            }
        });
    //fftshift(result.as_slice().unwrap())
    result.to_vec()
}

fn windows_mut_each<T>(v: &mut [T], n: usize, mut f: impl FnMut(&mut [T])) {
    let mut start = 0;
    let mut end = n;
    while end <= v.len()  {
        f(&mut v[start..end]);
        start += 1;
        end += 1;
    }
}

pub fn convolve<U, T>(signal: &mut [U], kernel:&[T], state: &mut Vec<U>)
where T: Copy,
    U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default,
{
    let tap=kernel.len();
    assert_eq!(state.len(), tap-1);
    state.extend_from_slice(signal);

    windows_mut_each(state, tap, |s|{
        let x=s.iter().zip(kernel.iter().rev()).map(|(&a, &b)| a*b).sum::<U>();
        s[0]=x;
    });

    //self.initial_state=self.initial_state[output_length..].to_vec();
    for (i, j) in (0..tap - 1).zip(state.len() + 1 - tap..state.len())
    {
        state[i] = state[j];
    }
    unsafe { state.set_len(tap - 1) };
}


pub fn convolve_fft<T>(signal: &mut [Complex<T>], kernel:&[Complex<T>], state: &mut Vec<Complex<T>>)
where T: Copy+FftNum+Default+NumCast,
{
    let tap=kernel.len();
    assert_eq!(state.len(), tap-1);
    let mut state1:Vec<_>=state.iter().chain(signal.iter()).cloned().collect();
    let noutput=signal.len();
    
    state.copy_from_slice(&state1[noutput..]);
    drop(state);

    let nfft=state1.len();
    if nfft&(nfft-1)!=0{
        eprintln!("Warning: length {} not optimal for fft", nfft);
    }

    let tailing_zeros=(0..(nfft-kernel.len())).map(|_| Complex::<T>::default());
    let mut kernel1:Vec<_>=kernel.iter().cloned().chain(tailing_zeros).collect();
    let mut planner=FftPlanner::<T>::new();
    let fft=planner.plan_fft_forward(nfft);
    let ifft=planner.plan_fft_inverse(nfft);
    
    fft.process(&mut state1);
    fft.process(&mut kernel1);
    let norm=T::from(nfft).unwrap();
    state1.iter_mut().zip(kernel1.iter()).for_each(|(a,&b)|{
        *a=*a*b/norm;
    });
    ifft.process(&mut state1);
    signal.copy_from_slice(&state1[(tap-1)..]);
}
