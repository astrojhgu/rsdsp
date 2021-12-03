use num::{
    traits::{
        Zero
        , Float, FloatConst
        , NumAssign
    }
};

use rustfft::{
    FftNum
};

use std::{
    iter::Sum
    , fmt::{
        Debug
    }
    , ops::{
        Add, Mul
    }
};

use crate::{
    filter::{
        Filter
    }
    , windowed_fir::{
        coeff
    }
};

pub fn insert_zeros<T>(input: &[T], up_sample_ratio: usize)->Vec<T>
where T: Zero+Copy
{
    let mut output=Vec::<T>::new();
    let zeros=vec![T::zero(); up_sample_ratio-1];
    for &x in input{
        output.push(x);
        output.extend_from_slice(&zeros);
    }
    output
}

pub struct UpSampler<U, T>{
    lp_filter: Filter<U, T>
    , up_sample_ratio: usize
}

impl<U, T> UpSampler<U, T>
where T: Float + FloatConst + NumAssign + std::iter::Sum<T> + Debug + FftNum + Copy,
U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default + Zero,
{
    pub fn new(tap: usize, up_sample_ratio:usize)->Self{
        let c=coeff(tap, T::one()/T::from(up_sample_ratio).unwrap());
        let filter=Filter::<U, T>::new(c);
        Self{
            lp_filter:filter
            , up_sample_ratio
        }
    }

    pub fn upsample(&mut self, input: &[U])->Vec<U>{
        let x=insert_zeros(input, self.up_sample_ratio);
        self.lp_filter.filter(&x)
    }
}
