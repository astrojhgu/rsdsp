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
    windowed_fir::{
        coeff
    }
};

pub struct DownSampler<U, T>{
    pub coeff_rev: Vec<T>,
    /// filter state
    pub initial_state: Vec<U>,
    down_sample_ratio: usize, 
}

impl<U, T> DownSampler<U, T>
where T: Float + FloatConst + NumAssign + std::iter::Sum<T> + Debug + FftNum + Copy,
U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default + Zero + Debug,
{
    pub fn new(tap: usize, down_sample_ratio:usize)->Self{
        let mut c=coeff(tap, T::one()/T::from(down_sample_ratio).unwrap());
        let tap=c.len();
        c.reverse();
        Self{
            coeff_rev:c,
            initial_state: vec![U::zero(); tap-1]
            , down_sample_ratio
        }
    }

    pub fn downsample(&mut self, input: &[U])->Vec<U>{
        self.initial_state.extend_from_slice(input);
        let tap=self.coeff_rev.len();
        let l=self.initial_state.len()-tap+1;
        let noutput=l/self.down_sample_ratio;
        let new_init_state=self.initial_state[noutput*self.down_sample_ratio..].iter().cloned().collect::<Vec<_>>();
        //println!("{:?}", self.initial_state);
        //println!("{:?}", new_init_state);
        let result=self.initial_state.windows(tap).step_by(self.down_sample_ratio).take(noutput).map(|x| {
            //println!("{:?}", x);
            x.iter().zip(self.coeff_rev.iter()).map(|(&a,&b)| a*b).sum::<U>()
        }).collect();
        self.initial_state=new_init_state;
        result
    }
}
