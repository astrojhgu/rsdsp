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
    }, 
};

use ndarray::{
    Array1
    , Array2
    , Axis
    , s
};

use crate::{
windowed_fir::{
        coeff
    }
};


pub struct UpSampler<U, T>{
    pub coeffs:Array2<T>
    , pub init_state:Vec<U>
    , pub up_sample_ratio: usize
}

impl<U, T> UpSampler<U, T>
where T: Float + FloatConst + NumAssign + std::iter::Sum<T> + Debug + FftNum + Copy,
U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default + Zero,
{
    pub fn new(tap_per_ch: usize, up_sample_ratio:usize)->Self{
        let c=coeff(tap_per_ch*up_sample_ratio, T::one()/T::from(up_sample_ratio).unwrap());
        let norm=c.iter().cloned().sum::<T>()/T::from(up_sample_ratio).unwrap();
        let c:Vec<_>=c.iter().map(|&x| x/norm).collect();

        Self::from_coeffs(&c, up_sample_ratio)
    }

    pub fn from_coeffs(c: &[T], up_sample_ratio:usize)->Self{
        let tap_per_ch=c.len()/up_sample_ratio;
        assert_eq!(tap_per_ch*up_sample_ratio, c.len());
        //let c:Vec<_>=(0..(tap_per_ch*up_sample_ratio)).map(|x| T::from(x).unwrap()).collect();
        let coeffs=Array1::from_vec(c.to_vec())
        .into_shape((tap_per_ch, up_sample_ratio)).unwrap()
        .t()
        .as_standard_layout().to_owned();

        let coeffs=coeffs.slice(s![..;-1,..]).to_owned();
        let init_state=vec![U::default(); tap_per_ch-1];
        Self{
            coeffs
            , init_state
            , up_sample_ratio
        }
    }

    pub fn up_sample(&mut self, input: &[U])->Vec<U>{
        let output_len=input.len();
        self.init_state.extend(input);
        let mut output=Array2::<U>::zeros((self.up_sample_ratio, output_len));
        let tap=self.coeffs.ncols();
        self.coeffs.axis_iter(Axis(0)).zip(output.axis_iter_mut(Axis(0))).for_each(|(c, mut dest)|{
            self.init_state.windows(tap).zip(dest.iter_mut()).for_each(|(x, t)|{
                *t=x.iter().zip(c.iter()).map(|(&a,&b)|{
                    a*b
                }).sum::<U>();
            })
        });
        
        self.init_state=self.init_state[output_len..].to_vec();
        output.t().as_standard_layout().into_shape(output_len*self.up_sample_ratio).unwrap().to_vec()
    }
}
