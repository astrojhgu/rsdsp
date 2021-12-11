extern crate rsdsp;
use num::{
    traits::{
        FloatConst
    }
    , complex::Complex, Zero
};

use rustfft::{
    FftPlanner
};


use rsdsp::{
    up_sample::UpSampler
    , utils::{
        convolve_fft
    }
};

fn main() {
    let mut signal=vec![Complex::<f64>::zero();16];
    signal[0]=1_f64.into();
    let kernel:Vec<_>=(0..17).map(|x| Complex::<f64>::from(x as f64)).collect();
    let mut state=vec![Complex::<f64>::zero();16];
    convolve_fft(&mut signal, &kernel, &mut state);
    for x in &signal{
        println!("{}", x.re);
        
    }
    println!("===");
    for x in &state{
        println!("{}", x.re);
    }

    println!("===");
    let mut signal=vec![Complex::<f64>::zero();16];
    convolve_fft(&mut signal, &kernel, &mut state);
    for x in &signal{
        println!("{}", x.re);
    }

    println!("===");
    for x in &state{
        println!("{}", x.re);
    }
}
