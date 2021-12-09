extern crate rsdsp;
use num::{
    traits::{
        FloatConst
    }
};


use rsdsp::{
    up_sample::UpSampler
};

fn main() {
    let dphi=f64::PI()/2.0;
    let mut upsampler=UpSampler::<f64,f64>::new(16, 8);
    let signal:Vec<_>=(0..1024).map(|i| (i as f64*dphi).cos()).collect();
    let signal1=upsampler.up_sample(&signal);
    for &x in &signal1{
        println!("{}", x);
    }
}
