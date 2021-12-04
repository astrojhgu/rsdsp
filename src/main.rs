extern crate rsdsp;
use num::{
    traits::{
        FloatConst
    }
};


fn main() {
    let data:Vec<_>=(0..1024).map(|x| x as f64).collect();
    let mut sampler=rsdsp::down_sample::DirectDownSampler::<f64>::new(12);
    let filtered=sampler.downsample(&data);
    for x in filtered{
        println!("{}", x);
    }
}
