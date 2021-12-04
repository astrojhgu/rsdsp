extern crate rsdsp;
use num::{
    traits::{
        FloatConst
    }
};


fn main() {
    let w=f64::PI()/8.0;
    //let signal:Vec<_>=(0..1024).map(|i| (Complex::<f64>::new(0.0, 1.0)*(i as f64*w)).exp()).collect();
    let signal:Vec<_>=(0..1024).map(|i| (i as f64*w).cos()).collect();
    //let signal:Vec<_>=(0..64).map(|i| i as f64).collect();
    let mut down_sampler=rsdsp::down_sample::DownSampler::<f64,f64>::new(4, 2);
    for x in down_sampler.downsample(&signal[..17]){
        println!("{}", x);
    }
    
    for x in down_sampler.downsample(&signal[17..]){
        println!("{}", x);
    }
}
