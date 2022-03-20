extern crate rsdsp;
use num::traits::FloatConst;

fn main() {
    let w = f64::PI() / 8.0;
    //let signal:Vec<_>=(0..1024).map(|i| (Complex::<f64>::new(0.0, 1.0)*(i as f64*w)).exp()).collect();
    let signal: Vec<_> = (0..1024).map(|i| (i as f64 * w).cos()).collect();
    let mut upsample = rsdsp::up_sample::UpSampler::<f64, f64>::new(32, 4);
    let signal1 = upsample.up_sample(&signal);
    for x in signal1 {
        println!("{}", x);
    }
}
