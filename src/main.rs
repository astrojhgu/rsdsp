extern crate rsdsp;
use ndarray::ArrayView1;
use ndarray_npy::write_npy;
use num::{complex::Complex, Zero};
use rsdsp::{ampl_resp::ampl_resp, ospfb::Analyzer, utils::convolve_fft, windowed_fir::pfb_coeff};

fn main() {
    let nch = 32;
    let tap = 16;
    let k = 1.1;
    let coeff = pfb_coeff::<f64>(nch / 2, tap, k).into_raw_vec();
    let mut pfb = Analyzer::<Complex<f64>, f64>::new(nch, ArrayView1::from(&coeff));
    let result = ampl_resp(&mut pfb, -1.0, 1.0, 512, 65536, 2);
    write_npy("a.npy", &result);
}
