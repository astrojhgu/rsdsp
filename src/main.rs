extern crate rsdsp;
use ndarray::{s, ArrayView1};
use ndarray_npy::write_npy;
use num::{complex::Complex, Zero};
use rsdsp::{ampl_resp::ampl_resp, ospfb::Analyzer, utils::convolve_fft, windowed_fir::pfb_coeff
    , batch_filter::{
        BatchFilter
        , BatchFilterFixed
    }
};

fn main() {
    let nch = 8;
    let tap = 4;
    let k = 1.1;
    let coeff = pfb_coeff::<f64>(nch, tap, k);

    let coeff = coeff
        .into_shape((tap, nch))
        .unwrap()
        .t()
        .as_standard_layout()
        .to_owned();
    let coeff = coeff.slice(s![..;-1,..]);
    println!("{:?}", coeff);


    let mut filter1=BatchFilter::<f64,_>::new(coeff.view());
    let mut filter2=BatchFilterFixed::<f64, _>::new(coeff.view());

    let signal=vec![1.0;nch];
    let output=filter1.filter(ArrayView1::from(&signal)).map(|x| x.re);
    println!("{:?}", output);

    let output=filter2.filter(&signal);
    println!("{:?}", output);


    for i in 0..tap{
        println!("=====");

        let signal=vec![0.0;nch];
        let output=filter1.filter(ArrayView1::from(&signal)).map(|x| x.re);
        println!("{:?}", output);
    
        let output=filter2.filter_par(&signal);
        println!("{:?}", output);
    }

}
