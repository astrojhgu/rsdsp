use crate::{csp_pfb::CspPfb, cspfb, oscillator::COscillator, ospfb};

use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use ndarray::{Array2, ArrayView1, Axis, ScalarOperand};
use rustfft::FftNum;
use serde::Serialize;

//use serde_yaml::to_writer;

#[allow(clippy::too_many_arguments)]
pub fn ampl_resp_2stages_1freq<T>(
    nch_coarse: usize,
    nch_fine: usize,
    coeff_coarse: &[T],
    coeff_fine: &[T],
    selected_coarse_ch: &[usize],
    freq: T,
    signal_len: usize,
    niter: usize,
) -> (Vec<T>, Vec<T>)
where
    T: Float + FloatConst + NumAssign + FftNum + Default + ScalarOperand + Serialize,
    Complex<T>: ScalarOperand,
{
    let mut coarse_pfb =
        ospfb::AnalyzerFixed::<Complex<T>, T>::new(nch_coarse, ArrayView1::from(&coeff_coarse));

    //to_writer(std::fs::File::create("./coarse_pfb.yaml").unwrap(), &coarse_pfb).unwrap();

    let fine_pfb =
        cspfb::AnalyzerFixed::<Complex<T>, T>::new(nch_fine * 2, ArrayView1::from(&coeff_fine));

    //to_writer(std::fs::File::create("./fine_pfb.yaml").unwrap(), &fine_pfb).unwrap();

    let tap_coarse = coeff_coarse.len() / (nch_coarse / 2);
    let tap_fine = coeff_fine.len() / (nch_fine * 2);
    assert_eq!(tap_coarse * nch_coarse / 2, coeff_coarse.len());
    assert_eq!(tap_fine * nch_fine * 2, coeff_fine.len());

    let mut csp = CspPfb::new(selected_coarse_ch, &fine_pfb);
    let mut osc = COscillator::new(T::zero(), freq);
    for _i in 0..niter - 1 {
        let mut signal = vec![Complex::<T>::default(); signal_len];
        signal.iter_mut().for_each(|x| *x = osc.get());
        let coarse_data = coarse_pfb.analyze(&signal);
        let _ = csp.analyze(coarse_data.view());
    }

    let mut signal = vec![Complex::<T>::default(); signal_len];
    signal.iter_mut().for_each(|x| *x = osc.get());
    let coarse_data = coarse_pfb.analyze(&signal);
    let coarse_spec = coarse_data.map(|x| x.norm_sqr()).sum_axis(Axis(1));

    let coarse_resp: Vec<_> = selected_coarse_ch.iter().map(|&c| coarse_spec[c]).collect();

    let fine_data = csp.analyze(coarse_data.view());

    let fine_resp = fine_data
        .map(|x| x.norm_sqr())
        .sum_axis(Axis(1))
        .into_raw_vec();
    (coarse_resp, fine_resp)
}

#[allow(clippy::too_many_arguments)]
pub fn ampl_resp<T>(
    pfb: &mut ospfb::AnalyzerFixed<Complex<T>, T>,
    f_min: T,
    f_max: T,
    n_freq: usize,
    signal_len: usize,
    niter: usize,
) -> Array2<T>
where
    T: Float + FloatConst + NumAssign + FftNum + Default + ScalarOperand + Serialize,
    Complex<T>: ScalarOperand,
{
    //let mut coarse_pfb =
    //    ospfb::AnalyzerFixed::<Complex<T>, T>::new(nch_coarse, ArrayView1::from(&coeff_coarse));
    let df = (f_max - f_min) / T::from(n_freq - 1).unwrap();
    let mut result = Array2::zeros((pfb.nch_total(), n_freq));

    result
        .axis_iter_mut(Axis(1))
        .enumerate()
        .for_each(|(i, mut x)| {
            let mut osc = COscillator::new(T::zero(), (T::from(i).unwrap() * df + f_min) * T::PI());
            let mut n = 0;
            println!("{}/{}", i, n_freq);
            loop {
                let signal: Vec<_> = (0..signal_len).map(|_| osc.get()).collect();
                let channelized = pfb.analyze(&signal);

                n += 1;
                if n == niter {
                    x.assign(
                        &channelized
                            .map(|y| y.norm_sqr())
                            .mean_axis(Axis(1))
                            .unwrap()
                            .view(),
                    );
                    break;
                }
            }
        });
    result
}
