use crate::{
    cfg::{PfbCfg, TwoStageCfg},
    csp_pfb::CspPfb,
    cspfb,
    oscillator::COscillator,
    ospfb,
    windowed_fir::pfb_coeff,
};

use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use ndarray::{parallel::prelude::*, Array1, Array2, ArrayView1, Axis, ScalarOperand};
use rustfft::FftNum;

use itertools_num::linspace;

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
    T: Float + FloatConst + NumAssign + FftNum + Default + ScalarOperand,
    Complex<T>: ScalarOperand,
{
    let mut coarse_pfb =
        ospfb::Analyzer::<Complex<T>, T>::new(nch_coarse, ArrayView1::from(&coeff_coarse));
    let fine_pfb =
        cspfb::Analyzer::<Complex<T>, T>::new(nch_fine * 2, ArrayView1::from(&coeff_fine));

    let mut csp = CspPfb::new(&selected_coarse_ch, &fine_pfb);
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
