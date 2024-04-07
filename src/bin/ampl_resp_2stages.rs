extern crate rsdsp;

use std::fs::File;

use rsdsp::{
    ampl_resp::ampl_resp_2stages_1freq,
    cfg::{PfbCfg, TwoStageCfg},
    windowed_fir::pfb_coeff,
};

use num::traits::FloatConst;

use serde_yaml::from_reader;

use clap::Parser;

use ndarray::{parallel::prelude::*, Array1, Array2, ArrayView1, Axis};

use itertools_num::linspace;

use ndarray_npy::NpzWriter;

type FloatType = f64;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// config
    #[clap(short('c'), long("cfg"), value_name = "config file")]
    ch_cfg_file:String, 

    #[clap(short('f'), long("fmin"), value_name="minimum freq", default_value="-1")]
    fmin: FloatType,

    #[clap(short('F'), long("fmax"), value_name="maximum freq", default_value="1")]
    fmax: FloatType,

    #[clap(short('n'), long("nfreq"), value_name="num of freq", default_value="1024")]
    nfreq: usize,

    #[clap(short('t'), long("niter"), value_name="niter", default_value="2")]
    niter: usize,

    #[clap(short('o'), long("out"), value_name="out")]
    outfile: String,
}


pub fn main() {
    let args=Args::parse();

    let mut cfg_file = File::open(args.ch_cfg_file).unwrap();
    let TwoStageCfg {
        coarse_cfg:
            PfbCfg {
                nch: nch_coarse,
                k: k_coarse,
                tap_per_ch: tap_coarse,
            },
        fine_cfg:
            PfbCfg {
                nch: nch_fine,
                k: k_fine,
                tap_per_ch: tap_fine,
            },
        selected_coarse_ch,
    } = from_reader(&mut cfg_file).unwrap();

    let fmin = args.fmin;
    let fmax = args.fmax;
    let nfreq = args.nfreq;
    let niter = args.niter;

    let coeff_coarse =
        pfb_coeff::<FloatType>(nch_coarse / 2, tap_coarse, k_coarse as FloatType).into_raw_vec();
    let coeff_fine =
        pfb_coeff::<FloatType>(nch_fine * 2, tap_fine, k_fine as FloatType).into_raw_vec();

    let signal_len = coeff_coarse.len() + coeff_fine.len() * nch_coarse / 2;
    println!("signal length={}", signal_len);
    let bandwidth = (fmax - fmin) * FloatType::PI();
    let df = bandwidth / (nfreq + 1) as FloatType;
    let freqs = Array1::from(
        linspace(FloatType::PI() * fmin, FloatType::PI() * fmax - df, nfreq).collect::<Vec<_>>(),
    );
    let mut coarse_spec = Array2::<FloatType>::zeros((nfreq, selected_coarse_ch.len()));
    let mut fine_spec = Array2::<FloatType>::zeros((nfreq, selected_coarse_ch.len() * nch_fine));
    println!("{:?}", freqs);

    fine_spec
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip_eq(coarse_spec.axis_iter_mut(Axis(0)).into_par_iter())
        .zip_eq(freqs.axis_iter(Axis(0)).into_par_iter())
        .for_each(|((mut fine_resp, mut coarse_resp), freq)| {
            let freq = freq[()];

            let (coarse_resp1, fine_resp1) = ampl_resp_2stages_1freq(
                nch_coarse,
                nch_fine,
                &coeff_coarse,
                &coeff_fine,
                &selected_coarse_ch,
                freq,
                signal_len,
                niter,
            );

            coarse_resp.assign(&ArrayView1::from(&coarse_resp1));
            fine_resp.assign(&ArrayView1::from(&fine_resp1));
        });

    let outfile = std::fs::File::create(args.outfile).unwrap();
    let mut npz = NpzWriter::new(outfile);
    npz.add_array("freq", &freqs).unwrap();
    npz.add_array("coarse", &coarse_spec).unwrap();
    npz.add_array("fine", &fine_spec).unwrap();
    npz.add_array(
        "coarse_ch",
        &ArrayView1::from(&selected_coarse_ch).map(|&x| x as i32),
    )
    .unwrap();
}
