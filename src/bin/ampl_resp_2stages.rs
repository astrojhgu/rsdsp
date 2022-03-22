extern crate rsdsp;

use std::fs::File;

use rsdsp::{
    ampl_resp::ampl_resp_2stages_1freq,
    cfg::{PfbCfg, TwoStageCfg},
    windowed_fir::pfb_coeff,
};

use num::traits::FloatConst;

use serde_yaml::from_reader;

use clap::{Arg, Command};

use ndarray::{parallel::prelude::*, Array1, Array2, ArrayView1, Axis};

use itertools_num::linspace;

use ndarray_npy::NpzWriter;

type FloatType = f64;

pub fn main() {
    let matches = Command::new("ampl_resp_2stages")
        .arg(
            Arg::new("chcfg")
                .short('c')
                .long("cfg")
                .takes_value(true)
                .value_name("config file")
                .required(true),
        )
        .arg(
            Arg::new("fmin")
                .short('f')
                .long("fmin")
                .allow_hyphen_values(true)
                .takes_value(true)
                .value_name("freq")
                .default_value("-1")
                .required(false),
        )
        .arg(
            Arg::new("fmax")
                .short('F')
                .long("fmax")
                .allow_hyphen_values(true)
                .takes_value(true)
                .value_name("freq")
                .default_value("1")
                .required(false),
        )
        .arg(
            Arg::new("nfreq")
                .short('n')
                .long("nfreq")
                .takes_value(true)
                .value_name("nfreq")
                .default_value("1024")
                .required(false),
        )
        .arg(
            Arg::new("niter")
                .short('t')
                .long("niter")
                .takes_value(true)
                .value_name("niter")
                .default_value("2")
                .required(false),
        )
        .arg(
            Arg::new("outfile")
                .short('o')
                .long("out")
                .takes_value(true)
                .value_name("output name")
                .required(true),
        )
        .get_matches();

    let mut cfg_file = File::open(matches.value_of("chcfg").unwrap()).unwrap();
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

    let fmin = matches
        .value_of("fmin")
        .unwrap()
        .parse::<FloatType>()
        .unwrap();
    let fmax = matches
        .value_of("fmax")
        .unwrap()
        .parse::<FloatType>()
        .unwrap();
    let nfreq = matches.value_of("nfreq").unwrap().parse::<usize>().unwrap();
    let niter = matches.value_of("niter").unwrap().parse::<usize>().unwrap();

    let coeff_coarse =
        pfb_coeff::<FloatType>(nch_coarse / 2, tap_coarse, k_coarse as FloatType).into_raw_vec();
    let coeff_fine =
        pfb_coeff::<FloatType>(nch_fine * 2, tap_fine, k_fine as FloatType).into_raw_vec();

    let signal_len=coeff_coarse.len()+coeff_fine.len()*nch_coarse/2;
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

    let outfile = std::fs::File::create(matches.value_of("outfile").unwrap()).unwrap();
    let mut npz = NpzWriter::new(outfile);
    let _ = npz.add_array("freq", &freqs).unwrap();
    let _ = npz.add_array("coarse", &coarse_spec).unwrap();
    let _ = npz.add_array("fine", &fine_spec).unwrap();
    let _ = npz.add_array(
        "coarse_ch",
        &ArrayView1::from(&selected_coarse_ch).map(|&x| x as i32),
    );
}
