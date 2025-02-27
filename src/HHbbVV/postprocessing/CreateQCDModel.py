from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import rhalphalib as rl

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False

signal_regions = ["pass"]
qcd_key = "QCD"


def main(args):
    with Path("./shape_vars.pkl").open("rb") as f:
        shape_vars = pickle.load(f)
    with Path("./templates_summed.pkl").open("rb") as f:
        templates_summed = pickle.load(f)
    with Path("./scaled_grids.pkl").open("rb") as f:
        scaled_grids = pickle.load(f)

    shape_var_mY, shape_var_mX = shape_vars
    m_obs = rl.Observable(shape_var_mY.name, shape_var_mY.bins)

    tf_MCtempl = rl.BasisPoly(
        "tf_MCtempl",
        (args.nTF[0], args.nTF[1]),
        [shape_var_mX.name, shape_var_mY.name],
        limits=(0, 10),
        basis="Bernstein",
    )

    qcd_eff = (
        templates_summed["pass"][qcd_key, ...].sum().value
        / templates_summed["fail"][qcd_key, ...].sum().value
    )

    # Build qcd MC pass+fail model and fit to polynomial
    # from https://github.com/nsmith-/rhalphalib/blob/61289a00488a014b3b6ca688e38166e3faf0193d/tests/test_rhalphalib.py#L46
    qcdmodel = rl.Model("qcdmodel")
    for mXbin in range(len(shape_var_mX.pts)):
        failCh = rl.Channel("mXbin%d%s" % (mXbin, "fail"))
        passCh = rl.Channel("mXbin%d%s" % (mXbin, "pass"))
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)
        failCh.setObservation(templates_summed["fail"][qcd_key, :, mXbin], read_sumw2=True)
        passCh.setObservation(templates_summed["pass"][qcd_key, :, mXbin], read_sumw2=True)

    tf_MCtempl_params = qcd_eff * tf_MCtempl(*scaled_grids)
    for mXbin in range(len(shape_var_mX.pts)):
        failCh = qcdmodel["mXbin%dfail" % mXbin]
        passCh = qcdmodel["mXbin%dpass" % mXbin]
        failObs = failCh.getObservation()[0]  # (obs, var)
        qcdparams = np.array(
            [
                rl.IndependentParameter("qcdparam_pXbin%d_mYbin%d" % (mXbin, i), 0)
                for i in range(m_obs.nbins)
            ]
        )
        sigmascale = 10.0
        scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
        fail_qcd = rl.ParametericSample(
            "mXbin%dfail_qcd" % mXbin, rl.Sample.BACKGROUND, m_obs, scaledparams
        )
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample(
            "mXbin%dpass_qcd" % mXbin,
            rl.Sample.BACKGROUND,
            tf_MCtempl_params[mXbin, :],
            fail_qcd,
        )
        passCh.addSample(pass_qcd)

    # need fake signal sample to render combine workspae
    dummy_nuisance = rl.NuisanceParameter("dummy", "lnN")
    dummy_signal = rl.TemplateSample(
        "mXbin9pass_signal", rl.Sample.SIGNAL, templates_summed["pass"][qcd_key, :, mXbin]
    )
    dummy_signal.setParamEffect(dummy_nuisance, 1.1)
    passCh.addSample(dummy_signal)
    qcdmodel.renderCombine(f"nTF{args.nTF[0]}{args.nTF[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nTF",
        required=True,
        help="QCD nTF",
        type=int,
        nargs=2,
    )
    args = parser.parse_args()
    main(args)
