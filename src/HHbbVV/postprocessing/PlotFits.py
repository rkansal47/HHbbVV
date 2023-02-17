import uproot
import plotting
from hist import Hist
import numpy as np
import os
import pickle

from hh_vars import sig_key, data_key

import utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fit-file",
    help="fitdiagnostics output root file",
    default="cards/test_tied_stats/fitDiagnosticsBlindedBkgOnly.root",
    type=str,
)
parser.add_argument("--plots-dir", help="plots directory", type=str)
utils.add_bool_arg(parser, "load-pickles", "load pre-saved shapes and data_err pickles")

args = parser.parse_args()


os.system(f"mkdir -p {args.plots_dir}")


hist_label_map = {
    "data": "Data",
    "bbWW_boosted_ggf_qcd_datadriven": "QCD",
    "ttbar": "TT",
    "ggHH_kl_1_kt_1_hbbhww4q": "HHbbVV",
}

hist_label_map_inverse = {val: key for key, val in hist_label_map.items()}
samples = list(hist_label_map.values())

# bb msd is final shape var
shape_var = ("bbFatJetMsd", r"$m^{bb}$ (GeV)")
shape_bins = [20, 50, 250]  # num bins, min, max
blind_window = [100, 150]
regions = {"fail": "Fail", "passCat1": "Pass Cat1"}
shapes = {
    "shapes_prefit": "Pre-Fit",
    "shapes_fit_s": "S+B Post-Fit",
    "shapes_fit_b": "B-only Post-Fit",
}


if not args.load_pickles:
    file = uproot.open(args.fit_file)

    for key in list(shapes.keys()):
        if key not in file:
            print(f"{key} not found!")
            del shapes[key]

    hists = {}
    data_errs = {}

    for shape in shapes:
        hists[shape] = {
            region: Hist.new.StrCat(samples, name="Sample")
            .Reg(*shape_bins, name=shape_var[0], label=shape_var[1])
            .Double()
            for region in regions
        }

        data_errs[shape] = {}

        for region in regions:
            h = hists[shape][region]
            for key, file_key in hist_label_map_inverse.items():
                if key != data_key:
                    data_key_index = np.where(np.array(list(h.axes[0])) == key)[0][0]
                    h.view(flow=False)[data_key_index] = file[shape][region][file_key].values() * 10

            data_key_index = np.where(np.array(list(h.axes[0])) == data_key)[0][0]
            h.view(flow=False)[data_key_index] = file[shape][region]["data"].values()[1] * 10

            data_errs[shape][region] = np.stack(
                (
                    file[shape][region]["data"].errors(which="low")[1] * 10,
                    file[shape][region]["data"].errors(which="high")[1] * 10,
                )
            )

    with open(f"{args.plots_dir}/hists.pkl", "wb") as f:
        pickle.dump(hists, f)

    with open(f"{args.plots_dir}/data_errs.pkl", "wb") as f:
        pickle.dump(data_errs, f)

    print("Saved pickles")
else:
    print("Loading pickles")

    with open(f"{args.plots_dir}/hists.pkl", "rb") as f:
        hists = pickle.load(f)

    with open(f"{args.plots_dir}/data_errs.pkl", "rb") as f:
        data_errs = pickle.load(f)


for shape, slabel in shapes.items():
    if shape not in hists:
        print(f"{shape} not found!")
        continue

    print(f"Plotting {shape}")

    for region, rlabel in regions.items():
        try:
            plotting.ratioHistPlot(
                hists[shape][region],
                ["QCD", "TT"],
                data_err=data_errs[shape][region],
                title=f"{rlabel} Region {slabel} Shapes",
                name=f"{args.plots_dir}/{shape}_{region}.pdf",
                show=False,
            )
        except AttributeError:
            print(f"Error in {region} region!")
