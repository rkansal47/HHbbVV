from __future__ import annotations

import ROOT


def _tolist(argset):
    return [x.GetName() for x in argset]


def getParameters(mcstats: bool):
    """Get nuisance parameters from workspace"""
    print("Getting nuisance parameters")
    print(
        "Remember to use an environment with Root 6.22 and run `ulimit -s unlimited` first to avoid memory issues!"
    )
    f = ROOT.TFile.Open("combined_withmasks.root", "READ")
    w = f.Get("w")
    ps = _tolist(w.allVars())
    pois = _tolist(w.set("ModelConfig_POI"))
    obs = _tolist(w.genobj("ModelConfig").GetObservables())

    ret_ps = []
    for p in ps:
        if not (
            "qcdparam" in p
            or "Blinded" in p
            or p.endswith(("_In", "__norm"))
            or p.startswith(("n_exp_", "mask_"))
            or p in pois
            or p in obs
        ):
            if "mcstat" in p and not mcstats:
                # skip mc stats in the fail region or if not specified
                continue
            ret_ps.append(p)

    return ret_ps


if __name__ == "__main__":
    ps = getParameters(True)
    print(len(ps), "nuisances")
    print(ps)
