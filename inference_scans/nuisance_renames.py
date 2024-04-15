from __future__ import annotations

years = ["2016APV", "2016", "2017", "2018"]

mc_label_map = {
    "ttbar": "TT",
    "wjets": "W+Jets",
    "zjets": "Z+Jets",
}

region_label_map = {
    "passggf": "ggF",
    "passvbf": "VBF",
    "fail": "Fail",
}

nuisance_label_map = {
    "ps_fsr": "Parton Showering (FSR)",
    "ps_isr": "Parton Showering (ISR)",
    "jmr": "JMR",
    "jms": "JMS",
    "CMS_scale_j": "JES",
    "CMS_res_j": "JER",
    "CMS_pileup": "Pileup",
}


def rename_nuisance(nuisance: str):
    if nuisance.startswith("CMS_bbWW_hadronic_"):
        nuisance = nuisance.split("CMS_bbWW_hadronic_")[1]

    if nuisance.startswith("tf_dataResidual_bbFatJetParticleNetMass_"):
        return "TF_" + nuisance.split("tf_dataResidual_bbFatJetParticleNetMass_")[1]

    if nuisance.startswith("tf_dataResidual_passggf_bbFatJetParticleNetMass_"):
        return (
            "ggF TF Param "
            + nuisance.split("tf_dataResidual_passggf_bbFatJetParticleNetMass_par")[1]
        )

    if nuisance.startswith("tf_dataResidual_passvbf_bbFatJetParticleNetMass_"):
        return (
            "VBF TF Param "
            + nuisance.split("tf_dataResidual_passvbf_bbFatJetParticleNetMass_par")[1]
        )

    if "mcstat" in nuisance:
        split = nuisance.split("_")

        return f"{region_label_map[split[0]]} {mc_label_map.get(split[1], split[1])} MCStats Bin {nuisance.split('bin')[1]}"

    if nuisance.startswith("lp_sf"):
        region = nuisance.split("lp_sf_")[1].split("_")[0]
        return f"Lund plane SF {region_label_map[region]} Region"

    if nuisance.split("_")[-1] in years:
        year = nuisance.split("_")[-1]
        nuisance = nuisance.split(f"_{year}")[0]
        return nuisance_label_map.get(nuisance, nuisance) + f" {year}"

    return nuisance_label_map.get(nuisance, nuisance)
