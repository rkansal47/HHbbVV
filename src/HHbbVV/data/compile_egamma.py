import correctionlib.schemav2 as clib
from correctionlib import convert
from correctionlib import CorrectionSet

corrections = {
    "trigger": {
        "electron": { # derived by D. Rankin
            "2016preVFP_UL": "egammaEffi_txt_trigger_EGM2D_UL2016preVFP.root:EGamma_EffMC2D",
            "2016postVFP_UL": "egammaEffi_txt_trigger_EGM2D_UL2016postVFP.root:EGamma_EffMC2D",
            "2017_UL": "egammaEffi_txt_trigger_EGM2D_UL2017.root:EGamma_EffMC2D",
            "2018_UL": "egammaEffi_txt_trigger_EGM2D_UL2018.root:EGamma_EffMC2D",
        },
    },
}

for corrName,corrDict in corrections.items():
    for lepton_type,leptonDict in corrDict.items():
        for year,ystring in leptonDict.items():
            corr = convert.from_uproot_THx(ystring)
            corr.name = "UL-Electron-Trigger-SF"
            corr.description = "Trigger Scale Factors"
            corr.inputs[0].name = "eta"
            corr.inputs[1].name = "pt"
            corr.data.inputs = ["eta","pt"]
            # TODO: need to add uncertainties

            cset = clib.CorrectionSet(schema_version=2, corrections=[])
            cset.corrections.append(corr)

            with open(f"{lepton_type}_{corrName}_{year}.json", "w") as fout:
                fout.write(cset.json(exclude_unset=True))
