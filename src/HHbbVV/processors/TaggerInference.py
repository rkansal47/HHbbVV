"""
Methods for deriving input variables for the tagger and running inference.

Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan
"""

from __future__ import annotations

import json

# import onnxruntime as ort
import time
import warnings
from pathlib import Path

import awkward as ak
import numpy as np
import numpy.ma as ma
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray
from numpy.typing import ArrayLike
from tqdm import tqdm


def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


def get_pfcands_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int | ArrayLike,
    jet: FatJetArray = None,
    fatjet_label: str = "FatJet",
    pfcands_label: str = "FatJetPFCands",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the pf_candidate features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    if jet is None:
        jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]

    jet_ak_pfcands = preselected_events[pfcands_label][
        preselected_events[pfcands_label].jetIdx == jet_idx
    ]
    jet_pfcands = preselected_events.PFCands[jet_ak_pfcands.pFCandsIdx]

    # sort them by pt
    pfcand_sort = ak.argsort(jet_pfcands.pt, ascending=False)
    jet_pfcands = jet_pfcands[pfcand_sort]

    # get features

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.ones_like(jet_pfcands.eta)
    eta_sign = eta_sign * (ak.values_astype(jet.eta > 0, int) * 2 - 1)
    feature_dict["pfcand_etarel"] = eta_sign * (jet_pfcands.eta - jet.eta)
    feature_dict["pfcand_phirel"] = jet_pfcands.delta_phi(jet)
    feature_dict["pfcand_abseta"] = np.abs(jet_pfcands.eta)

    feature_dict["pfcand_pt_log_nopuppi"] = np.log(jet_pfcands.pt)
    feature_dict["pfcand_e_log_nopuppi"] = np.log(jet_pfcands.energy)

    pdgIds = jet_pfcands.pdgId
    feature_dict["pfcand_isEl"] = np.abs(pdgIds) == 11
    feature_dict["pfcand_isMu"] = np.abs(pdgIds) == 13
    feature_dict["pfcand_isChargedHad"] = np.abs(pdgIds) == 211
    feature_dict["pfcand_isGamma"] = np.abs(pdgIds) == 22
    feature_dict["pfcand_isNeutralHad"] = np.abs(pdgIds) == 130

    feature_dict["pfcand_charge"] = jet_pfcands.charge
    feature_dict["pfcand_VTX_ass"] = jet_pfcands.pvAssocQuality
    feature_dict["pfcand_lostInnerHits"] = jet_pfcands.lostInnerHits
    feature_dict["pfcand_quality"] = jet_pfcands.trkQuality
    feature_dict["pfcand_normchi2"] = np.floor(jet_pfcands.trkChi2)

    if "Cdz" in jet_ak_pfcands.fields:
        feature_dict["pfcand_dz"] = jet_ak_pfcands["Cdz"][pfcand_sort]
        feature_dict["pfcand_dxy"] = jet_ak_pfcands["Cdxy"][pfcand_sort]
        feature_dict["pfcand_dzsig"] = jet_ak_pfcands["Cdzsig"][pfcand_sort]
        feature_dict["pfcand_dxysig"] = jet_ak_pfcands["Cdxysig"][pfcand_sort]
    else:
        # this is for old PFNano (<= v2.3)
        feature_dict["pfcand_dz"] = jet_pfcands.dz
        feature_dict["pfcand_dxy"] = jet_pfcands.d0
        feature_dict["pfcand_dzsig"] = jet_pfcands.dz / jet_pfcands.dzErr
        feature_dict["pfcand_dxysig"] = jet_pfcands.d0 / jet_pfcands.d0Err

    feature_dict["pfcand_px"] = jet_pfcands.px
    feature_dict["pfcand_py"] = jet_pfcands.py
    feature_dict["pfcand_pz"] = jet_pfcands.pz
    feature_dict["pfcand_energy"] = jet_pfcands.E

    for var in tagger_vars["pf_features"]["var_names"]:
        if "btag" in var:
            feature_dict[var] = jet_ak_pfcands[var[len("pfcand_") :]][pfcand_sort]

    feature_dict["pfcand_mask"] = (
        ~(
            ma.masked_invalid(
                ak.pad_none(
                    feature_dict["pfcand_abseta"],
                    tagger_vars["pf_features"]["var_length"],
                    axis=1,
                    clip=True,
                ).to_numpy()
            ).mask
        )
    ).astype(np.float32)

    # if no padding is needed, mask will = 1.0
    if isinstance(feature_dict["pfcand_mask"], np.float32):
        feature_dict["pfcand_mask"] = np.ones(
            (len(feature_dict["pfcand_abseta"]), tagger_vars["pf_features"]["var_length"])
        ).astype(np.float32)

    repl_values_dict = {
        "pfcand_normchi2": [-1, 999],
        "pfcand_dz": [-1, 0],
        "pfcand_dzsig": [1, 0],
        "pfcand_dxy": [-1, 0],
        "pfcand_dxysig": [1, 0],
    }

    # convert to numpy arrays and normalize features
    for var in set(
        tagger_vars["pf_features"]["var_names"] + tagger_vars["pf_vectors"]["var_names"]
    ):
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["pf_features"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        a = np.nan_to_num(a)

        # replace values to match PKU's
        if var in repl_values_dict:
            vals = repl_values_dict[var]
            a[a == vals[0]] = vals[1]

        if normalize:
            if var in tagger_vars["pf_features"]["var_names"]:
                info = tagger_vars["pf_features"]["var_infos"][var]
            else:
                info = tagger_vars["pf_vectors"]["var_infos"][var]

            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_svs_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int | ArrayLike,
    jet: FatJetArray = None,
    fatjet_label: str = "FatJet",
    svs_label: str = "JetSVs",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the sv features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    if jet is None:
        jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]

    jet_svs = preselected_events.SV[
        preselected_events[svs_label].sVIdx[
            (preselected_events[svs_label].sVIdx != -1)
            * (preselected_events[svs_label].jetIdx == jet_idx)
        ]
    ]

    # sort by dxy significance
    jet_svs = jet_svs[ak.argsort(jet_svs.dxySig, ascending=False)]

    # get features

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.ones_like(jet_svs.eta)
    eta_sign = eta_sign * (ak.values_astype(jet.eta > 0, int) * 2 - 1)

    feature_dict["sv_etarel"] = eta_sign * (jet_svs.eta - jet.eta)
    feature_dict["sv_phirel"] = jet_svs.delta_phi(jet)
    feature_dict["sv_abseta"] = np.abs(jet_svs.eta)
    feature_dict["sv_mass"] = jet_svs.mass
    feature_dict["sv_pt_log"] = np.log(jet_svs.pt)

    feature_dict["sv_ntracks"] = jet_svs.ntracks
    feature_dict["sv_normchi2"] = jet_svs.chi2
    feature_dict["sv_dxy"] = jet_svs.dxy
    feature_dict["sv_dxysig"] = jet_svs.dxySig
    feature_dict["sv_d3d"] = jet_svs.dlen
    feature_dict["sv_d3dsig"] = jet_svs.dlenSig
    svpAngle = jet_svs.pAngle
    feature_dict["sv_costhetasvpv"] = -np.cos(svpAngle)

    feature_dict["sv_px"] = jet_svs.px
    feature_dict["sv_py"] = jet_svs.py
    feature_dict["sv_pz"] = jet_svs.pz
    feature_dict["sv_energy"] = jet_svs.E

    feature_dict["sv_mask"] = (
        ~(
            ma.masked_invalid(
                ak.pad_none(
                    feature_dict["sv_etarel"],
                    tagger_vars["sv_features"]["var_length"],
                    axis=1,
                    clip=True,
                ).to_numpy()
            ).mask
        )
    ).astype(np.float32)

    # if no padding is needed, mask will = 1.0
    if isinstance(feature_dict["sv_mask"], np.float32):
        feature_dict["sv_mask"] = np.ones(
            (len(feature_dict["sv_abseta"]), tagger_vars["sv_features"]["var_length"])
        ).astype(np.float32)

    # convert to numpy arrays and normalize features
    for var in set(
        tagger_vars["sv_features"]["var_names"] + tagger_vars["sv_vectors"]["var_names"]
    ):
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["sv_features"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        a = np.nan_to_num(a)

        if normalize:
            if var in tagger_vars["sv_features"]["var_names"]:
                info = tagger_vars["sv_features"]["var_infos"][var]
            else:
                info = tagger_vars["sv_vectors"]["var_infos"][var]

            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_met_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int,
    fatjet_label: str = "FatJetAK15",
    met_label: str = "MET",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the MET features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]
    met = preselected_events[met_label]

    # get features
    feature_dict["met_relpt"] = met.pt / jet.pt
    feature_dict["met_relphi"] = met.delta_phi(jet)

    for var in tagger_vars["met_features"]["var_names"]:
        a = (
            # ak.pad_none(
            #     feature_dict[var], tagger_vars["met_features"]["var_length"], axis=1, clip=True
            # )
            feature_dict[var]  # just 1d, no pad_none
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        if normalize:
            info = tagger_vars["met_features"]["var_infos"][var]
            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_lep_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet_idx: int,
    fatjet_label: str = "FatJet",
    muon_label: str = "Muon",
    electron_label: str = "Electron",
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extracts the lepton features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    jet = ak.pad_none(preselected_events[fatjet_label], 2, axis=1)[:, jet_idx]
    jet_muons = preselected_events.Muon[preselected_events[muon_label].jetIdx == jet_idx]
    jet_electrons = preselected_events.Electron[
        preselected_events[electron_label].jetIdx == jet_idx
    ]

    # get features of leading leptons
    leptons = ak.concatenate([jet_muons, jet_electrons], axis=1)
    index_lep = ak.argsort(leptons.pt, ascending=False)
    leptons = leptons[index_lep]
    lepton_cand = ak.firsts(leptons)
    lepton = build_p4(lepton_cand)

    # print(ak.firsts(leptons).__repr__)
    electron_iso = jet_electrons.pfRelIso03_all
    electron_pdgid = np.abs(jet_electrons.charge) * 11
    muon_iso = jet_muons.pfRelIso04_all
    muon_pdgid = np.abs(jet_muons.charge) * 13
    feature_dict["lep_iso"] = (
        ak.firsts(ak.concatenate([muon_iso, electron_iso], axis=1)[index_lep])
        .to_numpy()
        .filled(fill_value=0)
    )
    feature_dict["lep_pdgId"] = (
        ak.firsts(ak.concatenate([muon_pdgid, electron_pdgid], axis=1)[index_lep])
        .to_numpy()
        .filled(fill_value=0)
    )

    # this is for features that are shared
    feature_dict["lep_dR_fj"] = lepton.delta_r(jet).to_numpy().filled(fill_value=0)
    feature_dict["lep_pt"] = (lepton.pt).to_numpy().filled(fill_value=0)
    feature_dict["lep_pt_ratio"] = (lepton.pt / jet.pt).to_numpy().filled(fill_value=0)
    feature_dict["lep_miniiso"] = lepton_cand.miniPFRelIso_all.to_numpy().filled(fill_value=0)

    # get features
    if "el_features" in tagger_vars:
        feature_dict["elec_pt"] = jet_electrons.pt / jet.pt
        feature_dict["elec_eta"] = jet_electrons.eta - jet.eta
        feature_dict["elec_phi"] = jet_electrons.delta_phi(jet)
        feature_dict["elec_mass"] = jet_electrons.mass
        feature_dict["elec_charge"] = jet_electrons.charge
        feature_dict["elec_convVeto"] = jet_electrons.convVeto
        feature_dict["elec_deltaEtaSC"] = jet_electrons.deltaEtaSC
        feature_dict["elec_dr03EcalRecHitSumEt"] = jet_electrons.dr03EcalRecHitSumEt
        feature_dict["elec_dr03HcalDepth1TowerSumEt"] = jet_electrons.dr03HcalDepth1TowerSumEt
        feature_dict["elec_dr03TkSumPt"] = jet_electrons.dr03TkSumPt
        feature_dict["elec_dxy"] = jet_electrons.dxy
        feature_dict["elec_dxyErr"] = jet_electrons.dxyErr
        feature_dict["elec_dz"] = jet_electrons.dz
        feature_dict["elec_dzErr"] = jet_electrons.dzErr
        feature_dict["elec_eInvMinusPInv"] = jet_electrons.eInvMinusPInv
        feature_dict["elec_hoe"] = jet_electrons.hoe
        feature_dict["elec_ip3d"] = jet_electrons.ip3d
        feature_dict["elec_lostHits"] = jet_electrons.lostHits
        feature_dict["elec_r9"] = jet_electrons.r9
        feature_dict["elec_sieie"] = jet_electrons.sieie
        feature_dict["elec_sip3d"] = jet_electrons.sip3d
        # convert to numpy arrays and normalize features
        for var in tagger_vars["el_features"]["var_names"]:
            a = (
                ak.pad_none(
                    feature_dict[var], tagger_vars["el_features"]["var_length"], axis=1, clip=True
                )
                .to_numpy()
                .filled(fill_value=0)
            ).astype(np.float32)

            if normalize:
                info = tagger_vars["el_features"]["var_infos"][var]
                a = (a - info["median"]) * info["norm_factor"]
                a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

            feature_dict[var] = a

    if "mu_features" in tagger_vars:
        feature_dict["muon_pt"] = jet_muons.pt / jet.pt
        feature_dict["muon_eta"] = jet_muons.eta - jet.eta
        feature_dict["muon_phi"] = jet_muons.delta_phi(jet)
        feature_dict["muon_mass"] = jet_muons.mass
        feature_dict["muon_charge"] = jet_muons.charge
        feature_dict["muon_dxy"] = jet_muons.dxy
        feature_dict["muon_dxyErr"] = jet_muons.dxyErr
        feature_dict["muon_dz"] = jet_muons.dz
        feature_dict["muon_dzErr"] = jet_muons.dzErr
        feature_dict["muon_ip3d"] = jet_muons.ip3d
        feature_dict["muon_nStations"] = jet_muons.nStations
        feature_dict["muon_nTrackerLayers"] = jet_muons.nTrackerLayers
        feature_dict["muon_pfRelIso03_all"] = jet_muons.pfRelIso03_all
        feature_dict["muon_pfRelIso03_chg"] = jet_muons.pfRelIso03_chg
        feature_dict["muon_segmentComp"] = jet_muons.segmentComp
        feature_dict["muon_sip3d"] = jet_muons.sip3d
        feature_dict["muon_tkRelIso"] = jet_muons.tkRelIso
        # convert to numpy arrays and normalize features
        for var in tagger_vars["mu_features"]["var_names"]:
            a = (
                ak.pad_none(
                    feature_dict[var], tagger_vars["mu_features"]["var_length"], axis=1, clip=True
                )
                .to_numpy()
                .filled(fill_value=0)
            ).astype(np.float32)

            if normalize:
                info = tagger_vars["mu_features"]["var_infos"][var]
                a = (a - info["median"]) * info["norm_factor"]
                a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

            feature_dict[var] = a

    return feature_dict


# adapted from https://github.com/lgray/hgg-coffea/blob/triton-bdts/src/hgg_coffea/tools/chained_quantile.py
class wrapped_triton:
    def __init__(
        self,
        model_url: str,
        batch_size: int,
        torchscript: bool = True,
    ) -> None:
        fullprotocol, location = model_url.split("://")
        _, protocol = fullprotocol.split("+")
        address, model, version = location.split("/")

        self._protocol = protocol
        self._address = address
        self._model = model
        self._version = version

        self._batch_size = batch_size
        self._torchscript = torchscript

    def __call__(self, input_dict: dict[str, np.ndarray]) -> np.ndarray:
        if self._protocol == "grpc":
            import tritonclient.grpc as triton_grpc

            client = triton_grpc.InferenceServerClient(url=self._address, verbose=False)
            triton_protocol = triton_grpc
        elif self._protocol == "http":
            import tritonclient.http as triton_http

            client = triton_http.InferenceServerClient(
                url=self._address,
                verbose=False,
                concurrency=12,
            )
            triton_protocol = triton_http
        else:
            raise ValueError(f"{self._protocol} does not encode a valid protocol (grpc or http)")

        # manually split into batches for gpu inference
        input_size = input_dict[next(iter(input_dict.keys()))].shape[0]
        print(f"size of input = {input_size}")

        outs = [
            self._do_inference(
                {key: input_dict[key][batch : batch + self._batch_size] for key in input_dict},
                triton_protocol,
                client,
            )
            for batch in tqdm(
                range(0, input_dict[next(iter(input_dict.keys()))].shape[0], self._batch_size)
            )
        ]

        return np.concatenate(outs) if input_size > 0 else outs

    def _do_inference(
        self, input_dict: dict[str, np.ndarray], triton_protocol, client
    ) -> np.ndarray:
        import tritonclient

        # Infer
        inputs = []

        for key, val in input_dict.items():
            input = triton_protocol.InferInput(key, val.shape, "FP32")
            input.set_data_from_numpy(val)
            inputs.append(input)

        out_name = "softmax__0" if self._torchscript else "softmax"

        output = triton_protocol.InferRequestedOutput(out_name)

        # in case of a server error, keep trying to connect for 5 minutes
        for _i in range(60):
            try:
                request = client.infer(
                    self._model,
                    model_version=self._version,
                    inputs=inputs,
                    outputs=[output],
                )
                break
            except tritonclient.utils.InferenceServerException as e:
                print("Triton Error:", e)
                time.sleep(5)

        return request.as_numpy(out_name)


def _derive_vars_new_tagger(
    jet_outputs: np.ndarray, jet_label: str, all_outputs: bool, tagger_vars: dict
):
    if len(jet_outputs):
        derived_vars = {
            f"{jet_label}FatJetParTMD_probQCD": np.sum(jet_outputs[:, 309:314], axis=1),
            f"{jet_label}FatJetParTMD_probT": np.sum(jet_outputs[:, :17], axis=1),
            **{
                f"{jet_label}FatJetParTMD_probH{l}": jet_outputs[:, 17 + i]
                for i, l in enumerate(["bb", "cc", "ss", "qq", "bc", "bs", "cs", "gg"])
            },
        }

        pnet_vars_all = {}

        if all_outputs:
            for i, output_name in enumerate(tagger_vars["output_names"]):
                pnet_vars_all[f"{jet_label}FatJetParTMD_{output_name}"] = jet_outputs[:, i]

        pvars = {**derived_vars, **pnet_vars_all}

    else:
        derived_vars = {
            f"{jet_label}FatJetParTMD_probQCD": np.array([]),
            f"{jet_label}FatJetParTMD_probT": np.array([]),
            **{
                f"{jet_label}FatJetParTMD_probH{l}": np.array([])
                for i, l in enumerate(["bb", "cc", "ss", "qq", "bc", "bs", "cs", "gg"])
            },
        }
        pnet_vars_all = {}

        if all_outputs:
            for output_name in tagger_vars["output_names"]:
                pnet_vars_all[f"{jet_label}FatJetParTMD_{output_name}"] = np.array([])

        pvars = {**derived_vars, **pnet_vars_all}

    return pvars


def _derive_vars(jet_outputs: np.ndarray, jet_label: str, all_outputs: bool, tagger_vars: dict):
    if len(jet_outputs):
        derived_vars = {
            f"{jet_label}FatJetParTMD_probQCD": np.sum(jet_outputs[:, 23:28], axis=1),
            f"{jet_label}FatJetParTMD_probHWW3q": np.sum(jet_outputs[:, 0:3], axis=1),
            f"{jet_label}FatJetParTMD_probHWW4q": np.sum(jet_outputs[:, 3:6], axis=1),
            f"{jet_label}FatJetParTMD_probT": np.sum(jet_outputs[:, 28:37], axis=1),
        }

        derived_vars[f"{jet_label}FatJetParTMD_THWW4q"] = (
            derived_vars[f"{jet_label}FatJetParTMD_probHWW3q"]
            + derived_vars[f"{jet_label}FatJetParTMD_probHWW4q"]
        ) / (
            derived_vars[f"{jet_label}FatJetParTMD_probHWW3q"]
            + derived_vars[f"{jet_label}FatJetParTMD_probHWW4q"]
            + derived_vars[f"{jet_label}FatJetParTMD_probQCD"]
        )

        derived_vars[f"{jet_label}FatJetParTMD_THWWvsT"] = (
            derived_vars[f"{jet_label}FatJetParTMD_probHWW3q"]
            + derived_vars[f"{jet_label}FatJetParTMD_probHWW4q"]
        ) / (
            derived_vars[f"{jet_label}FatJetParTMD_probHWW3q"]
            + derived_vars[f"{jet_label}FatJetParTMD_probHWW4q"]
            + derived_vars[f"{jet_label}FatJetParTMD_probQCD"]
            + derived_vars[f"{jet_label}FatJetParTMD_probT"]
        )

        pnet_vars_all = {}

        if all_outputs:
            for i, output_name in enumerate(tagger_vars["output_names"]):
                pnet_vars_all[f"{jet_label}FatJetParTMD_{output_name}"] = jet_outputs[:, i]

        pvars = {**derived_vars, **pnet_vars_all}

    else:
        derived_vars = {
            f"{jet_label}FatJetParTMD_probQCD": np.array([]),
            f"{jet_label}FatJetParTMD_probHWW3q": np.array([]),
            f"{jet_label}FatJetParTMD_probHWW4q": np.array([]),
            f"{jet_label}FatJetParTMD_probT": np.array([]),
            f"{jet_label}FatJetParTMD_THWW4q": np.array([]),
            f"{jet_label}FatJetParTMD_THWWvsT": np.array([]),
        }
        pnet_vars_all = {}

        if all_outputs:
            for output_name in tagger_vars["output_names"]:
                pnet_vars_all[f"{jet_label}FatJetParTMD_{output_name}"] = np.array([])

        pvars = {**derived_vars, **pnet_vars_all}

    return pvars


def runInferenceTriton(
    tagger_resources_path: Path,
    events: NanoEventsArray,
    num_jets: int = 2,
    in_jet_idx: ArrayLike = None,
    jets: FatJetArray = None,
    ak15: bool = False,
    all_outputs: bool = False,
    jet_label: str = None,
    new_tagger: bool = False,
) -> dict:
    """Runs inference with the triton server.

    Args:
        tagger_resources_path (str): Path to server, model configs.
        events (NanoEventsArray): Nano events.
        num_jets (int, optional): # jets for which to run inference. Defaults to 2.
        in_jet_idx (ArrayLike, optional): run inference for a specific jet per event. Only possible
          if ``num_jets`` is 1. Defaults to None.
        jets (FatJetArray, optional): give the jets to inference directly. ``in_jet_idx`` should be
          specified to get the correct pfcands/svs matching. Defaults to None.
        ak15 (bool, optional): AK15 or AK8 jets Defaults to False.
        all_outputs (bool, optional): Get all ParT outputs (i.e., incl. fine-tuning neurons).
          Defaults to False.
        jet_label (str, optional): custom label for output jets. Defaults to 'ak8' or 'ak15'.

    Returns:
        dict: output probabilities / discriminants per event
    """
    from scipy.special import softmax

    if in_jet_idx is not None and num_jets > 1:
        raise ValueError("Can't give in_jet_idx for num_jets != 1")

    if jets is not None and in_jet_idx is None:
        warnings.warn(
            "Input jets given without in_jet_idx - pfcands and svs matching will likely be incorrect!",
            stacklevel=2,
        )

    total_start = time.time()

    if jet_label is None:
        jet_label = "ak15" if ak15 else "ak8"

    if new_tagger:
        config_name = "triton_config_new_tagger_ak8"
    elif not ak15:
        config_name = "triton_config_ak8"
    else:
        config_name = "triton_config_ak15"

    with (tagger_resources_path / f"{config_name}.json").open() as f:
        triton_config = json.load(f)

    with (tagger_resources_path / f"{triton_config['model_name']}.json").open() as f:
        tagger_vars = json.load(f)

    triton_model = wrapped_triton(
        triton_config["model_url"], triton_config["batch_size"], torchscript=False
    )

    fatjet_label = "FatJetAK15" if ak15 else "FatJet"
    pfcands_label = "FatJetAK15PFCands" if ak15 else "FatJetPFCands"
    svs_label = "JetSVsAK15" if ak15 else "FatJetSVs"

    # prepare inputs for fat jets
    tagger_inputs = []
    feature_dicts = []
    for j in range(num_jets):
        if in_jet_idx is None:
            jet_idx = j
            jets = ak.pad_none(events[fatjet_label], 2, axis=1)[:, jet_idx]
        else:
            jet_idx = in_jet_idx
            if jets is None:
                jets = ak.pad_none(events[fatjet_label], 2, axis=1)[np.arange(len(events)), jet_idx]

        feature_dict = {
            **get_pfcands_features(tagger_vars, events, jet_idx, jets, fatjet_label, pfcands_label),
            **get_svs_features(tagger_vars, events, jet_idx, jets, fatjet_label, svs_label),
            # **get_lep_features(tagger_vars, events, jet_idx, fatjet_label, muon_label, electron_label),
        }

        feature_dicts.append(feature_dict)

        for input_name in tagger_vars["input_names"]:
            for key in tagger_vars[input_name]["var_names"]:
                np.expand_dims(feature_dict[key], 1)

        tagger_inputs.append(
            {
                f"{input_name}": np.concatenate(
                    [
                        np.expand_dims(feature_dict[key], 1)
                        for key in tagger_vars[input_name]["var_names"]
                    ],
                    axis=1,
                )
                for i, input_name in enumerate(tagger_vars["input_names"])
            }
        )

    # run inference for both fat jets
    tagger_outputs = []
    for jet_idx in range(num_jets):
        print(f"Running inference for Jet {jet_idx + 1}")
        start = time.time()
        out = triton_model(tagger_inputs[jet_idx])
        time_taken = time.time() - start
        print(f"Inference took {time_taken:.1f}s")

        if triton_config["num_reg"] > 0:
            # separate class and regression outputs
            out_cls, out_reg = (
                out[:, : -triton_config["num_reg"]],
                out[:, -triton_config["num_reg"] :],
            )
            # apply softmax if not already included in model
            if triton_config["softmax"] == "False":
                out_cls = softmax(out_cls, axis=1)

            out = np.concatenate((out_cls, out_reg), axis=1)
        elif triton_config["softmax"] == "False":
            out = softmax(out, axis=1)

        tagger_outputs.append(out)

    pnet_vars_list = []

    for jet_idx in range(num_jets):
        if new_tagger:
            pnet_vars_list.append(
                _derive_vars_new_tagger(
                    tagger_outputs[jet_idx], jet_label, all_outputs, tagger_vars
                )
            )
        else:
            pnet_vars_list.append(
                _derive_vars(tagger_outputs[jet_idx], jet_label, all_outputs, tagger_vars)
            )

    print(f"Total time taken: {time.time() - total_start:.1f}s")

    if num_jets == 2:
        return {
            key: np.concatenate(
                [pnet_vars_list[0][key][:, np.newaxis], pnet_vars_list[1][key][:, np.newaxis]],
                axis=1,
            )
            for key in pnet_vars_list[0]
        }
    else:
        return pnet_vars_list[0]
