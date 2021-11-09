from typing import Optional, List, Dict

import numpy as np
import awkward as ak
from coffea.nanoevents.methods.base import NanoEventsArray

import json
import onnxruntime as ort

import time

import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http


def get_pfcands_features(
    tagger_vars: dict, preselected_events: NanoEventsArray, jet_idx: int
) -> Dict[str, np.ndarray]:
    """
    Extracts the pf_candidate features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    jet = preselected_events.FatJetAK15[:, jet_idx]
    jet_pfcands = preselected_events.PFCands[
        preselected_events.JetPFCandsAK15.candIdx[
            preselected_events.JetPFCandsAK15.jetIdx == jet_idx
        ]
    ]

    # get features

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.values_astype(jet_pfcands.eta > 0, int) * 2 - 1
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

    feature_dict["pfcand_dz"] = jet_pfcands.dz
    feature_dict["pfcand_dxy"] = jet_pfcands.d0
    feature_dict["pfcand_dzsig"] = jet_pfcands.dz / jet_pfcands.dzErr
    feature_dict["pfcand_dxysig"] = jet_pfcands.d0 / jet_pfcands.d0Err

    feature_dict["pfcand_mask"] = (
        ~(
            ak.pad_none(
                feature_dict["pfcand_etarel"],
                tagger_vars["pf_points"]["var_length"],
                axis=1,
                clip=True,
            )
            .to_numpy()
            .mask
        )
    ).astype(np.float32)

    # convert to numpy arrays and normalize features
    for var in tagger_vars["pf_features"]["var_names"]:
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["pf_points"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        info = tagger_vars["pf_features"]["var_infos"][var]
        a = (a - info["median"]) * info["norm_factor"]
        a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    var = "pfcand_normchi2"
    info = tagger_vars["pf_features"]["var_infos"][var]
    # finding what -1 transforms to
    chi2_min = -1 - info["median"] * info["norm_factor"]
    feature_dict[var][feature_dict[var] == chi2_min] = info["upper_bound"]

    return feature_dict


def get_svs_features(tagger_vars: dict, preselected_events: NanoEventsArray, jet_idx: int) -> dict:
    """
    Extracts the sv features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    jet = preselected_events.FatJetAK15[:, jet_idx]
    jet_svs = preselected_events.SV[
        preselected_events.JetSVsAK15.svIdx[
            (preselected_events.JetSVsAK15.svIdx != -1)
            * (preselected_events.JetSVsAK15.jetIdx == jet_idx)
        ]
    ]

    # get features

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.values_astype(jet_svs.eta > 0, int) * 2 - 1
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

    feature_dict["sv_mask"] = (
        ~(
            ak.pad_none(
                feature_dict["sv_etarel"], tagger_vars["sv_points"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .mask
        )
    ).astype(np.float32)

    # convert to numpy arrays and normalize features
    for var in tagger_vars["sv_features"]["var_names"]:
        a = (
            ak.pad_none(
                feature_dict[var], tagger_vars["sv_points"]["var_length"], axis=1, clip=True
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        info = tagger_vars["sv_features"]["var_infos"][var]
        a = (a - info["median"]) * info["norm_factor"]
        a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def runInferenceOnnx(tagger_resources_path: str, events: NanoEventsArray) -> dict:
    total_start = time.time()

    with open(f"{tagger_resources_path}/pnetmd_ak15_hww4q_preprocess.json") as f:
        tagger_vars = json.load(f)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    tagger_session = ort.InferenceSession(
        f"{tagger_resources_path}/pnetmd_ak15_hww4q_model.onnx", sess_options=opts
    )

    # prepare inputs for both fat jets
    tagger_inputs = []
    for jet_idx in range(2):
        feature_dict = {
            **get_pfcands_features(tagger_vars, events, jet_idx),
            **get_svs_features(tagger_vars, events, jet_idx),
        }

        tagger_inputs.append(
            {
                input_name: np.concatenate(
                    [
                        np.expand_dims(feature_dict[key], 1)
                        for key in tagger_vars[input_name]["var_names"]
                    ],
                    axis=1,
                )
                for input_name in tagger_vars["input_names"]
            }
        )

    # run inference for both fat jets
    tagger_outputs = []
    for jet_idx in range(2):
        print(f"Running inference for Jet {jet_idx + 1}")
        start = time.time()
        tagger_outputs.append(tagger_session.run(None, tagger_inputs[jet_idx])[0])
        time_taken = time.time() - start
        print(f"Inference took {time_taken}s")

    pnet_vars_list = []
    for jet_idx in range(2):
        pnet_vars_list.append(
            {
                "ak15FatJetParticleNetHWWMD_probQCD": tagger_outputs[jet_idx][:, 3],
                "ak15FatJetParticleNetHWWMD_probHWW4q": tagger_outputs[jet_idx][:, 0],
                "ak15FatJetParticleNetHWWMD_THWW4q": tagger_outputs[jet_idx][:, 0]
                / (tagger_outputs[jet_idx][:, 0] + tagger_outputs[jet_idx][:, 3]),
            }
        )

    pnet_vars_combined = {
        key: np.concatenate(
            [pnet_vars_list[0][key][:, np.newaxis], pnet_vars_list[1][key][:, np.newaxis]], axis=1
        )
        for key in pnet_vars_list[0]
    }

    print(f"Total time taken: {time.time() - total_start}s")
    return pnet_vars_combined


# from https://github.com/lgray/hgg-coffea/blob/triton-bdts/src/hgg_coffea/tools/chained_quantile.py
class wrapped_triton:
    def __init__(
        self,
        model_url: str,
    ) -> None:
        fullprotocol, location = model_url.split("://")
        _, protocol = fullprotocol.split("+")
        address, model, version = location.split("/")

        self._protocol = protocol
        self._address = address
        self._model = model
        self._version = version

    def __call__(self, input_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if self._protocol == "grpc":
            client = triton_grpc.InferenceServerClient(url=self._address, verbose=False)
            triton_protocol = triton_grpc
        elif self._protocol == "http":
            client = triton_http.InferenceServerClient(
                url=self._address,
                verbose=False,
                concurrency=12,
            )
            triton_protocol = triton_http
        else:
            raise ValueError(f"{self._protocol} does not encode a valid protocol (grpc or http)")

        # Infer
        inputs = []

        for key in input_dict:
            input = triton_protocol.InferInput(key, input_dict[key].shape, "FP32")
            input.set_data_from_numpy(input_dict[key])
            inputs.append(input)

        output = triton_protocol.InferRequestedOutput("softmax")

        request = client.infer(
            self._model,
            model_version=self._version,
            inputs=inputs,
            outputs=[output],
        )

        out = request.as_numpy("softmax")

        return out


def runInferenceTriton(tagger_resources_path: str, events: NanoEventsArray) -> dict:
    total_start = time.time()

    with open(f"{tagger_resources_path}/pnetmd_ak15_hww4q_preprocess.json") as f:
        tagger_vars = json.load(f)

    with open(f"{tagger_resources_path}/triton_config.json") as f:
        triton_config = json.load(f)

    triton_model = wrapped_triton(triton_config.model_url)

    # prepare inputs for both fat jets
    tagger_inputs = []
    for jet_idx in range(2):
        feature_dict = {
            **get_pfcands_features(tagger_vars, events, jet_idx),
            **get_svs_features(tagger_vars, events, jet_idx),
        }

        tagger_inputs.append(
            {
                input_name: np.concatenate(
                    [
                        np.expand_dims(feature_dict[key], 1)
                        for key in tagger_vars[input_name]["var_names"]
                    ],
                    axis=1,
                )
                for input_name in tagger_vars["input_names"]
            }
        )

    # run inference for both fat jets
    tagger_outputs = []
    for jet_idx in range(2):
        print(f"Running inference for Jet {jet_idx + 1}")
        start = time.time()
        tagger_outputs.append(triton_model(tagger_inputs[jet_idx]))
        time_taken = time.time() - start
        print(f"Inference took {time_taken}s")

    pnet_vars_list = []
    for jet_idx in range(2):
        pnet_vars_list.append(
            {
                "ak15FatJetParticleNetHWWMD_probQCD": tagger_outputs[jet_idx][:, 3],
                "ak15FatJetParticleNetHWWMD_probHWW4q": tagger_outputs[jet_idx][:, 0],
                "ak15FatJetParticleNetHWWMD_THWW4q": tagger_outputs[jet_idx][:, 0]
                / (tagger_outputs[jet_idx][:, 0] + tagger_outputs[jet_idx][:, 3]),
            }
        )

    pnet_vars_combined = {
        key: np.concatenate(
            [pnet_vars_list[0][key][:, np.newaxis], pnet_vars_list[1][key][:, np.newaxis]], axis=1
        )
        for key in pnet_vars_list[0]
    }

    print(f"Total time taken: {time.time() - total_start}s")
    return pnet_vars_combined
