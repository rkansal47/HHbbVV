from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http


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

    def __call__(self, input_dict: dict[str, np.ndarray]) -> np.ndarray:
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

        output = triton_protocol.InferRequestedOutput("softmax__0")

        request = client.infer(
            self._model,
            model_version=self._version,
            inputs=inputs,
            outputs=[output],
        )

        out = request.as_numpy("softmax__0")

        return out


# model_url = "triton+grpc://ailab01.fnal.gov:8001/particlenet_hww/1"
# model_url = "triton+grpc://prp-gpu-1.t2.ucsd.edu:8001/particlenet_hww/1"
model_url = "triton+grpc://67.58.49.52:8001/particlenet_hww_ul_4q_3q/1"
# model_url = "triton+grpc://localhost:8001/particlenet_hww_ul_4q_3q/1"
triton_model = wrapped_triton(model_url)

batch_size = 10
pfs = 100
svs = 7

input_dict = {
    "pf_points": np.random.rand(batch_size, 2, pfs).astype("float32"),
    "pf_features": np.random.rand(batch_size, 19, pfs).astype("float32"),
    "pf_mask": (np.random.rand(batch_size, 1, pfs) > 0.2).astype("float32"),
    "sv_points": np.random.rand(batch_size, 2, svs).astype("float32"),
    "sv_features": np.random.rand(batch_size, 11, svs).astype("float32"),
    "sv_mask": (np.random.rand(batch_size, 1, svs) > 0.2).astype("float32"),
}

input_dict = {
    "pf_points__0": np.random.rand(batch_size, 2, pfs).astype("float32"),
    "pf_features__1": np.random.rand(batch_size, 19, pfs).astype("float32"),
    "pf_mask__2": (np.random.rand(batch_size, 1, pfs) > 0.2).astype("float32"),
    "sv_points__3": np.random.rand(batch_size, 2, svs).astype("float32"),
    "sv_features__4": np.random.rand(batch_size, 11, svs).astype("float32"),
    "sv_mask__5": (np.random.rand(batch_size, 1, svs) > 0.2).astype("float32"),
}

print("running inference")

with Path("tensors.pkl").open("wb") as f:
    pickle.dump(input_dict, f)


model_url = "triton+grpc://67.58.49.52:8001/particlenet_hww_ul_4q_3q/1"
triton_model = wrapped_triton(model_url)
output = triton_model(input_dict)
