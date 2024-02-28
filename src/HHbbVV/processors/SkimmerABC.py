"""
Skimmer Base Class - common functions for all skimmers.
Author(s): Raghav Kansal
"""

from abc import abstractmethod
import numpy as np
import awkward as ak
import pandas as pd

from coffea import processor

import pathlib
import pickle, json, gzip
import os

from typing import Dict, Tuple

from .common import LUMI
from . import corrections

import logging

logging.basicConfig(level=logging.INFO)


class SkimmerABC(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
        save_ak15 (bool, optional): save ak15 jets as well, for HVV candidate
    """

    XSECS = None

    def to_pandas(self, events: Dict[str, np.array]):
        """
        Convert our dictionary of numpy arrays into a pandas data frame
        Uses multi-index columns for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        return pd.concat(
            # [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in events.items()],
            [pd.DataFrame(v) for k, v in events.items()],
            axis=1,
            keys=list(events.keys()),
        )

    def dump_table(self, pddf: pd.DataFrame, fname: str, odir_str: str = None) -> None:
        """
        Saves pandas dataframe events to './outparquet'
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        local_dir = os.path.abspath(os.path.join(".", "outparquet"))
        if odir_str:
            local_dir += odir_str
        os.system(f"mkdir -p {local_dir}")

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, f"{local_dir}/{fname}")

    def pileup_cutoff(self, events, year, cutoff: float = 4):
        pweights = corrections.get_pileup_weight(year, events.Pileup.nPU.to_numpy())
        pw_pass = (
            (pweights["nominal"] <= cutoff)
            * (pweights["up"] <= cutoff)
            * (pweights["down"] <= cutoff)
        )
        logging.info(f"Passing pileup weight cut: {np.sum(pw_pass)} out of {len(events)} events")
        events = events[pw_pass]
        return events

    def get_dataset_norm(self, year, dataset):
        """
        Cross section * luminosity normalization for a given dataset and year.
        This still needs to be normalized with the acceptance of the pre-selection in post-processing.
        (Done in postprocessing/utils.py:load_samples())
        """
        if dataset in self.XSECS or "XToYHTo2W2BTo4Q2B" in dataset:
            # 1 fb xsec for resonant signal
            xsec = self.XSECS[dataset] if dataset in self.XSECS else 1e-3  # in pb
            weight_norm = xsec * LUMI[year]
        else:
            logging.warning("Weight not normalized to cross section")
            weight_norm = 1

        return weight_norm

    @abstractmethod
    def add_weights(self) -> Tuple[Dict, Dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        pass
