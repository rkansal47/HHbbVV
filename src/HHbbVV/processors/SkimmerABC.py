"""
Skimmer Base Class - common functions for all skimmers.
Author(s): Raghav Kansal
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from coffea import processor

from HHbbVV.hh_vars import LUMI

from . import corrections

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

    def to_pandas(self, events: dict[str, np.array]):
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
        import pyarrow as pa
        import pyarrow.parquet as pq

        local_dir = (Path() / "outparquet").resolve()
        if odir_str:
            local_dir += odir_str
        local_dir.mkdir(parents=True, exist_ok=True)

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, local_dir / fname)

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
            xsec = self.XSECS[dataset] if dataset in self.XSECS else 1e-3  # in pb  # noqa: SIM401
            weight_norm = xsec * LUMI[year]
        else:
            logging.warning("Weight not normalized to cross section")
            weight_norm = 1

        print("weight_norm", weight_norm)

        return weight_norm

    @abstractmethod
    def add_weights(self) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
