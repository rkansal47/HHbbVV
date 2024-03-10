from __future__ import annotations

import importlib.metadata

import HHbbVV as m


def test_version():
    assert importlib.metadata.version("HHbbVV") == m.__version__
