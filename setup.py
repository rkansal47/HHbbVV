#!/usr/bin/env python
from __future__ import annotations

from setuptools import setup

# packages = find_packages(
#     where="src",
#     include=["HHbbVV"],  # alternatively: `exclude=['additional*']`
# )

# package_dir = {"": "src"}

setup(
    name="HHbbVV",
    version="0.0.1",
    description="HHbbVV analysis code",
    author="Raghav Kansal",
    #   author_email='',
    url="https://github.com/rkansal47/HHbbVV",
    packages=["HHbbVV"],
    package_dir={"": "./src/"},
)
