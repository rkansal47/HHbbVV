#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2034

TAG=25Jun12BiggerFonts

python PlotFitsRes.py --cards-tag 25Mar11nTFQCD11nTF21 --plots-tag $TAG --mxmy 900 80 --hists1d --hists2d --toy-uncs
python PlotFitsRes.py --cards-tag 25Feb12UncorrNPFixes --plots-tag ${TAG}VR --mxmy 900 80 --hists1d --no-hists2d --b-only --toy-uncs
