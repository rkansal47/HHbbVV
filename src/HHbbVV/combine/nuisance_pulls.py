from __future__ import annotations

import ROOT

nuis_file = ROOT.TFile.Open("nuisance_pulls.root")
nuis_can = nuis_file.Get("nuisances")
nuis_can.Print("nuisance_pulls.pdf", "pdf")
nuis_file.Close()
