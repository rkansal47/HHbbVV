from __future__ import annotations

import warnings
from pathlib import Path

import ROOT


def _reduced_corr_matrix(fit_result, varsToIgnore=None, varsOfInterest=None, threshold=0):
    if varsOfInterest is None:
        varsOfInterest = []
    if varsToIgnore is None:
        varsToIgnore = []
    if threshold < 0:
        raise ValueError(
            "Threshold for correlation matrix values to plot must be a positive number."
        )

    ROOT.gStyle.SetOptStat(0)
    # ROOT.gStyle.SetPaintTextFormat('.3f')
    CM = fit_result.correlationMatrix()
    finalPars = fit_result.floatParsFinal()

    nParams = CM.GetNcols()
    finalParamsDict = {}
    for cm_index in range(nParams):
        if varsOfInterest == []:
            vname = finalPars.at(cm_index).GetName()
            if vname not in varsToIgnore:
                finalParamsDict[finalPars.at(cm_index).GetName()] = cm_index
        else:
            if finalPars.at(cm_index).GetName() in varsOfInterest:
                finalParamsDict[finalPars.at(cm_index).GetName()] = cm_index

    for paramXName in sorted(finalParamsDict.keys()):
        cm_index_x = finalParamsDict[paramXName]

        if not any(
            abs(CM[cm_index_x][cm_index_y]) > threshold
            for cm_index_y in range(nParams)
            if cm_index_y != cm_index_x
        ):
            finalParamsDict.pop(paramXName)

    nFinalParams = len(finalParamsDict.keys())
    out = ROOT.TH2D(
        "correlation_matrix",
        "correlation_matrix",
        nFinalParams,
        0,
        nFinalParams,
        nFinalParams,
        0,
        nFinalParams,
    )
    out_txt = ""

    for out_x_index, paramXName in enumerate(sorted(finalParamsDict.keys())):
        cm_index_x = finalParamsDict[paramXName]

        # if not any(
        #     [
        #         abs(CM[cm_index_x][cm_index_y]) > threshold
        #         for cm_index_y in range(nParams)
        #         if cm_index_y != cm_index_x
        #     ]
        # ):
        #     continue

        for out_y_index, paramYName in enumerate(sorted(finalParamsDict.keys())):
            cm_index_y = finalParamsDict[paramYName]
            if cm_index_x > cm_index_y:
                out_txt += "%s:%s = %s\n" % (paramXName, paramYName, CM[cm_index_x][cm_index_y])
            out.Fill(out_x_index + 0.5, out_y_index + 0.5, CM[cm_index_x][cm_index_y])

        out.GetXaxis().SetBinLabel(out_x_index + 1, finalPars.at(cm_index_x).GetName())
        out.GetYaxis().SetBinLabel(out_x_index + 1, finalPars.at(cm_index_x).GetName())

    out.SetMinimum(-1)
    out.SetMaximum(+1)

    return out, out_txt


def plot_correlation_matrix(varsToIgnore, threshold=0, corrText=False):
    fit_result_file = ROOT.TFile.Open("fitDiagnostics.root")
    for fittag in ["s", "b"]:
        fit_result = fit_result_file.Get("fit_" + fittag)
        if hasattr(fit_result, "correlationMatrix"):
            corrMtrx, corrTxt = _reduced_corr_matrix(
                fit_result, varsToIgnore=varsToIgnore, threshold=threshold
            )
            corrMtrxCan = ROOT.TCanvas("c", "c", 1400, 1000)
            corrMtrxCan.cd()
            corrMtrxCan.SetBottomMargin(0.22)
            corrMtrxCan.SetLeftMargin(0.17)
            corrMtrxCan.SetTopMargin(0.06)

            corrMtrx.GetXaxis().SetLabelSize(0.01)
            corrMtrx.GetYaxis().SetLabelSize(0.01)
            corrMtrx.Draw("colz text" if corrText else "colz")
            corrMtrxCan.Print(f"correlation_matrix_{fittag}.png", "png")
            corrMtrxCan.Print(f"correlation_matrix_{fittag}.pdf", "pdf")

            with Path(f"correlation_matrix_{fittag}.txt").open("w") as corrTxtFile:
                corrTxtFile.write(corrTxt)

        else:
            warnings.warn("Not able to produce correlation matrix.", RuntimeWarning, stacklevel=2)

    fit_result_file.Close()


if __name__ == "__main__":
    plot_correlation_matrix(varsToIgnore=[], threshold=0.1, corrText=False)
