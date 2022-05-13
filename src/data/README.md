Easy one-liner for making filelists: \
`for sample in /eos/uscms/store/user/lpcdihiggsboost/cmantill/PFNano/2017_UL_ak15/*; do find $sample/ -name "*.root" > 2017_UL_nano/$(basename "$sample").txt; done`
