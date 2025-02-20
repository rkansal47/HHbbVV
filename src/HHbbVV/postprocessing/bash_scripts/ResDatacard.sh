#!/bin/bash
sig_sample=NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250
templates_dir=/eos/uscms/store/user/rkansal/bbVV/templates/25Feb8XHYFix
 python3 -u postprocessing/CreateDatacard.py --templates-dir "${templates_dir}/${sig_sample}" --bg-templates-dir "${templates_dir}/backgrounds" --sig-separate --resonant --model-name 25Feb12UncorrNPFixes --sig-sample "${sig_sample}"
