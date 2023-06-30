# HH Inference scans

For HH limits, kl/c2v scans, etc. Adapted from https://github.com/LPC-HH/inference_scans.

- [HH Inference scans](#hh-inference-scans)
  - [Setup](#setup)


## Setup

1.  Make a fresh area on lxplus, for example:

```bash
mkdir ~/work/HH
```

2. Setup inference as follows
  - Set up the environment in a new folder with a clean environment (no `cmsenv`, ETC.)
  - Use `some_name` as the name of the environment.

```bash
git clone ssh://git@gitlab.cern.ch:7999/hh/tools/inference.git
cd inference
source setup.sh c1
```

TODO: update this with better datacard dirs and setting up card repos.

3. Add the following to the `.setups/v1.sh` file 
```bash
export DHI_DATACARDS_RUN2="/afs/cern.ch/user/r/rkansal/work/hh/datacards_run2"
export DHI_DATA="/afs/cern.ch/user/r/rkansal/work/hh/inference/data"
export DHI_STORE="/eos/user/r/rkansal/bbVV/inference"
export Cbbww4q="bbww_hadronic/v1"
export VERSION="dev"
export UNBLINDED="False"
```

4. Checkout this repo outside of `inference`

```bash
cd ..
git clone https://github.com/rkansal47/HHbbVV.git
```


5. Finally, from the `inference_scans` directory in `screen` or `tmux`, run the desired script:

```bash
cd inference
source setup.sh v1 # required before running a scan
cd ../HHbbVV/inference_scans/
source run_likelihood_scan_2d_kl_c2v.sh
``` 

Note! Need to make persistent screens to log off lxplus while running: 
https://hsf-training.github.io/analysis-essentials/shell-extras/persistent-screen.html

Also keep track of the specific lxplus node.