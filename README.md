# HHbbVV

### For [coffea-casa](https://coffea-casa.readthedocs.io/en/latest/cc_user.html):
1. after following instructions ^ set up an account, open the coffea-casa GUI (https://cmsaf-jh.unl.edu) and create an image
2. open `runCoffeaCasa.ipynb` 
3. import your desired processor, specify it in the `run_uproot_job` function, and specify your filelist
4. run the first three cells


### To submit with normal condor:

```
git clone https://github.com/rkansal47/HHbbVV/
cd HHbbVV
# replace 'rkansal' in homedir var in condor/submit.py and the proxy address in condor/submit.templ.jdl 
TAG=Aug18_skimmer
python condor/submit.py --processor skimmer --tag $TAG --files-per-job 20  # will need python3 (recommended to set up via miniconda)
for i in condor/$TAG/*.jdl; do condor_submit $i; done
```

Pickle files will be saved in eos directory of specified user at path `~/eos/bbVV/<processor type>/<tag>/outfiles/`, in the format `{'nevents': int, 'skimmed_events': dict of coffea 'column_accumulator's}`

After jobs finish, they can be combined (and normalized by total events in the case of MC) via
```
python condor/combine_pickles.py --year 2017 --indir /eos/uscms/store/user/rkansal/bbVV/skimmer/$TAG/outfiles/ --r True --norm True --combine-further True
```

The `--combine-further` argument combines them into broader categories as well, saved in the `<indir>/<year>_combined/` directory.



Check out more args for both scripts with the `--help` arg (e.g. `python condor/submit.py --help`)


To test locally, can do e.g.:

`python run.py --starti 0 --endi 1 --year 2017 --processor skimmer --condor --samples '2017_HHToBBVVToBBQQQQ_cHHH1'`


#### TODO: instructions for lpcjobqueue (currently quite buggy)
