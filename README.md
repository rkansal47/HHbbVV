# HHbbVV

#### For [coffea-casa](https://coffea-casa.readthedocs.io/en/latest/cc_user.html):
1. after following instructions ^ set up an account, open the coffea-casa GUI (https://cmsaf-jh.unl.edu) and create an image
2. open `runCoffeaCasa.ipynb` 
3. import your desired processor, specify it in the `run_uproot_job` function, and specify your filelist
4. run the first three cells


#### To submit with normal condor:

```
git clone https://github.com/rkansal47/HHbbVV/
cd HHbbVV
git checkout condor
# replace 'rkansal' in homedir var in condor/submit.py and the proxy address in condor/submit.templ.jdl 
python condor/submit.py Jul1 run.py 20  # will need python3
for i in condor/Jul1/*.jdl; do condor_submit $i; done
```




#### TODO: instructions for lpcjobqueue (currently quite buggy)
