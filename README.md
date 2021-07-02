# HHbbVV

To submit with normal condor:

```
git clone https://github.com/rkansal47/HHbbVV/
cd HHbbVV
git checkout condor
# replace 'rkansal' in homedir in submit.py and the proxy address in submit.templ.jdl 
python condor/submit.py Jul1 run.py 20  # will need python3
for i in condor/Jul1/*.jdl; do condor_submit $i; done
```
