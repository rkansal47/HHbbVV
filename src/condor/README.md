# Condor scripts

- [Condor scripts](#condor-scripts)
    - [Check jobs](#check-jobs)
    - [Combine pickles](#combine-pickles)


### Check jobs

Check that all jobs completed by going through output files:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/check_jobs.py --tag $TAG --processor trigger (--submit) --year $year; done
```


### Combine pickles

Combine all output pickles into one:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/combine_pickles.py --tag $TAG --processor trigger --r --year $year; done
```