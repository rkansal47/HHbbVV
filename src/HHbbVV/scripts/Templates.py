from PostProcess import PostProcess

if __name__ == "__main__":
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument(
         "--data-dir",
         dest="data_dir",
         default="../../../../data/skimmer/Apr28/",
         help="path to skimmed parquet",
         type=str,
     )
     parser.add_argument(
          "--year",
          default="year",
          choices=["2016","2016APV","2017","2018"],
          type=str,
     )

     # Both Jet's Msds > 50 & at least one jet with Txbb > 0.8
     filters = [
          [
               ("('ak8FatJetMsd', '0')", ">=", "50"),
               ("('ak8FatJetMsd', '1')", ">=", "50"),
               ("('ak8FatJetParticleNetMD_Txbb', '0')", ">=", "0.8"),
          ],
          [
               ("('ak8FatJetMsd', '0')", ">=", "50"),
               ("('ak8FatJetMsd', '1')", ">=", "50"),
               ("('ak8FatJetParticleNetMD_Txbb', '1')", ">=", "0.8"),
          ],
     ]
     
     p = PostProcess(args.data_dir,args.year,filters)
     p.apply_weights()
     p.get_bdt()
     hists = p.get_templates()
