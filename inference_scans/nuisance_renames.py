def rename_nuisance(nuisance: str):
    if nuisance.startswith("CMS_bbWW_hadronic_"):
        nuisance = nuisance.split("CMS_bbWW_hadronic_")[1]
    
    if nuisance.startswith("tf_dataResidual_bbFatJetParticleNetMass_"):
        return "TF_" + nuisance.split("tf_dataResidual_bbFatJetParticleNetMass_")[1]
    
    return nuisance