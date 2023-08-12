# contains the functions that I used to plot some of the data in bbVV skimmer before selections were applied.
# Most data stored as text or output as warning logs that I then filtered.

# this script also contains the extra code that I didn't use but is still nice to keep since it was tricky to come up with.

# Code used for debugging

# counts = [len(event) for event in jets]
# logger.warning(f"Jets total elements: {counts} len: {len(jets)}")
# counts = [len(event) for event in ak.any(e_pairs_mask, axis=-1)]
# logger.warning(f"Jets total elements: {counts} len: {len(ak.any(e_pairs_mask, axis=-1))}")

# logging.warning(f"Jets eta {jets.eta}")
# logging.warning(f"Jets pt {jets.pt}")

# here we can print some information about the jets so that we can study the selections a bit.  (this is when studying gen info)
# jet_pairs = ak.cartesian({"reco": jets, "gen": vbfGenJets[:,0:2]})
# # Calculate delta eta and delta phi for each pair
# delta_eta = jet_pairs["reco"].eta - jet_pairs["gen"].eta
# delta_phi = np.pi - np.abs(np.abs(jet_pairs["reco"].phi - jet_pairs["gen"].phi) - np.pi)
# # Calculate delta R for each pair
# delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
# # Apply a mask for a low delta R value
# mask_low_delta_R = delta_R < 0.4
# num_per_event= ak.sum(mask_low_delta_R, axis=-1)
# logging.warning(" | ".join([f"Percentage of original events with {i} true values: {ak.sum(num_per_event == i) / len(num_per_event) * 100:.2f}% | Total: {len(num_per_event)} | Count: {ak.sum(num_per_event == i)}" for i in range(3)]))
# mask_VVJet = VVJet.pt > -9000
# mask_bbJet = bbJet.pt > -9000
# m2 = mask_VVJet & mask_bbJet
# #logging.warning(f' num of events with both higs jets pt above -9000 {ak.sum(m2)}')
# mask_VVJet = (VVJet.pt >= 300)  & (np.abs(VVJet.eta) < 4.7) & ((VVJet.M >= 50) | (VVJet.M <= 250))
# mask_bbJet = (bbJet.pt >= 300) & (bbJet.M > 50)
# m2 = mask_VVJet & mask_bbJet
# #logging.warning(f' num of events with both higs jets passing ak8 jet cuts {ak.sum(m2)}')
# # we want to see reconstruction accuracy on unfiltered events. simply count number of events w 0 1 and 2 correctly matched jets.
# #self.getVBFGenMatchCount( events,jets,bbJet,VVJet,ak4_jet_mask & electron_muon_overlap_mask,ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][bb_mask])
# # we want to see how the fatjet delta R affects this (changing both at once). 0.8 - 2 in incremeents of 0.5. calculate 0 1 and 2 correct rate
# # then we want to see how the pt of the WW affects width.
# #self.getVBFGenPtvsR2data( events,jets,bbJet,VVJet,ak4_jet_mask & electron_muon_overlap_mask)
# # for a single event, I want to print the eta, phi, pt of each jet. Then print the eta phi pt of the sorted jets, candidates, and fatjets in that event.print each in list format.
# # here we want to collect the distribution of the vbf jets' delta R from the higgs jets in the original events to see what it looks like
# #self.getTruthDeltaRDistribution( events,jets,bbJet,VVJet,ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][bb_mask])
# #self.getVBFGenvsgenHH2ddata(events,ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][bb_mask])
# #logging.warning(events.FatJet.fields)
# #logging.warning(events.fields)
# logging.warning(f"Gen Jets eta {vbfGenJets.eta}")
# logging.warning(f"Gen Jets mass {vbfGenJets.mass}")
# logging.warning(f"Gen Jets pt {vbfGenJets.pt}")
# logging.warning(f"Gen Jets pdgId {vbfGenJets.pdgId}")

# Code used for sorting method tests.

# Generating the three different sorting methods: (pt, dijet eta, dijet mass). For each we keep the two best

# # dijet eta sorting
# jet_pairs = ak.combinations(vbfJets, 2, axis=1, fields=["j1", "j2"])
# delta_eta = np.abs(jet_pairs.j1.eta - jet_pairs.j2.eta)
# eta_sorted_pairs = jet_pairs[ak.argsort(delta_eta, axis=1,ascending= False)] # picks the two furthest jets.
# eta_first_pairs = ak.firsts(eta_sorted_pairs, axis=1)
# eta_sorted_mask = ak.any((vbfJets[:, :, None].eta == eta_first_pairs.j1.eta) | (vbfJets[:, :, None].eta == eta_first_pairs.j2.eta), axis=2)
# vbfJets_sorted_eta = vbfJets[eta_sorted_mask]

# # dijet mass sorting
# jj = jet_pairs.j1 + jet_pairs.j2
# mass_sorted_pairs = jet_pairs[ak.argsort(jj.mass, axis=1,ascending= False)] # picks the two furthest jets.
# mass_first_pairs = ak.firsts(mass_sorted_pairs, axis=1)
# mass_sorted_mask = ak.any((vbfJets[:, :, None].mass == mass_first_pairs.j1.mass) | (vbfJets[:, :, None].mass == mass_first_pairs.j2.mass), axis=2)
# vbfJets_sorted_mass = vbfJets[mass_sorted_mask]

# # Compute dijet eta and dijet mass cuts
# jj_sorted_mass = mass_first_pairs.j1 + mass_first_pairs.j2 # we update dijet since the previous one had many per event. this should be one number per event.
# mass_jj_cut_sorted_mass = jj_sorted_mass.mass > 500
# eta_jj_cut_sorted_mass = np.abs(mass_first_pairs.j1.eta - mass_first_pairs.j2.eta)  > 4.0
# vbfJets_mask_sorted_mass = vbfJets_mask * mass_jj_cut_sorted_mass * eta_jj_cut_sorted_mass

# jj_sorted_eta = eta_first_pairs.j1 + eta_first_pairs.j2
# mass_jj_cut_sorted_eta = jj_sorted_eta.mass  > 500
# eta_jj_cut_sorted_eta = np.abs(eta_first_pairs.j1.eta - eta_first_pairs.j2.eta)  > 4.0
# vbfJets_mask_sorted_eta = vbfJets_mask * mass_jj_cut_sorted_eta * eta_jj_cut_sorted_eta

# n_good_vbf_jets = ak.fill_none(ak.sum(vbfJets_mask, axis=1),0) #* eta_jj_mask * mass_jj_mask # filters out the events where the vbf jets are too close and mass too small., May need to convert to 0, 1 array instead of mask.
# n_good_vbf_jets_sorted_mass = ak.fill_none(ak.sum(vbfJets_mask_sorted_mass, axis=1),0)
# n_good_vbf_jets_sorted_eta = ak.fill_none(ak.sum(vbfJets_mask_sorted_eta, axis=1),0)

# self.getVBFGenMatchCount(events,jets)
# self.getVBFGenMatchCount(events,vbfJets)
# self.getVBFGenMatchCount(events,vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2])
# self.getVBFGenMatchCount(events,vbfJets_sorted_pt)
# self.getVBFGenMatchCount(events,vbfJets_sorted_mass)
# self.getVBFGenMatchCount(events,vbfJets_sorted_eta)

# logging.warning(np.sum(ak.sum(ak4_jet_mask, axis=1).to_numpy())) #," ," and final: {",np.sum(ak.sum(vbfJets_mask, axis=1).to_numpy())," compared to initial: {",np.sum(ak.sum(jets, axis=1).to_numpy()))
# logging.warning(np.sum(ak.sum(electron_muon_overlap_mask, axis=1).to_numpy()))
# logging.warning(np.sum(ak.sum(fatjet_overlap_mask, axis=1).to_numpy()))
# logging.warning(np.sum(ak.sum(vbfJets_mask, axis=1).to_numpy()))
# logging.warning(ak.sum(ak.num(jets, axis=1) ))

# dijet mass must be greater than 500
# dijet = vbfJets[:,0:1] + vbfJets[:,1:2]
# mass_jj_mask = dijet.mass > 500
# eta_jj_mask = (np.abs(vbfJets[:,0:1].eta -vbfJets[:,1:2].eta) > 4.0)
# vbfJet1, vbfJet2 = vbfJets[:,0],vbfJets[:,1]

# vbfVars[f"vbfptSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].pt, 2, axis=1)
# vbfVars[f"vbfetaSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].eta, 2, axis=1)
# vbfVars[f"vbfphiSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].phi, 2, axis=1)
# vbfVars[f"vbfMSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].mass, 2, axis=1)

# vbfVars[f"vbfptSortedM"] = pad_val(vbfJets_sorted_mass.pt, 2, axis=1)
# vbfVars[f"vbfetaSortedM"] = pad_val(vbfJets_sorted_mass.eta, 2, axis=1)
# vbfVars[f"vbfphiSortedM"] = pad_val(vbfJets_sorted_mass.phi, 2, axis=1)
# vbfVars[f"vbfMSortedM"] = pad_val(vbfJets_sorted_mass.mass, 2, axis=1)

# vbfVars[f"vbfptSortedeta"] = pad_val(vbfJets_sorted_eta.pt, 2, axis=1)
# vbfVars[f"vbfetaSortedeta"] = pad_val(vbfJets_sorted_eta.eta, 2, axis=1)
# vbfVars[f"vbfphiSortedeta"] = pad_val(vbfJets_sorted_eta.phi, 2, axis=1)
# vbfVars[f"vbfMSortedeta"] = pad_val(vbfJets_sorted_eta.mass, 2, axis=1)

# vbfVars[f"nGoodVBFJetsUnsorted"] = n_good_vbf_jets.to_numpy() # the original one does not have jj cuts since it assumes no sorting.
# vbfVars[f"nGoodVBFJetsSortedM"] = n_good_vbf_jets_sorted_mass.to_numpy()
# vbfVars[f"nGoodVBFJetsSortedeta"] = n_good_vbf_jets_sorted_eta.to_numpy()


def getVBFGenMatchCount(
    self, events: ak.Array, jets: ak.Array, bbJet, VVJet, base_mask, bbJet_Txbb
):
    """Tests how fatjet overlap delta R affects vbf selecting."""

    vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]

    for R1 in np.arange(0.5, 2, 0.1):
        for R2 in np.arange(0.5, 2, 0.1):
            fatjet_overlap_mask = (np.abs(jets.delta_r(bbJet)) > R1) & (
                np.abs(jets.delta_r(VVJet)) > R2
            )

            # compute n_good_vbf_jets + incorporate eta_jj > 4.0
            vbfJets_mask = base_mask & fatjet_overlap_mask
            # vbfJets_mask = fatjet_overlap_mask # this is for unflitered events
            vbfJets = jets[vbfJets_mask]
            vbfJets_sorted_pt = vbfJets[ak.argsort(vbfJets.pt, ascending=False)]
            vbfJets_sorted_pt = ak.pad_none(vbfJets_sorted_pt, 2, clip=True)
            jj_sorted_pt = vbfJets_sorted_pt[:, 0:1] + vbfJets_sorted_pt[:, 1:2]
            mass_jj_cut_sorted_pt = jj_sorted_pt.mass > 500
            eta_jj_cut_sorted_pt = (
                np.abs(vbfJets_sorted_pt[:, 0:1].eta - vbfJets_sorted_pt[:, 1:2].eta) > 4.0
            )
            vbfJets_mask_sorted_pt = vbfJets_mask * mass_jj_cut_sorted_pt * eta_jj_cut_sorted_pt
            num_sorted_pt = ak.fill_none(ak.sum(vbfJets_mask_sorted_pt, axis=1), 0)

            # here we can print some information about the jets so that we can study the selections a bit.
            # jet_pairs = ak.cartesian({"reco": vbfJets, "gen": vbfGenJets[:,0:2]}) # this is only for unfiltered events
            jet_pairs = ak.cartesian({"reco": vbfJets_sorted_pt[:, 0:2], "gen": vbfGenJets[:, 0:2]})

            # Calculate delta eta and delta phi for each pair
            delta_eta = jet_pairs["reco"].eta - jet_pairs["gen"].eta
            delta_phi = np.pi - np.abs(np.abs(jet_pairs["reco"].phi - jet_pairs["gen"].phi) - np.pi)

            # Calculate delta R for each pair
            delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

            # Apply a mask for a low delta R value
            mask_low_delta_R = delta_R < 0.4
            num_per_event = ak.sum(mask_low_delta_R, axis=-1)  # miscounts 0's since some are empty

            # logging.warning(VVJet.pt)
            # logging.warning(bbJet.pt)
            # logging.warning(f'{num_sorted_pt} {len(VVJet)} {len(num_sorted_pt)} {len(num_per_event)}')
            # logging.warning(num_per_event)

            # Create mask for ak8 jet requirements
            mask_VVJet = (
                (VVJet.pt >= 300) & (np.abs(VVJet.eta) < 4.7) & ((VVJet.M >= 50) | (VVJet.M <= 250))
            )
            mask_bbJet = (
                (bbJet.pt >= 300) & (bbJet.M > 50) & (np.abs(VVJet.eta) < 4.7) & (bbJet_Txbb > 0.8)
            )
            m2 = mask_VVJet & mask_bbJet
            mask_sorted_pt = num_sorted_pt > 1

            # Combine masks with logical 'and' operation
            total_mask = (
                m2 & mask_sorted_pt
            )  # this is comendd out for og events debug applying this since the sizes ndont match

            # Apply the mask to filter num_per_event
            # logging.warning(f'{len(total_mask)} {len(num_per_event)}')
            num_per_event = num_per_event[
                total_mask
            ]  # num_per_event = np.where(np.isnan(num_per_event.astype(float)), 0, num_per_event).astype(int)
            # logging.warning(num_per_event[0:4])

            # logging.warning(" | ".join([f"Num {val}: {perc:.2f}% (Count: {cnt})" for val, perc, cnt in zip(*np.unique(num_per_event, return_counts=True), np.bincount(num_per_event) / len(num_per_event) * 100)]) + f" | Total: {len(num_per_event)}")

            counts = np.bincount(
                num_per_event, minlength=3
            )  # Ensure counts for all [0, 1, 2] even if some might be missing
            # logging.warning(f"[{counts[0]}, {counts[1]}, {counts[2]}, {len(num_per_event)}, {R1:.1f}, {R2:.1f}]")

            result = [
                (counts[1] + counts[2]) / ak.sum(m2),
                R1,
                R2,
            ]  # normalize count by number of events passing ak8 jet selections
            # logging.warning(f'{result} ,{ak.sum(m2)}, { len(m2)}, {len(num_per_event)},{counts[0]},{counts[1]},{counts[2]} ')

            # Write the result to the output file, appending if the file already exists
            if True:
                output_file = "heatmap_fatjet_exclusion_ak8selections_normalized_selected.txt"
                with open(output_file, "a") as f:
                    f.write(str(result) + "\n")

    # we need to make a seperate look that tests how pt of VVjet is related to the selection rate.

    # we want to see reconstruction accuracy on unfiltered events. simply count number of events w 0 1 and 2 correctly matched jets.

    # we want to see how the fatjet delta R affects this (changing both at once). 0.8 - 2 in incremeents of 0.5. calculate 0 1 and 2 correct rate
    # then we want to see how the pt of the WW affects width.

    # for a single event, I want to print the eta, phi, pt of each jet. Then print the eta phi pt of the sorted jets, candidates, and fatjets in that event.print each in list format.
    return


def getVBFGenPtvsR2data(self, events: ak.Array, jets: ak.Array, bbJet, VVJet, base_mask):
    """"""
    vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]

    # Path to the output file
    output_file = "outputR2vsPtVVJet.txt"

    # If you need to reset the file, uncomment the following lines:
    # if os.path.exists(output_file):
    #     os.remove(output_file)
    logging.warning(f"{np.sum(VVJet.pt < 0)} {len(VVJet)}")  # fatjets
    R1 = 1.2
    for i in range(len(VVJet)):  # loop over the events
        # Get the pt of the VVJet for the current event
        pt = VVJet[i].pt

        if pt < 0:
            continue
        # logging.warning(f'VVJet: {pt},{VVJet[i].eta},{VVJet[i].phi}')

        for R2 in np.arange(0.6, 2, 0.1):
            # Apply the mask to the i-th event
            fatjet_overlap_mask = (np.abs(jets[i].delta_r(bbJet[i])) > R1) & (
                np.abs(jets[i].delta_r(VVJet[i])) > R2
            )
            vbfJets_mask = base_mask[i] & fatjet_overlap_mask

            vbfJets = jets[i][vbfJets_mask]
            vbfJets_sorted_pt = vbfJets[ak.argsort(vbfJets.pt, ascending=False)]
            # logging.warning(f'1021 {vbfJets_sorted_pt}')
            # while len(vbfJets_sorted_pt) < 2:
            #    vbfJets_sorted_pt = np.append(vbfJets_sorted_pt, None)
            # logging.warning(f'1024')
            if len(vbfJets_sorted_pt) < 2:
                break
            jj_sorted_pt = vbfJets_sorted_pt[0:1] + vbfJets_sorted_pt[1:2]
            mass_jj_cut_sorted_pt = jj_sorted_pt.mass > 500
            eta_jj_cut_sorted_pt = (
                np.abs(vbfJets_sorted_pt[0:1].eta - vbfJets_sorted_pt[1:2].eta) > 4.0
            )
            vbfJets_mask_sorted_pt = vbfJets_mask * mass_jj_cut_sorted_pt * eta_jj_cut_sorted_pt
            # logging.warning(f'1028')

            # Calculate delta R for the four possible pairings
            reco_1, reco_2 = vbfJets_sorted_pt[:2]
            gen_1, gen_2 = vbfGenJets[i, :2]

            delta_eta_11 = reco_1.eta - gen_1.eta
            delta_phi_11 = np.pi - np.abs(np.abs(reco_1.phi - gen_1.phi) - np.pi)
            delta_R_11 = np.sqrt(delta_eta_11**2 + delta_phi_11**2)

            delta_eta_12 = reco_1.eta - gen_2.eta
            delta_phi_12 = np.pi - np.abs(np.abs(reco_1.phi - gen_2.phi) - np.pi)
            delta_R_12 = np.sqrt(delta_eta_12**2 + delta_phi_12**2)

            delta_eta_21 = reco_2.eta - gen_1.eta
            delta_phi_21 = np.pi - np.abs(np.abs(reco_2.phi - gen_1.phi) - np.pi)
            delta_R_21 = np.sqrt(delta_eta_21**2 + delta_phi_21**2)

            delta_eta_22 = reco_2.eta - gen_2.eta
            delta_phi_22 = np.pi - np.abs(np.abs(reco_2.phi - gen_2.phi) - np.pi)
            delta_R_22 = np.sqrt(delta_eta_22**2 + delta_phi_22**2)

            # Compute a binary indicator of successful reconstruction
            successfulMatch = int(
                ((delta_R_11 < 0.4) and (delta_R_22 < 0.4))
                or ((delta_R_12 < 0.4) and (delta_R_21 < 0.4))
            )

            # Store the quantities in a list
            result = [pt, R2, successfulMatch]
            # logging.warning(f'{result} , {delta_R_11} {delta_R_21} {delta_R_12} {delta_R_22} ')

            # Write the result to the output file, appending if the file already exists
            if False:
                with open(output_file, "a") as f:
                    f.write(str(result) + "\n")


def getTruthDeltaRDistribution(self, events, vbfJets, bbJet, VVJet, bbJet_Txbb):
    vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]

    genjet1 = vbfGenJets[:, 0:1]
    genjet2 = vbfGenJets[:, 1:2]
    drj1bb = genjet1.delta_r(bbJet)
    drj2bb = genjet2.delta_r(
        bbJet
    )  # I believe this will have some empty values hopefully still len 9600
    drj1VV = genjet1.delta_r(VVJet)
    drj2VV = genjet2.delta_r(VVJet)

    # Create mask for ak8 jet requirements
    mask_VVJet = (
        (VVJet.pt >= 300) & (np.abs(VVJet.eta) < 4.7) & ((VVJet.M >= 50) | (VVJet.M <= 250))
    )
    mask_bbJet = (bbJet.pt >= 300) & (bbJet.M > 50) & (bbJet_Txbb > 0.8)
    m2 = mask_VVJet & mask_bbJet

    # Combine masks with logical 'and' operation
    total_mask = m2

    # Apply the mask to the new delta R values
    drj1bb_filtered = drj1bb[total_mask]
    drj2bb_filtered = drj2bb[total_mask]
    drj1VV_filtered = drj1VV[total_mask]
    drj2VV_filtered = drj2VV[total_mask]

    # logging.warning(f'testing truth thing{drj1VV[total_mask]} and {drj2VV[total_mask]} and {VVJet.phi[total_mask]}  and {genjet1.phi[total_mask]} and {genjet2.phi[total_mask]}')

    # Update the result list to include the new delta R values
    result = [
        [drj1bb_val, drj2bb_val, drj1VV_val, drj2VV_val]
        for drj1bb_val, drj2bb_val, drj1VV_val, drj2VV_val in zip(
            drj1bb_filtered, drj2bb_filtered, drj1VV_filtered, drj2VV_filtered
        )
    ]

    # Write the result to the output file
    if True:
        output_file = "output_deltaRtruthinfo_bbtagging.txt"
        with open(output_file, "a") as f:
            for pair in result:
                cleaned_pair = [
                    float(val[0]) for val in pair
                ]  # Convert each awkward array to a simple float
                f.write(str(cleaned_pair) + "\n")


# function that allows us to examine unfiltered events gen info relationships.
def getVBFGenvsgenHH2ddata(self, events: ak.Array, bbJet_Txbb):
    """"""
    vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]

    # Path to the output file
    output_file = "outputR2vsPtgenHH_posptreq.txt"

    # find higgs boson gen variables:
    # finding the two gen higgs
    GEN_FLAGS = ["fromHardProcess", "isLastCopy"]
    b_PDGID = 5
    Z_PDGID = 23
    W_PDGID = 24
    HIGGS_PDGID = 25
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    # finding bb and VV children
    higgs_children = higgs.children
    is_bb = abs(higgs_children.pdgId) == b_PDGID
    is_VV = (abs(higgs_children.pdgId) == W_PDGID) + (abs(higgs_children.pdgId) == Z_PDGID)

    # fatjet gen matching
    Hbb = higgs[ak.sum(is_bb, axis=2) == 2]
    Hbb.eta
    Hbb = ak.pad_none(Hbb, 1, axis=1, clip=True)[:, 0]

    HVV = higgs[ak.sum(is_VV, axis=2) == 2]
    HVV = ak.pad_none(HVV, 1, axis=1, clip=True)[:, 0]  # maybe this is ruining it

    mask_VVJet = (HVV.pt >= 300) & (np.abs(HVV.eta) < 4.7) & ((HVV.mass >= 50) | (HVV.mass <= 250))
    mask_bbJet = (Hbb.pt >= 300) & (Hbb.mass > 50)

    vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]
    # genjet1 = vbfGenJets[:,0:1]
    # genjet2 = vbfGenJets[:,1:2]

    for i in range(len(HVV)):  # loop over the events
        # logging.warning(HVV[i])

        # if mask_VVJet[i] == False or mask_bbJet[i] == False:
        #   continue
        if bbJet_Txbb[i] < 0:
            continue

        # Calculate delta R for the four possible pairings of higgs to quarks (all generator info)
        reco_1, reco_2 = Hbb[i], HVV[i]
        gen_1, gen_2 = vbfGenJets[i, :2]

        if reco_1 is None or reco_2 is None or gen_1 is None or gen_2 is None:
            continue

        delta_eta_11 = reco_1.eta - gen_1.eta
        delta_phi_11 = np.pi - np.abs(np.abs(reco_1.phi - gen_1.phi) - np.pi)
        delta_R_11 = np.sqrt(delta_eta_11**2 + delta_phi_11**2)

        delta_eta_12 = reco_1.eta - gen_2.eta
        delta_phi_12 = np.pi - np.abs(np.abs(reco_1.phi - gen_2.phi) - np.pi)
        delta_R_12 = np.sqrt(delta_eta_12**2 + delta_phi_12**2)

        delta_eta_21 = reco_2.eta - gen_1.eta
        delta_phi_21 = np.pi - np.abs(np.abs(reco_2.phi - gen_1.phi) - np.pi)
        delta_R_21 = np.sqrt(delta_eta_21**2 + delta_phi_21**2)

        delta_eta_22 = reco_2.eta - gen_2.eta
        delta_phi_22 = np.pi - np.abs(np.abs(reco_2.phi - gen_2.phi) - np.pi)
        delta_R_22 = np.sqrt(delta_eta_22**2 + delta_phi_22**2)

        # we wish to track the delta r distributions and the pts of the HVV jets.
        # Store the quantities in a list
        result = [
            reco_1.pt,
            reco_2.pt,
            delta_R_11,
            delta_R_12,
            delta_R_21,
            delta_R_22,
            bbJet_Txbb[i],
        ]
        logging.warning(f"{result} , {delta_R_11} {delta_R_21} {delta_R_12} {delta_R_22} ")

        # Write the result to the output file, appending if the file already exists
        if True:
            with open(output_file, "a") as f:
                f.write(str(result) + "\n")
