from __future__ import annotations

HLTs = {
    "2016": [
        "AK8DiPFJet250_200_TrimMass30_BTagCSV_p20",
        "AK8DiPFJet280_200_TrimMass30_BTagCSV_p20",
        #
        "AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20",
        "AK8PFHT700_TrimR0p1PT0p03Mass50",
        #
        "AK8PFJet360_TrimMass30",
        "AK8PFJet450",
        "PFJet450",
        #
        "PFHT800",
        "PFHT900",
        "PFHT1050",
        #
        "PFHT750_4JetPt50",
        "PFHT750_4JetPt70",
        "PFHT800_4JetPt50",
    ],
    "2017": [
        "PFJet450",
        "PFJet500",
        #
        "AK8PFJet400",
        "AK8PFJet450",
        "AK8PFJet500",
        #
        "AK8PFJet360_TrimMass30",
        "AK8PFJet380_TrimMass30",
        "AK8PFJet400_TrimMass30",
        #
        "AK8PFHT750_TrimMass50",
        "AK8PFHT800_TrimMass50",
        #
        "PFHT1050",
        #
        "AK8PFJet330_PFAK8BTagCSV_p17",
    ],
    "2018": [
        "PFJet500",
        #
        "AK8PFJet500",
        #
        "AK8PFJet360_TrimMass30",
        "AK8PFJet380_TrimMass30",
        "AK8PFJet400_TrimMass30",
        "AK8PFHT750_TrimMass50",
        "AK8PFHT800_TrimMass50",
        #
        "PFHT1050",
        #
        "AK8PFJet330_TrimMass30_PFAK8BTagCSV_p17_v",
    ],
}

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
btagWPs = {
    "deepJet": {
        "2016APV": {
            "L": 0.0508,
            "M": 0.2598,
            "T": 0.6502,
        },
        "2016": {
            "L": 0.0480,
            "M": 0.2489,
            "T": 0.6377,
        },
        "2017": {
            "L": 0.0532,
            "M": 0.3040,
            "T": 0.7476,
        },
        "2018": {
            "L": 0.0490,
            "M": 0.2783,
            "T": 0.7100,
        },
    }
}
