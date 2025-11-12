# VBFjet-tag   
Interface providing feature computation for CMS RunIII VBFjet-tag NN to the identification of the reconstructed vbf-jets for the final state of hh->bbtautau

# Installation

```sh
cmsrel CMSSW_12_4_10 # or above
cd CMSSW_12_4_10/src
cmsenv
git clone git@github.com:elviramartinv/VBFjTag.git HHTools/VBFjTag
scram b
```

# Testing
1. vbfjtagTest
Will produce the resulting scores for the input values for two events

# Features

- The reconstructed jets are selected using bbtautau baseline selection for the b jet candidates, describe at the twiki:
https://twiki.cern.ch/twiki/bin/viewauth/CMS/DoubleHiggsToBBTauTauWorkingLegacyRun2#Baseline_selection.

- The maximum number of jets considered considered is 10 for event.

- The jets must be ordered using the eta of the jets

- The taus are selected as specified for the baseline at the working twiki.

- The range used to calculate delta phi should be between [-pi, pi]

- The features used for the training are:

1. jet_pt: pT of each of the jets.
1. jet_eta: eta of each of the jets.
1. rel_jet_M_pt: Relative mass of the b-jet candidate: jet M / jet pT.
1. rel_jet_E_pt Relative energy of the b-jet candidate: jet E / jet pT.
1. jet_htt_deta: Eta between the b-jet and the visible 4-momentum of the HTT.
1. jet_btagScore: The score of the b-jet candidate given by the b-tagger : ParticleNet
1. jet_htt_dphi: Phi between the b-jet and the visible 4-momentum of the HTT.
1. jet_isbjet: 1 for jets tagged as b-jet; 0 for non tagged b-jets
1. jet_centrality: 1 - (2 * |eta_j - eta_c|) / (eta_max - eta_min), con eta_c = (eta_max + eta_min)/2
1. jet_isolation: min DeltaR(jj)
1. era_id: 0 - 2022preEE, 1 - 2022postEE, 2 - 2023preBPix, 3 - 2023postBPix, 4 - 2024
1. channelId: 0 - MuTau, 1 - ETau, 2 - TauTau, 3 - MuMu, 4 - EE, 5 - EMu
1. htt_pt: pT of visible 4-momentum of the HTT candidate.
1. htt_eta: eta of HTT visible 4-momentum of the HTT candidate.
1. htt_met_dphi: Phi between the visible 4-momentum of the HTT candidate and the MET.
1. rel_met_pt_htt_pt: Relative MET:  MET / pT of the visible 4-momentum of the HTT candidate.
1. htt_scalar_pt: Sum of the pT of the 2 selected taus.

# Output

The output of the VBFjet-tag interfase will be a vector with the score for each of the jets.
The two jets with the highest score will be more likely to be vbf jets
