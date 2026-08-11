#include <map>
#include "../interface/VBFjetTag.h"

int main(int argc, char *argv[])
{
    std::array <std::string, 2> models;
    for(size_t n = 0; n < 2; ++n) {
        std::ostringstream ss_model;
        ss_model << "HHTools/VBFjTag/models/VBFjTag_" << "_par_" << n;
        models.at(n) = ss_model.str();
    }
    vbf_tagger::VBFjetTag test(models);

    struct inputVariables {
        std::vector<float> jet_pt;
        std::vector<float> jet_eta;
        std::vector<float> rel_jet_M_pt;
        std::vector<float> rel_jet_E_pt;
        std::vector<float> jet_centrality;
        std::vector<float> jet_isolation;
        std::vector<float> jet_btagScore;
        std::vector<bool> isbjet;
        int sample_year;
        int channelId;

        inputVariables() {}
         inputVariables(std::vector<float> _jet_pt, std::vector<float> _jet_eta, std::vector<float> _rel_jet_M_pt,
         std::vector<float> _rel_jet_E_pt, std::vector<float> _jet_centrality, std::vector<float> _jet_isolation,
         std::vector<float> _jet_btagScore, const std::vector<bool>& _isbjet, int _sample_year, int _channelId)
            : jet_pt(_jet_pt), jet_eta(_jet_eta), rel_jet_M_pt(_rel_jet_M_pt), rel_jet_E_pt(_rel_jet_E_pt),
              jet_centrality(_jet_centrality), jet_isolation(_jet_isolation), jet_btagScore(_jet_btagScore), isbjet(_isbjet),
              sample_year(_sample_year), channelId(_channelId){}
    };

    std::vector<inputVariables> input_variables = {
        inputVariables({39.625, 20.671875,  56.84375}, // example values from event number with parity odd
                       {-4.058593, -2.851562,  -0.773681},
                       {0.17015, 0.07431, 0.13249},
                       {28.9555, 8.68660, 1.32118},
                       {0, 0.7348940, 0},
                       {2.4465339, 2.2977974, 2.2977974},
                       {-1, -1, 0.0022845},
                       {0, 0, 0},
                       2024, 1),
        inputVariables({58.28125, 80.75}, // example values from event number with parity even
                       {-4.65625, 1.4614257},
                       {0.07131, 0.08973},
                       {52.6251, 8.68674},
                       {0, 0},
                       {6.3263831, 6.3263831},
                       {-1, 0.0006128},
                       {0, 0},
                       2024, 1)

    };

    for (int i = 0; i < static_cast<int>(input_variables.size()); ++i){
        auto scores = test.VBFjetTag::GetScore(input_variables.at(i).jet_pt, input_variables.at(i).jet_eta,
                                             input_variables.at(i).rel_jet_M_pt, input_variables.at(i).rel_jet_E_pt,
                                             input_variables.at(i).jet_centrality, input_variables.at(i).jet_isolation,
                                             input_variables.at(i).jet_btagScore, input_variables.at(i).isbjet,
                                             input_variables.at(i).sample_year,
                                             input_variables.at(i).channelId, i);

         std::cout << "*************** parity " << i << " ***************" << "\n";
         for (size_t n_jet = 0; n_jet < scores.size(); ++n_jet)
            std::cout << "jet #" << n_jet << " score = " << scores.at(n_jet) << "\n";
    }

}
