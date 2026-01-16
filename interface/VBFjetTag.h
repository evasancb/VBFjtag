#include <vector>
#include <string>
#include <memory>
#include <array>

namespace tensorflow {
    class Session;
    class MetaGraphDef;
}

namespace vbf_tagger{

namespace InputVars{
    enum vars { jet_valid = 0, jet_pt = 1, jet_eta = 2, rel_jet_M_pt = 3, rel_jet_E_pt = 4, jet_htt_deta = 5,
                jet_btagScore = 6, jet_htt_dphi = 8, jet_isbjet = 9, jet_centrality = 10, jet_isolation = 11, 
                sample_year= 12, channelId = 13, htt_pt = 14, htt_eta = 15,
                htt_met_dphi = 16, rel_met_pt_htt_pt = 17, htt_scalar_pt = 18
    };
 }

class VBFjetTag {
public:
    static constexpr size_t n_models = 2;
    static constexpr size_t max_n_jets = 10;
    static constexpr size_t n_variables = 18;

    VBFjetTag(const std::array <std::string, n_models>& models);
    ~VBFjetTag();



    struct NNDescriptor {
        std::unique_ptr<tensorflow::MetaGraphDef> graph;
        tensorflow::Session* session;
        std::string input_layer;
        std::string output_layer;
    };

    std::vector<float> GetScore(const std::vector<float>& jet_pt, const std::vector<float>& jet_eta,
                                const std::vector<float>& rel_jet_M_pt, const std::vector<float>& rel_jet_E_pt,
                                const std::vector<float>& jet_htt_deta, const std::vector<float>& jet_btagScore,
                                const std::vector<float>& jet_htt_dphi, const std::vector<bool>& jet_isbjet, 
                                const std::vector<float>& jet_centrality, const std::vector<float>& jet_isolation, 
                                int sample_year, int channelId, float htt_pt, float htt_eta, 
                                float htt_met_dphi, float rel_met_pt_htt_pt,
                                float htt_scalar_pt, unsigned long long parity);

private:
    std::array<NNDescriptor, n_models> nn_descs;
};
}// namespace vbf_tagger
