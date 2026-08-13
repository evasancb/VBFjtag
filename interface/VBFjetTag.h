#include <vector>
#include <string>
#include <memory>
#include <array>

// Forward-declare to avoid JIT of tensorflow header
namespace tensorflow {
    class Session;
    class MetaGraphDef;
    class GraphDef;
}

namespace vbf_tagger{

namespace InputVars{
    enum vars { jet_valid = 0, jet_pt = 1, jet_eta = 2, rel_jet_M_pt = 3, rel_jet_E_pt = 4,
                jet_centrality = 5, jet_isolation = 6, jet_btagScore = 7, jet_isbjet = 8, sample_year= 9, channelId = 10
    };
 }

class VBFjetTag {
public:
    static constexpr size_t n_models = 2;
    static constexpr size_t max_n_jets = 10;
    static constexpr size_t n_variables = 11;

    VBFjetTag(const std::array <std::string, n_models>& models, bool useMetaGraph=true);
    ~VBFjetTag();



    struct NNDescriptor {
        // We use either metaGraph (SavedModel format) OR graph (frozen graph format, faster loading)
        std::unique_ptr<tensorflow::MetaGraphDef> metaGraph;
        std::unique_ptr<tensorflow::GraphDef> graph;
        tensorflow::Session* session;
        std::string input_layer;
        std::string output_layer;
        int64_t callableHandle; // To work with tensorflow::Session::RunCallable
    };

    std::vector<float> GetScore(const std::vector<float>& jet_pt, const std::vector<float>& jet_eta,
                                const std::vector<float>& rel_jet_M_pt, const std::vector<float>& rel_jet_E_pt,
                                const std::vector<float>& jet_centrality, const std::vector<float>& jet_isolation,
                                const std::vector<float>& jet_btagScore, const std::vector<bool>& jet_isbjet,
                                int sample_year, int channelId, unsigned long long parity);

private:
    std::array<NNDescriptor, n_models> nn_descs;
    bool useMetaGraph_;
};
}// namespace vbf_tagger
