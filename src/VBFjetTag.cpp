#include "../interface/VBFjetTag.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

using tensorflow::RewriterConfig;
namespace vbf_tagger{

VBFjetTag::VBFjetTag(const std::array<std::string, VBFjetTag::n_models>& models)
{
    tensorflow::Options default_options{};
    // Disable Grappler optimizations. from https://github.com/tensorflow/tensorflow/blob/181a9a13e629ed545ace9e154310c8706ab202e7/tensorflow/core/grappler/clusters/cluster.cc#L82
    auto rewriter_config =
        default_options._options.config.mutable_graph_options()->mutable_rewrite_options();
    rewriter_config->set_layout_optimizer(RewriterConfig::OFF);
    rewriter_config->set_disable_model_pruning(true);
    rewriter_config->set_function_optimization(RewriterConfig::OFF);
    rewriter_config->set_arithmetic_optimization(RewriterConfig::OFF);
    rewriter_config->set_loop_optimization(RewriterConfig::OFF);
    rewriter_config->set_dependency_optimization(RewriterConfig::OFF);
    rewriter_config->set_constant_folding(RewriterConfig::OFF);
    rewriter_config->set_memory_optimization(RewriterConfig::NO_MEM_OPT);
    rewriter_config->set_shape_optimization(RewriterConfig::OFF);
    rewriter_config->set_remapping(RewriterConfig::OFF);

    rewriter_config->set_common_subgraph_elimination(RewriterConfig::OFF); // added

    rewriter_config->set_pin_to_host_optimization(RewriterConfig::OFF);
    rewriter_config->mutable_auto_parallel()->set_enable(false);
    rewriter_config->clear_optimizers();

    //          rewrite_cfg.common_subgraph_elimination() != RewriterConfig::OFF ||

    for(size_t n = 0; n < VBFjetTag::n_models; ++n) {
        nn_descs.at(n).graph.reset(tensorflow::loadMetaGraphDef(models.at(n)));
        nn_descs.at(n).session = tensorflow::createSession(nn_descs.at(n).graph.get(), models.at(n), default_options);
        nn_descs.at(n).input_layer = "serving_default_input_1:0";
        nn_descs.at(n).output_layer = "StatefulPartitionedCall:0";
    }
}

std::vector<float> VBFjetTag::GetScore(const std::vector<float>& jet_pt, const std::vector<float>& jet_eta,
                                     const std::vector<float>& rel_jet_M_pt, const std::vector<float>& rel_jet_E_pt,
                                     const std::vector<float>& jet_htt_deta, const std::vector<float>& jet_btagScore,
                                     const std::vector<float>& jet_htt_dphi, const std::vector<bool>& jet_isbjet,
                                     const std::vector<float>& jet_centrality, const std::vector<float>& jet_isolation,
                                     int sample_year, int channelId, float htt_pt, float htt_eta, float htt_met_dphi, 
                                     float rel_met_pt_htt_pt, float htt_scalar_pt, unsigned long long parity)
                                     
{
    tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape{1, VBFjetTag::max_n_jets, VBFjetTag::n_variables});
    x.flat<float>().setZero();

    const size_t n_jets_evt = std::min(jet_pt.size(), VBFjetTag::max_n_jets);
    for (size_t jet_index = 0; jet_index < n_jets_evt; ++jet_index) {
        const int n_jet = static_cast<int>(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_valid) = 1;
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_pt) = jet_pt.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_eta) = jet_eta.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::rel_jet_M_pt) = rel_jet_M_pt.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::rel_jet_E_pt) = rel_jet_E_pt.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_htt_deta) = jet_htt_deta.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_btagScore) = jet_btagScore.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_htt_dphi) = jet_htt_dphi.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_isbjet) = jet_isbjet.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_centrality) = jet_centrality.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::jet_isolation) = jet_isolation.at(jet_index);
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::sample_year) = sample_year;
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::channelId) = channelId;
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::htt_pt) = htt_pt;
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::htt_eta) = htt_eta;
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::htt_met_dphi) = htt_met_dphi;
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::rel_met_pt_htt_pt) = rel_met_pt_htt_pt;
        x.tensor<float, 3>()(0, n_jet, InputVars::vars::htt_scalar_pt) = htt_scalar_pt;
    }
    std::vector<tensorflow::Tensor> pred_vec;
    parity = parity % n_models;
    tensorflow::run(nn_descs.at(parity).session, { { nn_descs.at(parity).input_layer, x } },
                    { nn_descs.at(parity).output_layer }, &pred_vec);

    std::vector<float> scores(jet_pt.size(), 0);
    for (size_t jet_index = 0; jet_index < n_jets_evt; ++jet_index) {
        const int n_jet = static_cast<int>(jet_index);
        scores.at(jet_index) = pred_vec.at(0).matrix<float>()(0, n_jet);
    }

    return scores;
}

VBFjetTag::~VBFjetTag()
{
    for(size_t n = 0; n < VBFjetTag::n_models; ++n)
        tensorflow::closeSession(nn_descs.at(n).session);
}

}// namespace vbf_tagger
