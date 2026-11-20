#pragma once

#include "ggml-backend.h"

#include <memory>
#include <string>

struct llama_model;
class llm_graph_result;

struct llama_moe_plastic_params {
    int32_t     n_keep;       // experts that the router can select in each MoE layer, 0 = disabled
    int32_t     n_skip;       // the first n_skip MoE layers keep all experts
    float       recruit;      // an expert out of the set joins it when the unmasked router gives it this weight
    float       half_life;    // tokens: the strength of an expert halves in half_life tokens without use
    bool        uniform;      // each layer keeps n_keep experts, instead of sharing n_keep*n_layer among the layers
    float       novelty;      // a prompt whose live coverage is under this fraction of the usual coverage is a new subject
    std::string memory;       // file with the strengths of the last sessions, read at start and written at end
    std::string vital;        // "il:e,il:e,..." experts that are never cut (super experts)
};

// plastic MoE: the router of each MoE layer selects only from a small live set of experts
// the set follows the experts that the unmasked router wants: use makes them strong, time makes them weak
// experts out of the set are never read, so a model larger than the RAM runs without disk reads after warm-up
class llama_moe_plastic {
public:
    llama_moe_plastic(const llama_model & model, const llama_moe_plastic_params & params);
    ~llama_moe_plastic();

    // F32 [n_expert] mask added to the router logits of layer il: 0 for the live experts, -INF for the others
    // nullptr if the layer keeps all its experts
    ggml_tensor * get_mask(int32_t il) const;

    // read the unmasked router choices of the last ubatch (see llm_graph_result::t_moe_pref_ids) and update the sets
    void observe(const llm_graph_result & res, ggml_backend_sched_t sched);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_plastic_ptr = std::unique_ptr<llama_moe_plastic>;
