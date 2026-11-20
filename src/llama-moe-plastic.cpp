#include "llama-moe-plastic.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace {

struct plastic_layer {
    bool active = false;                 // the router of this layer is masked

    std::vector<float>   strength;       // [n_expert] decayed sum of the router weights, at t_last
    std::vector<int64_t> t_last;         // [n_expert] clock of the last update
    std::vector<float>   memory;         // [n_expert] strengths of the last sessions, 0..1
    std::vector<uint8_t> live;           // [n_expert] the router can select the expert
    std::vector<uint8_t> vital;          // [n_expert] never cut

    std::vector<float> mask_data;        // [n_expert] 0 or -INF
    ggml_tensor *      mask  = nullptr;
    bool               dirty = false;

    ggml_tensor * exps[4] = {};          // host experts, for the prefetch
};

} // namespace

struct llama_moe_plastic::impl {
    llama_moe_plastic_params params;

    std::string model_name;
    int32_t     n_expert = 0;

    std::vector<plastic_layer> layers;

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr>                   bufs;

    int64_t clock        = 0;   // tokens seen
    int64_t last_rebuild = 0;

    // stats
    double  w_wanted   = 0.0;   // router weight that the unmasked router gave
    double  w_live     = 0.0;   // ... of it, to live experts
    size_t  n_recruit  = 0;
    size_t  n_create   = 0;
    size_t  n_cut      = 0;

    // prefetch: reads the host data of the experts that join a live set, so the next token does not wait for the disk
    std::thread                         worker;
    std::mutex                          mtx;
    std::condition_variable             cv;
    std::deque<std::pair<int32_t, int32_t>> queue;
    bool                                stop = false;
    std::atomic<size_t>                 n_prefetched{0};

    impl(const llama_model & model, const llama_moe_plastic_params & p) : params(p) {
        const auto & hp = model.hparams;
        n_expert   = (int32_t) hp.n_expert;
        model_name = model.name;

        layers.resize(model.layers.size());

        const bool no_alloc = model.no_alloc();

        int32_t n_moe = 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & ml = model.layers[il];
            if (ml.ffn_gate_inp == nullptr || ml.ffn_gate_inp->ne[1] != n_expert) {
                continue;
            }
            if (n_moe++ < params.n_skip) {
                continue;
            }
            plastic_layer & l = layers[il];
            l.active = true;
            l.strength.assign(n_expert, 0.0f);
            l.t_last.assign(n_expert, 0);
            l.memory.assign(n_expert, 0.0f);
            l.live.assign(n_expert, 1);
            l.vital.assign(n_expert, 0);
            l.mask_data.assign(n_expert, 0.0f);

            l.exps[0] = ml.ffn_up_exps;
            l.exps[1] = ml.ffn_gate_exps;
            l.exps[2] = ml.ffn_down_exps;
            l.exps[3] = ml.ffn_gate_up_exps;

            // the mask lives next to the router weights, so adding it needs no copy
            ggml_backend_buffer_type_t buft = ml.ffn_gate_inp->buffer ? ggml_backend_buffer_get_type(ml.ffn_gate_inp->buffer) : ggml_backend_cpu_buffer_type();
            auto it = ctxs.find(buft);
            if (it == ctxs.end()) {
                ggml_init_params ip = {
                    /*.mem_size   =*/ model.layers.size()*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                it = ctxs.emplace(buft, ggml_context_ptr(ggml_init(ip))).first;
            }
            l.mask = ggml_new_tensor_1d(it->second.get(), GGML_TYPE_F32, n_expert);
            ggml_format_name(l.mask, "moe_plastic.mask-%zu", il);
        }

        for (auto & [buft, ctx] : ctxs) {
            bufs.push_back(llama_memory_bufs::alloc_buf(ctx.get(), buft, no_alloc, "MoE plastic masks"));
            if (!no_alloc) {
                ggml_backend_buffer_clear(bufs.back().get(), 0); // all experts live until the first choices are seen
            }
        }

        parse_vital(params.vital);
        if (!params.memory.empty()) {
            load(params.memory);
        }

        int32_t n_active = 0;
        for (auto & l : layers) {
            n_active += l.active;
        }
        LLAMA_LOG_INFO("%s: plastic MoE: %d of %d experts live per layer (%s), %d layers (first %d MoE layers whole), recruit at %.2f, half-life %.0f tokens\n",
                __func__, params.n_keep, n_expert, params.uniform ? "same in each layer" : "on average, shared among the layers", n_active, params.n_skip, params.recruit, params.half_life);

        if (!no_alloc) {
            worker = std::thread([this] { prefetch_loop(); });
        }
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        if (clock > 0) {
            LLAMA_LOG_INFO("%s: plastic MoE: %lld tokens, %.1f%% of the wanted router weight on live experts, %zu recruited, %zu connections created, %zu cut, %zu experts prefetched\n",
                    __func__, (long long) clock, w_wanted > 0 ? 100.0*w_live/w_wanted : 0.0, n_recruit, n_create, n_cut, n_prefetched.load());
            if (!params.memory.empty()) {
                save(params.memory);
            }
        }
    }

    void parse_vital(const std::string & s) {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int il = -1;
            int e  = -1;
            if (sscanf(item.c_str(), "%d:%d", &il, &e) == 2 && il >= 0 && il < (int) layers.size() && e >= 0 && e < n_expert && layers[il].active) {
                layers[il].vital[e] = 1;
            }
        }
    }

    float decayed(const plastic_layer & l, int32_t e) const {
        const double dt = (double) (clock - l.t_last[e]);
        return l.strength[e]*(float) std::exp2(-dt/params.half_life);
    }

    // score used to choose the live set: the strength now, the memory of the last sessions, the vital experts first
    float score(const plastic_layer & l, int32_t e, float s_max) const {
        if (l.vital[e]) {
            return 1e30f;
        }
        const float s = s_max > 0.0f ? decayed(l, e)/s_max : 0.0f;
        return s + 0.25f*l.memory[e];
    }

    // choose the live set of a layer: the n_keep best scores, with hysteresis so that close experts do not swap
    void rebuild(int32_t il) {
        plastic_layer & l = layers[il];

        float s_max = 0.0f;
        for (int32_t e = 0; e < n_expert; ++e) {
            s_max = std::max(s_max, decayed(l, e));
        }
        if (s_max == 0.0f && *std::max_element(l.memory.begin(), l.memory.end()) == 0.0f) {
            return; // nothing learned yet: keep all experts live
        }

        std::vector<int32_t> order(n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            order[e] = e;
        }
        std::vector<float> sc(n_expert);
        for (int32_t e = 0; e < n_expert; ++e) {
            sc[e] = score(l, e, s_max) + (l.live[e] ? 0.05f : 0.0f); // hysteresis
        }
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return sc[a] > sc[b]; });

        std::vector<uint8_t> live(n_expert, 0);
        const int32_t n_keep = std::min(params.n_keep, n_expert);
        for (int32_t i = 0; i < n_expert; ++i) {
            if (i < n_keep || l.vital[order[i]]) {
                live[order[i]] = 1;
            }
        }
        apply(il, live);
    }

    // choose the live sets of all layers together: the total budget goes to the experts with the largest share of the
    // routing of their layer, so layers with a spread routing get more experts and peaked layers get fewer
    void rebuild_all() {
        struct cand { float share; int32_t il; int32_t e; };
        std::vector<cand> cands;
        std::vector<std::vector<uint8_t>> live(layers.size());
        std::vector<int32_t> n_live(layers.size(), 0);

        const int32_t n_min = std::max(4, params.n_keep/4);
        int64_t budget = 0;

        for (size_t il = 0; il < layers.size(); ++il) {
            plastic_layer & l = layers[il];
            if (!l.active) {
                continue;
            }
            std::vector<float> d(n_expert);
            double d_sum = 0.0;
            double m_sum = 0.0;
            for (int32_t e = 0; e < n_expert; ++e) {
                d[e]   = decayed(l, e);
                d_sum += d[e];
                m_sum += l.memory[e];
            }
            if (d_sum == 0.0 && m_sum == 0.0) {
                continue; // nothing learned yet: keep all experts live
            }
            budget += std::min(params.n_keep, n_expert);
            live[il].assign(n_expert, 0);

            std::vector<cand> lc(n_expert);
            for (int32_t e = 0; e < n_expert; ++e) {
                float share = 0.0f;
                if (d_sum > 0.0) {
                    share += (m_sum > 0.0 ? 0.8f : 1.0f)*(float) (d[e]/d_sum);
                }
                if (m_sum > 0.0) {
                    share += (d_sum > 0.0 ? 0.2f : 1.0f)*(float) (l.memory[e]/m_sum);
                }
                if (l.live[e]) {
                    share *= 1.1f; // hysteresis
                }
                lc[e] = { share, (int32_t) il, e };
            }
            std::sort(lc.begin(), lc.end(), [](const cand & a, const cand & b) { return a.share > b.share; });
            for (int32_t i = 0; i < n_expert; ++i) {
                const int32_t e = lc[i].e;
                if (i < n_min || l.vital[e]) {
                    live[il][e] = 1;
                    n_live[il]++;
                    budget--;
                } else {
                    cands.push_back(lc[i]);
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.share > b.share; });
        for (const cand & c : cands) {
            if (budget <= 0) {
                break;
            }
            live[c.il][c.e] = 1;
            n_live[c.il]++;
            budget--;
        }
        for (size_t il = 0; il < layers.size(); ++il) {
            if (!live[il].empty()) {
                apply((int32_t) il, live[il]);
            }
        }
    }

    void rebuild_sets() {
        if (params.uniform) {
            for (size_t il = 0; il < layers.size(); ++il) {
                if (layers[il].active) {
                    rebuild((int32_t) il);
                }
            }
        } else {
            rebuild_all();
        }
    }

    void apply(int32_t il, const std::vector<uint8_t> & live) {
        plastic_layer & l = layers[il];
        for (int32_t e = 0; e < n_expert; ++e) {
            if (live[e] == l.live[e]) {
                continue;
            }
            l.dirty = true;
            if (live[e]) {
                n_create++;
                enqueue(il, e);
            } else {
                n_cut++;
            }
        }
        l.live = live;
        if (l.dirty) {
            for (int32_t e = 0; e < n_expert; ++e) {
                l.mask_data[e] = l.live[e] ? 0.0f : -INFINITY;
            }
        }
    }

    // a strong unmasked choice out of the live set: it joins the set now, and the weakest live expert leaves
    void recruit(int32_t il, int32_t e) {
        plastic_layer & l = layers[il];

        float   s_max  = 0.0f;
        for (int32_t i = 0; i < n_expert; ++i) {
            s_max = std::max(s_max, decayed(l, i));
        }
        int32_t weakest = -1;
        float   w_score = 0.0f;
        for (int32_t i = 0; i < n_expert; ++i) {
            if (!l.live[i] || l.vital[i]) {
                continue;
            }
            const float s = score(l, i, s_max);
            if (weakest < 0 || s < w_score) {
                weakest = i;
                w_score = s;
            }
        }
        std::vector<uint8_t> live = l.live;
        live[e] = 1;
        if (weakest >= 0) {
            live[weakest] = 0;
        }
        n_recruit++;
        apply(il, live);
    }

    void observe(const llm_graph_result & res, ggml_backend_sched_t sched) {
        const auto & ids_t = res.t_moe_pref_ids;
        const auto & w_t   = res.t_moe_pref_w;

        bool any = false;
        for (size_t il = 0; il < layers.size() && il < ids_t.size(); ++il) {
            any = any || (layers[il].active && ids_t[il] != nullptr);
        }
        if (!any) {
            return;
        }
        ggml_backend_sched_synchronize(sched);

        std::vector<int32_t> ids;
        std::vector<float>   ws;
        int64_t n_tokens = 0;

        for (size_t il = 0; il < layers.size() && il < ids_t.size(); ++il) {
            plastic_layer & l = layers[il];
            if (!l.active || ids_t[il] == nullptr || w_t[il] == nullptr) {
                continue;
            }
            const ggml_tensor * ti = ids_t[il];
            const ggml_tensor * tw = w_t[il];
            const int64_t n_used = ti->ne[0];
            n_tokens = ti->ne[1];

            ids.resize(ggml_nelements(ti));
            ws.resize(ggml_nelements(tw));
            ggml_backend_tensor_get(ti, ids.data(), 0, ggml_nbytes(ti));
            ggml_backend_tensor_get(tw, ws.data(),  0, ggml_nbytes(tw));

            for (int64_t t = 0; t < n_tokens; ++t) {
                const int64_t now = clock + t;
                for (int64_t k = 0; k < n_used; ++k) {
                    const int32_t e = ids[t*n_used + k];
                    const float   w = ws[t*n_used + k];
                    if (e < 0 || e >= n_expert) {
                        continue;
                    }
                    const double dt = (double) (now - l.t_last[e]);
                    l.strength[e] = l.strength[e]*(float) std::exp2(-dt/params.half_life) + w;
                    l.t_last[e]   = now;

                    w_wanted += w;
                    if (l.live[e]) {
                        w_live += w;
                    } else if (w >= params.recruit && n_tokens == 1) {
                        recruit((int32_t) il, e);
                    }
                }
            }
        }
        clock += n_tokens;

        // the live sets follow the strengths: after a prompt, and every 16 generated tokens
        if (n_tokens > 1 || clock - last_rebuild >= 16) {
            rebuild_sets();
            last_rebuild = clock;
        }

        for (auto & l : layers) {
            if (l.active && l.dirty && l.mask->buffer) {
                ggml_backend_tensor_set(l.mask, l.mask_data.data(), 0, ggml_nbytes(l.mask));
                l.dirty = false;
            }
        }
    }

    void enqueue(int32_t il, int32_t e) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.emplace_back(il, e);
        }
        cv.notify_one();
    }

    void prefetch_loop() {
        for (;;) {
            std::pair<int32_t, int32_t> job;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return stop || !queue.empty(); });
                if (stop) {
                    return;
                }
                job = queue.front();
                queue.pop_front();
            }
            const plastic_layer & l = layers[job.first];
            bool done = false;
            for (const ggml_tensor * t : l.exps) {
                if (t == nullptr || t->data == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
                    continue;
                }
                const volatile uint8_t * p = (const uint8_t *) t->data + job.second*t->nb[2];
                uint8_t sum = 0;
                for (size_t i = 0; i < t->nb[2]; i += 4096) {
                    sum += p[i];
                }
                (void) sum;
                done = true;
            }
            n_prefetched += done;
        }
    }

    void save(const std::string & path) const {
        std::ofstream f(path);
        if (!f) {
            LLAMA_LOG_WARN("%s: cannot write the plastic memory to %s\n", __func__, path.c_str());
            return;
        }
        f << "llama-moe-plastic 1 " << model_name << "\n" << layers.size() << " " << n_expert << "\n";
        for (size_t il = 0; il < layers.size(); ++il) {
            const plastic_layer & l = layers[il];
            if (!l.active) {
                continue;
            }
            float s_max = 0.0f;
            for (int32_t e = 0; e < n_expert; ++e) {
                s_max = std::max(s_max, decayed(l, e));
            }
            f << "L " << il;
            for (int32_t e = 0; e < n_expert; ++e) {
                const float s = s_max > 0.0f ? decayed(l, e)/s_max : 0.0f;
                f << " " << 0.7f*l.memory[e] + 0.3f*s;
            }
            f << "\n";
            for (int32_t e = 0; e < n_expert; ++e) {
                if (l.vital[e]) {
                    f << "V " << il << " " << e << "\n";
                }
            }
        }
        LLAMA_LOG_INFO("%s: plastic memory saved to %s\n", __func__, path.c_str());
    }

    void load(const std::string & path) {
        std::ifstream f(path);
        if (!f) {
            LLAMA_LOG_INFO("%s: no plastic memory in %s yet, starting from scratch\n", __func__, path.c_str());
            return;
        }
        std::string line;
        std::getline(f, line);
        if (line.rfind("llama-moe-plastic 1 ", 0) != 0 || line.substr(20) != model_name) {
            LLAMA_LOG_WARN("%s: %s is the memory of another model, ignored\n", __func__, path.c_str());
            return;
        }
        size_t n_layer_f = 0;
        int    n_expert_f = 0;
        f >> n_layer_f >> n_expert_f;
        if (n_layer_f != layers.size() || n_expert_f != n_expert) {
            LLAMA_LOG_WARN("%s: %s does not match the model, ignored\n", __func__, path.c_str());
            return;
        }
        std::string tag;
        while (f >> tag) {
            size_t il = 0;
            f >> il;
            if (il >= layers.size()) {
                break;
            }
            plastic_layer & l = layers[il];
            if (tag == "L") {
                for (int32_t e = 0; e < n_expert; ++e) {
                    float v = 0.0f;
                    f >> v;
                    if (l.active) {
                        l.memory[e] = v;
                    }
                }
            } else if (tag == "V") {
                int32_t e = 0;
                f >> e;
                if (l.active && e >= 0 && e < n_expert) {
                    l.vital[e] = 1;
                }
            }
        }
        rebuild_sets();
        for (auto & l : layers) {
            if (l.active && l.dirty && l.mask->buffer && ggml_backend_buffer_get_size(l.mask->buffer) > 0) {
                ggml_backend_tensor_set(l.mask, l.mask_data.data(), 0, ggml_nbytes(l.mask));
                l.dirty = false;
            }
        }
        n_create = 0;
        LLAMA_LOG_INFO("%s: plastic memory loaded from %s\n", __func__, path.c_str());
    }
};

llama_moe_plastic::llama_moe_plastic(const llama_model & model, const llama_moe_plastic_params & params) :
    pimpl(new impl(model, params)) {
}

llama_moe_plastic::~llama_moe_plastic() = default;

ggml_tensor * llama_moe_plastic::get_mask(int32_t il) const {
    if (il < 0 || il >= (int32_t) pimpl->layers.size() || !pimpl->layers[il].active) {
        return nullptr;
    }
    return pimpl->layers[il].mask;
}

void llama_moe_plastic::observe(const llm_graph_result & res, ggml_backend_sched_t sched) {
    pimpl->observe(res, sched);
}
