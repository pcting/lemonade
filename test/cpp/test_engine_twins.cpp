// Proves engine-twin synthesis: a checkpoint that an engine's catalog can serve
// gains a twin entry on that engine's recipe for models registered under other
// recipes, additively and idempotently, and only for servable repositories.

#include "lemon/engine_twins.h"
#include "lemon/runtime_config.h"

#include "lemon/backends/gufo/gufo_compat.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using lemon::GgufTensorInfo;
using lemon::GgufTensorTypes;
using lemon::ModelInfo;
using lemon::ModelType;
using lemon::RuntimeConfig;
using lemon::backends::gufo::GufoCompat;
using namespace lemon::engine_twins;

namespace {

int failures = 0;

void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool has_label(const ModelInfo& info, const std::string& label) {
    return std::find(info.labels.begin(), info.labels.end(), label) != info.labels.end();
}

ModelInfo make_model(const std::string& id,
                     const std::string& recipe,
                     const std::string& main_checkpoint,
                     std::vector<std::string> labels,
                     std::map<std::string, std::string> extra_checkpoints = {}) {
    ModelInfo info;
    info.model_name = id;
    info.recipe = recipe;
    info.checkpoints["main"] = main_checkpoint;
    for (auto& [role, cp] : extra_checkpoints) {
        info.checkpoints[role] = cp;
    }
    info.labels = std::move(labels);
    return info;
}

}  // namespace

int main() {
    // Helpers
    check("checkpoint repository split",
          checkpoint_repository("unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL") ==
              "unsloth/Qwen3.8-27B-GGUF");
    check("checkpoint repository without variant",
          checkpoint_repository("Qwen/Qwen3-ASR-1.7B") == "Qwen/Qwen3-ASR-1.7B");
    check("engine qualifier", engine_qualifier("gufo") == "Gufo");
    check("engine qualifier with separators", engine_qualifier("sd-cpp") == "SdCpp");

    // Servable repositories come from the engine's built-in catalog entries only.
    std::map<std::string, ModelInfo> models;
    models.emplace("Qwen3.8-27B-GGUF",
                   make_model("Qwen3.8-27B-GGUF", "gufo",
                              "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
                              {"chat", "reasoning"}));
    models.emplace("Qwen3-ASR-1.7B",
                   make_model("Qwen3-ASR-1.7B", "gufo", "Qwen/Qwen3-ASR-1.7B",
                              {"transcription"}));
    models.emplace("user.RegisteredGufo",
                   make_model("user.RegisteredGufo", "gufo",
                              "someone-else/NotCatalogued:Q4_K_M", {"chat"}));

    const ServableRepos repos = servable_repositories(models, "gufo");
    check("catalog repository is servable",
          repos.count("unsloth/Qwen3.8-27B-GGUF") == 1);
    check("variant-pinned catalog shape recorded",
          repos.at("unsloth/Qwen3.8-27B-GGUF").variant_pinned);
    check("catalog variant recorded",
          repos.at("unsloth/Qwen3.8-27B-GGUF").variant == "UD-Q4_K_XL");
    check("directory catalog shape recorded",
          !repos.at("Qwen/Qwen3-ASR-1.7B").variant_pinned &&
              repos.at("Qwen/Qwen3-ASR-1.7B").mode_label == "transcription");
    check("user registrations never widen servability",
          repos.count("someone-else/NotCatalogued") == 0);

    // A registered chat model in a servable repository gains a loadable twin
    // that keeps every source label (capability badges must not regress).
    models.emplace("Qwen3.8-27B-GGUF-UD-Q8_K_XL",
                   make_model("Qwen3.8-27B-GGUF-UD-Q8_K_XL", "llamacpp",
                              "unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL",
                              {"chat", "vision", "mtp", "custom"},
                              {{"mmproj", "unsloth/Qwen3.8-27B-GGUF:mmproj-BF16.gguf"}}));
    const auto added = add_twins(models, "gufo", repos);
    check("one twin generated",
          added.size() == 1 && added[0] == "extra.Qwen3.8-27B-GGUF-UD-Q8_K_XL-Gufo");

    const ModelInfo& twin = models.at("extra.Qwen3.8-27B-GGUF-UD-Q8_K_XL-Gufo");
    const ModelInfo& source = models.at("Qwen3.8-27B-GGUF-UD-Q8_K_XL");
    check("twin runs the engine recipe", twin.recipe == "gufo");
    check("twin reuses the source checkpoints", twin.checkpoints == source.checkpoints);
    check("twin preserves source labels and adds the engine label",
          has_label(twin, "chat") && has_label(twin, "vision") &&
              has_label(twin, "mtp") && has_label(twin, "custom") &&
              has_label(twin, "gufo"));
    check("twin has no duplicate labels",
          std::set<std::string>(twin.labels.begin(), twin.labels.end()).size() ==
              twin.labels.size());
    check("twin is suggested and locally sourced",
          twin.suggested && twin.source == TWIN_SOURCE);
    check("twin routes as chat", twin.type == ModelType::LLM);
    check("twin defers download state", !twin.downloaded && twin.resolved_paths.empty());
    check("source entry untouched",
          source.recipe == "llamacpp" && !has_label(source, "gufo") &&
              source.labels.size() == 4);

    // Only servable repositories and only matching checkpoint shapes twin.
    models.emplace("LFM2-1.2B-GGUF",
                   make_model("LFM2-1.2B-GGUF", "llamacpp",
                              "unsloth/LFM2-1.2B-GGUF:Q4_K_M", {"chat"}));
    models.emplace("WholeRepo27B",
                   make_model("WholeRepo27B", "llamacpp", "unsloth/Qwen3.8-27B-GGUF",
                              {"chat"}));
    add_twins(models, "gufo", repos);
    check("no twin for a non-servable repository", models.count("extra.LFM2-1.2B-GGUF-Gufo") == 0);
    check("no twin for a whole-repo shape in a variant-pinned family",
          models.count("extra.WholeRepo27B-Gufo") == 0);

    // Modality twins inherit the catalog entry's deployment label.
    models.emplace("MyTranscriber",
                   make_model("MyTranscriber", "whispercpp", "Qwen/Qwen3-ASR-1.7B",
                              {"transcription", "custom"}));
    add_twins(models, "gufo", repos);
    check("modality twin generated", models.count("extra.MyTranscriber-Gufo") == 1);
    check("modality twin routes as transcription",
          models.at("extra.MyTranscriber-Gufo").type == ModelType::TRANSCRIPTION &&
              models.at("extra.MyTranscriber-Gufo").labels[0] == "transcription");

    // Generated ids never overwrite an existing model.
    models.emplace("extra.Collide-Gufo",
                   make_model("extra.Collide-Gufo", "llamacpp", "unsloth/Other:x",
                              {"chat"}));
    models.emplace("Collide",
                   make_model("Collide", "llamacpp",
                              "unsloth/Qwen3.8-27B-GGUF:Q6_K", {"chat"}));
    add_twins(models, "gufo", repos);
    check("collision is qualified, not overwritten",
          models.count("extra.Collide-Gufo-2") == 1);
    check("pre-existing twin id owner untouched",
          models.at("extra.Collide-Gufo").recipe == "llamacpp");

    // An engine that can read only some quantizations refuses candidates
    // through the filter; the rest of the repository still twins.
    {
        std::map<std::string, ModelInfo> filtered_models;
        filtered_models.emplace("FilterKeep",
                                make_model("FilterKeep", "llamacpp",
                                           "unsloth/Qwen3.8-27B-GGUF:Q8_0", {"chat"}));
        filtered_models.emplace("FilterDrop",
                                make_model("FilterDrop", "llamacpp",
                                           "unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL", {"chat"}));
        int consulted = 0;
        const auto filtered_added = add_twins(
            filtered_models, "gufo", repos,
            [&consulted](const ModelInfo& info, const ServableRepo&) {
                ++consulted;
                return info.model_name != "FilterDrop";
            });
        check("filter consulted once per candidate", consulted == 2);
        check("filter keeps the readable candidate",
              filtered_added.size() == 1 &&
                  filtered_models.count("extra.FilterKeep-Gufo") == 1);
        check("filter drops the unreadable candidate",
              filtered_models.count("extra.FilterDrop-Gufo") == 0);
    }

    // What the engine decides is the whole entry, so the draft travels into the
    // filter with the main checkpoint: a draft it refuses withholds the twin and the
    // sibling candidate in the same repository is unaffected.
    {
        std::map<std::string, ModelInfo> drafted;
        drafted.emplace("DraftedQ4K",
                        make_model("DraftedQ4K", "llamacpp",
                                   "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL", {"chat", "mtp"},
                                   {{"draft", "z-lab/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M"}}));
        drafted.emplace("MainOnly",
                        make_model("MainOnly", "llamacpp",
                                   "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL", {"chat"}));
        int consulted = 0;
        int shown_a_draft = 0;
        add_twins(drafted, "gufo", repos, [&](const ModelInfo& info, const ServableRepo&) {
            ++consulted;
            if (info.checkpoints.count("draft")) ++shown_a_draft;
            // The answer verdict_entry() gives this pair: Q4_K matrices are outside
            // what the DFlash2 draft's own groups accept.
            return info.model_name != "DraftedQ4K";
        });
        check("one consultation per candidate, whatever it declares", consulted == 2);
        check("the filter is handed the declared draft too", shown_a_draft == 1);
        check("a refused draft withholds the twin", drafted.count("extra.DraftedQ4K-Gufo") == 0);
        check("a sibling candidate in the same repository still twins",
              drafted.count("extra.MainOnly-Gufo") == 1);
        check("a refused candidate keeps its own entry and draft intact",
              drafted.at("DraftedQ4K").recipe == "llamacpp" &&
                  drafted.at("DraftedQ4K").checkpoints.at("draft") ==
                      "z-lab/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M");
    }

    // Permissive by absence, decided by the shipped table rather than a stand-in:
    // tensors no group of the family claims are ones the loader never looks up, so
    // the table answers servable and the twin appears. A table could otherwise hide
    // a model that loads, which is the failure mode this whole gate must not have.
    {
        std::map<std::string, ModelInfo> permissive;
        permissive.emplace("UnclaimedNames",
                           make_model("UnclaimedNames", "llamacpp",
                                      "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL", {"chat"}));
        GgufTensorTypes unclaimed;
        for (const char* name : {"blk.0.attn_qkv.bias", "blk.0._down.weight", "v.sum"}) {
            GgufTensorInfo tensor;
            tensor.name = name;
            tensor.dimensions = {4096, 4096};
            tensor.type = 10;  // Q2_K, which no group of this family accepts
            unclaimed.tensors.push_back(std::move(tensor));
        }
        const GufoCompat& manifest = GufoCompat::current();
        check("the shipped table claims none of those names",
              manifest.usable() &&
                  std::none_of(unclaimed.tensors.begin(), unclaimed.tensors.end(),
                               [&](const GgufTensorInfo& t) {
                                   return manifest.group_claiming("qwen38-dense", t.name) !=
                                          nullptr;
                               }));
        GgufTensorTypes one_claimed = unclaimed;
        GgufTensorInfo projection;
        projection.name = "blk.0.attn_q.weight";
        projection.dimensions = {4096, 4096};
        projection.type = 10;
        one_claimed.tensors.push_back(std::move(projection));
        check("the same tensors are refused once a name is claimed by a group",
              !manifest.verdict("qwen38-dense", one_claimed).accept);
        add_twins(permissive, "gufo", repos, [&](const ModelInfo&, const ServableRepo&) {
            return manifest.verdict("qwen38-dense", unclaimed).accept;
        });
        check("a checkpoint no rule speaks about gains its twin",
              permissive.count("extra.UnclaimedNames-Gufo") == 1);
    }

    // The switch: default on, explicit off, non-boolean rejected.
    RuntimeConfig defaults(nlohmann::json::object());
    check("engine twins default on", defaults.engine_twins_enabled());

    RuntimeConfig disabled(nlohmann::json{{"engine_twins", false}});
    check("engine twins can be disabled", !disabled.engine_twins_enabled());

    RuntimeConfig configurable(nlohmann::json::object());
    configurable.set(nlohmann::json{{"engine_twins", false}});
    check("engine twins switch is settable", !configurable.engine_twins_enabled());

    bool rejected = false;
    try {
        configurable.set(nlohmann::json{{"engine_twins", "yes"}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check("non-boolean switch value rejected", rejected);

    if (failures == 0) {
        std::printf("\nAll engine-twins tests PASSED.\n");
    } else {
        std::printf("\n%d engine-twins test(s) FAILED.\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
