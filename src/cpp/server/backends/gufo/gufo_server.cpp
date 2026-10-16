#include "lemon/backends/gufo/gufo_server.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/gufo/gufo.h"
#include "lemon/backends/gufo/gufo_compat.h"
#include "lemon/backends/hf_cache_util.h"
#include "lemon/backends/llamacpp/llamacpp_gguf.h"
#include "lemon/gguf_reader.h"
#include "lemon/model_manager.h"
#include "lemon/model_types.h"
#include "lemon/runtime_config.h"
#include "lemon/utils/custom_args.h"
#include "lemon/utils/http_client.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/process_manager.h"
#include <lemon/utils/aixlog.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace lemon::utils;

namespace lemon {
namespace backends {

using gufo::kMinGufoVersion;

namespace {

// gufo publishes no managed binaries (OCI image / build-from-source only), so
// every failure path must tell the user how to supply the executable.
const char* gufo_remediation() {
    return "gufo was not found. Gufo is a system-managed backend: install it "
           "(https://github.com/gufo-org/gufo — Nix or CMake build) or point "
           "Lemonade at an existing build with the gufo.server_bin config key "
           "or the LEMONADE_GUFO_SERVER_BIN environment variable.";
}

// Exit code gufo uses when the HIP device enters the lost state; the only
// recovery is reloading the model after the GPU recovers.
constexpr int kExitDeviceLost = 75;

std::string resolve_gufo_binary_or_empty() {
    std::string bin = BackendUtils::find_external_backend_binary("gufo", "");
    if (!bin.empty()) {
        std::string expanded = bin;
        if (expanded.size() >= 2 && (expanded[0] == '~')) {
            const char* home = std::getenv("HOME");
            if (home) {
                expanded = std::string(home) + expanded.substr(1);
            }
        }
        if (is_safe_executable_path(expanded) && fs::exists(expanded)) {
            return expanded;
        }
        return "";
    }
    return find_executable_in_path("gufo");
}

std::string resolve_gufo_binary() {
    std::string bin = resolve_gufo_binary_or_empty();
    if (bin.empty()) {
        throw std::runtime_error(gufo_remediation());
    }
    return bin;
}

// Parses "gufo version 0.11.2 (abc1234)" or "gufo version development (rev)".
// Returns the version token, "development", or "" when unparseable.
std::string query_gufo_version(const std::string& executable) {
    std::string output;
    int rc = ProcessManager::run_command("\"" + executable + "\" --version 2>/dev/null",
                                         output, 10);
    if (rc != 0) {
        return "";
    }
    std::smatch match;
    std::regex version_re(R"(gufo version (\S+))");
    if (!std::regex_search(output, match, version_re)) {
        return "";
    }
    return match[1].str();
}

// Numeric compare of "X.Y.Z"; returns true when version >= minimum.
bool version_at_least(const std::string& version, const std::string& minimum) {
    auto parts = [](const std::string& v) {
        std::vector<int> out;
        std::stringstream ss(v);
        std::string item;
        while (std::getline(ss, item, '.')) {
            out.push_back(std::atoi(item.c_str()));
        }
        while (out.size() < 3) {
            out.push_back(0);
        }
        return out;
    };
    auto v = parts(version);
    auto m = parts(minimum);
    for (size_t i = 0; i < 3; ++i) {
        if (v[i] != m[i]) {
            return v[i] > m[i];
        }
    }
    return true;
}

std::string mime_for_audio(const std::string& filename) {
    std::string ext = fs::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".mp3") return "audio/mpeg";
    if (ext == ".flac") return "audio/flac";
    if (ext == ".ogg") return "audio/ogg";
    if (ext == ".mp4" || ext == ".m4a") return "audio/mp4";
    if (ext == ".webm") return "audio/webm";
    return "audio/wav";
}

// Storage types gufo's reader and dequant paths implement: unquantized
// F32/F16/BF16 plus the quantized set in its GgmlType enum (Q4_0..Q8_1,
// Q2_K..Q8_K, IQ2_XXS/IQ3_S/IQ4_XS/IQ4_NL, I32). Anything else (IQ1_*,
// IQ2_S/IQ2_XS, IQ3_XXS, MXFP4, ...) has no dequant path and fails at load.
bool gufo_supported_ggml_type(uint32_t type) {
    switch (type) {
        case 0: case 1: case 2: case 3: case 6: case 7: case 8: case 9:
        case 10: case 11: case 12: case 13: case 14: case 15:
        case 16: case 20: case 21: case 23: case 26: case 30:
            return true;
        default:
            return false;
    }
}

// The flat half of the decision: every tensor the file declares must use a type
// gufo implements. Reached only while gufo_compat.json is unusable, so it can
// refuse a type without naming the group that would have refused it.
// Unsloth-style layouts keep the KV header and split metadata in shard 1 and the
// tensors in the following shards, so an empty tensor list follows one sibling
// before judging. Unreadable headers stay permissive: the engine's own load
// error is the backstop, and refusing here would hide models that work.
bool gufo_can_read_gguf(const std::string& path) {
    GgufTensorTypes info;
    if (!gufo::gufo_read_inventory(info, path)) return true;
    for (const auto& tensor : info.tensors) {
        if (!gufo_supported_ggml_type(tensor.type)) return false;
    }
    return true;
}

// Twins-exposure gate for an artifact that is on disk: the manifest verdict when
// the manifest is usable, the flat storage-type check when it is not, so a broken
// table degrades the decision instead of hiding models.  Registry cache builds
// re-check the same artifacts often, so the result is cached on both artifacts'
// identity plus the manifest's own version: a re-pull, a swapped draft or an
// updated table changes the key instead of returning a stale answer.
bool gufo_checkpoint_servable(const ModelInfo& info, const std::string& main_path) {
    const gufo::GufoCompat& manifest = gufo::GufoCompat::current();
    std::error_code ec;
    const auto identity = [&ec](const std::string& path) {
        const auto mtime = fs::last_write_time(path, ec);
        const auto size = fs::file_size(path, ec);
        return path + "|" +
               std::to_string(static_cast<long long>(mtime.time_since_epoch().count())) + "|" +
               std::to_string(size);
    };
    const std::string resolved_draft = info.resolved_path("draft");
    const std::string draft =
        fs::is_regular_file(path_from_utf8(resolved_draft), ec) ? resolved_draft : std::string();

    const std::string key = identity(main_path) +
        (draft.empty() ? std::string() : "+" + identity(draft)) + "|" +
        (manifest.usable() ? manifest.stamp() : std::string("flat-check"));

    static std::mutex cache_mutex;
    static std::map<std::string, bool> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }
    const bool servable = manifest.usable()
        ? gufo::verdict_entry(main_path,
                              engine_twins::checkpoint_repository(info.checkpoint("main")),
                              draft,
                              engine_twins::checkpoint_repository(info.checkpoint("draft")),
                              info.labels)
              .accept
        : gufo_can_read_gguf(main_path);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[key] = servable;
    }
    return servable;
}

// Variant portion of a "repo:variant" checkpoint, empty when there is none.
std::string checkpoint_variant(const std::string& checkpoint) {
    const auto colon = checkpoint.find(':');
    return colon == std::string::npos ? std::string() : checkpoint.substr(colon + 1);
}

// Locate the artifact a "repo:variant" checkpoint names straight in the hub
// cache. Resolution is manifest-driven, and a pull of one variant rewrites the
// manifest for the whole snapshot, so a sibling variant that is on disk can
// resolve to nothing; the readability gate still has to judge the file that is
// there. The variant names either a file/directory under the snapshot root
// ("UD-Q3_K_XL/", "mmproj-BF16.gguf") or part of a file name
// ("…-UD-Q8_K_XL.gguf", "…-UD-Q8_K_XL-00001-of-000NN.gguf").
std::string gufo_find_artifact_on_disk(const std::string& repo,
                                       const std::string& variant,
                                       const std::string& registry_source) {
    if (repo.empty() || variant.empty()) return {};

    const fs::path snapshots =
        fs::path(get_hf_cache_dir()) /
        hf_cache::repo_id_to_cache_dir_name(repo, registry_source) / "snapshots";
    std::error_code ec;
    if (!fs::is_directory(snapshots, ec)) return {};

    const auto ends_with = [](const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    std::vector<std::string> matches;
    std::error_code it_ec;
    for (const auto& snapshot : fs::directory_iterator(snapshots, hf_cache::dir_options(), it_ec)) {
        if (!snapshot.is_directory(ec)) continue;
        const fs::path snapshot_dir = snapshot.path();

        const fs::path direct = snapshot_dir / variant;
        if (fs::is_regular_file(direct, ec)) return path_to_utf8(direct);
        if (fs::is_directory(direct, ec)) {
            std::vector<std::string> shards;
            for (const auto& file : fs::directory_iterator(direct, hf_cache::dir_options(), ec)) {
                if (file.is_regular_file(ec) && file.path().extension() == ".gguf") {
                    shards.push_back(path_to_utf8(file.path()));
                }
            }
            if (!shards.empty()) {
                std::sort(shards.begin(), shards.end());
                return shards.front();
            }
        }

        for (const auto& file : fs::directory_iterator(snapshot_dir, hf_cache::dir_options(), ec)) {
            if (!file.is_regular_file(ec) || file.path().extension() != ".gguf") continue;
            const std::string stem = file.path().stem().string();
            if (stem == variant || ends_with(stem, "-" + variant) ||
                stem.find("-" + variant + "-") != std::string::npos) {
                matches.push_back(path_to_utf8(file.path()));
            }
        }
    }

    if (matches.empty()) return {};
    std::sort(matches.begin(), matches.end());
    return matches.front();
}

}  // namespace

InstallParams GufoServer::get_install_params(const std::string& backend,
                                             const std::string& version) {
    (void)backend;
    (void)version;
    // Nothing is installable here: the engine is system-managed. Refuse in a way
    // the endpoint reports as a client error, rather than returning empty params,
    // which it would report as a successful install.
    throw BackendNotInstallable(gufo_remediation());
}

GufoServer::GufoServer(const std::string& log_level, ModelManager* model_manager,
                       BackendManager* backend_manager)
    : WrappedServer("gufo-server", log_level, model_manager, backend_manager) {
}

void GufoServer::load(const std::string& model_name,
                      const ModelInfo& model_info,
                      const RecipeOptions& options,
                      bool do_not_upgrade) {
    (void)do_not_upgrade;

    std::string gufo_backend;
    if (json v = options.get_option("gufo_backend"); v.is_string()) {
        gufo_backend = v.get<std::string>();
    }
    std::string gufo_args;
    if (json v = options.get_option("gufo_args"); v.is_string()) {
        gufo_args = v.get<std::string>();
    }
    int ctx_size = 0;
    if (json v = options.get_option("ctx_size"); v.is_number()) {
        ctx_size = v.get<int>();
    }

    RuntimeConfig::validate_backend_choice("gufo", gufo_backend);

    std::string executable = resolve_gufo_binary();

    std::string version = query_gufo_version(executable);
    if (version == "development") {
        LOG(WARNING, "Gufo") << "Using development gufo build; "
                             << "minimum supported release is " << kMinGufoVersion
                             << std::endl;
    } else if (!version.empty() && !version_at_least(version, kMinGufoVersion)) {
        std::ostringstream err;
        err << "gufo " << version << " is too old (minimum " << kMinGufoVersion
            << "). " << gufo_remediation();
        throw std::runtime_error(err.str());
    } else if (version.empty()) {
        LOG(WARNING, "Gufo") << "Could not parse gufo --version output; "
                             << "continuing without a version gate." << std::endl;
    }

    const char* modality = nullptr;
    switch (model_info.type) {
        case ModelType::LLM: modality = "llm"; break;
        case ModelType::TRANSCRIPTION: modality = "asr"; break;
        case ModelType::TTS: modality = "tts"; break;
        case ModelType::IMAGE: modality = "image"; break;
        default:
            throw std::runtime_error("gufo does not serve this model type");
    }

    std::string main_path = model_info.resolved_path("main");
    if (main_path.empty() || !fs::exists(main_path)) {
        throw std::runtime_error("gufo model checkpoint not found on disk");
    }

    std::vector<std::string> args;
    args.push_back("serve");
    args.push_back(modality);
    args.push_back("--model");
    args.push_back(main_path);
    args.push_back("--host");
    args.push_back("127.0.0.1");

    port_ = choose_port();
    args.push_back("--port");
    args.push_back(std::to_string(port_));
    // gufo 404s requests whose model field mismatches; serve under the name
    // clients actually send. The router loads namespaced entries (user.*,
    // extra.*) by canonical id while clients use the public alias, so map it.
    std::string served_name = model_name;
    if (model_manager_ != nullptr) {
        const std::string public_name = model_manager_->get_public_model_name(model_name);
        if (!public_name.empty()) {
            served_name = public_name;
        }
    }
    args.push_back("--served-model-name");
    args.push_back(served_name);

    std::set<std::string> reserved_flags = {
        "-m", "--model", "-i", "--host", "-p", "--port",
        "--served-model-name", "--api-key",
    };

    if (std::string(modality) == "llm") {
        std::string mmproj = model_info.resolved_path("mmproj");
        if (!mmproj.empty() && fs::exists(mmproj)) {
            args.push_back("--mmproj");
            args.push_back(mmproj);
        }

        std::string draft = model_info.resolved_path("draft");
        if (!draft.empty() && fs::exists(draft)) {
            // One label decision drives both the engine flags here and the
            // compatibility verdict in serves_model_checkpoint(); if they ever
            // disagreed, a draft would be launched under rules it was not checked
            // against.
            const std::string speculative = gufo::gufo_speculative_label(model_info.labels);
            if (speculative == "mtp") {
                args.push_back("--speculative");
                args.push_back("mtp");
                args.push_back("--mtp-model");
                args.push_back(draft);
            } else if (speculative == "dspark") {
                args.push_back("--speculative");
                args.push_back("dspark");
                args.push_back("--dspark-model");
                args.push_back(draft);
            } else {
                args.push_back("--speculative");
                args.push_back("dflash2");
                args.push_back("--dflash-model");
                args.push_back(draft);
            }
            reserved_flags.insert({"--speculative", "--dflash-model",
                                   "--mtp-model", "--dspark-model"});
        }
    }

    // ctx_size is an LLM-shaped knob (the router auto-tunes it for chat);
    // asr/tts/image have small per-chunk defaults in gufo that a 32K window
    // would blow past, so only the llm modality forwards it.
    if (ctx_size > 0 && std::string(modality) == "llm") {
        args.push_back("--context");
        args.push_back(std::to_string(ctx_size));
        reserved_flags.insert({"-c", "--context"});
    }

    if (!gufo_args.empty()) {
        std::string error = validate_custom_args(gufo_args, reserved_flags);
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
        std::vector<std::string> extra = parse_custom_args(gufo_args);
        args.insert(args.end(), extra.begin(), extra.end());
    }

    LOG(INFO, "Gufo") << "Starting gufo serve " << modality << " for "
                     << model_name << " on port " << port_ << std::endl;

    bool inherit_output = (log_level_ == "info") || is_debug();
    // gufo binds its listener only after weights are resident, so the ready
    // endpoint answering means the model is ready to serve.
    ServerCommand command;
    command.program = executable;
    command.args = std::move(args);
    command.port = port_;
    command.ready_endpoint = "/health";
    try {
        start_server(std::make_unique<NativeProcess>(ProcessOutput{inherit_output, true}, ""),
                     command, HttpClient::get_default_timeout());
    } catch (const ServerStartFailure& e) {
        // gufo exits 75 when the HIP context is lost. The base class carries the
        // status out of a failed start, so the failure can name the lost context
        // and its recovery instead of reporting a generic startup timeout.
        if (e.exit_code() && *e.exit_code() == kExitDeviceLost) {
            throw std::runtime_error("gufo lost the GPU context during model load (device_lost, "
                                     "exit 75). Wait for the GPU to recover, then load the "
                                     "model again.");
        }
        throw;
    }

    device_type_ = DEVICE_GPU;
    LOG(INFO, "Gufo") << "Model " << model_name << " ready on port " << port_
                      << std::endl;
}

std::string GufoServer::get_backend_health_state() const {
    // The watchdog reaps a dead child before a health check can look at it, so
    // the status that reap recorded is the source of truth here, not the process.
    if (last_stopped_exit_code() == kExitDeviceLost) {
        return "device_lost (gufo exit 75: GPU context lost, reload required)";
    }
    return WrappedServer::get_backend_health_state();
}

json GufoServer::chat_completion(const json& request) {
    return forward_request("/v1/chat/completions", request);
}

json GufoServer::completion(const json& request) {
    return forward_request("/v1/completions", request);
}

json GufoServer::responses(const json& request) {
    return forward_request("/v1/responses", request);
}

json GufoServer::audio_transcriptions(const json& request) {
    try {
        if (!request.contains("file_data")) {
            throw std::runtime_error("Missing 'file_data' in request");
        }

        std::string audio_data = request["file_data"].get<std::string>();
        std::string filename = request.value("filename", "audio.wav");

        std::vector<MultipartField> fields;
        fields.push_back({"file", audio_data, filename, mime_for_audio(filename)});
        if (request.contains("model") && request["model"].is_string()) {
            fields.push_back({"model", request["model"].get<std::string>(), "", ""});
        }
        if (request.contains("language") && request["language"].is_string()) {
            fields.push_back({"language", request["language"].get<std::string>(), "", ""});
        }
        if (request.contains("prompt") && request["prompt"].is_string()) {
            fields.push_back({"prompt", request["prompt"].get<std::string>(), "", ""});
        }
        // The engine implements the OpenAI contract itself: it serves json, text
        // and verbose_json (the last with task/language/duration/segments) and
        // refuses anything else with a client error, so the requested format is
        // passed through — except `text`, whose answer is a plain-text body that
        // this forward path expects to be a JSON document. The endpoint renders a
        // text request from the engine's compact answer instead, which is where
        // the plain-text conversion lives anyway.
        if (request.contains("response_format") && request["response_format"].is_string()) {
            const std::string format = request["response_format"].get<std::string>();
            if (format != "text") {
                fields.push_back({"response_format", format, "", ""});
            }
        }
        if (request.contains("temperature") && request["temperature"].is_number()) {
            fields.push_back({"temperature", std::to_string(request["temperature"].get<double>()), "", ""});
        }

        return forward_multipart_request("/v1/audio/transcriptions", fields, 0);
    } catch (const std::exception& e) {
        return json{
            {"error", {
                {"message", std::string("Transcription failed: ") + e.what()},
                {"type", "audio_processing_error"}
            }}
        };
    }
}

void GufoServer::audio_speech(const json& request, httplib::DataSink& sink) {
    forward_streaming_request("/v1/audio/speech", request.dump(), sink, false);
}

json GufoServer::image_generations(const json& request) {
    return forward_request("/v1/images/generations", request, 0);
}

json GufoServer::image_edits(const json& request) {
    try {
        std::string prompt = request.value("prompt", "");

        std::vector<MultipartField> fields;
        fields.push_back({"prompt", prompt, "", ""});
        if (request.contains("model") && request["model"].is_string()) {
            fields.push_back({"model", request["model"].get<std::string>(), "", ""});
        }
        // gufo's edit route takes references as image[] parts (up to 10) and
        // has no separate mask input; a masked edit is rejected rather than
        // silently ignored.
        if (request.contains("mask_data")) {
            return json{
                {"error", {
                    {"message", "gufo image edits do not support masks"},
                    {"type", "unsupported_capability_error"}
                }}
            };
        }
        if (request.contains("image_data")) {
            std::string image_binary = JsonUtils::base64_decode(
                request["image_data"].get<std::string>());
            fields.push_back({"image[]", image_binary, "image.png", "image/png"});
        }

        return forward_multipart_request("/v1/images/edits", fields, 0);
    } catch (const std::exception& e) {
        return json{
            {"error", {
                {"message", std::string("Image edit failed: ") + e.what()},
                {"type", "image_processing_error"}
            }}
        };
    }
}

json GufoServer::image_variations(const json& request) {
    (void)request;
    return unsupported_capability_error("image variations");
}

namespace gufo {

// System-managed binary: no managed install flow, version comes from the
// resolved executable itself, and unavailability points at build instructions.
class GufoOps : public BackendOps {
public:
    InstallCheck check_install(const std::string& backend,
                               bool binary_found) const override {
        (void)backend;
        (void)binary_found;  // no managed install dir ever exists
        std::string bin = resolve_gufo_binary_or_empty();
        if (bin.empty()) {
            return {false, gufo_remediation()};
        }
        return {true, ""};
    }

    std::string resolve_version(const std::string& backend,
                                const std::string& file_version) const override {
        (void)backend;
        if (!file_version.empty() && file_version != "unknown") {
            return file_version;
        }
        std::string bin = resolve_gufo_binary_or_empty();
        if (bin.empty()) {
            return "unknown";
        }
        std::string version = query_gufo_version(bin);
        return version.empty() ? "unknown" : version;
    }

    std::optional<UnavailableState> classify_unavailable(
        const std::string& backend, const std::string& install_error,
        const std::string& default_install_command) const override {
        (void)backend;
        (void)install_error;
        (void)default_install_command;
        UnavailableState state;
        state.state = "action_required";
        state.message = "gufo is system-managed: Lemonade cannot download it. "
                        "Install gufo or set gufo.server_bin to an existing build.";
        state.action = "https://github.com/gufo-org/gufo";
        return state;
    }

    // Gate downloads on the system-managed binary being present: a user on a
    // perfectly gfx1151-capable host shouldn't burn multi-gigabyte bandwidth
    // for models their machine cannot yet run (spec: pull gating).
    void download_model(const ModelInfo& info, bool do_not_upgrade,
                        DownloadProgressCallback progress,
                        const BackendOpsContext& ctx) const override {
        if (resolve_gufo_binary_or_empty().empty()) {
            throw std::runtime_error(gufo_remediation());
        }
        BackendOps::download_model(info, do_not_upgrade, progress, ctx);
    }

    // Engine twins: repositories the gufo catalog already names are servable
    // for entries registered under other recipes too, so a downloaded model can
    // be run on gufo without re-registration (see lemon/engine_twins.h).
    engine_twins::ServableRepos servable_checkpoint_repositories(
        const std::map<std::string, ModelInfo>& models) const override {
        return engine_twins::servable_repositories(models, descriptor.recipe);
    }

    // Only advertise a checkpoint gufo can actually read. A downloaded file is
    // judged by its tensor storage types (gufo implements only some
    // quantizations); before that, the catalog's declared variant is the only
    // verifiable signal, so unverifiable quants stay out.
    bool serves_model_checkpoint(const ModelInfo& info,
                                 const engine_twins::ServableRepo& repo) const override {
        const std::string main = info.resolved_path("main");
        std::error_code ec;
        if (!main.empty() && fs::is_regular_file(path_from_utf8(main), ec)) {
            return gufo_checkpoint_servable(info, main);
        }

        // Directory-shaped catalog entries (ASR/TTS/image) are served whole:
        // the repository match is the check.
        if (!repo.variant_pinned) return true;

        const std::string checkpoint = info.checkpoint("main");
        const std::string variant = checkpoint_variant(checkpoint);
        const std::string on_disk = gufo_find_artifact_on_disk(
            engine_twins::checkpoint_repository(checkpoint), variant,
            info.registry_source.empty() ? "huggingface" : info.registry_source);
        if (!on_disk.empty()) return gufo_checkpoint_servable(info, on_disk);

        return variant == repo.variant;
    }

    // Chat mains are GGUF files whose checkpoint variants are quant names
    // (flat repos) or shard folders (unsloth sharded layout); the shared GGUF
    // resolver handles both and returns the first shard for sharded families.
    // Directory checkpoints (asr/tts/image, no variant) and sidecars keep the
    // shared default.
    std::string resolve_checkpoint_path(const ModelInfo& info,
                                        const CheckpointResolveContext& ctx) const override {
        if (ctx.type == "main" && !ctx.variant.empty()) {
            return llamacpp::resolve_gguf_path(ctx.model_cache_path, ctx.variant);
        }
        return BackendOps::resolve_checkpoint_path(info, ctx);
    }

    void populate_metadata(ModelInfo& info,
                           const BackendOpsContext& ctx) const override {
        (void)ctx;
        // GGUF chat checkpoints carry the trained context length; directory
        // checkpoints (asr/tts/image) have none.
        std::string main_path = info.resolved_path("main");
        if (main_path.empty() || !fs::exists(main_path)) {
            return;
        }
        GgufMetadata meta;
        if (read_gguf_metadata(meta, main_path) && meta.context_length > 0) {
            info.max_context_window = meta.context_length;
        }
    }
};

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<GufoServer>(ctx);
}

const BackendSpec* spec() {
    return make_spec<GufoServer>(descriptor);
}

const BackendOps* ops() {
    return single_ops<GufoOps>();
}

}  // namespace gufo

}  // namespace backends
}  // namespace lemon
