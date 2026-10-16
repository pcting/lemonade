#pragma once

#include <string>

#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/wrapped_server.h"

namespace lemon {
namespace backends {

// One `gufo serve <modality>` subprocess per loaded model. Gufo serves exactly
// one model and one modality per process, so the modality is chosen from the
// model's deployment mode at load() time.
class GufoServer : public WrappedServer,
                   public ITranscriptionServer,
                   public ITextToSpeechServer,
                   public IImageServer {
public:
    // The system-managed binary has no downloadable artifact; any attempt to
    // install it through the managed flow is refused with remediation text.
    static InstallParams get_install_params(const std::string& backend,
                                            const std::string& version);

    GufoServer(const std::string& log_level, ModelManager* model_manager,
               BackendManager* backend_manager);

    void load(const std::string& model_name, const ModelInfo& model_info,
              const RecipeOptions& options, bool do_not_upgrade = false) override;

    json chat_completion(const json& request) override;
    json completion(const json& request) override;
    json responses(const json& request) override;

    json audio_transcriptions(const json& request) override;
    void audio_speech(const json& request, httplib::DataSink& sink) override;

    json image_generations(const json& request) override;
    json image_edits(const json& request) override;
    json image_variations(const json& request) override;

    std::string get_backend_health_state() const override;
};

namespace gufo {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
// Lets BackendModeContractTest check supported_modes against the real class.
constexpr uint32_t capabilities() { return capability_mask_of<GufoServer>(); }
}  // namespace gufo

}  // namespace backends
}  // namespace lemon
