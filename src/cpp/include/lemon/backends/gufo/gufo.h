#pragma once

#include "lemon/backends/backend_descriptor.h"

namespace lemon {
namespace backends {
namespace gufo {

// Minimum `gufo --version` accepted at load time — older builds lack the server
// surface this backend drives (served-model-name, /health semantics).  The
// compatibility manifest's floor field must equal this value; the release the
// manifest *mirrors* is a separate, higher field, checked against the release
// the shipped image bundles (gufo_compat.json engine.pin), so a pin bump moves
// the mirror without silently moving this floor.
inline constexpr const char* kMinGufoVersion = "0.11.0";

// The gufo backend descriptor (plain data). Gufo is a vertical inference
// engine built for AMD Strix Halo (gfx1151); upstream publishes no binaries
// (OCI image / build-from-source only), so this backend resolves a
// system-managed `gufo` executable via gufo.server_bin or PATH.
inline const BackendDescriptor descriptor = {
    /*recipe*/          "gufo",
    /*display_name*/    "Gufo (Strix Halo)",
    /*binary*/          "gufo",
    /*config_section*/  "",  // defaults to recipe
    /*default_device*/  DEVICE_GPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ true,
    /*uses_ctx_size*/   true,
    /*dynamic_models*/  false,
    /*options*/ {
        {"gufo_backend", "--gufo", "", "BACKEND",
         "Gufo backend to use", "Gufo Options"},
        {"gufo_args", "--gufo-args", "", "ARGS",
         "Custom arguments to pass to gufo serve", "Gufo Options"},
    },
    /*support*/ {
        {"rocm", BackendTier::Experimental, BackendFormat::Native, {"linux"},
         {{"amd_gpu", {"gfx1151"}}}, "AMD Strix Halo (Radeon 8060S)"},
    },
    /*supported_modes*/ {"chat", "transcription", "tts", "image"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*web_display_name*/ "Gufo Strix Halo",
    /*rocm_channels*/   {},  // single rocm artifact, no stable/nightly channels
    /*exposes_prometheus_metrics*/ true,  // gufo serve exposes /metrics (llama.cpp-style gauges)
    /*rocm_requires_cwsr_fix*/ false,     // gufo needs the stock AMD ROCm stack, not TheRock
    /*version_policy*/  VersionPolicy::AtLeast,  // system-managed binary, minimum version gate
    /*self_manages_downloads*/ false,
    /*takes_args*/      true,
    /*arg_variants*/    {},
    /*bin_variants*/    {},  // override via gufo.server_bin (empty-backend bin key)
    /*config_extra*/    nlohmann::json::object(),
};

}  // namespace gufo
}  // namespace backends
}  // namespace lemon
