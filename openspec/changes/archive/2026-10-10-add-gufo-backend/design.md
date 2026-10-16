# Design: add-gufo-backend

## Context

See `proposal.md` for motivation and the two specs for the behavior contract. Design-relevant facts about the two codebases:

**gufo runtime contract** (from `../gufo/docs/SERVER.md`, `docs/CLI.md`, `src/cli/serve/serve.cpp`):

- One binary (`gufo`, v0.11.0), one `gufo serve <modality>` process per model, one model per process. No multi-model serving, no HTTP model swap.
- Modalities and their `--model` shapes: `llm` (GGUF file or first shard of a sharded set), `asr` (HF repo dir), `tts` (HF repo dir), `image` (HF pipeline dir), `video` (out of scope).
- HTTP: OpenAI-compatible under `/v1` only, plus bare `/health`, `/ready`, `/metrics`, llama-server compat routes. SSE streaming with `[DONE]` terminator. Strict `model` field matching (404 on mismatch) unless `--served-model-name` is set. Errors are OpenAI-shaped JSON.
- Readiness ordering is orchestrator-friendly: the listener binds only after weights are resident, so a successful `/health` response implies model-ready. Port flag `-p/--port` (default 8080), host defaults to `127.0.0.1`.
- Exit codes: 0 clean, 1 load/bind failure, 2 usage, **75 = HIP context lost (device_lost), restart is the only recovery**.
- Hardware: Linux x86_64 + `gfx1151` only, enforced at build time and again at runtime; ROCm 7.2.x runtime, `/dev/kfd` + `/dev/dri` access. Device-wide ceiling of ~4 concurrent HIP processes (8 compute queues; llm uses 2, asr/tts 1, image/video more).
- Distribution: **no GitHub release binaries** — Nix flake, plain CMake build, or OCI image `ghcr.io/gufo-org/toolboxes/gufo-runtime`. `gufo --version` prints `gufo version <X.Y.Z|development> (<rev>)`.
- Speculation flags: `--speculative dflash2 --dflash-model <p>`, `--mtp-model <p>` (implies mtp), `--speculative dspark --dspark-model <p>`.
- Concurrency is in-process (`--sessions`, admission queue); gufo handles concurrent Lemonade requests itself and propagates client disconnects to abort generation.

**lemonade backend framework** (from `docs/dev/adding-a-backend.md` and the llamacpp/ds4/flm implementations): descriptor header + `WrappedServer` subclass + `LEMON_BACKENDS` line; `BackendManager` install machinery is GitHub-releases-based; `WrappedServer` provides `choose_port()`, `wait_for_ready()`, process handle ownership, request/streaming proxying with cancel tokens (invariant 11), and a process watchdog; `BackendUtils` provides `server_bin`-style external binary overrides (`gufo.server_bin` / `LEMONADE_GUFO_SERVER_BIN`) and PATH lookup; whole-repo directory checkpoints are already supported (colon-less `checkpoint: "repo"` — used by ryzenai, kokoro, vllm); `BackendModeContractTest` ties `supported_modes` to implemented capability interfaces.

**Reference implementation choice**: the **vllm backend** (`src/cpp/server/backends/vllm/`) is the most recently added gfx1151 ROCm backend and the closest precedent for gufo's serve contract — single ROCm variant, experimental, loopback bind with explicit `--host 127.0.0.1`, `--served-model-name` model-id alignment, `/health` readiness, whole-repo HF checkpoints. This design anchors its serve contract and descriptor conventions on vllm rather than the older llamacpp implementation; llamacpp remains the reference only for multi-capability proxying (multipart/TTS/image patterns) it uniquely has.

## Goals / Non-Goals

**Goals**

- A `gufo` recipe that behaves like any first-party backend on Strix Halo: visible in `/system-info`, drivable by CLI flags, testable with `--wrapped-server gufo`, invisible-as-noise elsewhere.
- Zero new endpoints, zero router edits: all four modalities map onto existing capability interfaces.
- Honest failure modes: missing binary, old binary, wrong hardware, device loss — each produces an actionable message, never a hang or a mystery 500.

**Non-Goals**

- Managed binary download (phase 2, see Decisions D2 and Migration).
- gufo video serving, embeddings/rerank (501 upstream), `--voice` enrollment for TTS Base models, gufo disk KV cache configuration, Anthropic-native passthrough to gufo's `/v1/messages` (Lemonade's own Anthropic gateway already converts to chat completions).
- Container-based execution (`podman run ghcr.io/gufo-org/...` as the "binary"): rejected — it hides the binary behind an image-fetch step, complicates model bind-mounts and `/dev/kfd` group handling, and conflicts with the subprocess model's direct-host-GPU intent. Users who prefer the container can point `gufo.server_bin` at a wrapper script.

## Decisions

### D1. Descriptor shape — vllm as the template, flm for version policy

```cpp
inline const BackendDescriptor descriptor = {
    /*recipe*/          "gufo",
    /*display_name*/    "Gufo Strix Halo",
    /*binary*/          "gufo",
    /*config_section*/  "",                       // "gufo"
    /*default_device*/  DEVICE_GPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ true,                  // vllm pattern: --gufo + gufo.backend config
    /*uses_ctx_size*/   true,                     // maps to --context
    /*dynamic_models*/  false,
    /*options*/ {
        {"gufo_backend", "--gufo", "", "BACKEND",
         "Gufo backend to use", "Gufo Options"},
        {"gufo_args", "--gufo-args", "", "ARGS",
         "Custom arguments to pass to gufo serve", "Gufo Options"},
    },
    /*support*/ {
        {"rocm", {"linux"}, {{"amd_gpu", {"gfx1151"}}}, "AMD Strix Halo (Radeon 8060S)"},
    },
    /*supported_modes*/ {"chat", "transcription", "tts", "image"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*experimental*/    true,                     // phase-1, single-hardware, system-managed
    /*web_display_name*/ "Gufo Strix Halo",
    /*rocm_channels*/   {},
    /*exposes_prometheus_metrics*/ true,           // gufo serves /metrics (llama.cpp-style gauges)
    /*rocm_requires_cwsr_fix*/ false,              // gufo needs stock ROCm 7.2, not TheRock
    /*version_policy*/  VersionPolicy::AtLeast,    // minimum 0.11.0
    /*self_manages_downloads*/ false,
    /*takes_args*/      true,
    /*arg_variants*/    {},
    /*bin_variants*/    {},                        // override via gufo.server_bin (empty-backend key)
    /*config_extra*/    nlohmann::json::object(),
};
```

Why `experimental: true` initially: single-hardware, system-managed binary, and the model set moves fast upstream; flip when the phase-2 managed install lands and CI has soaked. `rocm_requires_cwsr_fix: false` because gufo uses the distro/AMD ROCm stack, not TheRock — so **gufo must NOT be added to `_THEROCK_RECIPES`** (unlike llamacpp-rocm and vllm-rocm).

`selectable_backend: true` follows the vllm precedent (vllm declares `--vllm` + a `vllm_backend` option despite having only the one `rocm` variant): it gives `--gufo` CLI parity with `--llamacpp`/`--vllm`, a `gufo.backend` config key defaulting to `"auto"`, and `RuntimeConfig::validate_backend_choice("gufo", ...)` enforcement in `load()` (valid choices: empty/auto/`rocm`). ds4/flm use `false`, but vllm is the newer convention and the one this backend aligns with.

### D2. System-managed binary; no `backend_versions.json` entry

`GufoOps::check_install` resolves the binary via `BackendUtils::find_external_backend_binary("gufo", "")` (config `gufo.server_bin` / env `LEMONADE_GUFO_SERVER_BIN`) falling back to PATH lookup of `gufo`; `resolve_version` shells out to `gufo --version` and parses `gufo version <v> (<rev>)`. Version gate: minimum `0.11.0` (first release whose SERVER.md contract this spec pins); a `development` version string (Nix dev builds — the common case for people checking out the repo) passes the gate with a warning. `spec()` returns a real `BackendSpec` but no install params are produced — `install_backend("gufo")` reports "system-managed: install gufo yourself or set gufo.server_bin" instead of attempting a download.

Why not vllm's managed route today: vllm converges on `lemonade-sdk/vllm-rocm` publishing per-arch release archives (`get_install_params` with `version_override` tags like `vllm0.20.1-rocm7.12.0-gfx1151`, `VersionPolicy::Exact`) — but the equivalent `lemonade-sdk/gufo-rocm` build infra (ROCm 7.2 + GCC 15 toolchain in CI) doesn't exist yet, and the user decision was system-first. **Phase 2 converges on exactly the vllm pattern**: `VLLMServer::get_install_params` is the template to copy, and the descriptor is shaped so phase 2 only adds `get_install_params` + a version pin (flipping `version_policy` to `Exact`) without touching lifecycle code.

Alternative considered: OCI image execution — rejected (Non-Goals).

### D3. Modality dispatch inside one server class

One `GufoServer : WrappedServer, ITranscriptionServer, ITextToSpeechServer, IImageServer` class; chat methods come from the `WrappedServer` defaults being overridden. `load()` reads the model's `ModelType` (from its deployment-mode label) and selects the serve subcommand:

| ModelType | subcommand | model arg |
|---|---|---|
| LLM | `serve llm` | `--model <main.gguf or first shard>` |
| TRANSCRIPTION | `serve asr` | `--model <dir>` |
| TTS | `serve tts` | `--model <dir>` |
| IMAGE | `serve image` | `--model <dir>` |

This matches gufo's one-process-per-modality design and lemonade's one-WrappedServer-instance-per-loaded-model routing. `supported_modes` ⇔ implemented interfaces is exactly what `BackendModeContractTest` checks.

### D4. argv construction in `load()` — modeled on `VLLMServer::load()`

Order of operations mirrors vllm's `load()`: read options → `RuntimeConfig::validate_backend_choice("gufo", gufo_backend)` → `backend_manager_->install_backend(...)` (no-op that reports the system-managed message, D2) → resolve checkpoints → `port_ = choose_port()` → build argv → spawn → `wait_for_ready("/health")`.

```
gufo serve <modality>
  --model <resolved main path>            (file for llm, directory for asr/tts/image)
  --host 127.0.0.1                        (explicit, vllm pattern — never rely on engine defaults)
  --port <choose_port()>
  --served-model-name <lemonade model id>
  [--mmproj <p>]                          when an mmproj checkpoint resolved
  [--context <ctx_size>]                  llm modality only, only when set (gufo default 0 = native)
  [--speculative dflash2 --dflash-model <p>]  draft + "dflash" label
  [--mtp-model <p>]                           draft + "mtp" label
  [--speculative dspark --dspark-model <p>]   draft + "dspark" label
  <parsed gufo_args>                      after validate_custom_args
```

Reserved flags rejected in `gufo_args`: `-m`, `--model`, `-i`, `--host`, `-p`, `--port`, `--served-model-name`, and the speculative flags when a draft checkpoint drove them (same reserved-flag discipline as ds4's `reserved_custom_arg_flags()`). `--api-key` is also reserved (Lemonade speaks to the subprocess unauthenticated on loopback; a user-injected key would break proxying).

`--served-model-name` is the linchpin for model-id alignment: gufo 404s on any `model` mismatch, and Lemonade rewrites/normalizes model ids at the router edge; setting the served name to the Lemonade id makes both agree by construction. This is the exact mechanism `VLLMServer::load()` already uses — vllm adopted it instead of llamacpp's older `normalize_response_model` response rewriter, and gufo follows vllm.

Deliberately **not** adopted from vllm: its `vllm_model_config.json` per-model argument resolution (`resolve_vllm_args`). gufo bakes per-model sampling/speculation presets into the engine itself, so there is nothing meaningful to configure per model from Lemonade; the descriptor `gufo_args` option plus checkpoint-driven flags suffice.

### D5. Readiness, watchdog, device loss

`wait_for_ready("/health")` unchanged from the base class — since gufo binds its listener only after weights are resident, health-OK implies model-ready; the 600 s budget covers an 80 GiB DeepSeek cold load on slow storage. The base watchdog already detects child death; on exit code 75 the error surfaced to clients names device loss and advises a GPU reset/reload (the process is gone; the router marks the model unloaded on next touch). No custom supervisor loop — gufo's docs recommend `Restart=on-failure`, but auto-restarting inside `lemond` would mask a broken GPU; a reload is the user-visible recovery.

### D6. Request paths

- Chat/completion/responses: base-class `forward_request` to `/v1/chat/completions|/v1/completions|/v1/responses`; streaming via `forward_streaming_request`. Invariant 11 (client-disconnect abort) comes free — do not bypass.
- Transcription: proxy multipart to `/v1/audio/transcriptions` following the whisper.cpp backend's existing multipart forwarding pattern.
- TTS: proxy JSON to `/v1/audio/speech` (kokoro precedent). gufo's voice-cloning `--voice` enrollment flags are per-process startup config for Base models; phase 1 does not expose them (CustomVoice variants need nothing).
- Image: proxy to `/v1/images/generations` and `/v1/images/edits` following the sd-cpp pattern; `variations` returns the standard unsupported error.
- gufo's strict field validation (rejects `logprobs`, `n>1`, etc.) is passed through as-is: Lemonade surfaces the upstream 400 JSON. No pre-validation — gufo's error shape is already OpenAI-compatible.

### D7. Model catalog entries

Whole-repo checkpoints (colon-less) for directory modalities, `repo:VARIANT` for GGUFs, sidecars as named checkpoints; revisions pinned to the commits gufo's model docs pin:

| id | recipe | checkpoint(s) | labels | size (GB) |
|---|---|---|---|---|
| `Qwen3.8-27B-GGUF` | gufo | main `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL`, draft `z-lab/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` | chat, hot, vision, reasoning, tool-calling, dflash | ~17 |
| `Qwen3.8-Flash-Next-GGUF` | gufo | main `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` (sharded), draft `unsloth/Qwen3.8-Flash-Next-GGUF:MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | chat, hot, vision, reasoning, tool-calling, mtp | ~101 |
| `DeepSeek-V4-Flash-GGUF` | gufo | main `antirez/deepseek-v4-gguf:DeepSeek-V4-Flash-IQ2XXS-...-0731.gguf`, draft `antirez/deepseek-v4-gguf:DeepSeek-V4-Flash-DSpark-support-0731.gguf` | chat, reasoning, dspark | ~81 |
| `Qwen3-ASR-1.7B` | gufo | main `Qwen/Qwen3-ASR-1.7B` (whole repo) | transcription | ~3.6 |
| `Qwen3-TTS-1.7B-CustomVoice` | gufo | main `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | tts | ~4 |
| `Qwen3-TTS-1.7B-VoiceDesign` | gufo | main `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | tts | ~4 |
| `Qwen3-TTS-1.7B-Base` | gufo | main `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | tts | ~4 |
| `Qwen-Image-2.1` | gufo | main `Qwen/Qwen-Image-2.1` (whole pipeline) | image | ~33 |

`suggested: true` only for the two Qwen3.8 entries (the flagship Strix Halo experience); the rest are discoverable but not default-pulled. `QwenOps::populate_metadata` reads GGUF metadata for `max_context_window` (Flash-Next reports 262,144) the way llamacpp's GGUF helper does; directory modalities have no GGUF and skip it.

Sharded-GGUF note: gufo discovers sibling shards from the first shard path, which matches how llamacpp's shard resolution already returns the first shard path — reuse `gguf_shard_utils` rather than writing new shard logic.

### D8. Concurrency and slot policy

`SlotPolicy::Standard` (counts toward per-type LRU slots). gufo's 4-HIP-processes-per-device ceiling aligns with lemonade's per-type LRU (4 model types ⇒ at most 4 gufo processes if all types are loaded); no NPU-exclusivity interplay. If a user pins more concurrent gufo models than the queue budget allows, gufo itself logs `queue_budget_exceeded` and still starts — we pass that signal through rather than adding a second admission layer.

### D9. Tests and CI

- `test/utils/capabilities.py`: gufo entries per modality with small test models (chat: `Qwen3.8-Flash-Next-GGUF` is the fastest-loading chat entry; transcription: `Qwen3-ASR-1.7B`; tts: `Qwen3-TTS-1.7B-CustomVoice`; image: `Qwen-Image-2.1`).
- Reuse existing scripts as `--wrapped-server gufo` cases: `server_llm.py`, `server_whisper.py` (transcription), the TTS and image scripts. No new test files unless multipart/proxy quirks demand one.
- CI matrix rows in `cpp_server_build_test_release.yml` on Strix Halo runners, modeled on the existing **vllm gfx1151 rows** (same runner pool, same `ci:backends` label convention). Unlike vllm, gufo is **not** in `_THEROCK_RECIPES` and has no managed install to warm the runner — instead the runner image must pre-provision a `gufo` build (phase-1 constraint of the system-binary decision — see Risks).

## Risks / Trade-offs

- [System-binary UX] A fresh Strix Halo user cannot `lemonade pull` a working gufo stack without first building gufo (ROCm 7.2 + GCC 15) — mitigated by the actionable availability error naming the exact remediation, and by phase 2 (managed releases) being the planned end state.
- [gufo flag drift] gufo is pre-1.0 and its CLI is explicitly undocumented-by-design ("use --help"); a rename of `--served-model-name` or a speculative flag breaks loads. Mitigation: pin minimum version 0.11.0, keep argv construction in one function, integration tests on the merge queue catch breakage at bump time.
- [CI provisioning] No prebuilt gufo binaries means CI runners must build or cache gufo themselves; a gfx1151 runner image build adds a ROCm-toolchain dependency. Mitigation: bake a pinned gufo build into the Strix Halo runner image; tests skip (not fail) with a clear "gufo not provisioned" state where the runner lacks it.
- [Device loss masquerade] Exit 75 can also surface as a plain process death if the watchdog races the exit code; worst case the user sees a generic backend error instead of "device lost" — acceptable, reload still recovers.
- [Queue budget] Loading 5+ gufo models (e.g., 2 LLMs + tts + asr + image) can exceed the 4-process HIP ceiling and degrade throughput rather than fail. Mitigation: Standard LRU eviction already bounds loaded models; document the ceiling in backends-reference.
- [Strict upstream validation] gufo rejects fields llamacpp tolerates (`logprobs`, `n>1`, some sampling combos); clients that relied on tolerance will see new 400s on this recipe only. Accepted — gufo's errors are explicit and OpenAI-shaped.
- [`development` version pass-through] Accepting `gufo version development` weakens the version gate for Nix dev builds. Accepted deliberately: this backend's phase-1 audience builds from source; the gate exists to protect against pre-0.11 contract mismatches.

## Migration Plan

Additive change; no data or API migrations. Deploy = ship the build; on non-gfx1151 hosts the recipe is inert. Rollback = revert the commit; nothing persisted changes shape (`defaults.json` regenerates; `server_models.json` entries are orphaned but harmless if the recipe disappears). Phase 2 (managed install) will add `backend_versions.json` + `get_install_params` without touching lifecycle code.

## Open Questions

- Which Strix Halo CI runner label pool to target for the gufo jobs (`lemon-prod` vs a dedicated gfx1151 pool) — an infra detail that doesn't change the spec or task breakdown.
- Whether to expose gufo's `--cache-disk` continuation cache as a descriptor option once its on-disk format is stable.
- TTS voice enrollment (`--voice NAME=PATH` for Base models) — needs a Lemonade-side voice-management surface that doesn't exist yet; revisit if requested.

## Appendix: Comparison with the llamacpp and vllm backends

Side-by-side of the existing `llamacpp` backend and the proposed `gufo` backend, grounded in the llamacpp descriptor and decisions D1–D9 above.

| Dimension | `llamacpp` (existing) | `gufo` (proposed) |
|---|---|---|
| **Engine** | llama.cpp (`llama-server`) | gufo (`gufo serve`), vertical Strix Halo engine |
| **OS support** | Windows, Linux, macOS | Linux only |
| **Hardware scope** | Broad: NVIDIA (cuda), AMD (rocm), Vulkan, CPU, Apple metal — many arch gates | Single arch: `gfx1151` (Strix Halo) only; hard fail elsewhere, no fallback |
| **Support rows** | 6 (`system, metal, cuda, vulkan, rocm, cpu`) | 1 (`rocm`) |
| **Backend variants** | `selectable_backend: true` — `--llamacpp` picks vulkan/rocm/cuda/...; ROCm has stable/nightly channels | `selectable_backend: true` (vllm pattern) — `--gufo` + `gufo.backend` config, but only one variant (`rocm`) |
| **Binary distribution** | Managed GitHub-release downloads (`backend_versions.json` pins per variant, e.g. `b10825`), atomic install into cache dir | **System-managed**: `gufo.server_bin` / `LEMONADE_GUFO_SERVER_BIN` → PATH; no download, no version pin |
| **Version policy** | `Exact` (pins like `b10825`) | `AtLeast` (min 0.11.0; `development` builds pass with warning) |
| **`system` PATH escape hatch** | One variant among many | The *only* mode of operation |
| **Supported modes** | `chat, embeddings, reranking` | `chat, transcription, tts, image` |
| **Capability interfaces** | `IEmbeddingsServer, IRerankingServer, ISlotsServer, ITokenizerServer` | `ITranscriptionServer, ITextToSpeechServer, IImageServer` |
| **Process model** | One `llama-server` per loaded model; `--parallel 1` forced | One `gufo serve <modality>` per loaded model; concurrency internal (`--sessions`) |
| **Device queue budget** | No device-wide process ceiling | ≤4 HIP processes/device (8 compute queues) — aligns with per-type LRU |
| **Readiness probe** | `/health` (server binds before model ready) | `/health` (listener binds **only after** weights resident — health ⇒ model-ready) |
| **Model-id handling** | `-m <path>` → response rewriter (`normalize_response_model`) to fix echoed ids | `--served-model-name <lemonade-id>` — aligned at startup, no rewriter needed |
| **Model artifacts** | GGUF + `mmproj` + `draft` sidecars | GGUF + `mmproj` + `draft`, **plus whole-repo HF directories** (asr/tts/image) |
| **Speculative decoding** | `--model-draft`, `--spec-type draft-dflash/draft-mtp` merged via `resolve_runtime_options` | `--speculative dflash2/dspark`, `--mtp-model` driven by `dflash`/`mtp`/`dspark` labels |
| **ctx_size** | Always passed (`--ctx-size`, embedding floor 8192) | Passed only for the `llm` modality when set (asr/tts per-chunk defaults would blow up on a chat-sized window); gufo otherwise defaults to native context |
| **Custom args flag** | `--llamacpp-args` (+ `--llamacpp-device`) | `--gufo-args`; reserved flags: `-m/--model`, `--host`, `--port`, `--served-model-name`, `--api-key` |
| **Special env/variant handling** | Heavy: TheRock lib paths, `OCL_SET_SVM_SIZE` for gfx1151, CUDA env, Windows DLL staging | None — stock ROCm 7.2 runtime; **not** in `_THEROCK_RECIPES` |
| **Failure semantics** | Generic process death → backend error | Distinct **exit 75 = device_lost**; named error, reload is the recovery |
| **Upstream API strictness** | Tolerant (accepts `logprobs`, `n>1`, etc.) | Strict: unknown/unsupported fields → explicit 400s; `model` mismatch → 404 |
| **Prometheus metrics** | `exposes_prometheus_metrics: true` | Also `true` (gufo serves llama.cpp-style `llamacpp:*` gauges) |
| **Model catalog** | Dozens of entries, `suggested` broadly | 8 entries, gufo's fixed catalog; `suggested` only on the two Qwen3.8 flagships |
| **Maturity flag** | `experimental: false` (flagship backend) | `experimental: true` initially (phase 1) |
| **CI hardware** | Many runner pools (vulkan, rocm, cuda, metal) | gfx1151 runners only + runner-image gufo provisioning |

Two structural asymmetries:

- **Distribution is inverted.** llamacpp's default path is "lemond downloads a pinned binary"; gufo's default path is "the user brings the binary" (D2). That inversion is the single biggest source of new code: `check_install`/`resolve_version` replace `get_install_params`, and phase 2 exists to converge the two models.
- **Modalities are disjoint except chat.** On Strix Halo the two backends are complements — llamacpp owns embeddings/rerank, gufo owns transcription/tts/image — so recipe routing stays unambiguous and neither backend needs to defer to the other.

### gufo vs vllm — the closest precedent

vllm is the most recently added gfx1151 ROCm backend and the template for gufo's serve contract (D1/D4). Where they agree and where they diverge:

| Dimension | `vllm` (existing) | `gufo` (proposed) |
|---|---|---|
| **Serve contract** | `--model` / explicit `--host 127.0.0.1` / `--port` / `--served-model-name`, `/health` readiness | Same pattern, verbatim (D4) |
| **Descriptor conventions** | `selectable_backend: true` + `--vllm` despite one variant; `experimental: true`; single `rocm` row incl. gfx1151; no ROCm channels | Same conventions (D1) |
| **Whole-repo HF checkpoints** | ✅ colon-less `checkpoint: "repo"` | ✅ for asr/tts/image entries |
| **Binary distribution** | Managed download from `lemonade-sdk/vllm-rocm` (per-arch tags, `version_override`) | System-managed only in phase 1; **phase 2 converges on this exact pattern** (D2) |
| **Version policy** | `Exact` (fork pins per arch) | `AtLeast` (min 0.11.0) — forced by the system-binary decision |
| **ROCm stack** | TheRock + CWSR fix (`rocm_requires_cwsr_fix: true`, in `_THEROCK_RECIPES`) | Stock AMD ROCm 7.2; **not** in `_THEROCK_RECIPES` |
| **Modalities** | `chat` only | 4 modes — the multi-capability proxying (multipart ASR, TTS, image) has no vllm precedent; llamacpp-family backends remain the reference there |
| **Binary nature** | Bundled Python env (PYTHONPATH startup shim, env surgery) | Single native binary; no env handling needed |
| **Per-model arg config** | `vllm_model_config.json` + `resolve_vllm_args` | Deliberately not adopted — gufo bakes per-model presets into the engine (D4) |
| **Prometheus metrics** | `exposes_prometheus_metrics: false` (scrapes internally for its own use) | `true` — gufo's `/metrics` is re-exposed |
| **CI** | gfx1151 rows, runner warms TheRock + managed install | Same runner pool/label convention; gufo needs a pre-provisioned binary baked into the runner image instead (D9) |
