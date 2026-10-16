# Proposal: add-gufo-backend

## Why

Gufo is a purpose-built inference engine for AMD Strix Halo (`gfx1151`, Ryzen AI MAX+ 395 / Radeon 8060S) that substantially outperforms the generic ROCm paths on that hardware — e.g. 1,700+ tok/s prefill and up to 184 aggregated tok/s decode with speculative decoding on Qwen3.8 Flash-Next — while covering text, speech, and image modalities from a single binary. Lemonade users on Strix Halo today get llamacpp-rocm or ds4; neither exposes gufo's model set (Qwen3.8 27B, Flash-Next with MTP, DeepSeek V4 Flash with DSpark, Qwen3-ASR/TTS, Qwen-Image-2.1) or its first-class concurrency/cancellation behavior.

Gufo is architecturally a perfect fit for the backend model: one subprocess running `gufo serve <modality>` that exposes an OpenAI-compatible HTTP API with `/health` and `/ready` endpoints — the same subprocess-plus-HTTP pattern as llamacpp and ds4.

## What Changes

- New `gufo` backend registered via the self-describing descriptor system (`LEMON_BACKENDS` one-liner + descriptor header + `WrappedServer` subclass), serving four existing capability interfaces: chat (`chat_completion`/`completion`/`responses`), `ITranscriptionServer`, `ITextToSpeechServer`, `IImageServer`. No new endpoints are introduced — all traffic flows through existing Lemonade API routes.
- Subprocess lifecycle: `lemond` spawns `gufo serve llm|asr|tts|image` per loaded model on a chosen port with `--served-model-name` set to the Lemonade model id, waits for readiness, proxies requests (including SSE streaming and client-disconnect abort via the base class), and terminates the process on unload/eviction.
- System-managed binary: gufo upstream publishes no GitHub release binaries (only OCI images and build-from-source), so the backend resolves `gufo` from `PATH` or an explicit `gufo_bin` config/env override — the flm "system-managed package" pattern, with `VersionPolicy::AtLeast` and a minimum version gate checked via `gufo --version`. No `backend_versions.json` download pin in this phase.
- Hardware gating: the support matrix restricts the recipe to Linux x86_64 with an AMD `gfx1151` GPU. Availability additionally degrades gracefully (clear message, not a crash) when the binary is missing, too old, or the device is absent.
- Full gufo model catalog added to `server_models.json` with recipe `gufo`: Qwen3.8-27B (GGUF + optional DFlash2 draft + mmproj), Qwen3.8-Flash-Next (sharded GGUF + MTP sidecar + mmproj), DeepSeek-V4-Flash (GGUF + DSpark support file), Qwen3-ASR-1.7B (HF directory), Qwen3-TTS-1.7B (CustomVoice/VoiceDesign/Base HF directories), Qwen-Image-2.1 (pipeline directory).
- Descriptor-driven surfaces come free and must be regenerated: CLI flags (`--gufo`, `--gufo-args`), `/system-info` recipes entry, `defaults.json`, and the generated docs regions (`backends-reference.md`, README matrix, CLI guide, etc.) via `gen_backend_boilerplate.py`.
- Integration test coverage: gufo added as a `--wrapped-server gufo` case for LLM/transcription/TTS/image test scripts, with per-modality `test_models` entries in `test/utils/capabilities.py`.
- Non-goals: gufo video serving (MiniMax H3 — no Lemonade endpoint exists), gufo embeddings/rerank (upstream returns 501), managed binary download from a fork repo (future phase), Windows/macOS support (gufo is Linux-gfx1151-only by construction), and container-based execution (violates the subprocess model's intent of direct host GPU access without an image-fetch step).

## Capabilities

### New Capabilities
- `gufo-backend`: Registration, availability gating, binary resolution, subprocess lifecycle (spawn, readiness, proxy, unload, crash/device-loss handling), option surface (`--gufo-args`, ctx size, speculative draft wiring), and request forwarding semantics for the gufo backend across its four modalities.
- `gufo-model-catalog`: The set of `server_models.json` entries with recipe `gufo`, their checkpoint shapes (single GGUF, sharded GGUF, HF directories), deployment-mode labels, and pull/load behavior against gufo's strict model-id and artifact requirements.

### Modified Capabilities

(none — no existing spec-level behavior changes; the backend registry, router, and endpoints are extended by data, not edited)

## Impact

- **New code**: `src/cpp/include/lemon/backends/gufo/{gufo.h,gufo_server.h}`, `src/cpp/server/backends/gufo/gufo_server.cpp`.
- **Edited**: root `CMakeLists.txt` (`LEMON_BACKENDS` += `"gufo|gufo"`), `src/cpp/resources/server_models.json` (new model entries), `src/cpp/resources/defaults.json` (regenerated), generated docs regions (regenerated).
- **Tests**: `test/utils/capabilities.py` (gufo test models), existing `test/server_llm.py` / `server_whisper.py` / TTS / image scripts gain a gufo case; CI matrix rows in `.github/workflows/cpp_server_build_test_release.yml` gated on Strix Halo runners.
- **Runtime dependencies**: none added to lemond; gufo itself requires ROCm 7.2.x runtime, `/dev/kfd` + `/dev/dri` access, and is capped at 4 HIP processes per device (queue budget), which the router's LRU slot policy must not fight.
- **Hardware scope**: Strix Halo (`gfx1151`) Linux hosts only; every other platform sees the recipe as unsupported and is unaffected.
