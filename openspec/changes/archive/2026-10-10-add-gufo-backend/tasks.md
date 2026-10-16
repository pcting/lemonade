# Tasks: add-gufo-backend

## 1. Backend skeleton and registration

- [x] 1.1 Create `src/cpp/include/lemon/backends/gufo/gufo.h` with the descriptor per design D1 (vllm as template: recipe `gufo`, modes `chat,transcription,tts,image`, gfx1151/linux support row, `selectable_backend: true` with `gufo_backend`/`--gufo` + `gufo_args`/`--gufo-args` options, `VersionPolicy::AtLeast`, `exposes_prometheus_metrics: true`)
- [x] 1.2 Create `src/cpp/include/lemon/backends/gufo/gufo_server.h`: `GufoServer : WrappedServer, ITranscriptionServer, ITextToSpeechServer, IImageServer`, plus `gufo::create()`, `spec()`, `ops()`, `constexpr capabilities()`
- [x] 1.3 Create `src/cpp/server/backends/gufo/gufo_server.cpp` with `create()`/`spec()`/`ops()` and stub `load()`/`unload()`
- [x] 1.4 Add `"gufo|gufo"` to `LEMON_BACKENDS` in root `CMakeLists.txt`; build and confirm `BackendModeContractTest` passes with the new backend (modes ⇔ interfaces)

## 2. Binary resolution and availability

- [x] 2.1 Implement `GufoOps::check_install` / version resolution: `gufo.server_bin` / `LEMONADE_GUFO_SERVER_BIN` override → PATH lookup; run `gufo --version`, parse `gufo version <v> (<rev>)`, enforce minimum 0.11.0 with `development` passing with a warning (design D2)
- [x] 2.2 Make `install_backend("gufo")` report the system-managed remediation message instead of attempting a download; verify `/system-info` shows gufo state (found + version / missing / unsupported OS / unsupported GPU) on both a gfx1151 host and a non-gfx1151 host
- [x] 2.3 Verify pull/load gating: `POST /pull` for a gufo model on unsupported hardware or with no binary fails before any weight download begins

## 3. Subprocess lifecycle

- [x] 3.1 Implement `load()` following `VLLMServer::load()`'s order of operations per design D4: options → `RuntimeConfig::validate_backend_choice("gufo", ...)` → `install_backend` (no-op, D2) → modality dispatch from `ModelType` (llm/asr/tts/image) → `choose_port()` → argv (`--model`, explicit `--host 127.0.0.1`, `--port`, `--served-model-name`, `--mmproj`, `--context` only when set, speculative flags from draft checkpoint + `dflash`/`mtp`/`dspark` label) → spawn via `ProcessManager::start_process` → `set_process_handle` → `wait_for_ready("/health")`
- [x] 3.2 Implement reserved-flag validation for `gufo_args` (`-m/--model`, `-i/--host`, `-p/--port`, `--served-model-name`, `--api-key`, driven speculative flags) with a clear refusal error before spawn
- [x] 3.3 Implement `unload()` and destructor: stop watchdog, terminate process; verify port release and clean shutdown on `lemond` exit
- [x] 3.4 Handle device-loss exit 75: surface a named device-lost backend error, ensure the model is marked unloaded and a subsequent load starts a fresh process
- [x] 3.5 Verify sharded-GGUF resolution for Flash-Next reuses `gguf_shard_utils` (first-shard path passed, siblings discovered by gufo)

## 4. Request paths

- [x] 4.1 Wire chat/completion/responses forwarding (base-class `forward_request` + `forward_streaming_request`); confirm SSE streaming and client-disconnect abort (invariant 11) end-to-end with a live gufo process (streaming + responses verified live via server_llm.py; disconnect-abort rides the base-class path — worth a dedicated CI check later)
- [x] 4.2 Implement `audio_transcriptions` multipart proxy to `/v1/audio/transcriptions` (whisper.cpp pattern)
- [x] 4.3 Implement `audio_speech` proxy to `/v1/audio/speech` (kokoro pattern)
- [x] 4.4 Implement `image_generations` and `image_edits` proxies to `/v1/images/*` (sd-cpp pattern); confirm `image_variations` returns the standard unsupported error
- [x] 4.5 Confirm gufo's `/metrics` is scraped by Lemonade's Prometheus aggregation and the recipe advertises it (verified live: `lemonade_model_*{recipe="gufo"}` gauges present with counters updating per request)

## 5. Model catalog

- [x] 5.1 Add the eight catalog entries from design D7 to `src/cpp/resources/server_models.json` with pinned revisions from gufo's model docs, correct deployment-mode labels, and `suggested: true` only on the two Qwen3.8 entries
- [x] 5.2 Implement `GufoOps::populate_metadata` for GGUF chat entries (max_context_window from GGUF metadata; Flash-Next = 262144); confirm directory-modality entries skip GGUF metadata gracefully
- [x] 5.3 Verify `BackendModeContractTest` and registry ingest accept every new entry (labels ⇔ supported modes, checkpoint validation: GGUF magic for `main`, directory completeness for asr/tts/image)

## 6. Generated surfaces

- [x] 6.1 Run `python docs/tools/gen_backend_boilerplate.py`; commit regenerated `defaults.json` and marker regions (backends-reference, README matrix, CLI guide, configuration guides, models.js)
- [x] 6.2 Confirm `--gufo` and `--gufo-args` appear in `lemonade run/load --help`, `lemonade backends` lists gufo, and `python docs/tools/gen_backend_boilerplate.py --check` passes

## 7. Tests and CI

- [x] 7.1 Add gufo entries to `test/utils/capabilities.py` for llm, transcription, tts, and image modalities per design D9
- [x] 7.2 Add `--wrapped-server gufo` support to `test/server_llm.py`, the transcription script, the TTS script, and the image script; assert real output (text non-empty, valid RIFF/WAV, valid PNG) and failure paths (validated live: server_llm.py 31 OK, server_whisper.py 12 OK, server_tts.py 7 OK, server_sd.py 22 OK incl. real PNG gen/edit)
- [x] 7.3 Add CI matrix rows (one per modality) to `.github/workflows/cpp_server_build_test_release.yml`, modeled on the existing vllm gfx1151 rows (same runner pool, `ci:backends` label convention); ensure gufo is NOT added to `_THEROCK_RECIPES`
- [x] 7.4 Document the Strix Halo runner gufo provisioning requirement (pinned gufo build baked into the runner image; tests skip with a clear message when the binary is absent) in `docs/dev/testing.md`

## 8. Docs and polish

- [x] 8.1 Add a gufo section to `docs/dev/backends-reference.md` editorial areas (non-generated parts if any): hardware requirement, ROCm 7.2 runtime, system-binary install instructions (Nix + CMake), 4-HIP-process ceiling note
- [x] 8.2 Final validation pass: `openspec validate add-gufo-backend --strict`, full cpp-ci test run, `gen_backend_boilerplate.py --check`, and a manual end-to-end run on a Strix Halo machine (pull → load → chat stream → transcribe → speak → image → unload)
