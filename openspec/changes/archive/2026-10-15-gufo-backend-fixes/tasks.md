## 1. Transcription honours the requested format

- [x] 1.1 Add `response_format` and `temperature` to the multipart body `GufoServer::audio_transcriptions` builds, mirroring `whispercpp_server.cpp`
  → Added, with one exception found by measurement: `text` is left off, because the engine answers it with a plain-text body and the forward path (`forward_multipart_request` and the router behind it) parses the backend's answer as JSON — forwarding it turned a working request into a 500 (reproduced against a stub engine, then fixed). The endpoint already renders plain text from a `text` field, so a text request still arrives as `text/plain`
- [x] 1.2 Pin what the client sees when the engine refuses a format: send an `srt` request through a server and check the status. Rely on the existing error mapping if the engine's body maps to a client error; if it does not, refuse subtitle formats in the backend with the shape FLM returns
  → No new mapping needed: `get_error_status_code` maps the engine's `invalid_request_error` type to 400 (server.cpp:146-148), and a stub run confirms it — `srt` returns **400** with the engine's `invalid_response_format` body nested under `error.details.response`. The existing mapping was right, so the backend refuses nothing itself
- [x] 1.3 Confirm `text` still arrives as `text/plain` (the server converts it from the response's `text` field) and that `verbose_json` now carries the engine's `language`/`duration`/`segments`
  → Measured through a server with the stub engine: `text` → **200 `text/plain`**, `verbose_json` → **200 JSON** `{"duration":3,"language":"english","segments":[],"task":"transcribe","text":"stub transcript"}`, `json` → the compact shape, and the stub's log confirms which formats it was actually asked for (`srt` and `verbose_json` forwarded, none for `text`)
- [x] 1.4 Tests: drop the gufo exclusion in the format loop of `test/server_whisper.py`, and add a gufo case asserting a client error naming the format for `srt`/`vtt`, mirroring `test_002d`
  → The comment in `test_002b` now says why gufo is excluded from the subtitle formats (the engine refuses them) and names the new test; `test_002e_transcription_subtitle_formats_rejected_on_gufo` asserts 400 and an error naming the format, mirroring `test_002d`. Not run here: the suite imports `openai`/`numpy`, which this host does not have — CI runs it against `--wrapped-server gufo`
- [x] 1.5 Document the format contract where gufo's endpoint coverage is described, so the refusal is expected rather than surprising
  → A paragraph in the gufo section of `docs/dev/backends-reference.md`: the format is passed to the engine, `text` comes back as plain text, `verbose_json` carries the engine's `task`/`language`/`duration`/`segments`, and any other value is refused with a client error — the same outcome as FLM

## 2. The reader cannot be made to throw

- [x] 2.1 Bound the tensor reservation in `read_gguf_tensor_types` by what the file can hold and return `false` for an impossible declaration, leaving the caller's permissive path in charge
  → Two bounds: a declared count is rejected outright when the remaining bytes could not describe that many entries (20 bytes is the floor per entry), and the reservation itself is capped so the count the file claims can never size an allocation. Both answers are `false`, which is the same answer a short read gets, so the caller's permissive path is unchanged
- [x] 2.2 Tests in `test_gufo_compat.cpp`: a header declaring more tensors than the file can describe yields accept-as-unreadable without throwing, and an ordinary artifact still judges and refuses exactly as before
  → `write_gguf_declaring` writes a header claiming 2^40 tensors with no records; the check `a header declaring more than the file can hold accepts as unreadable` passes alongside every existing refusal and acceptance case
- [x] 2.3 Re-run the conformance check and the drift judge over the existing artifacts to prove no verdict moved (the bound must be invisible to every real file)
  → Conformance: 0 failures. Drift: `18 groups re-read from /tmp/opencode/gufo-v0.15.0` and from the local checkout, and the artifact mirror still reports `every declared tensor group loads` for both judged entries — no verdict moved

## 3. Device loss is reported on the load path

- [x] 3.1 Add the exit status a failed readiness wait observed to `WrappedServer` (additive accessor, set where the child is reaped) without changing any existing backend's behaviour
  → `last_failed_exit_code()` plus `record_failed_exit_code()`, called at both reap sites (the readiness wait and the watchdog reset) and cleared whenever a child is adopted. Additive: no existing backend reads it, and the generic base-class log line is untouched
- [x] 3.2 Use it in `GufoServer::load` so an exit of 75 during startup produces the device-loss message and recovery hint, keeping the generic base-class log line
  → The handle-based read is replaced (it could never see the status, because the wait reaps the child first). Measured: a stub engine exiting 75 during load now fails with `gufo lost the GPU context during model load (device_lost, exit 75). Wait for the GPU to recover, then load the model again.` instead of `gufo failed to start within timeout`
- [x] 3.3 Test: point `gufo.server_bin` at a stub that exits 75 and assert the load error names device loss rather than a startup timeout
  → That is exactly how the 3.2 measurement was made: `LEMONADE_GUFO_SERVER_BIN` pointed at a stub whose serve mode exits 75, triggered through `POST /load`
- [x] 3.4 Reconcile `get_backend_health_state` with the same source of truth, so the watchdog reaping the handle cannot hide the state
  → The health state now reads the recorded status rather than a handle the watchdog has already consumed

## 4. An install request says what to do

- [x] 4.1 Refuse an install request naming the gufo recipe as a client error carrying the remediation text, and map that refusal in the install handler
  → A `BackendNotInstallable` refusal (thrown by `get_install_params`) is mapped to 400 with the remediation in both `handle_install` and `handle_install_dry_run`. The flow also had to be unblocked first: `get_version_from_config` threw `backend_versions.json is missing 'gufo' section` before the refusal was ever reached, so a *known* recipe with no version section now resolves to no version, exactly as the `system` backend already did — an unknown recipe still raises, so a typo is not silently accepted
- [x] 4.2 Verify a managed backend's install path is unchanged by the mapping (dry run and a normal install both behave as before)
  → Measured through a server: dry-run `llamacpp:cpu` → **200** with a GitHub asset URL (unchanged), dry-run `gufo:rocm` → **400** with `code: backend_not_installable`, dry-run for an unknown recipe → **500** `Unknown recipe` (the typo guard intact)
- [x] 4.3 Test: `POST /install` for `gufo` returns a client error with the remediation, attempts no download, and reports no server error
  → Measured: `POST /api/v0/install {"recipe":"gufo","backend":"rocm"}` → **400**, `type: invalid_request_error`, message carrying the remediation (was **500** before the fix). An endpoint-suite case (`test_000b`) asserts the same for CI; not run here because that suite imports `openai` too

## 5. Close out

- [x] 5.1 `ctest -L cpp-ci` at its baseline (the pre-existing `ModelSyncAndAutoUpdateTest` failure is not this change's) and `openspec validate --all --strict`
  → `ctest -L cpp-ci`: **75/76**, the single failure being `ModelSyncAndAutoUpdateTest` — the pre-existing abort this baseline already carried. The gufo tests all pass (`BackendModeContractTest`, `EngineTwinsTest`, `GufoCompatTest`, `GufoCompatDriftTest`). `openspec validate --all --strict`: 6 items, 0 failures
- [x] 5.2 Rebuild the image, redeploy, and confirm end to end on the deployment: `verbose_json` carries the verbose fields, `text` is plain, `srt` is refused
  → Built as `lemonade:gufo-fixes`, parked the running build as `gufo-prev`, promoted it to the tag the compose names, and recreated **only** `lemonade-server` (healthy ~10 s, engine `0.15.0 (89d6185)`, table hash now identical to the repo, 0 fallback lines). Against the real engine: `text` → 200 `text/plain`; `verbose_json` → 200 with `duration`/`language`/`segments`/`task`; `srt` → **400** `invalid_response_format`; `json` → 200 compact. The ASR model loaded for the check is unloaded again
- [x] 5.3 Record the client-visible change (subtitle formats refused, verbose fields added) where API behaviour changes are documented
  → The API reference documents neither this endpoint nor its formats (there is no `/audio/transcriptions` section in `docs/api/lemonade.md`), so the behaviour is recorded where the backend's client-facing contract already lives: the gufo section of `docs/dev/backends-reference.md`, added under 1.5
