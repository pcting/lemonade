## Context

See `proposal.md` — Why. The constraints that shape the fixes, all measured against a deployed build whose `lemond`/`gufo` binaries match this branch:

- The transcription handler already hands `response_format` to the backend (server.cpp:4665) and already turns a plain-text format into `text/plain` when the backend returns a `text` field (server.cpp:4742). Gufo builds its multipart body from four fields and drops the rest, so the engine never sees the request's format.
- The engine itself is stricter than Lemonade currently is: asked directly it serves `text`, serves `json`/`verbose_json` (with `task`, `language`, `duration`, `segments`) and answers `srt` with `400 {"code":"invalid_response_format", …}`. Every other transcription backend forwards the field; the project already has a test asserting that FLM *refuses* subtitle formats with 400 rather than returning plain text.
- `read_gguf_tensor_types` reserves from the header's `tensor_count` before reading anything (gguf_reader.h:583). The pre-existing metadata reader never reserves. A declared-count mismatch therefore throws `bad_alloc`/`length_error` where the documented permissive path expects `false`.
- `wait_for_ready` consumes and reaps the child when it exits (wrapped_server.cpp:601-605), so the exit status a backend can still read afterwards is empty — which is exactly the case `GufoServer::load` checks for exit 75.
- `GufoServer::get_install_params` throws, while the two system-managed precedents (`FastFlowLMServer`, `LlamaCppServer`'s `system` variant) return empty parameters. Gufo's backend id is `rocm`, so the manager's `system` early-return does not apply and the throw becomes a 500.

## Goals / Non-Goals

**Goals:**
- The response format a transcription client asks for is the format it gets, and what the engine refuses to produce is refused rather than silently downgraded.
- No input file can make the compatibility reader throw out of a registry build.
- A device-loss during load is reported as one.
- An install request for a system-managed engine says what to do instead, in the status class it belongs to.

**Non-Goals:**
- Changing the manifest, its pin, the twin mechanism, the engine pin or the image.
- The review's three observations that are not defects: the flat fallback type list, `image_variations`, and the absence of prompt-token fitting.
- Re-deriving or re-verifying any pairing record; no verdict changes here.

## Decisions

### D1 — Forward the format, except where the engine's answer is not JSON

Add `response_format` (and `temperature`, which is dropped the same way) to the multipart body in `GufoServer::audio_transcriptions`, mirroring `whispercpp_server.cpp:468-471`. The engine then supplies the verbose shape for `verbose_json` and its own 400 for anything else.

`text` is the exception, found by running the change against a stub engine: the engine answers `text` with a plain-text body, and the forward path (`forward_multipart_request` and the router behind it) parses the backend's answer as JSON, so forwarding it turned a working request into a 500. The endpoint already renders plain text from a `text` field, and the engine's compact answer carries one, so the format is left off for `text` and the rendering happens where it already did.

The status the client sees when the engine refuses a format needs no new mapping: `get_error_status_code` maps the engine's `invalid_request_error` type to 400, which a stub-engine run confirmed.

**Alternative rejected:** rejecting every format except `json` in the backend. `text` works today, and `verbose_json` would lose the engine's fields for no reason.

### D2 — Bound the reader's reservation by what the file can hold

Before reserving, require the declared count to be plausible for the file's size — the tensor-info section cannot be larger than the file, and each entry needs at least a name length, a dimension count, a type and an offset — and return `false` (unreadable) when it is not. That is the same failure the reader already returns for a short read, so the caller's permissive path is unchanged and the spec's "unreadable header" now covers it.

**Alternatives rejected:** catching `bad_alloc` at the call site (hides the cause and leaves the reader's contract unclear); clamping the reserve and continuing to parse (would judge a file we cannot trust).

### D3 — Let the base readiness wait hand back the exit status

The status is already observed and thrown away by `wait_for_ready`. Add an additive accessor on `WrappedServer` for the exit status seen by a failed readiness wait, and have `GufoServer::load` name device loss from it. The base change is additive: no existing backend reads the new value, so their behaviour is unchanged, and the generic base-class log line stays as it is.

**Alternatives rejected:** the current in-backend read of the process handle (proven unreachable, because the wait has already consumed it); dropping the special case entirely (keeps a documented recovery hint out of the message, and the spec now asks for it).

### D4 — Refuse the install as a client error rather than a no-op

The precedents return empty parameters, which the endpoint reports as success: for a system-managed engine that misstates the outcome — nothing was installed and nothing will be. Keep the refusal, but make it a client-actionable error carrying the remediation text, so the status class matches the situation and the message matches what a missing-binary load already says. Implement it as a distinguishable refusal the install handler maps to a client error, leaving the other recipes' install behaviour untouched.

**Alternative rejected:** returning empty parameters for consistency with FLM/llama.cpp's `system` variant. Consistency of code shape is worth less here than not telling an operator that an engine they do not have was installed.

## Risks / Trade-offs

- **The transcription change is client-visible.** A client that currently receives a 200 with an untimestamped transcript for `srt`/`vtt` will now receive a client error, and `verbose_json` responses gain fields. That is the intended fix, but it belongs in the docs and in the release notes, and the format tests must cover gufo rather than exclude it.
- **A shared-surface change for a one-backend diagnostic (D3).** Kept additive and opt-in; `ctest -L cpp-ci` must stay at its baseline (one pre-existing failure, unrelated).
- **The reader bound must not reject valid files.** The bound is derived from the file size rather than a constant, and the drift judge re-judges the existing artifacts, so any regression in the verdicts shows up there before anything ships.
- **The install mapping touches an endpoint shared by every recipe.** Verification must cover a managed backend's install path (or at least confirm it is unchanged) in addition to gufo's refusal.
