## Why

A review of the gufo backend against its closest counterpart (vLLM) and the project's other backends found four defects. One is user-visible: Lemonade drops `response_format` before it reaches the engine, so a transcription request for `srt` or `vtt` returns HTTP 200 with an untimestamped transcript instead of the refusal the project already produces on FLM, and `verbose_json` loses the language, duration and segment metadata the engine would have supplied.

The other three are robustness and consistency:

- the tensor-info reader allocates from a tensor count the file declares, so a header claiming an implausible count throws out of the registry build instead of taking the documented "unreadable header → accept" path;
- the device-loss diagnostic on the load path cannot fire, because the base readiness wait reaps the exited child before the backend can read its status;
- an explicit install request for the system-managed engine surfaces as a server error, where the two existing system-managed backends treat installation as nothing to do.

## What Changes

- Forward `response_format` and `temperature` to `gufo serve asr`, so the engine's own contract applies: `json`, `text` and `verbose_json` are served with their full shapes, and a format the engine cannot produce reaches the client as the same client error FLM produces rather than a silently degraded 200.
- Bound the tensor-info reader's reservation to what a file of that size can plausibly contain, so an implausible declaration is treated as an unreadable header — accepted, permissively, as before.
- Make the device-loss status reachable on the load path, so a load that fails because the GPU context was lost reports that instead of a generic startup timeout.
- Answer an explicit install request for gufo with a client-actionable refusal carrying the remediation text, instead of a server error.
- Extend the tests: gufo joins the transcription format tests instead of being excluded from them, plus a rejection case for formats it cannot serve, a reader bound case, a load-path device-loss case and an install-request case.

Excluded, because the review judged them observations rather than defects: the flat storage-type fallback list duplicating engine knowledge beside the manifest, `image_variations` returning the standard unsupported-capability error while `sdcpp` implements it, and the absence of vLLM-style prompt-token fitting (llama.cpp does not fit either).

## Capabilities

### New Capabilities

- None — this change modifies existing capabilities.

### Modified Capabilities

- `gufo-backend`: the transcription endpoint's response-format contract, the device-loss diagnostic on the load path, and the outcome of an explicit install request for a system-managed engine.
- `gufo-checkpoint-compatibility`: the per-artifact verdict's definition of an unreadable header, extended to cover a header that parses but declares what the file cannot hold.

## Impact

- `src/cpp/server/backends/gufo/gufo_server.cpp` — multipart fields for transcription, the load-path exit status handling, and the install-params refusal.
- `src/cpp/include/lemon/gguf_reader.h` — a bound on the tensor reservation before it is made.
- Tests: `test/server_whisper.py` (gufo covered by the format tests rather than excluded, plus a rejection case), `test/cpp/test_gufo_compat.cpp` (reader-bound cases that keep the permissive path intact).
- Untouched: the compatibility manifest and its pin, engine twins, the container image and its smoke check, the model catalog.
- Shipped behaviour needs a rebuilt image: the deployed 0.15.0 image predates these fixes, and the transcription change is visible to API clients.
