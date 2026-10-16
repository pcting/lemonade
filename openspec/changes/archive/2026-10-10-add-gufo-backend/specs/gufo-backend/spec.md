# gufo-backend

## Purpose

Exposes the gufo Strix Halo inference engine (`gfx1151`) as a Lemonade backend recipe, so clients can run gufo's text, transcription, speech-synthesis, and image models through the standard Lemonade APIs on supported hardware.

## ADDED Requirements

### Requirement: Recipe registration and visibility
The system SHALL expose a backend recipe named `gufo` with display name "Gufo (Strix Halo)", declared supported modes `chat`, `transcription`, `tts`, and `image`, and it SHALL appear in every descriptor-driven surface without per-surface edits: `GET /system-info` recipes, `lemonade backends`, the generated docs matrix, and the CLI option table (including a `--gufo` backend-selection flag, a `--gufo-args` custom-arguments flag, and the shared `ctx_size` option, following the vllm backend's flag conventions).

#### Scenario: Recipe listed on a supported host
- **WHEN** `lemonade backends` or `GET /api/v1/system-info` is queried on a Linux x86_64 host with a gfx1151 GPU
- **THEN** the `gufo` recipe is present with its four supported modes and its state reflects whether a usable gufo binary was found

#### Scenario: CLI flag generated from descriptor
- **WHEN** `lemonade run --help` or `lemonade load --help` is invoked
- **THEN** `--gufo` and `--gufo-args` are listed among the per-recipe options

### Requirement: Platform and hardware gating
The `gufo` recipe SHALL be available only on Linux x86_64 hosts whose AMD GPU reports architecture `gfx1151`. On any other platform the recipe SHALL be reported as unsupported, and attempts to pull or load a gufo-recipe model SHALL fail with a message naming the hardware requirement rather than launching a doomed process.

#### Scenario: Unsupported GPU
- **WHEN** a client requests `POST /pull` for a gufo-recipe model on a host without a gfx1151 GPU
- **THEN** the request is refused with a clear unsupported-hardware error and no `gufo` process is started

#### Scenario: Non-Linux host
- **WHEN** `GET /system-info` is queried on Windows or macOS
- **THEN** the `gufo` recipe reports unsupported for the current OS

### Requirement: System-managed binary resolution
Because upstream publishes no downloadable release binaries, the system SHALL resolve the `gufo` executable from (in order) an explicit binary override (config key `gufo.server_bin` or env `LEMONADE_GUFO_SERVER_BIN`) and then `PATH`. No managed download SHALL be attempted. If no executable is found, load and install checks SHALL report an actionable message telling the user how to install gufo or point Lemonade at an existing build.

#### Scenario: Binary found on PATH
- **WHEN** a gufo model is loaded and `gufo` is present on PATH and passes the version gate
- **THEN** the load proceeds using that binary

#### Scenario: Binary missing
- **WHEN** a gufo model is loaded and no `gufo` executable can be resolved
- **THEN** the load fails with an error explaining that gufo must be installed (build-from-source or package) or configured via the binary override, and no model state is left half-loaded

#### Scenario: Version below minimum
- **WHEN** the resolved binary reports a version older than the backend's documented minimum via `gufo --version`
- **THEN** the load fails with a message stating the found and required versions

### Requirement: Per-modality subprocess lifecycle
For each loaded gufo model the system SHALL run exactly one `gufo serve` subprocess for the model's modality (`llm`, `asr`, `tts`, or `image`), passing the resolved model artifact via `--model`, a server-chosen free port via `--port`, and binding loopback only. The system SHALL consider the backend ready only after the subprocess's HTTP interface answers, SHALL proxy subsequent requests to it, and SHALL terminate the process on unload, eviction, or server shutdown.

#### Scenario: Chat model load
- **WHEN** a gufo chat model is loaded via `POST /load`
- **THEN** a `gufo serve llm` subprocess is running with the model path and a Lemonade-chosen port, and the readiness probe on its HTTP interface has succeeded

#### Scenario: Unload stops the process
- **WHEN** the model is unloaded or evicted
- **THEN** the gufo subprocess is terminated and its port is released

#### Scenario: Subprocess dies during idle
- **WHEN** the gufo process exits unexpectedly while the model is loaded
- **THEN** subsequent requests for that model fail with a backend error rather than hanging, and a subsequent load starts a fresh process

### Requirement: Model identity alignment
The system SHALL start each gufo subprocess with `--served-model-name` set to the Lemonade model identifier, so that requests routed through Lemonade (which name the Lemonade model id) match gufo's strict model-id check and never receive a model-not-found error from the subprocess.

#### Scenario: Chat request model field
- **WHEN** a client sends `POST /v1/chat/completions` with `model` set to the Lemonade model name for a loaded gufo model
- **THEN** the request completes successfully and the response's `model` field identifies the requested Lemonade model

### Requirement: Streaming and client-disconnect passthrough
Chat, completion, and responses traffic (including SSE streaming) SHALL be proxied to the subprocess with Lemonade's existing non-streaming and streaming cancellation semantics: a client disconnect or timeout SHALL abort the upstream request to the gufo process without re-entering router locks and without triggering a model reload.

#### Scenario: Streaming chat
- **WHEN** a client requests a streaming chat completion on a loaded gufo chat model
- **THEN** SSE chunks from gufo are relayed to the client and the stream terminates normally

#### Scenario: Client disconnects mid-generation
- **WHEN** a non-streaming or streaming client disconnects while gufo is generating
- **THEN** the upstream request to gufo is aborted, the model remains loaded, and other concurrent requests are unaffected

### Requirement: Custom argument passthrough with reserved flags
The system SHALL append user-supplied `gufo_args` (config or `--gufo-args`) to the subprocess command line, and SHALL reject custom arguments that attempt to override Lemonade-reserved flags: the modality subcommand, `--model`/`-m`, `--host`/`-i`, `--port`/`-p`, and `--served-model-name`.

#### Scenario: Allowed tuning args
- **WHEN** a model is loaded with `gufo_args` containing `--sessions 4`
- **THEN** the subprocess command line includes `--sessions 4`

#### Scenario: Reserved flag rejected
- **WHEN** `gufo_args` contains `--port 9`
- **THEN** the load is refused with an error naming the reserved flag before any process is started

### Requirement: Speculative decoding wiring
For chat models whose registry entry declares a speculative draft artifact, the system SHALL pass the appropriate gufo speculative flags for the draft type: `--speculative dflash2 --dflash-model <path>`, `--mtp-model <path>`, or `--speculative dspark --dspark-model <path>`. Models without a draft artifact SHALL start with gufo's default (no speculation).

#### Scenario: DFlash2 draft
- **WHEN** a gufo chat model with a DFlash2 draft checkpoint is loaded
- **THEN** the subprocess runs with `--speculative dflash2` and the resolved draft file path

#### Scenario: No draft
- **WHEN** a gufo chat model without any draft checkpoint is loaded
- **THEN** no speculative flags are present on the command line

### Requirement: Device-loss recovery
When the gufo subprocess exits with the device-lost status (75) after a GPU context loss, the system SHALL treat the backend as failed: report a backend error to in-flight and subsequent requests, mark the model unloaded, and allow a later load to start a fresh process (the only recovery for context loss).

#### Scenario: GPU context loss
- **WHEN** the gufo process exits with status 75 while serving
- **THEN** requests fail with a backend error mentioning device loss, and a subsequent `POST /load` succeeds once the GPU is usable again

### Requirement: Modality endpoint coverage
Gufo models SHALL be reachable through the existing Lemonade endpoints for their mode: chat/completions/responses for `chat`, `POST /audio/transcriptions` for `transcription`, `POST /audio/speech` for `tts`, and `POST /images/generations` (and image edits, which gufo supports) for `image`. Image variations, which gufo does not implement, SHALL return the standard unsupported-capability error.

#### Scenario: Transcription
- **WHEN** audio is posted to `POST /api/v1/audio/transcriptions` with a loaded gufo ASR model
- **THEN** a transcription result is returned from the gufo process

#### Scenario: Unsupported variation call
- **WHEN** `POST /images/variations` is called against a gufo image model
- **THEN** the response is the standard unsupported-capability error, not a crash or hang

### Requirement: Metrics exposure
Because the gufo subprocess serves Prometheus-format metrics, the system SHALL scrape and re-expose them through Lemonade's existing metrics aggregation, and the recipe SHALL advertise this capability in `/system-info`.

#### Scenario: Metrics scraped
- **WHEN** a gufo chat model is loaded and Lemonade's metrics endpoint is scraped
- **THEN** gufo-sourced gauges are present alongside other backend metrics
