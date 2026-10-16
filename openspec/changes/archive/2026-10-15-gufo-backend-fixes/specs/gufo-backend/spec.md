## MODIFIED Requirements

### Requirement: System-managed binary resolution

Because upstream publishes no downloadable release binaries, the system SHALL resolve the `gufo` executable from (in order) an explicit binary override (config key `gufo.server_bin` or env `LEMONADE_GUFO_SERVER_BIN`) and then `PATH`. No managed download SHALL be attempted, and an install request naming this recipe SHALL be answered as a client-actionable refusal carrying the same remediation text as a missing binary — never as a server error, and never as a success. If no executable is found, load and install checks SHALL report an actionable message telling the user how to install gufo or point Lemonade at an existing build.

#### Scenario: Binary found on PATH
- **WHEN** a gufo model is loaded and `gufo` is present on PATH and passes the version gate
- **THEN** the load proceeds using that binary

#### Scenario: Binary missing
- **WHEN** a gufo model is loaded and no `gufo` executable can be resolved
- **THEN** the load fails with an error explaining that gufo must be installed (build-from-source or package) or configured via the binary override, and no model state is left half-loaded

#### Scenario: Version below minimum
- **WHEN** the resolved binary reports a version older than the backend's documented minimum via `gufo --version`
- **THEN** the load fails with a message stating the found and required versions

#### Scenario: Install requested for a system-managed engine
- **WHEN** an install request names the `gufo` recipe
- **THEN** the response is a client error carrying the remediation text, no download is attempted, and the outcome is not reported as an internal server error

### Requirement: Device-loss recovery

When the gufo subprocess exits with the device-lost status (75) after a GPU context loss, the system SHALL treat the backend as failed: report a backend error to in-flight and subsequent requests, mark the model unloaded, and allow a later load to start a fresh process (the only recovery for context loss). A device-lost exit that happens while a load is still in progress SHALL be reported as such, so the load failure names the lost context and its recovery instead of a generic startup timeout.

#### Scenario: GPU context loss
- **WHEN** the gufo process exits with status 75 while serving
- **THEN** requests fail with a backend error mentioning device loss, and a subsequent `POST /load` succeeds once the GPU is usable again

#### Scenario: Context lost during a load
- **WHEN** the gufo process exits with status 75 before it ever reports ready
- **THEN** the load fails with a message naming the device-lost status and how to recover, rather than reporting only that the process did not become ready in time

## ADDED Requirements

### Requirement: Transcription response formats

For a loaded gufo ASR model, the client SHALL receive the response format it asked for: `text` as plain text, `json` and `verbose_json` as JSON carrying the engine's own fields (the verbose shape including the detected language and segment metadata when the engine reports them), and a format the engine does not implement as a client error naming that format, matching how the same request is refused on FLM. No requested format SHALL be answered with a different shape.

#### Scenario: Verbose metadata requested
- **WHEN** `POST /audio/transcriptions` requests `response_format=verbose_json` against a loaded gufo ASR model
- **THEN** the response is JSON carrying the engine's verbose fields, including the detected language and segment metadata when the engine reports them

#### Scenario: Plain text requested
- **WHEN** the same request asks for `response_format=text`
- **THEN** the response body is the transcript as plain text, with no JSON envelope around it

#### Scenario: Format the engine does not implement
- **WHEN** the request asks for a subtitle format the gufo engine refuses, such as `srt` or `vtt`
- **THEN** the client receives a client error naming the unsupported format instead of a success carrying an untimestamped transcript

#### Scenario: Transcript that is itself valid JSON
- **WHEN** `response_format=text` is requested and the transcript text happens to be valid JSON
- **THEN** the body is still that text, returned verbatim rather than parsed or re-wrapped
