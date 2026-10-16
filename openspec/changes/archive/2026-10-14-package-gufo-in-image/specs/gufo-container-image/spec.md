# gufo-container-image

## Purpose

Makes the shipped Lemonade container image self-sufficient for the gufo backend: the gufo engine is built at image build time from a pinned upstream source revision and installed into the image, so a fresh container on supported hardware reports the gufo recipe as installed and can load models with zero operator configuration.

## ADDED Requirements

### Requirement: Image bundles the gufo engine at a pinned upstream revision
The container image build SHALL compile the gufo engine from an upstream source revision recorded as a build-configurable pin (repository URL and version tag), and the pinned version SHALL satisfy the gufo backend's minimum-version gate as an exact release — not a `development` build.

#### Scenario: Fresh container reports the pinned version
- **WHEN** `gufo --version` is executed inside a freshly built image
- **THEN** the output reports the pinned release version and exits successfully

#### Scenario: Version gate passes without warning
- **WHEN** a gufo model is loaded inside the container using the bundled engine
- **THEN** the backend's minimum-version check passes on the parsed release version and no `development build` warning is emitted

### Requirement: Bundled engine is discoverable with zero configuration
The engine SHALL be installed at a location on the standard `PATH` inside the image, so the existing system-managed resolution order (explicit override → `PATH`) finds it without `gufo.server_bin` config or environment variables, while explicit overrides set by an operator still take precedence over the bundled copy.

#### Scenario: Recipe installed out of the box
- **WHEN** `GET /system-info` is queried on a Linux gfx1151 host running the image with no gufo overrides set
- **THEN** the `gufo` recipe backend state is `installed` with the bundled version

#### Scenario: Operator override wins
- **WHEN** the container runs with `LEMONADE_GUFO_SERVER_BIN` pointing at a different executable
- **THEN** that executable is used instead of the bundled one

### Requirement: Engine runtime dependencies are satisfied in-image
The image SHALL contain the engine's runtime closure: the HIP/rocBLAS/hipBLAS/hipBLASLt/hsa runtime libraries the engine links against and the `ffmpeg`/`ffprobe` executables it invokes for media handling.

#### Scenario: Engine links cleanly
- **WHEN** any dynamic loader check of the bundled binary runs inside the final image
- **THEN** no `shared libraries: cannot open shared object file` errors occur

#### Scenario: Serving starts on supported hardware
- **WHEN** `gufo serve asr --model <staged ASR checkpoint> --port <free>` is started inside the container on a gfx1151 host with the GPU devices exposed
- **THEN** the server binds and its `/health` endpoint answers

### Requirement: Container smoke test asserts the bundled engine
The Docker container CI smoke workflow SHALL fail when the built image lacks the engine, reports a version below the backend's minimum, or lacks the `ffmpeg` executable.

#### Scenario: Regression caught
- **WHEN** an image build omits or breaks the engine installation
- **THEN** the container smoke test fails with a message naming the missing check

### Requirement: Bundling does not alter host behavior
Outside containers, the gufo backend's resolution, version gating, remediation messaging, and the no-managed-download invariant SHALL be unchanged: an image-less host with no gufo install continues to report the actionable remediation and never attempts a download.

#### Scenario: Non-container host unaffected
- **WHEN** lemond runs directly on a host with no gufo binary and no overrides
- **THEN** pulling a gufo model fails with the build-instructions remediation before any weight download, exactly as before
