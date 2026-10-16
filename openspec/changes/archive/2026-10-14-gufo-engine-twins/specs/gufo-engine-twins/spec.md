# gufo-engine-twins

## Purpose

Exposes locally registered or already-downloaded checkpoints that gufo can serve as additional gufo-recipe entries, so a user who has a model staged for another engine (typically llama.cpp) can select the gufo engine for it without re-registering by hand and without downloading anything twice.

## ADDED Requirements

### Requirement: Twins for servable local checkpoints

For every locally registered or locally downloaded model entry whose recipe is not `gufo` and whose main checkpoint belongs to a gufo-servable family, the registry SHALL expose an additional entry with recipe `gufo`. The twin SHALL copy the source entry's checkpoints (`main`, `mmproj`, and `draft` when present) verbatim, use a distinct id derived from the source entry's name and a `Gufo` qualifier, preserve every label of the source entry, ensure the deployment-mode label is present, and include a label identifying the gufo engine.

#### Scenario: Registered chat checkpoint gains a gufo twin
- **WHEN** the registry contains a `llamacpp` model whose main checkpoint is a gufo-servable GGUF (for example `unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`) and that checkpoint is already present in the cache
- **THEN** the model list contains a second entry with recipe `gufo`, the same checkpoint strings, a `gufo` label, and it reports as downloaded without starting any new download

#### Scenario: Twin is loadable on the gufo engine
- **WHEN** the gufo twin is loaded on a supported host
- **THEN** it runs on the gufo engine using the already-staged checkpoint files

### Requirement: Servability gate

Twins SHALL be produced only for checkpoint families gufo supports — the model families the gufo catalog declares (Qwen3.8 chat families, DeepSeek-V4-Flash, Qwen3 ASR/TTS, Qwen-Image) including their variants — and SHALL NOT be produced for arbitrary checkpoints. Within a servable family, a twin SHALL additionally be produced only when the engine can read that specific checkpoint: a downloaded artifact whose tensor storage types are all ones gufo implements, or, before a download, one whose variant matches the catalog entry's declared variant (or a directory-shaped catalog entry, where the repository is the artifact).

#### Scenario: Unsupported checkpoint
- **WHEN** a registered model's main checkpoint belongs to a family gufo does not serve
- **THEN** no gufo twin is produced for it

#### Scenario: Supported variant of a servable family
- **WHEN** a registered model uses a readable quant variant (`UD-Q8_K_XL`, `Q8_0`, …) of a gufo-servable repository
- **THEN** a gufo twin is produced `main` its variant, alongside the existing entries for the repository's other variants

#### Scenario: Unreadable quantization of a servable family
- **WHEN** a registered model's main checkpoint belongs to a gufo-servable repository but its tensor storage types include one gufo does not implement (for example `UD-Q3_K_XL` of the Flash-Next family, whose experts are `IQ3_XXS`)
- **THEN** no gufo twin is produced for it, so the model list never offers an engine/quantization combination that cannot load

#### Scenario: Split-layout artifact is judged on its tensor shard
- **WHEN** a downloaded candidate's first shard carries only metadata (zero tensors) and the following shard carries the tensors
- **THEN** the tensor types of the following shard decide servability

### Requirement: Additive, idempotent, non-shadowing

Twin synthesis SHALL leave the source entry unchanged and SHALL be idempotent across registry cache rebuilds. A twin SHALL never reuse an id already owned by a registered or discovered model; when the derived id is taken, the implementation SHALL qualify it rather than overwrite or silently drop another model.

#### Scenario: Rebuild stability
- **WHEN** the registry cache is rebuilt repeatedly
- **THEN** exactly one twin exists per servable source entry and the source entry is unchanged

#### Scenario: Id already taken
- **WHEN** a model with the twin's derived id already exists
- **THEN** the twin is registered under a qualified unique id and no existing model is replaced

### Requirement: Modality twins

Servable directory checkpoints for non-chat modalities SHALL gain twins with the matching deployment-mode label (`transcription`, `tts`, or `image`) so they are routable on the gufo engine.

#### Scenario: ASR directory checkpoint
- **WHEN** a Qwen3-ASR directory checkpoint is registered under a non-gufo recipe
- **THEN** its twin carries the `transcription` deployment label and routes to the gufo ASR subprocess

### Requirement: Disable switch

Twin synthesis SHALL be suppressible through runtime configuration (`engine_twins`), defaulting to enabled, so an operator can keep the model list free of generated entries.

#### Scenario: Switch off
- **WHEN** `engine_twins` is set to false
- **THEN** no twins are synthesized and the model list contains only explicitly registered and discovered entries

#### Scenario: Switch on (default)
- **WHEN** `engine_twins` is unset or true
- **THEN** twins are synthesized for servable local checkpoints
