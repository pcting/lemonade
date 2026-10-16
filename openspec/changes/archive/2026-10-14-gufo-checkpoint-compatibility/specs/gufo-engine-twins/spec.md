## MODIFIED Requirements

### Requirement: Servability gate
Twins SHALL be produced only for checkpoint families gufo supports — the model families the gufo catalog declares (Qwen3.8 chat families, DeepSeek-V4-Flash, Qwen3 ASR/TTS, Qwen-Image) including their variants — and SHALL NOT be produced for arbitrary checkpoints. Within a servable family, a twin SHALL additionally be produced only when the gufo checkpoint compatibility verdict accepts it: for an artifact already on disk, every tensor group it declares (in the main checkpoint and in a declared draft companion) must be one the engine loads in that form; before a download, one whose variant matches the catalog entry's declared variant (or a directory-shaped catalog entry, where the repository is the artifact). The family the verdict resolves an artifact to SHALL be determined by the same repository membership the family gate uses, so the two can never disagree about whether a repository is gufo's. A twin SHALL NOT be offered for an artifact the engine refuses, whatever the reason recorded by the verdict.

#### Scenario: Unsupported checkpoint
- **WHEN** a registered model's main checkpoint belongs to a family gufo does not serve
- **THEN** no gufo twin is produced for it

#### Scenario: Supported variant of a servable family
- **WHEN** a registered model uses a readable quant variant (`UD-Q8_K_XL`, `Q8_0`, …) of a gufo-servable repository
- **THEN** a gufo twin is produced `main` its variant, alongside the existing entries for the repository's other variants

#### Scenario: Unreadable quantization of a servable family
- **WHEN** a registered model's main checkpoint belongs to a gufo-servable repository but a tensor group uses a storage type the engine does not load in that group (for example `UD-Q3_K_XL` of the Flash-Next family, whose experts are `IQ3_XXS`)
- **THEN** no gufo twin is produced for it, so the model list never offers an engine/quantization combination that cannot load

#### Scenario: Split-layout artifact is judged on its tensor shard
- **WHEN** a downloaded candidate's first shard carries only metadata (zero tensors) and the following shard carries the tensors
- **THEN** the tensor groups of the following shard decide servability

#### Scenario: Draft companion makes an otherwise readable main unloadable
- **WHEN** a registered model's main checkpoint passes the verdict but its `draft` companion declares a tensor format the engine rejects for that group (for example an MTP draft whose mixer tensors are `Q4_K`)
- **THEN** no gufo twin is produced, and the refusal is diagnosable from the recorded reason

#### Scenario: Group the manifest does not describe
- **WHEN** a downloaded candidate declares a tensor whose name matches no group the manifest describes for that family
- **THEN** the candidate is still treated as servable, leaving the engine's load error as the backstop
