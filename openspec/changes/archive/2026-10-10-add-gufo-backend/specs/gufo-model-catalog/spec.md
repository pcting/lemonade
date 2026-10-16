# gufo-model-catalog

## Purpose

Defines the set of models registered with recipe `gufo` — their identifiers, deployment-mode labels, and checkpoint shapes — so that pull, load, and routing behave correctly for gufo's fixed, hardware-specific model set.

## ADDED Requirements

### Requirement: Catalog membership and mode labels
The model registry SHALL contain gufo-recipe entries covering every modality the backend declares, with exactly one deployment-mode label per entry drawn from the gufo supported modes:

| Model id | Mode | Notes |
|---|---|---|
| `Qwen3.8-27B-GGUF` | chat | suggested; Q4_K_XL default variant; Q8_K_XL variant |
| `Qwen3.8-Flash-Next-GGUF` | chat | suggested; sharded Q4_K_XL; MTP draft sidecar |
| `DeepSeek-V4-Flash-GGUF` | chat | IQ2XXS quant; DSpark support sidecar |
| `Qwen3-ASR-1.7B` | transcription | BF16 HF directory |
| `Qwen3-TTS-1.7B-CustomVoice` | tts | HF directory |
| `Qwen3-TTS-1.7B-VoiceDesign` | tts | HF directory |
| `Qwen3-TTS-1.7B-Base` | tts | HF directory |
| `Qwen-Image-2.1` | image | full pipeline directory |

No gufo entry SHALL carry a mode label outside the backend's supported modes (e.g. embeddings, reranking), which gufo does not implement.

#### Scenario: Mode contract holds
- **WHEN** the backend mode-contract check runs over the registry
- **THEN** every gufo-recipe entry's deployment-mode label is one of chat, transcription, tts, or image, and each such mode is covered by at least one entry

#### Scenario: Pull a catalog model on supported hardware
- **WHEN** `POST /pull` requests `Qwen3.8-Flash-Next-GGUF` on a gfx1151 host with an installed gufo binary
- **THEN** the entry resolves, downloads stage, and a subsequent load succeeds

### Requirement: Checkpoint shapes per artifact type
Each catalog entry SHALL declare checkpoints in the shape gufo consumes:
- Single-file GGUF models declare `main` as `repo:VARIANT` (e.g. an unsloth GGUF quant).
- Sharded GGUF models declare `main` such that resolution yields the first shard path; gufo discovers the remaining shards from it.
- Sidecar artifacts (vision projector, speculative draft) are declared as named checkpoints (`mmproj`, `draft`) resolved to local paths at load time.
- Directory-based models (ASR, TTS, image pipeline) declare checkpoints that download the complete required file set of the Hugging Face repository (config plus all safetensors shards), and load SHALL pass the local directory path — not an individual file — to the subprocess.

#### Scenario: Sharded GGUF resolution
- **WHEN** `Qwen3.8-Flash-Next-GGUF` finishes downloading
- **THEN** load passes the first shard's local path to `gufo serve llm --model` and inference proceeds with all shards present

#### Scenario: Directory checkpoint
- **WHEN** `Qwen3-ASR-1.7B` finishes downloading
- **THEN** the staged directory contains the model config and both safetensors shards, and load passes that directory path to `gufo serve asr --model`

### Requirement: Speculative draft declarations
Chat entries that have a published speculative companion SHALL declare it as a `draft` checkpoint with a label identifying the speculative family so the backend selects the right gufo flags: `dflash` for Qwen3.8-27B (DFlash2), `mtp` for Qwen3.8-Flash-Next, `dspark` for DeepSeek-V4-Flash. Entries without a companion SHALL declare no draft checkpoint.

#### Scenario: Draft staged with parent
- **WHEN** `Qwen3.8-27B-GGUF` is pulled
- **THEN** the DFlash2 companion file is staged as the `draft` checkpoint and load wires it as `--speculative dflash2 --dflash-model <path>`

### Requirement: Pull and load gating on unsupported hosts
Pulling or loading any gufo-recipe model on a host that fails the backend's platform gate (non-Linux, non-x86_64, or non-gfx1151) or lacks a usable gufo binary SHALL fail with the backend's actionable availability error before multi-gigabyte downloads begin, so users are not charged bandwidth for models their machine cannot run.

#### Scenario: Early refusal before download
- **WHEN** `POST /pull` requests `DeepSeek-V4-Flash-GGUF` (~81 GiB) on a host without gfx1151
- **THEN** the request is refused immediately and no weight download starts

### Requirement: Context defaults
Gufo chat entries SHALL default to the model's native context (gufo's own default) when no `ctx_size` is configured, and an explicit `ctx_size` SHALL be forwarded as the per-session context capacity. The registry entry for Qwen3.8-Flash-Next SHALL reflect its 262,144-token native context as its maximum advertised context window.

#### Scenario: Native context by default
- **WHEN** `Qwen3.8-Flash-Next-GGUF` is loaded with no ctx_size override
- **THEN** no `--context` flag is passed and the served model reports its native context capacity
