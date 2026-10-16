## MODIFIED Requirements

### Requirement: Speculative draft declarations
Chat entries that have a published speculative companion SHALL declare it as a `draft` checkpoint with a label identifying the speculative family so the backend selects the right gufo flags: `dflash` for Qwen3.8-27B (DFlash2), `mtp` for Qwen3.8-Flash-Next checkpoints, `dspark` for DeepSeek-V4-Flash. Entries without a companion SHALL declare no draft checkpoint.

Every companion a gufo entry declares — built-in catalog row or registration shipped with the container image — SHALL pair with a `main` checkpoint in a combination the gufo checkpoint compatibility manifest accepts, so a shipped pairing is never one the engine refuses at load.

#### Scenario: Draft staged with parent
- **WHEN** `Qwen3.8-27B-GGUF` is pulled
- **THEN** the DFlash2 companion file is staged as the `draft` checkpoint and load wires it as `--speculative dflash2 --dflash-model <path>`

#### Scenario: Shipped pairings conform
- **WHEN** the compatibility conformance check runs over the built-in gufo catalog rows and the registrations bundled with the container image
- **THEN** every declared `main`/`draft` pairing is accepted, and the check names the entry and offending tensor for any pairing that is not

#### Scenario: A rejected companion cannot ship
- **WHEN** a pairing is declared whose draft carries a tensor format the engine rejects for that group (for example a Flash-Next MTP draft with `Q4_K` mixer tensors)
- **THEN** the conformance check fails for that entry
