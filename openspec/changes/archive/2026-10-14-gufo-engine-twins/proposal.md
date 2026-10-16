# Proposal: gufo-engine-twins

## Why

Users download checkpoints that gufo can serve (e.g. `unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`, already staged in the HF cache) but register them under `llamacpp`, because that is the recipe they were added with — and the model-options UI only ever shows the *recipe's device backends* (`Auto · Vulkan · ROCm · CPU`), never another engine. The engine is fixed at registration, the UI has no recipe edit for existing entries, and the gufo catalog covers only its own default quants. The practical result: a downloaded model cannot be configured to run on the gufo engine without hand-re-registering it (`lemonade pull user.X --checkpoint <repo:variant> --recipe gufo`).

Separately, the client's static recipe-options mirror (`recipeOptionsConfig.ts`) has no `gufo` entry, so even gufo-recipe models render an empty options panel (no context size, no gufo arguments).

## What Changes

- For every locally registered or downloaded model whose checkpoints gufo can serve, the registry exposes an equivalent **gufo-recipe twin** — `extra.<base>-Gufo`, checkpoints copied verbatim (the Hugging Face cache is shared, so no re-download), `suggested: true`, and a `gufo` label so the engine is visible in the list.
- The original entry is never modified or shadowed; twins are additive and idempotent across cache rebuilds.
- A servability gate limits twins to checkpoint families gufo actually supports (the catalog's chat/ASR/TTS/image families and their variants) — arbitrary GGUFs never gain a gufo twin. Within a supported family, a twin is offered only for a checkpoint the engine can really read (its tensor storage types for a downloaded artifact, the catalog's variant before a download), so the list never advertises an engine/quantization pair that fails at load.
- A configuration switch (`engine_twins`, default on) turns twin generation off.
- The client model-options surface exposes a gufo-recipe model's configurable options (context size, gufo backend, gufo arguments) — the frontend mirror entry our earlier change left out. Per repository policy, frontend changes are prepared for core-maintainer review; the fork may land it locally.
- Docs describe the twin behavior and the manual registration path for engines without a twin.
- The built-in `Qwen3.8-Flash-Next-GGUF` catalog row is removed; that family ships as a `recipe: gufo` registration seeded into the container image (`docker/user_models.json`), so a fresh config volume starts with the loadable checkpoint pairing instead of a catalog row that duplicates the user's own registration (design D10).

## Capabilities

### New Capabilities

- `gufo-engine-twins`: locally registered/downloaded checkpoints that gufo supports are automatically exposed as gufo-recipe entries, reusing cached artifacts, so clients can select the gufo engine for models the user already has.

### Modified Capabilities

- `gufo-backend` (ADDED requirement, no existing behavior changed): gufo-recipe models SHALL expose their configurable options in client model-option surfaces, so the recipe is configurable rather than showing an empty panel.
- `gufo-model-catalog` (MODIFIED requirement: catalog membership): the built-in rows no longer include `Qwen3.8-Flash-Next-GGUF`; the sharded-resolution and context-default requirements are restated over checkpoints rather than that row, which is how they are implemented.

## Impact

- **Server**: `model_manager.cpp` (registry cache build gains twin synthesis and the candidate filter), `backends/backend_ops.h` (the `serves_model_checkpoint` hook), `engine_twins.h` (catalog variant + filter), `gguf_reader.h` (tensor storage-type scan), gufo backend ops (servability matcher + readability check), `runtime_config.cpp` (switch), unit tests under `test/cpp/`.
- **Registry**: `src/cpp/resources/server_models.json` loses the `Qwen3.8-Flash-Next-GGUF` row (design D10).
- **Frontend (maintainer-scoped)**: `src/app/src/renderer/recipes/recipeOptionsConfig.ts` (add `gufo` to `RECIPE_OPTIONS_MAP` + `gufoBackend`/`gufoArgs` definitions); no other renderer change required for twins to appear.
- **Container**: `Dockerfile` seeds `docker/user_models.json` into the image's config path, so a new config volume starts with the starter registrations.
- **Docs**: `docs/dev/backends-reference.md` (gufo section: twins + manual registration), `docs/guide/install/docker.md` (container note).
- **Ops (verification)**: rebuild `lemonade:gufo`, redeploy `genai`, confirm the twin of the user's `UD-Q8_K_XL` entry appears, loads on gufo, and transcribes/chats without re-download.
- **Unaffected**: `backend_versions.json`, the subprocess model, and existing recipes' behavior.
