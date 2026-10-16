# Design: gufo-engine-twins

## Context

Facts established while diagnosing the reported gap (strixbeemo production container, 2026-10-11):

- **Registrations fix the engine.** Locally added models live in `user_models.json` (e.g. `Qwen3.8-27B-GGUF-UD-Q8_K_XL` with `main: unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`, `recipe: llamacpp`, labels `[chat, custom]`). The recipe is part of the registration; nothing in the UI rewrites it for an existing entry.
- **The options dropdown is device backends, not engines.** `ModelOptionsModal` renders `supportedRecipes[recipe]` — for `llamacpp` that is `Auto · Vulkan · ROCm · CPU`. gufo cannot be a value there; it is a different recipe.
- **The gufo catalog only covers its own quants.** The gufo 27B entry uses `UD-Q4_K_XL`; the user's downloads are `UD-Q8_K_XL` / `Q8_0`, so no gufo entry matches them.
- **Discovery is not a hook for this.** The HF-cache/extra-dir scanner (`discover_extra_models`) hardcodes `EXTRA_MODEL_RECIPE = "llamacpp"`, and gufo's descriptor has `dynamic_models: false` (per-descriptor discovery is for provider-native catalogs like flm/cloud, not local files).
- **Registry sources** merged into the cache: builtin (`server_models.json`), user registrations, extra-dir discoveries, and dynamic per-descriptor discovery. `add_extra_model` already implements collision-safe id qualification for synthesized entries.
- **Client mirror drift.** `recipeOptionsConfig.ts` (`RECIPE_OPTIONS_MAP`, `OPTION_DEFINITIONS`) has no `gufo` entry and no recipe-tail keys; `getOptionsForRecipe('gufo')` returns `[]`, so gufo models render an empty options panel. Server-side recipe keys are descriptor-driven (`get_keys_for_recipe`) and already expose `ctx_size` + `gufo_backend` + `gufo_args`.
- **Single-backend recipes render no device selector** by design (`hasMultipleBackends` requires >1). gufo has one backend (`rocm`); this is expected, not a bug.

## Goals / Non-Goals

**Goals**

- A downloaded/registered checkpoint that gufo can serve gains a selectable gufo-recipe entry, with zero re-download and no operator action.
- The source entry, its id, and its behavior are untouched.
- The generated entries are conservative (only gufo-servable families), idempotent, and suppressible.
- gufo-recipe models (twins and catalog entries) expose their configurable options in clients.

**Non-Goals**

- Changing the model options' *device-backend* dropdown semantics (engines are not device backends).
- A UI recipe editor for existing registrations (frontend, core-maintainer scope — noted as the long-term upstream UX).
- Auto-assigning the gufo recipe to arbitrary discovered GGUFs (unsafe: gufo serves a fixed family set).
- Adding new quants to the catalog as a substitute for twins.

## Decisions

### D1. Twin synthesis during the registry cache build (chosen)

After all sources merge and before indexing, iterate the merged entries; for each non-gufo entry whose main checkpoint's repository is gufo-servable, synthesize a twin and insert it.

| Alternative | Why rejected |
|---|---|
| UI recipe editor for existing models | Frontend; AGENTS.md scopes UI to core maintainers; larger change; doesn't help API/CLI clients |
| Documented `pull --recipe gufo` only | Works today but is manual, per-model, and leaves the user's list unchanged (not "an option in the UI") |
| Per-descriptor `discover_models` hook | Wrong layer: local registrations/extra-dir models are scanned by the hardcoded extra-dir path, not the dynamic discovery loop |
| Auto-recipe guessing for discovered GGUFs | gufo supports a fixed family set; guessing produces entries that fail at load |

### D2. Servability derives from the gufo catalog itself, then from the engine's own reader

Two gates, both needed.

**Gate 1 — the catalog.** The matcher takes the repository portion (before `:`) of every **builtin gufo entry** and treats any checkpoint whose repository matches as servable; the twin inherits that catalog entry's deployment-mode label **and its pinned variant**. No second allowlist is maintained — adding a gufo catalog model automatically extends twin coverage, and removing one narrows it.

**Gate 2 — the engine's reader.** A repository match is not sufficient: gufo implements only the quantizations in its `GgmlType` enum, so a sibling quant of a servable family can still be unreadable (the reported case: `UD-Q3_K_XL` of `unsloth/Qwen3.8-Flash-Next-GGUF`, whose experts are `IQ3_XXS`, absent from gufo's quant table — its load aborts at parse time). Before generating a twin the engine is asked whether it can actually serve that candidate:

- **Downloaded candidate** → its tensor storage types are read (`lemon/gguf_reader.h`'s tensor-type scan) and every type must be one gufo implements. Unsloth-style layouts keep the KV header and split metadata in shard 1 and the tensors in the following shards, so a zero-tensor first shard follows one sibling; unreadable headers stay permissive and let the engine's own load error be the backstop.
- **Candidate the registry cannot resolve** → resolution is manifest-driven, and a pull of one variant of a repository rewrites the manifest for the whole snapshot (see D9.5), so a sibling variant that is on disk can resolve to nothing. The artifact is therefore looked up in the hub cache directly (the variant names either a file/directory under the snapshot or part of a file name) and judged by its tensor types like any other downloaded file.
- **Nothing on disk at all** → the catalog's declared variant is the only verifiable signal: the candidate twins only when its variant matches the catalog's (or the catalog entry is directory-shaped, where the repository itself is the artifact). This keeps a twin from advertising a quant nobody has, whose load would fail after a multi-gigabyte pull.

Note what this deliberately does **not** do: exclude by variant name. `UD-Q8_K_XL` of the 27B family is not the catalog's pinned variant but is perfectly readable, and it twins — exactly the case the mechanism exists for. The decision is made on what the artifact contains, not on how it is labelled.

### D3. Identity: `extra.<base>-Gufo`, collision-qualified

Twins are named `<EXTRA_MODEL_PREFIX><base>-Gufo` and inserted with the existing collision-safe qualification used by `add_extra_model` (append `-2`, `-3`, … rather than overwrite). `suggested` is true (so the entry is visible where suggested models are listed) and labels **preserve every source label** — `vision`, `mtp`, `reasoning`, `tool-calling`, `custom`, … — with the deployment-mode label ensured present and the engine label (`gufo`) added. Preserving them matters: client capability badges are rendered from labels, and the twin serves the same artifacts (mmproj, draft) as its source. The qualifier makes the engine obvious in the id and prevents shadowing any existing model.

### D4. Checkpoints copied verbatim; no invented sidecars

The twin copies the source's `checkpoints` map exactly (`main`, `mmproj`, `draft`). Because resolution reuses the HF cache, an already-downloaded source yields an immediately loadable twin and no new network traffic. A twin never fabricates a speculative draft the source entry does not have — the engine simply runs without speculation, and the catalog's own entries still provide the drafted variants.

### D5. `engine_twins` runtime switch, default on

A server-side boolean in runtime configuration (`engine_twins`, default true) suppresses synthesis. Rationale: the entries are generated, not registered; operators need a way to keep the list purely explicit without renaming models. Default-on matches the request that started this change.

### D6. Modality behavior

Twins inherit the deployment label of the matching catalog entry, so ASR/TTS/image registrations produce entries that route to the correct gufo subprocess (`transcription`/`tts`/`image`) instead of defaulting to `chat`.

### D7. Client options surface (maintainer-scoped)

The frontend's `recipeOptionsConfig.ts` needs a `gufo` entry mirroring the server's descriptor-driven keys (`ctxSize`, `gufoBackend`, `gufoArgs`). This is prepared as a reviewable patch; per AGENTS.md the frontend is core-maintainer territory, and the fork may land it locally. Twins function without it (loadable by name from the CLI/API and the model list); the options panel stays empty until the mirror lands. Note for reviewers: gufo's single supported backend means the *device*-backend dropdown remains absent by design.

### D8. Rejected: listing engines in the backend dropdown

Tempting ("gufo should be in the list") but wrong: the dropdown's contract is the recipe's device backends. Adding engine names there would break per-recipe option round-trips and the `validate_backend_choice` checks. The twin mechanism gives the same outcome through the engine dimension the architecture actually has.

### D9. Implementation findings (fixed and verified live)

Findings that surfaced only once twins ran against the real container:

1. **Twins reported not-downloaded** while their source entry reported downloaded. `are_required_checkpoints_complete()` reads `ModelInfo::resolved_paths`, which ingestion fills before the download pass; clearing the map in synthesis made the twin look absent. `build_cache()` now calls `resolve_all_model_paths()` for each generated twin, so the download pass sees the same cache state as the source.
2. **Namespaced ids broke gufo's strict model check.** The router loads by canonical id (`extra.*` for twins) while clients send the public alias; gufo 404s requests whose `model` field mismatches, so the twin was unreachable for chat. `GufoServer::load()` now serves `ModelManager::get_public_model_name(canonical)` (falling back to the canonical id), which is also what makes extra-dir-discovered models safe on a strict backend.
3. **Twins under-reported their capabilities.** Synthesis replaced the source's label list with `mode + engine + custom`, so a twin of a vision/MTP model lost the `vision`/`mtp`/`reasoning`/`tool-calling` badges in clients even though it carries the same mmproj/draft artifacts. Caught by comparing the live model list (source row showed Vision, its `-Gufo` twin did not); twins now preserve every source label and add the engine label.
4. **Twins were offered for quantizations the engine cannot read.** Repository-level matching generated a `-Gufo` twin for `UD-Q3_K_XL` of the Flash-Next family, and loading it failed with `Invalid GGUF tensor shape or unsupported storage type: blk.0.ffn_gate_exps.weight` — its MoE experts are `IQ3_XXS`, a type gufo's reader has no dequant path for. Settled by D2's second gate (tensor-type check for downloaded artifacts, catalog-variant equality before a download). Verified on the real files with the reader: the Q3 first shard is a zero-tensor stub whose sibling shard declares `IQ3_XXS` → refused; the working 27B `UD-Q8_K_XL` declares only `F32/Q8_0/BF16` → twin kept.
5. **A pull of one variant rewrites the manifest for the whole snapshot.** `.download_manifest.json` records the file list of the pull that wrote it, keyed by repository snapshot, so pulling `Qwen3.8-27B-GGUF:Q8_0` (triggered by a client's auto-load request) dropped `Qwen3.8-27B-UD-Q8_K_XL.gguf` from the manifest while the 31 GB file stayed on disk. Two consequences: that entry (and its twin) read as not-downloaded, and the readability gate — which asks `resolved_path("main")` — saw an unresolvable artifact and applied the not-downloaded rule, hiding a twin for a model that is demonstrably loadable. Two fixes: the gate now probes the hub cache for the named variant and judges the file it finds (D2), and the twin survives a manifest rewrite as long as the artifact is on disk. The registry's own downloaded/unresolved state for such a sibling variant remains upstream behavior and is reported, not changed here. A *stale* manifest (left behind by an interrupted completion) additionally makes the whole snapshot read as not-downloaded, including the variant it lists; `model_manager.cpp`'s completeness check treats any manifest as "interrupted download", so removing the stale file restores every variant in that snapshot.
6. **The storage-type gate cannot capture per-tensor format expectations.** `Qwen3.8-Flash-Next`'s `UD-Q4_K_XL` passes the gate — every type it declares is one gufo implements — yet loading failed with `Failed to create Qwen3.8-Flash-Next model: tensor blk.48.hc_attn_down.weight has unsupported format Q4_K`. gufo's Flash-Next loader accepts only `Q8_0/BF16/F16/F32` for the `hc_attn_*` mixer tensors (`src/models/qwen38_flash_next/weights.cpp`, `Mixer()`), and `blk.48` belongs to the **MTP draft**, not the main model (the main's blocks stop at 47, all `hc_attn_down` = `Q8_0`). The user's registration pairs the UD-Q4 main with `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, whose `blk.48.hc_attn_down` is `Q4_K` → rejected; the catalog entry pairs the same main with `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (`Q8_0`) → loads. This is why the catalog's draft choice is not arbitrary, and why a twin that mirrors a source's draft verbatim (D4) can be unloadable even when its main is fine. No cheap generic check exists (the rejected type is one gufo implements elsewhere), so the catalog stays the authority for loadable combinations and the engine's load error remains the backstop — verified live: the catalog entry loaded in 18 s and generated at 39–42 tok/s with MTP acceptance 52/88, 83 GiB of the 108 GiB GTT pool in use.

### D10. The built-in Flash-Next row is dropped; the family ships as a registration

`Qwen3.8-Flash-Next-GGUF` was removed from the built-in catalog (`src/cpp/resources/server_models.json`). Once twins exist, a built-in row for a 107 GB family duplicates whatever the user registers themselves: the same checkpoint pair appears twice, one row permanently "not downloaded" for anyone who pulled it under another engine. The container image ships the equivalent registration instead — `docker/user_models.json` carries `Qwen3.8-Flash-Next-GGUF-Gufo` (recipe `gufo`, `UD-Q4_K_XL` main with the `shared-Q8_0` MTP draft, i.e. the loadable pairing from D9.6), COPYed into the image's config path so a freshly created config volume starts with these rows while an existing volume keeps its own file.

The consequence is taken deliberately: `unsloth/Qwen3.8-Flash-Next-GGUF` is no longer a built-in gufo row, so gate 1 (D2) no longer marks it servable and the family gains no automatic twins. Servability follows the catalog in both directions — *"removing one narrows it"* — and nobody is offered a 107 GB engine they did not ask for. Reaching the family means registering it or starting from the image's seeded row, and what ships is the draft pairing proven to load.

## Risks / Trade-offs

- **List growth**: twins double the rows for servable checkpoints (source + `-Gufo`). Mitigated by the `-Gufo` id, the `gufo` label, and the D5 switch; judged acceptable because the alternative is an invisible engine option.
- **Servability drift via the catalog (D2)**: a mistaken gufo catalog entry would widen twin coverage to families gufo cannot actually load. Narrowed twice over: the catalog stays the curated source of truth, and the engine's own reader vets each candidate's tensor types before a twin is generated. The residual case is a candidate whose file cannot be read at all (absent, corrupt, or a header the scan refuses) — judged permissively and caught at load time by the engine's own error.
- **Frontend dependency (D7)**: full "configure it in the UI" requires the mirror patch; without it the engine is selectable but the options panel is empty. Flagged as maintainer-owned so it is not silently forgotten.
- **Naming collisions**: qualified ids (D3) can produce surprising suffixes when a user already used `-Gufo`; qualification keeps behavior safe at the cost of a less pretty id.
- **Stale state**: synthesis is derived, so deleting the source entry removes its twin at the next rebuild; no orphan cleanup code is needed.
- **Seeded registrations are image policy (D10)**: `docker/user_models.json` is the starter registry for a *new* config volume only — Docker copies it while the volume is empty, so editing it never rewrites an existing deployment, and its machine-flavored entries (local NPU paths, cloud registrations) travel with the image.
- **Twin of an undownloaded source**: the twin shows as not-downloaded and pulling it downloads the same files (shared HF cache dedups); acceptable, and consistent with how `extra.*`-style entries behave.
