# Tasks: gufo-engine-twins

## 1. Servability matcher

- [x] 1.1 Add a `GufoOps` (or gufo-backend) helper `servable_checkpoint_repos()` that returns the repository set derived from the registry's builtin gufo entries (`main` checkpoint repository portion, before `:`), plus a `is_servable(repo)` predicate and a `deployment_label_for(repo)` lookup returning the matching catalog entry's mode label (design D2/D6)
- [x] 1.2 Unit-test the matcher under `test/cpp/` and wire it into the `cpp-ci` aggregate target: servable repo with an arbitrary variant (e.g. `unsloth/Qwen3.8-27B-GGUF` with `UD-Q8_K_XL`) matches; a non-gufo repository (e.g. LFM2) does not; ASR/TTS/image repositories return their modality labels

## 2. Twin synthesis in the registry cache build

- [x] 2.1 In `ModelManager`'s cache build, after all sources merge and before indexing: for each non-`gufo` entry whose `main` checkpoint repository `is_servable`, synthesize a twin — id `<EXTRA_MODEL_PREFIX><base>-Gufo`, checkpoints copied verbatim from the source, `recipe: gufo`, `suggested: true`, labels = the source's labels (deployment-mode label ensured present) plus the engine label, recipe options empty (design D3/D4)
- [x] 2.2 Insert twins through the existing collision-qualification path (never overwrite, never shadow an existing id; append `-2`, `-3`, … on conflict)
- [x] 2.3 Unit-test synthesis with a fake registry: twin appears for a servable `llamacpp` chat entry (checkpoint strings identical, no download triggered); no twin for a non-servable checkpoint; modality entry gains the right deployment label; repeated builds yield exactly one twin; a pre-existing `-Gufo` id forces qualification; the source entry is byte-identical before/after

## 3. Disable switch

- [x] 3.1 Add the `engine_twins` boolean to runtime configuration (default true), following the existing validation/round-trip patterns for server config keys; document it in the configuration reference's key list
- [x] 3.2 Unit-test the switch: disabled ⇒ no twins synthesized; default/unset ⇒ twins synthesized

## 4. Client model-option surface (maintainer-scoped, design D7)

- [x] 4.1 Prepare the `recipeOptionsConfig.ts` patch: add `gufo` to `RECIPE_OPTIONS_MAP` (`ctxSize`, `gufoBackend`, `gufoArgs`, `mergeArgs`, `pinned`, `saveOptions`) and the `gufoBackend`/`gufoArgs` definitions (`isBackendOption: true` for the backend key), mirroring the server's descriptor-driven keys
- [x] 4.2 Verify in a locally built image that a gufo model's options panel shows context size and gufo arguments (no device-backend selector: single backend, D8) and that saving `gufo_args` reaches the subprocess command line (spec: "Options round-trip")
- [x] 4.3 File/flag the mirror patch for core-maintainer review (AGENTS.md: UI changes are maintainer-owned); note in the patch description that gufo's single supported backend means the device-backend dropdown stays absent by design

## 5. Documentation

- [x] 5.1 `docs/dev/backends-reference.md` gufo section: describe twin behavior (`extra.<base>-Gufo`, additive, servability from the catalog, the `engine_twins` switch) and the manual equivalent (`lemonade pull user.X --checkpoint <repo:variant> --recipe gufo`);
- [x] 5.2 `docs/guide/install/docker.md`: one line in the gufo section that downloaded models of supported families appear with a `-Gufo` companion entry inside the container
- [x] 5.3 Re-run `python docs/tools/gen_backend_boilerplate.py --check` and confirm zero drift (docs edits touch non-generated regions only)

## 6. Verification and rollout (strixbeemo)

- [x] 6.1 Build `lemond` and run the full `ctest -L cpp-ci` suite; confirm the new unit tests pass and the baseline is unchanged (72/73 pre-existing network test)
- [x] 6.2 Rebuild `lemonade:gufo` with the change, redeploy `genai`, and confirm the user's `Qwen3.8-27B-GGUF-UD-Q8_K_XL` (registered `llamacpp`, downloaded) now has a visible `-Gufo` twin in the model list
- [x] 6.3 Functional proof: load the twin on the production container and run a chat completion (no re-download; confirm the gufo subprocess and telemetry show the gufo recipe), then unload cleanly
- [x] 6.4 Confirm the switch: set `engine_twins=false` on a test instance and verify the twin disappears from the list; restore the default

## 7. Final gates

- [x] 7.1 `openspec validate gufo-engine-twins --strict` passes
- [x] 7.2 `git status` clean of stray artifacts; `backend_versions.json` unchanged; the only catalog change is the Flash-Next row removal recorded in D10, with `openspec/specs/gufo-model-catalog/spec.md` updated to match the registry

## 8. Checkpoint readability gate (design D2, second gate)

- [x] 8.1 Extend `lemon/gguf_reader.h` with a tensor storage-type scan (one type id per tensor, plus split metadata) beside the existing metadata reader
- [x] 8.2 Add the backend hook `serves_model_checkpoint(info, repo)` (default: every candidate qualifies) and pass it into twin synthesis as the candidate filter, so only gufo needs an opinion
- [x] 8.3 Implement gufo's check: a downloaded artifact must declare only tensor types its own reader implements, following one split-stub shard when the first carries no tensors; an artifact the registry cannot resolve is looked up in the hub cache by variant first (a sibling variant's pull rewrites the snapshot manifest — D9.5); with nothing on disk, the catalog's declared variant must match; directory-shaped catalog entries accept on repository match. Results are cached by path+mtime+size
- [x] 8.4 Unit-test the filter (a refused candidate yields no twin while its siblings still twin) and prove the reader against the real artifacts: the `UD-Q3_K_XL` first shard is a zero-tensor stub whose sibling declares `IQ3_XXS` (refused), the 27B `UD-Q8_K_XL` declares `F32/Q8_0/BF16` (kept)
- [x] 8.5 Prove the gate's decision path against a staged cache tree that mirrors the production one (clobbered manifest included): the unresolved-but-present 27B `UD-Q8_K_XL` is admitted, a `.partial`-only `Q8_0` is refused, the `UD-Q3_K_XL` shard directory is refused, an absent non-catalog variant is refused and an absent catalog variant admitted
- [x] 8.6 Rebuild/redeploy and confirm the `…-UD-Q3_K_XL-LlamaCPP` `-Gufo` twin is gone from the model list while the 27B `UD-Q8_K_XL` and `…-UD-Q4_K_XL` twins are present

## 9. Built-in Flash-Next row removal and image starter registry (design D10)

- [x] 9.1 Drop the built-in `Qwen3.8-Flash-Next-GGUF` row from `src/cpp/resources/server_models.json`; the family is served by the `Qwen3.8-Flash-Next-GGUF-Gufo` registration, so gate 1 no longer auto-twins it
- [x] 9.2 Seed `docker/user_models.json` in the Dockerfile runtime stage, so a freshly created config volume starts with the starter registrations and an existing volume keeps its own file
- [x] 9.3 Bring `openspec/specs/gufo-model-catalog/spec.md` to the implemented state (membership table without Flash-Next, sharded-resolution and context-default requirements restated over checkpoints)
- [x] 9.4 Confirm `python docs/tools/gen_backend_boilerplate.py --check` reports zero drift and `test_engine_twins` passes after the registry change
