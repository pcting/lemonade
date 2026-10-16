## Why

The servability gate that `gufo-engine-twins` shipped judges a checkpoint against a **flat set of GGML storage-type ids** — `gufo_supported_ggml_type()` accepts any type gufo's `core::GgmlType` enum can name. That is not what gufo enforces. gufo accepts types **per tensor group, per model package**:

- `gufo/src/models/qwen/weights.cpp` whitelists by `TensorRole`: `kNorm` and `kSsmParameter` are **F32 only** ("HIP norm, convolution, and recurrence kernels consume these tensors as float pointers"), `kProjection` accepts F32/BF16/Q8_K/Q8_0/Q5_K/Q6_K/Q4_K/Q3_K/IQ4_NL/IQ4_XS/IQ3_S — **not** Q4_0, Q4_1, Q5_0, Q5_1, Q8_1 or Q2_K, all of which the flat set admits.
- `gufo/src/models/qwen38_flash_next/weights.cpp` whitelists by tensor group: the `hc_attn_*` mixer tensors take only **Q8_0/BF16/F16/F32**, and any format whose block size does not divide `cols` is refused.

Change `gufo-engine-twins` recorded this as finding **D9.6** and concluded *"No cheap generic check exists … the catalog stays the authority for loadable combinations and the engine's load error remains the backstop."* The conclusion was reached with the check implemented as code. The whitelists above show a generic check **does** exist, provided the per-group rules are treated as data rather than as a type-id switch.

Both failure directions are live today:

- **False offer** — a `UD-Q4_K_XL` main paired with an `mtp-…-Q4_K_M.gguf` draft passes the flat gate and aborts at load with `tensor blk.48.hc_attn_down.weight has unsupported format Q4_K`. **Live-verified** (finding D9.6, Strix Halo). A Q4_0-quantized 27B main would pass for the same reason — that case is **derived from the quoted source**, not measured; no such published artifact was found to test against.
- **False refusal** — a single tensor in a group gufo never reads (a projector tensor left in the main file, a tensor the loader skips) hides a twin whose load would have succeeded, because the flat rule is *every declared tensor must be in the set*. Measured: 342 of the 866 tensor names in the cached 27B `UD-Q4_K_XL` belong to groups the engine has no rule for.

## What Changes

- Add a declarative, versioned **compatibility manifest** at `src/cpp/resources/gufo_compat.json`: for each gufo family, named tensor groups mapped to their accepted storage types, plus the shard/dimension rule and the rules that apply to a speculative draft companion. Families are selected by the **repositories** they cover. Every rule set carries the gufo release tag and the engine source file *and line range* it mirrors, so an engine bump forces re-validation and a reviewer can diff one group against one call site.
- `GufoServer` judges artifacts against the manifest instead of the flat type set. `read_gguf_tensor_types()` is extended into a tensor **inventory** (name + dims + type) so group membership can be decided; group membership is matched on tensor names against declared patterns in declared order, mirroring the engine's own loader logic. A repository no family claims — any third-party registration — is accepted permissively.
- **Draft checkpoints are judged too**, with the family's draft rule set — this is what closes D9.6: a twin that mirrors its source's draft verbatim stops being offered when that draft's mixer tensors are Q4_K.
- Unknown tensor groups inside a known family stay **permissive** (the engine's load error remains the backstop), and every refusal is logged with the offending tensor name, its type, the group it matched, and the manifest version — so a refusal is explainable from `lemonade logs` without reproducing a load.
- The verdict gates **twins exposure** only. Catalog rows, `pull` and `load` are unchanged at runtime; the pairings the catalog and the container seed ship are instead checked in CI against a table of recorded `main`/`draft` keys that states the evidence behind each row, so a shipped pairing cannot silently regress into a combination the engine rejects, and a pairing nobody could test is labelled as declared rather than measured.
- **Amends** `gufo-engine-twins` D9.6: the catalog stops being the *only* authority for loadable combinations; it remains the authority for *which* companion a catalog entry ships.
- Explicitly **out of scope**, no files touched: `src/cpp/server/backends/vllm/**` and `src/cpp/resources/vllm_model_config.json`. The separate finding that `VLLMServer::load()` passes the canonical id to `--served-model-name` (so a namespaced `user.*` / `extra.*` entry would 404 on a strict engine) is recorded here as a known gap and left for its own change.
- **Reviewer-legible footprint**: changes that exist only because gufo is a backend stay additive and small in shared files, so a reviewer reads this as one new backend plus a few hooks rather than a sweep across the tree. The branch's diff against `origin/main` is audited to that standard (design D9).

## Capabilities

### New Capabilities
- `gufo-checkpoint-compatibility`: what the gufo engine can load — a versioned, engine-pinned description of per-family tensor group format rules, and the verdict (accept / reject with a reason) it produces for a given artifact, including speculative draft companions.

### Modified Capabilities
- `gufo-engine-twins`: the *Servability gate* requirement's second gate now consults the checkpoint-compatibility verdict (per tensor group, per artifact, covering the draft companion) instead of a flat storage-type set; refusals are explained.
- `gufo-model-catalog`: *Speculative draft declarations* now requires every declared `main`/`draft` pairing to satisfy the compatibility manifest.

## Impact

**Code**
- `src/cpp/resources/gufo_compat.json` — new resource, embedded with the other server resources.
- `src/cpp/include/lemon/gguf_reader.h` — `GgufTensorTypes` gains tensor names and dims (inventory); reader already parses both, they are currently discarded.
- `src/cpp/server/backends/gufo/gufo_server.cpp` — manifest load + group matching + verdict logging; `gufo_supported_ggml_type()`'s flat switch is replaced by manifest lookup.
- `src/cpp/include/lemon/backends/backend_ops.h` — **no signature change**: `serves_model_checkpoint()` keeps its current shape and gufo keeps the reason internal, so no other backend is affected.

**Tests**
- `test/cpp/test_engine_twins.cpp` — Q4_K-draft rejection, per-group acceptance, unknown-group permissive path.
- `test/cpp/test_gufo_compat.cpp` — new: manifest parses, covers every family in the gufo catalog rows, and reproduces gufo's documented whitelists. Also re-reads each group's cited engine source range from a local gufo checkout and fails on drift (skips loudly when the checkout is absent). Must be labeled `cpp-ci` **and** added as a dependency of the `cpp-ci-tests` aggregate to run in CI.

**Docs**
- `docs/dev/backends-reference.md` — gufo section: how the manifest is maintained and what to update on a gufo version bump.

**Review footprint** (measured against `origin/main` at merge-base `b83cc1b55`, before this change)
- 12 code files carry **zero deleted lines** (+1,438 total): gufo's three files (+812), the new shared `engine_twins.h` (+189) and `gguf_reader.h` (+77), the `backend_ops.h` hook (+21), and new/extended test utilities. This includes `model_manager.cpp` (+31) and `runtime_config.cpp` (+12) — the twin pass and the config key are pure insertions.
- Only 7 code files delete anything (+199/−57), and **54 of the 57 deleted lines are the four pre-existing Python integration tests** (`server_sd.py` −20, `server_tts.py` −18, `server_llm.py` −14, `server_whisper.py` −2). The remaining three are single statements: `server.cpp`, `test_backend_mode_contract.cpp`, `recipeOptionsConfig.ts`.
- So the C++ already reads as "one new backend plus hooks". The Python edits that carry those deletions are capability-gated with the prior behavior as the default, and they are gufo-driven rather than incidental — gufo is the first reasoning-by-default, wav-only-TTS backend those suites exercise — so D9 requires them to be **disclosed**, not relocated (design D9).

**Engine side**
- No change to the `gufo` repository. The manifest is a mirror; drift is caught by the pinned tag, the per-group source citations and the catalog-conformance test, not by a build-time dependency on engine sources. Worth requesting upstream: a checkpoint-validation verb (none exists today — `diagnose --validate-artifact` validates benchmark JSON, not weights) would delete the transcription risk and demote the manifest to an offline fallback.
