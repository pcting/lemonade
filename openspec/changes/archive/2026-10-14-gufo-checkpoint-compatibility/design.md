## Context

`GufoServer` already implements the `BackendOps::serves_model_checkpoint()` hook added by `gufo-engine-twins`. Today it answers "can gufo read this?" with `gufo_supported_ggml_type()` — a switch over GGML type ids that mirrors nothing more specific than gufo's `core::GgmlType` enum — applied to *every tensor the file declares*. `read_gguf_tensor_types()` parses names and dims and throws them away.

The engine is stricter in some places and looser in others, and the strictness is **positional**:

| Engine loader | How it decides |
|---|---|
| `gufo/src/models/qwen/weights.cpp` | `SupportsTensorType(type, TensorRole)` over four roles (`:59-90`). `kNorm` **and** `kSsmParameter` are **F32 only** (`:73-77`); `kProjection` (`:78-87`) and `kEmbedding` (`:62-72`) take curated K-quant/I-quant sets and reject Q4_0/Q4_1/Q5_0/Q5_1/Q8_1/Q2_K. Roles are bound to names at the `ValidateTensor` call sites, **not** by name shape: `token_embd.weight` alone carries `kEmbedding`, while `output.weight` is `kProjection` (`:166`) — the two sets differ only by `Q8_K` |
| `gufo/src/models/qwen38_flash_next/weights.cpp` | A `Binder::Get(name, cols, rows, experts, {accepted types})` per tensor. `dense` = Q8_0/BF16/F16/F32 and `experts` = **Q4_K/Q5_K/Q6_K/Q5_1/Q8_0** are declared once per layer (`:140-144`) and reused; `Mixer()` (`:117-131`) builds **both** `hc_attn*` and `hc_ffn*` (`:146-147`) with `*_norm` = F32, `*_down`/`*_up` = dense, `*_inject` = F32/Q8_0/BF16 (**no F16**, unlike its own `*_down`). `ssm_alpha`/`ssm_beta` = F32/BF16/Q8_0 (`:156-159`) while `ssm_conv1d`/`ssm_dt.bias`/`ssm_a`/`ssm_norm` = F32. Optional shapes (`linear`, `ple`, `nextn`) guard whole groups (`:149`, `:193`). Additionally `cols % format.block == 0` |
| both | They validate **the tensors they load**. Tensors the loader never looks up are not rejected |

Two facts from that table are load-bearing and easy to get wrong, so they are stated up front:

1. **A group is an engine role, not a name family.** `attn_qkv.weight` is `dense` (Q8_0/BF16/F16/F32) in Flash-Next but a `kProjection` in the dense family, whose set also admits Q4_K/Q3_K/IQ4_XS/… . The same tensor name therefore needs opposite rules in two families, which is why rules are keyed by family first and name pattern second.
2. **The engine's own sets are not closed under union.** Any sketch that merges two groups "because the lists look nearly the same" silently widens or narrows one of them. Every group in the manifest must come from one call site, never from a blend.

Three consequences drive the design:

1. The flat set produces **false offers** — live-verified for the `Q4_K_M` MTP mixer draft (D9.6, aborted at `blk.48.hc_attn_down.weight`) and derived from the quoted source for a Q4_0/Q2_K projection or embedding — and **false refusals** (one unused or unmatched tensor rejects a whole checkpoint, because the flat rule is *every declared tensor must be in the set*).
2. Routed experts are the one place where a K-quant is *required* (`experts` accepts Q4_K/Q5_K/Q6_K/Q5_1/Q8_0 but not BF16), so a manifest that narrows rules globally would break the checkpoints that work today.
3. `gufo-engine-twins` D9.6 concluded "no cheap generic check exists … the catalog stays the authority". That holds while the rules are code. As data, the check is a table lookup.

**Measured baseline.** The rule table that an earlier draft of this design sketched was transcribed into a throwaway GGUF inventory probe and run against the artifacts cached on the Strix Halo host. All three artifacts known to load were accepted (27B `UD-Q4_K_XL`, 866 tensors; the `z-lab` DFlash2 `Q4_K_M` draft, 81 tensors; `mmproj-BF16`, 334 tensors), so the permissive-unmatched rule does not hide working models. The same probe found 342 of the 27B's 866 tensor names unmatched, and reading the engine's `Get()` call sites afterwards found **four transcription errors in that sketch** — `output.weight` given the embedding set instead of the projection set, `hc_attn_inject.weight` given `F16` the engine refuses, `ssm_alpha`/`ssm_beta` given F32-only, and `hc_ffn*` missing entirely. That base rate is the input to D1: the risk in this design is not the matching algorithm, it is hand-copying the rules.

Constraints: registry rebuild runs on every model-list refresh; the gate must work on hosts with **no gufo binary** (unsupported hardware, pre-install), where the list still renders; verdicts are cached behind a mutex (invariant 7); `src/cpp/server/backends/vllm/**` is out of scope for this change.

## Goals / Non-Goals

**Goals:**
- One authoritative, reviewable description of gufo's format rules, version-pinned to the engine release it mirrors.
- Verdicts that are wrong in the *permissive* direction only: a tensor group the manifest does not describe never causes a refusal.
- The same verdict logic serving both the twins gate and catalog conformance, so a shipped pairing cannot silently become unloadable.
- Refusals explainable from logs alone.

**Non-Goals:**
- Mirroring gufo's **shape** checks (`Binder::Get`'s exact `cols/rows/experts`). A shape mismatch means the artifact is not that architecture at all, which the family gate owns; duplicating per-arch config into lemonade would be a second model registry.
- Mirroring the DS4 / Qwen / Flash-Next *numerical* differences, or per-build state-ABI identities.
- Any change to another backend, to gufo's own loader, or to the "catalog decides which companion ships" rule.
- Replacing gufo's load error as the backstop.

## Decisions

### D1. A lemonade-side manifest mirroring the engine, transcribed under a drift guard

`src/cpp/resources/gufo_compat.json`, embedded like the other server resources, parsed once at backend construction.

**Polarity, the axis that matters more than data-vs-code.** With unmatched-is-permissive (D2), the gate's exposure is exactly the transcription fidelity of the *matched* groups: a group the manifest describes is enforced as written, and the manifest is copied by hand from engine source. A denylist learned from load failures would be unfalsifiable in that direction — it can only ever be looser than the engine — but it cannot fix false offers, which is half the point of the change, and the engine itself is a whitelist, so mirroring it is what the truth looks like. The chosen shape therefore keeps the whitelist and buys down the copying risk instead:

- Every group carries `mirrors` as **file plus line range**, and one line of `why` quoting the engine's own comment where it has one (the F32-only norm rule, "HIP norm, convolution, and recurrence kernels consume these tensors as float pointers", is precisely the kind of constraint a later editor would otherwise relax).
- A `cpp-ci` test re-reads each cited range from the pinned engine checkout when it is present (`/home/pcting/workspace/gufo` on a dev box) and fails on drift between the declared type list and the list in the source. Where the checkout is absent the test skips loudly rather than passing silently.
- One group per engine call site, never a union of two groups. The four sketch errors found during review all came from blending or paraphrasing.
- `mirrors` is a machine-readable citation, not a footnote: `path:first-range, :more-ranges accepted at :range…`. The **first** range must yield a set of `GgmlType::k…` tokens *exactly equal* to the group's declared types — it is the mirrored expression itself. Every further range must yield a **subset** (empty is fine: those ranges are the call sites that bind the same expression to more names). A range wider than the group is precisely the blending hazard the bullet above forbids, so the drift test refuses it rather than merely reading it.

*Found by writing that second rule.* `scalar_f32` originally cited `weights.cpp:154-164` as its mirrored expression; re-reading that range yielded `{F32, BF16, Q8_0}`, because lines 156-159 belong to `ssm_alpha`/`ssm_beta`, which take `recurrent_gate`'s wider set. The blend would have made the group's citation unreadable as a mirror and quietly widened what a future editor could add. Re-cited to `:162` (a pure `{GgmlType::kF32}` site) with the remaining F32-only sites listed as `accepted at`, and the exclusion named in `why` so nobody "restores" those two lines. All 18 shipped groups now pass both halves of the rule.

*Considered and rejected — ask the engine:* shelling out per artifact is unusable here. Registry rebuild judges every candidate on every refresh, the binary is absent on non-gfx1151 hosts where the list must still render, and a subprocess per artifact would put inference-engine startup on the model-list path. Checked for an existing dry-run verb and found none: `gufo diagnose --validate-artifact` validates *benchmark JSON* against the diagnostics schema, and `--fingerprint` emits a machine identity — neither is a checkpoint verdict. *Deferred:* if gufo exposes a checkpoint-validation verb (worth requesting upstream, since it deletes the transcription risk entirely), the manifest becomes the offline fallback and the engine becomes authority — the interface shape (`verdict(artifact) -> accept|reject+reason`) is already that narrow.

*Considered and rejected — encode the rules in C++ as today:* same behavior, but every engine bump becomes a code change buried in a switch, and neither the catalog-conformance test nor the drift guard above would have anything declarative to read.

Six more went the same way while writing the manifest for real. Walking the call sites instead of paraphrasing the table found: `Bind()`'s top-level dense set accepts **Q6_K** and `Layer()`'s does not, so one `dense` group covering `token_embd.weight` and the layer projections is a blend of two expressions; `ffn_gate_inp_shexp.weight` is F32 while the `ffn_*_shexp` matrices beside it are dense, so a `*.ffn_*_shexp.weight` glob is wrong in the permissive direction; `per_layer_token_embd.weight` (IQ4_NL/BF16, the family's only IQ4_NL, bound only when the config declares a ple layer) was missing entirely; the `output_hc_*` and `nextn.hc_head_*` mixer instances and `nextn.enorm/hnorm/eh_proj` were missing; `Mixer()` emits `*_inject` only at its `with_inject=true` call sites, so `output_hc_inject.weight` and `nextn.hc_head_inject.weight` must stay unmatched **on purpose**; and the DFlash2 companion is a separate tensor namespace with its own, narrower sets (`src/models/qwen/dflash_weights.cpp:46-54` — matrices F32/BF16/Q8_0/Q4_K/Q6_K, vectors F32/BF16), not a partial checkpoint of its host. Judging that companion under the host family's groups would both offer companions the engine refuses (Q3_K matrices) and refuse ones it accepts (BF16 norms).

Re-checked afterwards: `deepseek_v4_flash` has no per-tensor-group whitelist at all — only a flat decoder set in `runtime/tensor_types.h` (F32, F16, Q8_0, Q2_K, Q4_K, IQ2_XXS, I32) — so that family declares no groups and stays permissive rather than guessing.

**Task 1.8 result.** Cross-checking the finished manifest against the artifacts on disk: the 27B `UD-Q4_K_XL` claims 862 of 866 tensors and the DFlash2 `Q4_K_M` draft claims 81 of 81, with **no tensor of either file falling outside its winning group** — the two files known to load are accepted. The four unmatched 27B names are `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`, which the dense loader never looks up (the MTP head ships inside the main file), so permissive is the engine's own answer. For Flash-Next, which has no published artifact cached anywhere on this machine, the cross-check ran against the 59 names `weights.cpp` looks up: all claimed except the two intentionally-unmatched inject names.

### D2. Families resolved by repository; groups keyed by tensor name, first declared match wins, unmatched is permissive

The manifest ships at `src/cpp/resources/gufo_compat.json`. It declares `qwen38-dense`, `qwen38-flash-next`, `qwen38-dflash2` (reached as a companion, see D4) and `deepseek-v4-flash` (no groups, see D1). Every group has this shape — a name, the names it claims, the accepted types, and the single engine expression it mirrors:

```json
{ "name": "mixer_inject",
  "patterns": ["*.hc_attn_inject.weight", "*.hc_ffn_inject.weight"],
  "types": ["F32", "Q8_0", "BF16"],
  "mirrors": "src/models/qwen38_flash_next/weights.cpp:127-129",
  "why": "the inject projection drops F16, which its own *_down and *_up accept; Mixer() emits inject only for its with_inject=true call sites" }
```

The sketch this section carried while the change was being proposed is deliberately not reproduced: it invented repositories no registry row references (`peer-diffusion/…-PolyModel-GGUF`, `MRae/…-Deckard-GGUF`), claimed a `dense` group that blended two different engine expressions, and missed four groups. Those errors are recorded in D1 rather than kept as a plausible-looking example.

Semantics, pinned rather than left to taste:

- **Family resolution is by repository.** `verdict()` needs a family and the gate is handed a repository, so each family declares `repos`. A repository that matches no family — any `user.*` or `extra.*` registration pointing at some other publisher's quant — yields **accept** with the reason "family unresolved". Family membership here mirrors the identity rule `gufo-engine-twins` already uses to group twins; it does not invent a second one. `repos` lists only repositories the built-in catalog or the container seed actually names — declaring one nothing ships would give the gate jurisdiction over an artifact nobody has.
- **`exact` before `patterns`.** An exact tensor name always beats any glob, so `output.weight` can never be swallowed by the `dense`/`projection` glob list of the same family.
- **First declared group wins.** `patterns` are fnmatch-style with `*`, evaluated in declared array order, and only the first match decides — no "specificity" heuristic, because specificity is not well defined over globs and the engine's own precedence is call-site order, not pattern shape. Declaration order is therefore load-bearing and reviewable top-to-bottom: `mixer_norm` before `mixer_lowrank`, `recurrent_gate` before anything that could claim a recurrent scalar. Overlap stayed *possible* while the manifest was a sketch — the sketch claimed 147 of the cached 27B's 866 names were double-claimed, because `normalization` used a `*_norm.weight` glob and `projection` used `blk.*.ffn_*.weight`, and `*` spans dots. Deriving each group's names from the engine's role bindings instead of from name families removed the contention: measured **zero** names claimed by two groups across the 27B (866), the DFlash2 draft (81) and the Flash-Next engine name set (59). Precedence is still pinned by test, because a future pattern can reintroduce overlap in one line and the current array's order is the only thing preventing it — `test/cpp/fixtures/gufo_compat/precedence.json` records the winning group for all 107 distinct name shapes, so reordering the array cannot silently move a name.
- **`variant_groups` names the groups the engine loads conditionally** (`ple`, the `linear` vs full-attention branch, `nextn` heads). A group listed there — or a name in it guarded by an engine comment saying so — is judged only for the tensors the artifact actually declares: absence is never a refusal, and a present tensor of a disallowed type always is. Variant-guarded names stay *inside* a broader group (`ple_*` sit in `dense` and `scalar_f32`) rather than becoming separate groups, because a separate group could only weaken those, and weakening is the direction that costs a working twin. The one shipped exception, `ple_table`, exists because `per_layer_token_embd.weight` has its own accepted-types expression at `qwen38_flash_next/weights.cpp:281-282` (IQ4_NL/BF16) that no other group can express — D1's one-expression-per-group rule outranks the preference here. The field exists so the drift check knows which names to re-verify against `Layer()`'s `if` guards.
- **Unmatched name → no rule → acceptable**, matching the engine, which validates only the tensors it loads. This is the property that keeps a manifest gap from hiding working models, and it is why the four unmatched names in the measured 27B file (`blk.64.nextn.*`, the MTP head bundled inside a file the dense loader reads only as a host) are not a defect.
- **`block_divides_cols`** reproduces the engine's `cols % format.block == 0` check where the engine performs it, using dims the inventory already carries.

Per-tensor exact lists for every name were rejected because they would have to be duplicated per block count and arch variant; regular expressions were rejected because this runs over every tensor of every candidate on every registry rebuild and GGML naming is a stable convention.

### D3. Type names as strings, resolved at load; a broken manifest degrades to today's behavior

Manifest uses GGML names (`Q4_K`, `IQ3_XXS`) as `core::ToString()` prints them, so review against engine source is a text diff, and unknown ids (`MXFP4`) can be listed without touching a C++ enum. If the manifest fails to parse or names an unknown type, the backend logs one prominent error and falls back to the current flat check for the whole capability: **a broken table must never hide models.** (The vLLM backend's invalid-config handling backs up the offending file; here the table ships embedded in the binary, so the file is the source of truth and the fallback is enough.)

### D4. Draft companions judged with the family's companion rule set

Each family declares which companion kinds it has (`dflash`, `mtp`, `dspark`), keyed by the label the catalog already uses for speculative selection, so the same label drives both flag wiring and validation. A companion resolves its rules through its **own** repository first; only when that repository is unknown to the manifest does it fall back to the host family's groups through the declared label.

The sketch wrote `extends: "self"` for `dflash` and hedged that the companion "may need its own narrower sets later". Reading `src/models/qwen/dflash_weights.cpp` settles it now rather than later: a DFlash2 file is a separate tensor namespace (`selector_predecessor/successor/hidden.weight`, `fc.weight`, `fc.scale`, `enc.output_norm.weight`, `dflash.blk.N.*`, `*_conv_base`) whose matrices take only F32/BF16/Q8_0/Q4_K/Q6_K (`:46-50`) and whose vectors take only F32/BF16 (`:52-54`). Judged under `qwen38-dense` instead, `dflash.blk.0.attn_q.weight` would hit the host's `projection` glob and a Q3_K companion would be offered although the engine refuses it, while `blk.0.attn_q_norm.weight` would be demanded as F32 although the engine accepts BF16 — one wrong direction produces a twin that fails on load, the other hides a twin that works. So `dflash` names its own family. `mtp` keeps `extends: "self"`, because that companion really is the same architecture: `MtpWeights::Bind` calls `Layer()` with `nextn=true` (`qwen38_flash_next/weights.cpp:296-319`), so an MTP file speaks the host family's names and the `nextn.*` entries exist for it.

The verdict for an entry is `verdict(main) AND verdict(draft when declared)`. This is the concrete closure of D9.6: the user's registration pairing `UD-Q4_K_XL` main with `mtp-…-Q4_K_M.gguf` is refused because `blk.48.hc_attn_down.weight` (draft file) hits `mixer_lowrank`, which takes only Q8_0/BF16/F16/F32; the catalog's `mtp-…-shared-Q8_0.gguf` passes the same check. No inference across files is needed — each GGUF is judged on its own tensors, and the block index that belongs to the draft is not special.

### D5. Verdict reason is internal to the backend; the hook signature does not change

`BackendOps::serves_model_checkpoint()` keeps returning `bool`. The reason is composed inside `gufo_server.cpp` and logged there.

*Alternative considered:* widen the hook to return a verdict struct. It would give richer logging in `engine_twins.h`, but it changes a shared interface that every backend inherits for the benefit of one implementation, in a change whose scope excludes every other backend. Logging where the decision is made also keeps the log line and the rule table in one file.

### D6. Cache key gains the manifest version

The existing path+mtime+size verdict cache keys on `schema_version` (and `engine.pin`) too, so a manifest bump invalidates cached verdicts without flushing anything else. The mutex-guarded map stays as is (invariant 7).

### D7. Conformance is checked against declared pairings, keyed by the verbatim registry checkpoint string

Registry checkpoint keys come in two shapes today, sometimes in one entry:

```
main   unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL                                      ← variant alias
mmproj unsloth/Qwen3.8-27B-GGUF:mmproj-BF16.gguf                                ← file name
draft  unsloth/Qwen3.8-Flash-Next-GGUF:MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

So a pairing entry is keyed by **`repository` plus the registry checkpoint string verbatim**, not by a resolved file name and not by a normalized alias:

```json
"pairings": [
  { "main": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
    "draft": "z-lab/Qwen3.8-27B-DFlash2-GGUF:Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    "evidence": "inventory-measured",
    "verified": "2026-10-14",
    "artifacts": { "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                   "z-lab/Qwen3.8-27B-DFlash2-GGUF:Qwen3.8-27B-DFlash2-Q4_K_M.gguf": "Qwen3.8-27B-DFlash2-Q4_K_M.gguf" },
    "note": "…" }
]
```

That choice makes D9.6 fall out for free rather than needing a rule: `MTP/mtp-…-Q4_K_M.gguf` is simply a *different key* that no shipped entry may use, and the conformance test refuses any gufo row in `server_models.json` or `docker/user_models.json` whose `main`/`draft` string is absent from the table. Resolving aliases to file names was rejected — it would require a download to compare, and it would let an alias and a file name silently claim the same provenance.

Three things the shipped shape adds over the sketch this decision started from, each forced by what could actually be measured:

- **`evidence` states what the record rests on**: `engine-loaded` (a gufo process loaded the artifact), `inventory-measured` (the artifact was read and every tensor it declares fell inside the groups that claim it), `catalog-declared` (no artifact was reachable, so the record asserts only that these two strings are a pairing the registry intends to ship). "Verified against a real engine build" — the requirement as first written — turned out to be satisfiable for exactly one of the four shipped combinations: of the gufo artifacts, only the 27B `UD-Q4_K_XL`, its DFlash2 `Q4_K_M` draft and `mmproj-BF16` exist on any host this change was built on. A tier makes the honest statement per row instead of either overstating the table or dropping the cross-check. Writing the tiers honestly then changed what was reachable: the Strix Halo box turned out to be running a container on the pinned gufo 0.11.0 against the seeded Flash-Next pair, drafting at ~70% acceptance, which is `engine-loaded`, and that container's cache held the tensor-info of the other seeded artifacts, which is `inventory-measured` — a tier the measured `UD-Q8_K_XL` row then left as well, on the same day, when the packaged image was tested by loading that 31.5 GB quant and generating from it, which is what `engine-loaded` means and what a read alone cannot say. Only the `antirez/deepseek-v4-gguf` pair stays declared. So read "no artifact was reachable" as a claim about the host that wrote the row, current as of its `verified` date — that is exactly what makes a tier re-checkable instead of permanent, and it is why the drift test re-judges whatever a hub cache holds rather than trusting the tier in the file.
- **No `family` field.** The family is what `resolve_family()` already answers for the main's repository, and for a draft published as a family of its own a recorded family would be ambiguous. A second, checked-but-redundant copy could only rot.
- **Directory-shaped checkpoints are out of scope.** A transcription, TTS or image row names a model directory the engine loads whole — there is no tensor inventory to judge, so no pairing to record. The check does not simply skip anything without a `.gguf`: it asserts that everything it excludes is labelled `transcription`/`tts`/`image`/`embedding`/`reranking`, so a chat row cannot escape the check by spelling its `main` without a variant.

`GufoCompat` does not parse this section at all. Provenance is a claim about the registry, not about a tensor, so the CI test reads it as JSON and the engine never sees it — which is what makes a stale or wrong pairing row incapable of changing a verdict. (An earlier revision loaded a flat `repository`/`checkpoint`/`provenance` triple into the class; nothing consumed it, and keeping it would have implied the gate consults provenance.)

The C++ conformance test cannot pull 100 GB, so the table is what makes `docker/user_models.json` provably loadable in CI, and it is the artifact that records "this combination was measured", which types alone never do. Where the real artifacts *are* on disk — the Strix Halo dev container that already caches both families — the live check runs the actual verdict on the actual files, which is how D9.6 was found in the first place. That check is `--drift --hub <dir>` (or `HF_HUB_CACHE`), and it reports each entry it could not find as *not downloaded* rather than as a pass: measured here, 1 of 4 shipped entries judged, 3 not downloaded. Its negative control is a synthetic GGUF planted at the recorded main's path with Q4_0 on `blk.0.ffn_down.weight`, which the entry point fails with exit 1 naming the tensor, `projection` and the manifest stamp.


### D8. The verdict gates twins exposure only; catalog and pull stay ungated

`verdict()` is reached from `serves_model_checkpoint()`, which `engine_twins` calls. Built-in rows are still pulled, listed and loaded without a verdict — they are trusted because the pairing table says so, and the table is enforced by CI rather than at runtime. This keeps the runtime blast radius of a wrong manifest row to "a twin was not offered", never "a shipped model became uninstallable". It also means the manifest is a *prediction* about the engine, always backed by the load error, and the pairings table is where that prediction is tied to evidence row by row: of the four shipped combinations one is measured from files on disk and three are declared, and `evidence` says which is which rather than letting the table read as more certain than it is.

### D9. A reviewer reads one new backend, not a sweep across the tree

Adding a backend should be legible as *addition*. A reviewer opens the diff expecting `src/cpp/server/backends/gufo/` and a handful of registration lines; what distracts them is unrelated-looking edits elsewhere, especially where pre-existing code was restructured to accommodate gufo. So the branch holds itself to:

| Surface | Rule |
|---|---|
| Gufo's own files | Everything engine-specific lives here. Verdicts, reasons, manifest parsing, flag wiring — nothing about gufo is readable in a shared file |
| Shared C++ | Additive only: a new header, a new hook with a default no-op, one call site, one config key. **An existing statement is not rewritten unless its current behavior is wrong**, and then that edit is a standalone commit with its own message |
| Long dispatch chains | `server.cpp`'s and `runtime_config.cpp`'s `else if` ladders cannot take a new branch without appearing in the diff; keep each to the minimum (the config key is three added lines, no reindentation) and never reformat them |
| Generated artifacts | `defaults.json` and the `docs/` regions regenerated by `gen_backend_boilerplate.py`, never hand-edited, so their diff is provably machine output |
| Pre-existing Python integration tests | Where a shared test had to accommodate gufo, the change is capability-gated so a default configuration selects exactly what it selected before, and the edits are named in the PR body rather than absorbed; new coverage goes in a new file (`test_gufo_compat.cpp`) instead of extending an unrelated one |
| Non-code noise | `.opencode/` workflow files, `openspec/` artifacts and the `docker/user_models.json` seed are declared as separate commits so nobody reads them as C++ review surface |

*Considered and rejected:* keeping the whole gufo effort in one commit for "atomicity". It buries the two genuinely shared-code edits (`model_manager.cpp`'s twin pass, the `engine_twins` config key) under ~5,600 lines of backend, tests and docs, where a reviewer cannot find them.

Measured against that standard, the branch nearly satisfies the C++ half: across all shared C++, TypeScript and headers, **three** statements are deleted (`server.cpp`'s `error`-null test, one assertion line in `test_backend_mode_contract.cpp`, the `RecipeName` union gaining `gufo`), with `model_manager.cpp` and `runtime_config.cpp` as pure insertions.

**What the test edits are, and are not.** `server_tts.py`, `server_llm.py`, `server_sd.py` and `server_whisper.py` carry 54 deleted lines, and they are *not* gratuitous restructuring: every one is behind a default-preserving guard — `_is_kokoro()` returns the original `TTS_MODEL`/mp3 path unless the suite is run for gufo, `_image_model()` returns `SD_MODEL` for sd-cpp, whisper's existing `is_flm` test widened to `("flm", "gufo")`, and `server_llm.py`'s `no_reasoning()` fires only under `supports("reasoning_by_default")`. They are also necessary: gufo is the first reasoning-by-default, wav-only-TTS backend these suites exercise, and `test/utils/capabilities.py` is a registry whose `gufo` row is part of `feat(gufo)`.

So extracting them into a preceding commit — the obvious reading of "separate concern" — would be **misleading rather than clarifying**: it creates an ancestor containing a gate no backend uses, and moves `capabilities.py` away from the backend that requires it, which breaks `feat(gufo)`'s own CI run. Separating *concerns* and separating *commits* are different acts; here only the former applies, and it is satisfied by the guards.

What is left is genuinely small: disclosure, plus one stray commit that the branch has since dropped.

| Finding | Status |
|---|---|
| `bdd2b1b28 "install vim and tree"` — July 22, the branch's first commit, 2 lines adding debug tools to the image's apt list | **Resolved:** dropped by `git rebase --onto bdd2b1b28^ bdd2b1b28`. The conflict it produced (the ROCm bundle commit appended `rocm-core` et al. directly below those two lines) was resolved keeping the ROCm packages, so the image's apt list now matches `origin/main` plus the runtime libraries gufo actually needs |
| `Dockerfile` rewrites two existing `RUN` lines (apt list, `chmod +x ./lemond ./lemonade`) | Legitimate container work; name it in the PR body |
| `fb137d512` mixes the image-edit multipart contract with sd-suite gating | Both changes are gufo-facing (`gufo_server.cpp` + a gated test), so the diff already reads as one concern — no split needed |
| Net `server_models.json` effect vs `origin/main` | 6 rows added, **0 removed**: `450e2741c`'s 18-line Flash-Next deletion removed a row this branch itself had introduced, never published (`git log -S` over `origin/main` is empty). The PR body must say so, because "a built-in model was removed" is the wrong conclusion a reader would otherwise draw |

| | How | When |
|---|---|---|
| **Disclose (chosen)** | Keep the suite edits in `feat(gufo)`, and state the guarantee in the PR body: every edit to an existing test is capability-gated with the prior behavior as the default | Always |
| **Split (rejected)** | Move the four suites into their own commit | Only if a suite edit is found that is *not* gated — none exists today |

## Risks / Trade-offs

- **Transcription fidelity is the dominant risk** — four errors survived one hand-copy of this table during review (blended groups, a dropped `F16`, an omitted `hc_ffn` prefix). Mitigated by one group per engine call site, per-group `mirrors` line ranges, a `why` line quoting the engine's constraint, and the checkout-drift test in D1. Residual failure is still bounded: a mis-transcribed group can only affect names it matches.
- **Mirror drift on an engine bump** (gufo widens or narrows a whitelist) → every family block names the engine source and release it mirrors; `engine.pin` is asserted against `kMinGufoVersion` by the conformance test; `docs/dev/backends-reference.md` gains a "when you bump gufo" step. Failure mode is bounded by permissiveness: drift makes the gate no better than today, not actively wrong in the hiding direction — except when the engine *widens*, which costs a false refusal; that is mitigated by the reason log and the existing `engine_twins=false` switch.
- **Name patterns too narrow or too broad** → unmatched names are permissive by construction; the risk is a missed narrowing, not a hidden model. A pattern that is too broad can still refuse a loadable checkpoint, which is why refusals log the matched group name for audit and why first-match order is declared rather than inferred.
- **Second source of truth** → the manifest is explicitly a proxy for engine behavior, never a gate the engine defers to; load errors remain the backstop and the reason remains authoritative for users.
- **Rebuild cost** → unchanged order of magnitude: the tensor list is already parsed header-only per artifact (measured: 866 tensors in a 31 GB file, header read only), and verdicts are cached by path+mtime+size+manifest-version. Group matching is a linear scan of a per-family list per tensor.
- **Permissiveness limits the guarantee** → the twins gate still cannot promise "loads successfully"; it promises "no known-rejected format". Stated in the spec rather than papered over.

## Migration Plan

1. Land the manifest + inventory change + verdict code behind the existing gate; no new user-visible surface, no config change.
2. Run the conformance test, then the live check on the dev container. The live check may only *confirm* verdicts on files that exist — it must never claim a reject for a quant nobody published. Expected on the artifacts this host caches: 27B `UD-Q4_K_XL` accept, DFlash2 `Q4_K_M` draft accept, `mmproj-BF16` accept. The reject cases (27B Q4_0 projection, Flash-Next `UD-Q3_K_XL`, Flash-Next `UD-Q4_K_XL` + `Q4_K_M` draft) are synthetic-inventory cases from 5.3 plus the one D9.6 live result, not new measurements. Confirm the seeded `Qwen3.8-Flash-Next-GGUF-Gufo` row still loads and generates.
3. Rollback is a revert: the flat gate returns, no persisted state to migrate.
