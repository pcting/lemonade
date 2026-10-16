## Why

The image ships gufo `v0.11.0` while upstream is at `v0.15.0` — four releases of work aimed at the family this table gates: Flash-Next serving wired to a component cache, cache pages pinned without compacting host memory, faster wide-prefill kernels, a ROCm managed-alloc fallback, and serve fixes (response schemas without `additionalProperties`, IPv6 listen addresses). The compatibility manifest is *engine-pinned* by design, so holding a stale pin means the table describes an engine nobody runs.

The bump is cheap on evidence — a re-diff of every mirrored call site shows the rules did not change across the whole range — but it exposes a coupling worth fixing: `gufo.h` declares one constant to be both the runtime minimum version *and* "the engine revision the checkpoint compatibility manifest transcribes", and the conformance check asserts the two agree. Under that invariant a pin bump drags the runtime floor up, refusing engines that still satisfy the floor's own stated criterion.

## What Changes

- Bump the bundled engine pin to upstream `v0.15.0` (`Dockerfile` `ARG GUFO_VERSION`, the manifest's derived-from comment, docs and fixtures that name it).
- Keep the manifest's declared floor at `0.11.0`, backed by evidence: every file the table mirrors (`src/models/qwen/weights.cpp`, `dflash_weights.cpp`, `src/models/qwen38_flash_next/weights.cpp`) is byte-identical between `v0.11.0` and `v0.15.0`.
- Move the runtime minimum out of that coupling: the manifest's floor must equal the backend's minimum version, while the mirrored release must equal the release the image bundles. A bump then moves the mirror without silently moving the floor.
- Re-earn the two `engine-loaded` pairing records under the new release, and have each record name the release that produced its evidence.
- **BREAKING** for nobody in practice, but state it: nothing in `0.11.0`…`0.14.0` is refused by this change; a deployment that wants the new engine still has to rebuild and redeploy its image.

## Capabilities

### New Capabilities

- None — this change modifies an existing capability.

### Modified Capabilities

- `gufo-checkpoint-compatibility`: the manifest's mirrored release and its floor get distinct, separately checkable meanings, and a change to the mirrored release obliges re-verifying records whose evidence came from the engine.

`gufo-container-image` needs no delta: its requirement is already "a build-configurable pin (repository URL and version tag) … as an exact release — not a `development` build", and `v0.15.0` satisfies it without changing what is required.

## Impact

- `Dockerfile` (`ARG GUFO_VERSION` and the comment above it), `src/cpp/resources/gufo_compat.json` (`engine.pin`, the justification for `engine.min_supported`, `_comment`, and the two pairing notes that name the old release), `src/cpp/include/lemon/backends/gufo/gufo.h` (the stale half of the `kMinGufoVersion` comment), the conformance check in `test/cpp/test_gufo_compat.cpp` together with `test/cpp/fixtures/gufo_compat/precedence.json`, `docs/dev/backends-reference.md`, `docs/guide/install/docker.md`.
- Hosts with a system-managed engine keep their floor: any `gufo` at or above `0.11.0` is still accepted, which the audited surface supports — the flags this backend drives (`--served-model-name`, `--api-key`) are present in both releases' `serve.cpp`, and the only `http_server.hpp` change in the range is an additive IPv6 listen-address check.
- The deployed container only gains the engine after a rebuild and redeploy; the previous image tag stays in place for rollback.
- Out of repo: the `komodo-halo` compose comment names the old engine version and would need its own edit.
