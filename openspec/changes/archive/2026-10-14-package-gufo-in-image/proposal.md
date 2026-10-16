# Proposal: package-gufo-in-image

## Why

The gufo backend resolves its engine from `gufo.server_bin` / `PATH`, but upstream publishes no installable binaries — so in the containerized deployment (lemonade-server in the `genai` Komodo stack) the recipe can only ever report `action_required`. An operator mounting the host's gufo build into the Ubuntu-based image does not work either: the host binary dynamically links `libicuuc.so.78`, a gcc-15-era `libstdc++`, and ROCm 7.2 libraries that an `ubuntu:24.04` runtime does not provide (verified on strixbeemo: `error while loading shared libraries`). Bundling a distro-consistent gufo build into the image is the only self-contained container story.

## What Changes

- Add a `gufo` builder stage to the root `Dockerfile` that compiles gufo from a pinned git ref on `ubuntu:24.04` with the ROCm 7.2 apt repository (HIP clang offloader, `rocwmma-dev` for headers, distro gcc for C++20 host code).
- Extend the runtime stage with the engine's runtime closure — ROCm HIP/hipBLAS/hipBLASLt/rocBLAS/hsa/comgr runtime packages, `ffmpeg`/`ffprobe` (gufo shells out for media), and the Ubuntu ICU/webp/jpeg libraries — and install the built engine as `/usr/local/bin/gufo` so the existing PATH resolution finds it with no configuration.
- The pinned ref is version `0.11.0`, exactly satisfying the backend's `AtLeast(0.11.0)` gate without the `development` warning path.
- Extend the Docker container smoke-test CI workflow to assert the engine is present and reports the pinned version.
- Document that official container images bundle the gufo engine (docker install guide + the backends-reference gufo section).
- Deploy consequence (out-of-repo, tracked in tasks for verification): rebuild the `lemonade:gufo` fork image on strixbeemo, redeploy the `genai` stack, and confirm the recipe flips from `action_required` to `installed` with a real model load from inside the container.

No changes to the gufo backend C++ code, the model catalog, or the system-managed resolution order: bundling simply means PATH finds the binary. Host installs without Docker are unaffected.

## Capabilities

### New Capabilities

- `gufo-container-image`: the shipped container image contains a working, version-pinned gufo engine, discoverable through the same resolution rules as any operator-provided install, with its media and GPU runtime dependencies satisfied in-image and smoke-tested in CI.

### Modified Capabilities

None. `gufo-backend`'s existing "System-managed binary resolution" requirement already permits this: the engine arriving via `PATH` is the specified fallback, and no managed download is attempted. `gufo-model-catalog` requirements are unchanged.

## Impact

- **Build**: root `Dockerfile` (new stage + runtime packages; image size grows roughly 1.5–2 GiB — ROCm runtime + ffmpeg; accepted tradeoff, noted in design).
- **CI**: `.github/workflows/docker-build-smoke-test.yml` gains engine assertions.
- **Docs**: `docs/guide/install/docker.md`, `docs/dev/backends-reference.md` (gufo editorial section: replace "system-managed, not in image" framing with "bundle-in-image for containers; system-managed elsewhere").
- **Ops (verification only)**: komodo-halo `stacks/genai/compose.yaml` NOTE comment update + `lemonade:gufo` image rebuild/redeploy on strixbeemo.
- **Unaffected**: `backend_versions.json` (still no managed download), CLI, server C++ code, model catalog.
