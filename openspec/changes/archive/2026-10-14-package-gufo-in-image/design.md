# Design: package-gufo-in-image

## Context

The root `Dockerfile` is two stages on `ubuntu:24.04`: a builder (lemonade C++ + web app) and a runtime stage that installs the *client* binaries only — inference backends arrive at run time via managed downloads into `/opt/lemonade/.cache` (a volume). gufo breaks that pattern: upstream ships no binaries, so the container has no way to obtain the engine.

Facts established on strixbeemo (2026-10-10) that constrain the design:

- The host's gufo binary **cannot run in the runtime image**: `error while loading shared libraries: libicuuc.so.78` — it links Arch's ICU 78 and gcc-15 `libstdc++`, and needs `/opt/rocm` (libamdhip64.so.7, libhipblas.so.3, libhipblaslt.so.1, librocblas.so.5, hsa/comgr transitives). None exist in `ubuntu:24.04` (ICU 74), and glibc/ld differences make LD_LIBRARY_PATH bundling a non-starter.
- gufo's `CMakeLists.txt` needs: C++20 host compiler, `find_package` ICU/CURL/PNG/JPEG/OpenSSL-crypto/PkgConfig/Threads/OpenMP, GPU path `find_package(hip/hipblas/hipblaslt/rocblas)` + `find_path(ROCWMMA_INCLUDE_DIR)`, and `ffmpeg`/`ffprobe` as invoked executables at runtime (media encode/decode).
- Upstream has a real release tag: checkout reports `v0.11.0-9-gd0f7c3a`, `version.txt = 0.11.0` — exactly the backend's `AtLeast(0.11.0)` gate.
- ROCm 7.2 publishes apt packages for Ubuntu 24.04 (`noble`) at `repo.radeon.com/rocm/apt/7.2` — HIP clang offloader, dev headers, and rocWMMA headers (`rocwmma-dev`) as plain debs. The gcc-15/rocWMMA extractions used for the host build were pure CachyOS workarounds, not gufo requirements.

## Goals / Non-Goals

**Goals**

- A freshly built container image on a gfx1151 host reports `gufo` → `installed` with zero operator configuration and can load models.
- Engine provenance is pinned and visible in the build (repository + version tag).
- No change to the backend C++ resolution order, the version gate, or the no-managed-download invariant.

**Non-Goals**

- Publishing managed gufo fork release binaries (the phase-2 OCI/asset story from add-gufo-backend D2).
- Per-model engine sidecar containers (the halogen pattern) — would violate the subprocess model for gufo and doubles GTT registration (one 62 GiB fits at a time on this host).
- Nix-based image builds.
- Changing `backend_versions.json` (gufo stays out of managed versions; the pin lives in the Dockerfile).

## Decisions

### D1. Build-time bundling in the lemonade image (chosen)

Compile gufo in a new Dockerfile stage and `COPY` the binary into the runtime stage.

| Alternative | Why rejected |
|---|---|
| Bind-mount host binary + libs | Cross-distro ABI: icu78, gcc-15 libstdc++, glibc interpreter mismatch — proven failing today |
| Run gufo in its own container | Lemonade spawns backends as subprocesses (critical invariant #3); cross-container exec would be a new architecture |
| On-demand download at first load | No upstream binaries exist — the premise of the system-managed design |
| Nix in the image | Pulls the world; apt already provides everything on noble |

### D2. `gufo-builder` stage recipe

`FROM ubuntu:24.04 AS gufo-builder`; add the ROCm 7.2 apt repo (key `https://repo.radeon.com/rocm/rocm.gpg.key`, distro `noble`); install `build-essential cmake ninja-build git pkg-config` + `libicu-dev libcurl4-openssl-dev libpng-dev libjpeg-turbo8-dev libwebp-dev libssl-dev` plus `rocm-core rocm-llvm rocm-llvm-dev hip-dev rocblas-dev hipblas-dev hipblaslt-dev hipcub-dev rocprim-dev rocwmma-dev`; clone `--depth 1 --branch "$GUFO_VERSION" "$GUFO_REPOSITORY"`; build with `cmake --preset release -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++` and `cmake --build --preset release`. Distro gcc-13 is the host compiler (C++20 satisfied; no gcc-16 `<format>`/HIP-`__noinline__` clash, which was the only reason the Arch host build needed a private gcc-15).

Implementer's findings, applied: ROCm 7.2's deb split renamed/retired 6.x names — `hip-clang` is now `rocm-llvm` (compiler) + `rocm-llvm-dev` (ships `/usr/lib/cmake/hip-lang/hip-lang-config.cmake`, required by `enable_language(HIP)`); gufo's `find_path` needs explicit `hipcub-dev`/`rocprim-dev`/`rocwmma-dev`; `libwebp-dev` for the `pkg_check_modules(libweb)` check; `rocm-device-libs` is mandatory (clang's HIP offload hard-fails "cannot find ROCm device library" without its `amdgcn/bitcode`). Debian also installs `hipcc`/`hipconfig` into `/usr/bin` while `hip-config.cmake` `set_and_check`s `${_IMPORT_PREFIX}/bin/<tool>`, and `hipcc` is a sh wrapper locating its perl sibling relative to `$0` — so the builder writes `exec`-shims into `/opt/rocm/bin` (a plain symlink breaks the perl lookup). The `release` preset does not set `CMAKE_HIP_COMPILER` (verified from CMakePresets.json), so it is passed explicitly.

The trap that cost the most: **Ubuntu universe carries a stale Debian-maintained ROCm fork** (hipcc 5.7.1, old `libamdhip64-dev` headers in `/usr/include/hip`, `rocminfo`/`libhsa-runtime64-1` 5.7.x), and because radeon versions some packages by *project* semver (`hipcc 1.1.1.70200` < universe `5.7.1-3`), plain apt version comparison picks the **universe** package — silently mixing 5.7-era headers into a 7.2 build (observed symptom: `gufo_diagnostics`' `.hip` file fails on undeclared `hipMemoryTypeUnregistered`/`__AMDGCN_WAVEFRONT_SIZE`). Both stages that touch ROCm debs write an apt preferences pin: `repo.radeon.com` at priority 900 (wins every shared binary name regardless of version strings) plus a `-1` blocklist of the fork's names at `archive.ubuntu.com`. Verified by clean-layer rebuild compiling the previously failing target.

### D3. Runtime closure, validated by `ldd` in the final image

Runtime stage installs: `rocm-core hip-runtime-amd rocblas hipblas hipblaslt hsa-rocr comgr` (the `.so`-only set — dev packages stay in the builder; note the runtime package is `comgr`, not the builder-era `amd-comgr`), plus `libicu74 libwebp7 libjpeg-turbo8 libpng16-16t64 ffmpeg`. The Debian ROCm packages do **not** register `/opt/rocm/lib` with the dynamic linker (no `ld.so.conf.d` entry — `ldconfig -p` stays empty), so both stages write `/etc/ld.so.conf.d/rocm.conf` + `ldconfig` themselves; otherwise the engine dies at `error while loading shared libraries: libhipblas.so.3`. Acceptance is mechanical, not guesswork: inside the final image `ldd /usr/local/bin/gufo` must resolve fully, `gufo --version` must exit 0, and `command -v ffmpeg ffprobe` must succeed. The concrete package list that made these pass is recorded in a Dockerfile comment so future ROCm minor bumps have a known baseline.

### D4. Pin expressed as build ARGs

`ARG GUFO_REPOSITORY=https://github.com/gufo-org/gufo.git` and `ARG GUFO_VERSION=v0.11.0` (upstream tag; `version.txt` = 0.11.0 = the backend gate). The build logs `git rev-parse` of the resolved tag so image provenance names the commit, and stamps the release identity into the binary via `-DGUFO_RELEASE_VERSION="${GUFO_VERSION#v}" -DGUFO_REVISION=$(git rev-parse --short HEAD)` — gufo's banner reads those cache variables (defaults "development"/"unknown"; release-please injects them in official builds), so without the injection a source-tag build self-reports as `development` and would trip the backend's dev-warning path. Tag (not SHA) as the default pin mirrors how `backend_versions.json` pins releases; a SHA override is available via the same ARG for supply-chain-tight builds.

### D5. Install path `/usr/local/bin/gufo` — no resolver changes

The existing resolution order (`gufo.server_bin` / `LEMONADE_GUFO_SERVER_BIN` → `PATH`) finds the bundled engine untouched, and operator overrides still win. The pinned release passes `AtLeast(0.11.0)` with no `development` warning. No C++ change; `backend_versions.json` untouched.

### D6. Always-on bundle (no build-time toggle)

The image grows substantially: measured **8.67 GB** vs the halogen-fork baseline's 1.13 GB — the ROCm runtime debs are heavy (rocBLAS/hipBLASLt ship multi-gigabyte precompiled Tensile kernel libraries for every gfx target; ffmpeg adds ~0.5 GB). A `GUFO_BUNDLE=off` variant was rejected: it doubles the build matrix in CI and packaging, and Lemonade's flagship target hardware is the very chip gufo serves. A gfx1151-only Tensile trim of the runtime libs is the obvious follow-up if size becomes contentious, not a per-backend ARG.

### D7. Unprivileged-user GPU access is verified, not assumed

The runtime stage `USER lemonade` (uid 10001). HIP needs `/dev/kfd` (+ DRI render nodes for media paths). Host device node ownership on strixbeemo decides the fix: prefer `group_add: [render]` (or the matching gid) in the stack's compose service — an operator-side capability grant — over baking group membership into the image. An implementation task on real hardware resolves this before docs claim it works. If kfd is mode 0666 on the host (some ROCm configs), no change is needed at all; record which case applied.

### D8. CI smoke checks binary presence on CPU runners; functional state on gfx1151

`docker-build-smoke-test.yml` (CPU runners) asserts: `gufo --version` prints the pinned version, `ffmpeg` is present, and `/system-info` lists the `gufo` recipe (state will be unsupported-hardware there — presence is what's tested). The flip to `installed` plus a live model load is the GPU-machine deploy task on strixbeemo, mirroring how the halogen rollout was validated.

### D9. Rollout on strixbeemo

1. `docker build --network=host -t lemonade:gufo .` from the branch (host-network workaround for strixbeemo's bridge veth issue, same as the existing fork builds).
2. Re-point `genai`'s `lemonade-server` image (tag stays `lemonade:gufo`), and replace the "ENGINE is not in this image" NOTE comment with the new provenance line.
3. Komodo deploy → verify `/system-info` reports `installed` with version 0.11.0, load `Qwen3-ASR-1.7B` (already staged in the same-path-mounted HF cache), transcribe the sample WAV, unload.
4. Keep a rollback: the previous `lemonade:gufo` image id remains available for `docker tag` if the container fails GPU access.

## Risks / Trade-offs

- **ROCm apt package name drift (7.2 → future minors)**: D3's ldd/`--version` gate converts silent breakage into a build failure with a named check; recorded package list in the Dockerfile comment gives the next bump a baseline.
- **Image size**: +1.5–2 GiB on all consumers of the official image. Accepted (D6); rollback lever is a slim target, not a toggle.
- **Builder time**: compiling HIP kernels adds minutes to every image build; acceptable for CI cadence here (the host cold build took ~10 minutes of kernel compilation).
- **Tag mutability** (D4): upstream could move `v0.11.0`; the logged commit SHA in build output preserves auditability, and SHA pinning is a documented override.
- **GPU access dead-end for the unprivileged user** (D7): worst case the container needs `group_add render` (compose-side) or, if kfd demands root, gufo-in-container ships with a clear doc caveat ("run the container with GPU access granted") — same posture as every ROCm container today. Verification happens in a task before docs promise the behavior.
- **ffmpeg pulls X11-adjacent deps** into the runtime (~300 MB incl. libraries): noted cost of D3; alternative of a stripped custom ffmpeg build was rejected for maintenance reasons.
