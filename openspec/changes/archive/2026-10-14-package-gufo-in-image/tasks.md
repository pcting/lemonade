# Tasks: package-gufo-in-image

## 1. Dockerfile — gufo builder stage

- [x] 1.1 Add `FROM ubuntu:24.04 AS gufo-builder` with the ROCm 7.2 apt repository (repo.radeon.com key + `noble` distro) and build deps per design D2: `build-essential cmake ninja-build git libicu-dev libcurl4-openssl-dev libpng-dev libjpeg-turbo8-dev libssl-dev pkg-config rocm-core hip-clang hip-dev rocblas-dev hipblas-dev hipblaslt-dev rocwmma-dev`
- [x] 1.2 Add `ARG GUFO_REPOSITORY=https://github.com/gufo-org/gufo.git` and `ARG GUFO_VERSION=v0.11.0`; shallow-clone the tag, log the resolved commit SHA during build (design D4)
- [x] 1.3 Configure and build: `cmake --preset release` with `-DCMAKE_PREFIX_PATH=/opt/rocm` (and `-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++` unless the preset already sets it); `cmake --build --preset release`; confirm the produced `build/release/gufo` is the only artifact copied downstream

## 2. Dockerfile — runtime closure

- [x] 2.1 In the runtime stage, install the `.so`-only ROCm set (`rocm-core hip-runtime-amd rocblas hipblas hipblaslt hsa-rocr amd-comgr` — adjust names to what the ldd gate in 3.1 proves) plus `libicu74 libwebp7 libjpeg-turbo8t64 ffmpeg`
- [x] 2.2 `COPY --from=gufo-builder` the engine to `/usr/local/bin/gufo`; chmod 755; no PATH or resolver changes anywhere (design D5)
- [x] 2.3 Record the final, proven-working apt package list as a Dockerfile comment next to the runtime install (baseline for future ROCm bumps)

## 3. Local image validation (strixbeemo)

- [x] 3.1 `docker build --network=host -t lemonade:gufo-rc1 .` then inside the final image: `ldd /usr/local/bin/gufo` resolves with zero "cannot open shared object file", `gufo --version` prints `gufo version 0.11.0 (...)`, `command -v ffmpeg ffprobe` succeed
- [x] 3.2 Start a test container mirroring the stack service (devices `/dev/kfd` `/dev/dri`, same-path HF cache bind, static IP not required) and query `/api/v1/system-info`: assert `gufo` recipe present; record the backend state (expected `installed` on this host)
- [x] 3.3 Resolve design D7 empirically: as the unprivileged `lemonade` user, run `gufo serve llm --model <staged 27B path> --port <free>` inside the container (or directly exec the binary), note host `/dev/kfd` + `/dev/dri` node modes/groups, and determine whether the fix is compose-side `group_add: [render]` / gid grant, host udev, or nothing; document the finding in the docker guide (5.1) and the stack compose comment

## 4. CI smoke extension

- [x] 4.1 In `.github/workflows/docker-build-smoke-test.yml`, add checks against the built image: `gufo --version` matches the pinned release version; `ffmpeg` present; `/system-info` lists the `gufo` recipe (presence only — CPU runners can't be `installed`; design D8)
- [x] 4.2 Verify the smoke job fails when `GUFO_VERSION` points at a tag below the backend minimum (temporary local run or a matrixed negative case, not a committed broken pin)

## 5. Documentation

- [x] 5.1 `docs/guide/install/docker.md`: state that official images bundle the gufo engine at the pinned version on supported hardware, that the engine is still system-managed on non-container installs (remediation path unchanged), and record the D7 GPU-grant finding
- [x] 5.2 Update the `docs/dev/backends-reference.md` gufo editorial section: replace "Upstream publishes no binaries → remediation" framing with "no upstream binaries; container images bundle a pinned source build" while keeping the Nix/CMake instructions for hosts
- [x] 5.3 Re-run `python docs/tools/gen_backend_boilerplate.py --check` (5.1/5.2 touch non-generated regions only; confirm zero drift)

## 6. Rollout on strixbeemo (verification, out-of-repo)

- [x] 6.1 Rebuild the fork image from this branch (`docker build --network=host -t lemonade:gufo .` after tasks 1–3), keeping the previous image id tagged for rollback (design D9.4)
- [x] 6.2 In komodo-halo `stacks/genai/compose.yaml`: replace the "gufo ENGINE is not in this image" NOTE with the bundled-provenance line (and the D7 group grant if required); push and `komodo deploy genai`
- [x] 6.3 Functional proof inside the deployed container: `/system-info` gufo state `installed` (0.11.0); load `Qwen3-ASR-1.7B` (already staged in the shared HF cache), transcribe `test/test_speech.wav` with non-empty text, unload cleanly with the gufo subprocess gone

## 7. Final gates

- [x] 7.1 `openspec validate package-gufo-in-image --strict` passes
- [x] 7.2 Full cpp-ci ctest run unchanged from baseline (72/73, pre-existing network test only) — this change adds no C++
- [x] 7.3 `docker build` of the plain (unforked) image succeeds on a clean checkout, so the packaging lands as a general image improvement rather than only a strixbeemo-local build
