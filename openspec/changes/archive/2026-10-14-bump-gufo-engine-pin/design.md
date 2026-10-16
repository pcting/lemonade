## Context

See `proposal.md` — Why. The constraints that shape the approach:

- The manifest is engine-pinned: `src/cpp/resources/gufo_compat.json` declares `engine.pin` and `engine.min_supported`, and every group cites a file-and-line range in the engine's own source.
- `src/cpp/include/lemon/backends/gufo/gufo.h` gives `kMinGufoVersion` two jobs — the runtime minimum-version gate, and "the engine revision the checkpoint compatibility manifest transcribes". Its comment states the conformance check asserts the two agree.
- `test/cpp/test_gufo_compat.cpp` does exactly that: `manifest.engine_pin()` is compared against `lemon::backends::gufo::kMinGufoVersion`. A bump therefore forces the runtime floor up with it.
- The two `engine-loaded` pairing records name the release that produced them, which is `0.11.0` today.

## Goals / Non-Goals

**Goals:**
- Ship the engine upstream released as `v0.15.0` without refusing engines that still satisfy the floor's own criterion.
- Give the manifest's two version fields distinct, separately checkable meanings.
- Keep every pairing record honest about which release produced its evidence.

**Non-Goals:**
- Re-deriving the tensor-group table: the re-diff shows the mirrored rules are unchanged, so the table's content is not in question (see Decisions).
- Adopting any other `v0.15.0` behaviour (component cache, managed-alloc fallback, IPv6 listen) as lemonade-visible features. The engine gains them; lemonade's surface does not claim them.
- Changing the checkpoint gate's permissive rules, its explainability, or which recipes it applies to.

## Decisions

### D1 — The mirrored release moves to `v0.15.0`; the floor stays at `0.11.0`

The floor's stated justification is the surface this backend drives (served-model-name, /health semantics), so the question is whether the range kept that surface. Audited against both tags:

| what | `v0.11.0` → `v0.15.0` |
|---|---|
| `src/models/qwen38_flash_next/weights.cpp` (11 groups mirror it) | byte-identical |
| `src/models/qwen/weights.cpp` (5 groups) | byte-identical |
| `src/models/qwen/dflash_weights.cpp` (2 groups) | byte-identical |
| `src/cli/serve/serve.cpp` — `--served-model-name`, `--api-key` | present in both (6 and 2 occurrences) |
| `src/cli/serve/http_server.{cpp,hpp}` | `/health` and `api_key` identical; only addition is `IsListenAddress` (IPv6) |
| `src/cli/main.cpp`, `arg_parser.hpp` | byte-identical |
| tree | 1187 → 1218 files, +31 additions, 0 removals |

So the table's rules hold across the whole range and nothing the backend drives was withdrawn. **Alternative considered and rejected:** bump the floor to `0.15.0` along with the pin (today's forced behaviour). It refuses `0.11.0`…`0.14.0` engines on a criterion that no longer holds — no evidence any of them lacks the surface — and it would break a host running the released `0.11.0` engine with no benefit to the table.

### D2 — Two anchors instead of one invariant

The conformance check keeps asserting an equality, but against different anchors than before:

```
  Dockerfile ARG GUFO_VERSION ──(image smoke test)──▶ gufo --version in the image
            │
            └──(conformance)──▶ manifest.pin ── mirrors ──▶ group citations
                                      
  backend kMinGufoVersion ◀──(conformance)── manifest.min_supported
```

`manifest.pin` must equal the release the image bundles; `manifest.min_supported` must equal `kMinGufoVersion`. **Alternative considered and rejected:** drop the pin check rather than re-anchor it. A pin nobody checks is how a table ends up describing an older engine — the very failure the original comment names.

### D3 — The guard against a stale table becomes an obligation to re-diff, not a version comparison

`v0.15.0` removed no files and did not touch the mirrored call sites, but the range *did* add a continuation adapter, a component text cache, and HIP heap placement. None introduces weight tensors, so no group changes. The rule this change encodes for the next bump: the floor is evidence-backed (a re-diff of the cited files), and a release that *does* change a cited call site obliges extending the table rather than moving the number.

### D4 — `engine-loaded` records are re-earned, and name the release that earned them

Both `engine-loaded` records were produced by `0.11.0`. After this change the shipped engine is `0.15.0`, so the claim has to be re-established against it (drift judge over the artifacts, plus a real load and generation), and the record must name `0.15.0`. Records resting on a read or on the registry's intent are untouched: neither claim was about an engine build.

## Risks / Trade-offs

- **The re-audit covers the surface we know the backend drives, not the engine's whole CLI.** Mitigation: the implementation verifies the engine's own `serve` help from the built image before the deployment is swapped, and the deployed container is smoke-tested by serving a real request.
- **Re-earning the records depends on this host holding the artifacts.** The 27B quant and the ASR directory are in the deployed cache volume, so the load checks are available; the Flash-Next pair is not loadable beside a running server (111 GB), so its record keeps its existing evidence and gains only the re-judged drift result if the files are reachable read-only.
- **The floor stays put while the engine moves, so a future engine could satisfy the floor number and still break a mirrored rule.** That is exactly what the re-diff obligation in D3 and the mirrored-source check in the manifest requirement are for; this change strengthens the first and relies on the second as-is.
- **Two version fields can drift apart in documentation.** The `_comment`'s derived-from line, the docs' guidance, and the out-of-repo compose comment all name versions; the implementation updates each so no artifact claims `0.11.0` is what ships.
