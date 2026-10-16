#pragma once

// Which checkpoint artifacts the gufo engine can load, described by
// resources/gufo_compat.json rather than by rules compiled into this binary: one
// family per model architecture, each holding tensor groups whose accepted
// storage types mirror one accepted-types expression in the engine's weight
// binder.  The engine's own load error stays the backstop for anything the
// mirror has not caught up with, so every degradation path here answers "yes".
//
// The manifest's `pairings` section is deliberately not parsed here.  It records
// provenance for the combinations the registry ships, which is a claim about the
// registry rather than about a tensor, so CI reads it and the engine never does:
// a stale or wrong pairing row can then not change a verdict.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "lemon/gguf_reader.h"

namespace lemon {
namespace backends {
namespace gufo {

// One tensor group from the manifest: the tensor names it claims and the storage
// types the mirrored engine expression accepts for them.
struct GufoCompatGroup {
    std::string name;
    std::vector<std::string> exact;
    std::vector<std::string> patterns;
    std::vector<uint32_t> types;
    // Engine file + line range this group mirrors, carried so a reviewer can
    // check the transcription and so a drift report can name it.
    std::string mirrors;
    std::string why;
};

struct GufoCompatFamily {
    std::string name;
    std::vector<std::string> repos;
    // Whether this family's loader rejects a type whose block size does not
    // divide the tensor's contiguous dimension (gufo's FormatOf gate).
    bool block_divides_cols = false;
    std::vector<std::string> variant_groups;
    std::vector<GufoCompatGroup> groups;
    // Speculative labels this family ships drafts for, each mapped to the family
    // that judges them; an empty target means the draft is the same architecture
    // and reuses this family's groups.
    std::vector<std::pair<std::string, std::string>> companions;
};

// Result of judging one artifact against one family.  A reject names the
// artifact, the tensor, the type found, the group that decided, and the manifest
// that produced it, so a refusal is diagnosable from the log alone.
struct GufoCompatVerdict {
    bool accept = true;
    // Machine-readable cause.  Also set on the documented permissive paths
    // ("family unresolved", "header unreadable"), so "allowed by a rule" and
    // "allowed because there is no rule" stay distinguishable.
    std::string reason;
    std::string artifact;
    std::string offending_tensor;
    std::string type_name;
    std::string group;
    std::string manifest;

    std::string explanation() const;
};

class GufoCompat {
public:
    // The shipped table, built on first use; static-local initialization keeps
    // concurrent registry rebuilds from racing on the parse.
    static const GufoCompat& current();

    GufoCompat();
    // Empty path selects resources/gufo_compat.json; a path is supplied by tests
    // that judge a manifest fixture.
    explicit GufoCompat(const std::string& path);

    // False when the manifest is missing, unparseable, or names a storage type
    // this engine release cannot name.  Callers must then fall back to the flat
    // type-set check: a broken table must never hide a working model.
    bool usable() const { return usable_; }
    const std::string& error() const { return error_; }
    int schema_version() const { return schema_version_; }
    const std::string& engine_pin() const { return engine_pin_; }
    const std::string& min_supported() const { return min_supported_; }
    std::string stamp() const;

    // Family a repository belongs to; empty when it belongs to none, which means
    // there are no rules it could break.
    std::string resolve_family(const std::string& repository) const;

    // Family that judges a draft companion: the companion's own repository when
    // the manifest declares it as a family, else the host family's declared rule
    // for the speculative label, else the host family.
    std::string companion_family(const std::string& host_family,
                                 const std::string& label,
                                 const std::string& draft_repository) const;

    GufoCompatVerdict verdict(const std::string& family,
                              const GgufTensorTypes& inventory) const;
    GufoCompatVerdict verdict(const std::string& family, const std::string& path) const;

    // Group whose name rules claim a tensor: any exact entry first, else the first
    // pattern in declared array order; null when no group claims it.  Public
    // because precedence is pinned by a golden fixture, and because "which rule
    // answered" is the first thing a drift report has to say.
    const GufoCompatGroup* group_claiming(const std::string& family,
                                          const std::string& tensor) const;

    const std::vector<GufoCompatFamily>& families() const { return families_; }

private:
    // Group that claims a tensor name: any exact entry first, else the first
    // pattern in declared array order.  Null when no group claims it.
    const GufoCompatGroup* claim(const GufoCompatFamily& family,
                                 const std::string& tensor) const;
    const GufoCompatFamily* family_named(const std::string& name) const;

    bool usable_ = false;
    std::string error_;
    int schema_version_ = 0;
    std::string engine_pin_;
    std::string min_supported_;
    std::vector<GufoCompatFamily> families_;
};

// Reads a checkpoint's tensor-info section, following unsloth-style layouts whose
// first shard is a metadata-only stub (0 tensors) to the sibling that declares
// tensors.  Reports the inventory either way: a stub with no readable sibling is
// still an artifact this gate has no rule about.
bool gufo_read_inventory(GgufTensorTypes& out, const std::string& path);

// Verdict for one registry entry: the main artifact, ANDed with the draft when
// the entry declares one.  A manifest that is unusable, a repository no family
// claims, and an unreadable header all report accept, so the caller can fall back
// to the flat storage-type check instead of deciding on a broken table.  Every
// reject is logged here, once per judged artifact.
GufoCompatVerdict verdict_entry(const std::string& main_path,
                                const std::string& main_repository,
                                const std::string& draft_path,
                                const std::string& draft_repository,
                                const std::vector<std::string>& labels);

// Speculative family a draft companion belongs to, from the model's labels.  The
// `--speculative` flag wiring and the compatibility verdict both call this so
// they cannot disagree about what a draft file is.
std::string gufo_speculative_label(const std::vector<std::string>& labels);

}  // namespace gufo
}  // namespace backends
}  // namespace lemon
