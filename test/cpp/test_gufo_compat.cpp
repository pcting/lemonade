// Proves the gufo checkpoint compatibility table decides what the gufo engine can
// load: every shipped group agrees with the engine source lines it cites, group
// precedence is pinned by a golden list measured from real artifacts, main and
// draft verdicts are ANDed through the family rules, every gufo row in either
// registry matches a recorded pairing on its verbatim checkpoint string, and every
// path that cannot decide — unresolved repository, unreadable header, broken table
// — answers "servable", because a table that fell behind the engine must never hide
// a model that loads.
//
// Run with --drift to re-read each group's cited engine ranges; that entry reports
// SKIP when no gufo checkout is available, so a machine without the engine source
// never reports the transcription as verified. With --hub <dir> it also re-judges
// the shipped pairings against the artifacts in that cache.

#include "lemon/backends/gufo/gufo.h"
#include "lemon/backends/gufo/gufo_compat.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

using lemon::GgufTensorInfo;
using lemon::GgufTensorTypes;
using lemon::backends::gufo::GufoCompat;
using lemon::backends::gufo::GufoCompatFamily;
using lemon::backends::gufo::GufoCompatGroup;
using lemon::backends::gufo::GufoCompatVerdict;

namespace {

int failures = 0;

void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// Column count for every synthetic tensor: a multiple of every block size gufo's
// FormatOf() knows, so a verdict is decided by the type rules unless a test asks
// for the block rule on purpose.
constexpr uint64_t kCols = 4096;

uint32_t type_id(const std::string& name) {
    uint32_t id = 0;
    if (!lemon::ggml_type_id(name, id)) {
        std::printf("FAIL test asks for storage type %s, which gufo cannot name\n", name.c_str());
        ++failures;
    }
    return id;
}

GgufTensorTypes inventory(std::initializer_list<std::pair<std::string, std::string>> named) {
    GgufTensorTypes out;
    for (const auto& entry : named) {
        GgufTensorInfo tensor;
        tensor.name = entry.first;
        tensor.dimensions = {kCols, kCols};
        tensor.type = type_id(entry.second);
        out.tensors.push_back(std::move(tensor));
    }
    out.tensor_count = out.tensors.size();
    return out;
}

fs::path scratch_dir() {
    // Wiped once, on first use: every artifact the suite writes has to stay
    // readable for the checks that follow.
    static const fs::path dir = [] {
        const fs::path fresh = fs::temp_directory_path() / "lemonade_gufo_compat_test";
        std::error_code ec;
        fs::remove_all(fresh, ec);
        fs::create_directories(fresh, ec);
        return fresh;
    }();
    return dir;
}

void put_u32(std::ostream& out, uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void put_u64(std::ostream& out, uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void put_string(std::ostream& out, const std::string& value) {
    put_u64(out, value.size());
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

// GGUF v3 with a header, optional split metadata and tensor-info records: the
// inventory reader stops at the end of the tensor list, so an artifact this suite
// judges needs no tensor data behind it.
fs::path write_gguf(const std::string& file,
                    const std::vector<std::pair<std::string, std::string>>& tensors,
                    int32_t split_no = 0,
                    int32_t split_count = 1) {
    const fs::path path = scratch_dir() / file;
    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4);
    put_u32(out, 3);
    put_u64(out, tensors.size());
    put_u64(out, split_count > 1 ? 2 : 0);
    if (split_count > 1) {
        put_string(out, "split.no");
        put_u32(out, 5);
        put_u32(out, static_cast<uint32_t>(split_no));
        put_string(out, "split.count");
        put_u32(out, 5);
        put_u32(out, static_cast<uint32_t>(split_count));
    }
    for (const auto& tensor : tensors) {
        put_string(out, tensor.first);
        put_u32(out, 2);
        put_u64(out, kCols);
        put_u64(out, kCols);
        put_u32(out, type_id(tensor.second));
        put_u64(out, 0);
    }
    out.close();
    return path;
}

// A header declaring a tensor count with no records behind it: nothing but the
// reader's own bound stands between that count and an allocation.
fs::path write_gguf_declaring(const std::string& file, uint64_t declared_tensors) {
    const fs::path path = scratch_dir() / file;
    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4);
    put_u32(out, 3);
    put_u64(out, declared_tensors);
    put_u64(out, 0);  // no KV records
    out.close();
    return path;
}

const GufoCompatFamily* family_of(const GufoCompat& manifest, const std::string& name) {
    const auto it = std::find_if(manifest.families().begin(), manifest.families().end(),
                                 [&](const GufoCompatFamily& f) { return f.name == name; });
    return it == manifest.families().end() ? nullptr : &*it;
}

GufoCompatVerdict verdict_at(const std::string& family, const fs::path& path) {
    return GufoCompat::current().verdict(family, path.string());
}

// ── The shipped table ─────────────────────────────────────────────────

void test_shipped_manifest() {
    const GufoCompat& manifest = GufoCompat::current();
    check("shipped gufo_compat.json loads", manifest.usable());
    if (!manifest.usable()) {
        std::printf("     %s\n", manifest.error().c_str());
        return;
    }
    check("schema version is the one this binary reads", manifest.schema_version() == 1);

    // Two anchors, not one: the floor is the oldest release this backend still
    // accepts (and must equal the runtime gate), while the mirror is the release
    // the shipped image bundles. The image smoke workflow asserts the mirror
    // against the Dockerfile pin, where both values are visible; here the floor is
    // what can be checked, plus the sanity that the mirror is not older than it.
    const std::string pin = manifest.engine_pin();
    const std::string floor = manifest.min_supported();
    const auto parts = [](const std::string& v) {
        std::vector<int> out;
        size_t start = 0;
        while (start <= v.size()) {
            const size_t dot = v.find('.', start);
            out.push_back(std::atoi(v.substr(start, dot - start).c_str()));
            if (dot == std::string::npos) break;
            start = dot + 1;
        }
        return out;
    };
    const std::string bare = pin.rfind("v", 0) == 0 ? pin.substr(1) : pin;
    check("declared floor is the version gate this backend enforces",
          floor == lemon::backends::gufo::kMinGufoVersion);
    check("mirrored release is not older than the declared floor",
          !bare.empty() && !(parts(bare) < parts(floor)));

    std::set<std::string> repositories;
    bool every_repo_declared = true;
    bool every_citation_present = true;
    bool companions_resolve = true;
    for (const auto& family : manifest.families()) {
        for (const auto& repo : family.repos) {
            if (!repositories.insert(repo).second) every_repo_declared = false;
        }
        if (family.repos.empty()) every_repo_declared = false;
        for (const auto& group : family.groups) {
            if (group.mirrors.find(".cpp:") == std::string::npos) every_citation_present = false;
            if (group.types.empty()) every_citation_present = false;
        }
        for (const auto& companion : family.companions) {
            if (!companion.second.empty() && !family_of(manifest, companion.second)) {
                companions_resolve = false;
            }
        }
    }
    // A repository claimed by two families would leave the first declaration to
    // decide an architecture, which is exactly the ambiguity the family list is
    // meant to remove.
    check("every family owns repositories and no repository is owned twice",
          every_repo_declared);
    check("every group cites engine source and declares the types it mirrors",
          every_citation_present);
    check("every draft rule points at a family that exists", companions_resolve);

    // A draft rule is what lets a paired checkpoint be judged at all; whether the
    // family declares tensor groups is a separate choice (see deepseek below).
    const char* chat_families[] = {"qwen38-dense", "qwen38-flash-next", "deepseek-v4-flash"};
    bool chat_declares_drafts = true;
    for (const char* name : chat_families) {
        const GufoCompatFamily* family = family_of(manifest, name);
        if (!family || family->companions.empty()) {
            chat_declares_drafts = false;
        }
    }
    check("every chat family declares a draft rule", chat_declares_drafts);

    // DeepSeek's runtime binds one flat decoder set, so there is nothing per-role
    // to mirror: the honest table is empty and the family accepts by having no
    // rules rather than by guessing at a shape.
    const GufoCompatFamily* deepseek = family_of(manifest, "deepseek-v4-flash");
    const GufoCompatFamily* flash_next = family_of(manifest, "qwen38-flash-next");
    const GufoCompatFamily* dense = family_of(manifest, "qwen38-dense");
    const GufoCompatFamily* dflash = family_of(manifest, "qwen38-dflash2");
    check("deepseek-v4-flash declares no groups (permissive by absence of rules)",
          deepseek && deepseek->groups.empty());
    check("only the flash-next loader is block-gated", flash_next && dense && dflash &&
          flash_next->block_divides_cols && !dense->block_divides_cols &&
          !dflash->block_divides_cols);
}

// ── Precedence, pinned by measured name shapes ────────────────────────

void test_precedence() {
#ifdef GUFO_COMPAT_FIXTURE_DIR
    const std::string path = std::string(GUFO_COMPAT_FIXTURE_DIR) + "/precedence.json";
#else
    const std::string path = "test/cpp/fixtures/gufo_compat/precedence.json";
#endif
    std::ifstream in(path);
    check("precedence golden list is readable", static_cast<bool>(in));
    if (!in) return;
    const json fixture = json::parse(in, nullptr, false);
    if (fixture.is_discarded()) {
        check("precedence golden list parses", false);
        return;
    }

    const GufoCompat& manifest = GufoCompat::current();
    size_t shapes = 0;
    size_t wrong_group = 0;
    size_t refused_exemplar = 0;
    for (const auto& entry : fixture["families"].items()) {
        const std::string family = entry.key();
        for (const auto& shape : entry.value().items()) {
            const std::string exemplar = shape.value().at("exemplar").get<std::string>();
            const json& group = shape.value().at("group");
            const std::string want = group.is_null() ? std::string() : group.get<std::string>();
            const GufoCompatGroup* got = manifest.group_claiming(family, exemplar);
            ++shapes;
            if ((got ? got->name : std::string()) != want) {
                ++wrong_group;
                std::printf("     %s %s: want %s, got %s\n", family.c_str(), exemplar.c_str(),
                            want.empty() ? "(no group)" : want.c_str(),
                            got ? got->name.c_str() : "(no group)");
            }
            const std::string measured = shape.value().value("type", std::string("n/a"));
            if (!want.empty() && measured != "n/a") {
                const GufoCompatVerdict verdict =
                    manifest.verdict(family, inventory({{exemplar, measured}}));
                if (!verdict.accept) {
                    ++refused_exemplar;
                    std::printf("     %s\n", verdict.explanation().c_str());
                }
            }
        }
    }
    // The count is asserted too: a fixture that silently lost its entries would
    // otherwise turn this into a vacuous pass.
    check("golden list: name shapes resolve to the pinned group", shapes >= 100 && wrong_group == 0);
    check("golden list: every measured exemplar type is accepted", refused_exemplar == 0);
}

// ── Group-level verdicts ──────────────────────────────────────────────

void test_group_verdicts() {
    const GufoCompat& manifest = GufoCompat::current();
    if (!manifest.usable()) return;

    // Narrowing is the point of the table: one accepted type in a family must not
    // generalize across roles.
    const GufoCompatVerdict embedding =
        manifest.verdict("qwen38-dense", inventory({{"token_embd.weight", "Q8_K"}}));
    check("embedding refuses Q8_K, which only the unembedding accepts",
          !embedding.accept && embedding.group == "embedding" &&
              embedding.type_name == "Q8_K");
    check("a reject names the manifest it came from",
          embedding.explanation().find("gufo_compat schema") != std::string::npos);

    const GufoCompatVerdict projection_bad =
        manifest.verdict("qwen38-dense", inventory({{"blk.3.attn_q.weight", "Q4_0"}}));
    check("projection refuses Q4_0", !projection_bad.accept && projection_bad.group == "projection");
    check("projection accepts Q4_K",
          manifest.verdict("qwen38-dense", inventory({{"blk.3.attn_q.weight", "Q4_K"}})).accept);

    const GufoCompatVerdict mixer_norm =
        manifest.verdict("qwen38-flash-next", inventory({{"blk.0.hc_attn_norm.weight", "Q8_0"}}));
    check("mixer_norm refuses Q8_0 and is not confused with mixer_lowrank",
          !mixer_norm.accept && mixer_norm.group == "mixer_norm");

    const GufoCompatVerdict experts_bf16 =
        manifest.verdict("qwen38-flash-next", inventory({{"blk.5.ffn_gate_exps.weight", "BF16"}}));
    check("experts refuses BF16", !experts_bf16.accept && experts_bf16.group == "experts");
    check("experts accepts Q4_K",
          manifest.verdict("qwen38-flash-next", inventory({{"blk.5.ffn_gate_exps.weight", "Q4_K"}})).accept);

    const GufoCompatVerdict inject =
        manifest.verdict("qwen38-flash-next", inventory({{"blk.0.hc_attn_inject.weight", "F16"}}));
    check("mixer_inject refuses F16, which mixer_lowrank accepts",
          !inject.accept && inject.group == "mixer_inject");
    check("mixer_lowrank accepts F16",
          manifest.verdict("qwen38-flash-next", inventory({{"blk.0.hc_attn_down.weight", "F16"}})).accept);

    // Tensors the engine never looks up carry no risk: an unmatched name is the
    // permissive case, not an omission.
    check("a name no group claims is accepted",
          manifest.group_claiming("qwen38-flash-next", "blk.0._down.weight") == nullptr &&
          manifest.verdict("qwen38-flash-next", inventory({{"blk.0._down.weight", "Q2_K"}})).accept);

    // The FormatOf gate: a type the group accepts still fails when its block does
    // not divide the contiguous dimension, and the gate is family-scoped.
    const GufoCompatVerdict block_rule =
        manifest.verdict("qwen38-flash-next", inventory({{"blk.5.ffn_up_exps.weight", "Q8_0"}}));
    check("block-gated family accepts a block-aligned Q8_0 expert matrix", block_rule.accept);
    GgufTensorTypes ragged = inventory({{"blk.5.ffn_up_exps.weight", "Q8_0"}});
    ragged.tensors[0].dimensions[0] = 33;
    const GufoCompatVerdict ragged_verdict = manifest.verdict("qwen38-flash-next", ragged);
    check("block-gated family refuses a Q8_0 whose columns are not block aligned",
          !ragged_verdict.accept && ragged_verdict.group == "experts" &&
              ragged_verdict.reason.find("block") != std::string::npos);
    GgufTensorTypes same_ragged = inventory({{"blk.3.attn_q.weight", "Q4_K"}});
    same_ragged.tensors[0].dimensions[0] = 33;
    check("the block rule is not applied to a family whose loader has no such gate",
          manifest.verdict("qwen38-dense", same_ragged).accept);

    // Permissive paths, each distinguishable in the reason so "allowed by a rule"
    // and "allowed because there is no rule" stay separable in the log.
    const GufoCompatVerdict unknown_family =
        manifest.verdict("family-that-does-not-exist", inventory({{"a.weight", "Q4_K"}}));
    check("an unknown family accepts, reported as unresolved",
          unknown_family.accept && unknown_family.reason == "family unresolved");
    const GufoCompatVerdict unreadable =
        manifest.verdict("qwen38-dense", (scratch_dir() / "absent.gguf").string());
    check("an unreadable header accepts, reported as such",
          unreadable.accept && unreadable.reason == "header unreadable");

    // A header that parses but declares more tensors than the file has bytes to
    // describe is unreadable too, and it has to be answered rather than believed:
    // the count comes from the file, so a reader that trusts it can be made to
    // allocate on demand.
    const fs::path impossible = write_gguf_declaring("declares-too-many.gguf", 1ull << 40);
    const GufoCompatVerdict declared =
        manifest.verdict("qwen38-dense", impossible.string());
    check("a header declaring more than the file can hold accepts as unreadable",
          declared.accept && declared.reason == "header unreadable");

    // Unsloth's split layout keeps tensors out of the first shard: the verdict for
    // shard one has to be the verdict for the shard that declares tensors.
    const fs::path stub = write_gguf("split-00001-of-00002.gguf", {}, 1, 2);
    write_gguf("split-00002-of-00002.gguf", {{"token_embd.weight", "Q8_K"}}, 2, 2);
    const GufoCompatVerdict followed =
        manifest.verdict("qwen38-dense", stub.string());
    check("a metadata-only first shard is judged by its sibling's tensors",
          !followed.accept && followed.offending_tensor == "token_embd.weight" &&
              followed.artifact == stub.string());
}

// ── Which family judges what ─────────────────────────────────────────

void test_family_resolution() {
    const GufoCompat& manifest = GufoCompat::current();
    if (!manifest.usable()) return;

    check("a shipped main repository resolves to its family",
          manifest.resolve_family("unsloth/Qwen3.8-27B-GGUF") == "qwen38-dense");
    check("an unlisted repository resolves to nothing",
          manifest.resolve_family("somebody/else-GGUF").empty());

    // Its own declaration wins over the host's rule: the DFlash2 draft is a
    // different tensor namespace, however it is paired.
    check("a draft from a declared companion repository is judged by its own family",
          manifest.companion_family("qwen38-dense", "dflash",
                                    "z-lab/Qwen3.8-27B-DFlash2-GGUF") == "qwen38-dflash2");
    check("a companion from an undeclared repository still follows the host's label rule",
          manifest.companion_family("qwen38-dense", "dflash", "peer/Qwen3.8-Draft-GGUF") ==
              "qwen38-dflash2");
    check("a draft of the host's own repository is judged by the host family",
          manifest.companion_family("qwen38-flash-next", "mtp",
                                    "unsloth/Qwen3.8-Flash-Next-GGUF") == "qwen38-flash-next");
    check("a label the host has no rule for is judged on the host's rules",
          manifest.companion_family("qwen38-flash-next", "dflash", "peer/Qwen3.8-Draft-GGUF") ==
              "qwen38-flash-next");

    check("the speculative label matches the flag wiring's ladder",
          lemon::backends::gufo::gufo_speculative_label({}) == "dflash" &&
          lemon::backends::gufo::gufo_speculative_label({"mtp", "vision"}) == "mtp" &&
          lemon::backends::gufo::gufo_speculative_label({"dspark"}) == "dspark" &&
          lemon::backends::gufo::gufo_speculative_label({"audio"}) == "dflash");
}

// ── Entry-level verdicts: main AND draft ──────────────────────────────

void test_entry_verdicts() {
    using lemon::backends::gufo::verdict_entry;

    const fs::path main_ok = write_gguf(
        "main-ok.gguf", {{"token_embd.weight", "Q4_K"}, {"blk.3.attn_q.weight", "Q4_K"}});
    check("a servable main with no draft is servable",
          verdict_entry(main_ok.string(), "unsloth/Qwen3.8-27B-GGUF", {}, {}, {}).accept);

    const fs::path main_bad =
        write_gguf("main-bad.gguf", {{"token_embd.weight", "Q8_K"}});
    const GufoCompatVerdict main_only =
        verdict_entry(main_bad.string(), "unsloth/Qwen3.8-27B-GGUF", {}, {}, {});
    check("with no draft declared, only the main checkpoint is judged",
          !main_only.accept && main_only.artifact == main_bad.string());

    check("a repository no family claims is servable for lack of rules",
          verdict_entry(main_bad.string(), "somebody/else-GGUF", {}, {}, {}).accept);

    // The DFlash2 draft is judged by its own narrower table: what its own family
    // accepts, the host family would refuse, so pairing a draft with a dense host
    // neither widens nor narrows it by accident.
    const fs::path draft_dflash =
        write_gguf("draft-dflash-ok.gguf", {{"blk.0.attn_norm.weight", "BF16"}});
    check("a BF16 draft norm is accepted by the DFlash2 family",
          verdict_at("qwen38-dflash2", draft_dflash).accept);
    check("the same BF16 draft norm would be refused under the host family",
          !verdict_at("qwen38-dense", draft_dflash).accept);
    check("a servable dense main plus a servable draft is servable",
          verdict_entry(main_ok.string(), "unsloth/Qwen3.8-27B-GGUF", draft_dflash.string(),
                        "z-lab/Qwen3.8-27B-DFlash2-GGUF", {})
              .accept);

    const fs::path draft_bad =
        write_gguf("draft-dflash-bad.gguf", {{"fc.weight", "Q5_K"}});
    const GufoCompatVerdict draft_verdict =
        verdict_entry(main_ok.string(), "unsloth/Qwen3.8-27B-GGUF", draft_bad.string(),
                      "z-lab/Qwen3.8-27B-DFlash2-GGUF", {});
    check("a refused draft decides the entry even when the main is servable",
          !draft_verdict.accept && draft_verdict.artifact == draft_bad.string() &&
              draft_verdict.group == "matrix");

    // A third-party draft of a host that ships MTP companions: judged on the host's
    // mixer rules, so a type the host's binder refuses cannot ride in on an unknown
    // repository.
    const fs::path flash_main = write_gguf("main-flash-next.gguf",
                                           {{"token_embd.weight", "Q6_K"},
                                            {"blk.0.hc_attn_norm.weight", "F32"}});
    const fs::path mtp_bad =
        write_gguf("draft-mtp-f16.gguf", {{"blk.0.hc_attn_inject.weight", "F16"}});
    const fs::path mtp_ok =
        write_gguf("draft-mtp-bf16.gguf", {{"blk.0.hc_attn_inject.weight", "BF16"}});
    const GufoCompatVerdict mtp_verdict = verdict_entry(
        flash_main.string(), "unsloth/Qwen3.8-Flash-Next-GGUF", mtp_bad.string(),
        "peer/Qwen3.8-Flash-Next-MTP-GGUF", {"mtp"});
    check("a third-party MTP draft is judged under the host family's rules and refused",
          !mtp_verdict.accept && mtp_verdict.group == "mixer_inject" &&
              mtp_verdict.artifact == mtp_bad.string());
    check("the same draft with an accepted type is servable",
          verdict_entry(flash_main.string(), "unsloth/Qwen3.8-Flash-Next-GGUF", mtp_ok.string(),
                        "peer/Qwen3.8-Flash-Next-MTP-GGUF", {"mtp"})
                .accept);

    check("a declared draft that is not on disk yet does not refuse the main",
          verdict_entry(main_ok.string(), "unsloth/Qwen3.8-27B-GGUF",
                        (scratch_dir() / "not-downloaded.gguf").string(),
                        "z-lab/Qwen3.8-27B-DFlash2-GGUF", {})
                .accept);
}

// ── A table that fell behind the engine takes itself out of service ───

json sound_table() {
    return json{
        {"schema_version", 1},
        {"engine", {{"name", "gufo"}, {"pin", "v0.15.0"}, {"min_supported", "0.11.0"}}},
        {"families",
         {{"scratch",
           {{"repos", {"me/scratch-GGUF"}},
            {"groups",
             {{{"name", "matrices"},
               {"exact", {"fc.weight"}},
               {"types", {"F32"}},
               {"mirrors", "src/models/scratch/weights.cpp:10-12"}}}}}}}},
        {"pairings", json::array()},
    };
}

void test_broken_table() {
    const std::string good = (scratch_dir() / "table-sound.json").string();
    {
        std::ofstream out(good);
        out << sound_table().dump();
    }
    check("the control table loads", GufoCompat(good).usable());

    struct Case {
        const char* slug;
        const char* what;
        const char* expected_error;
        void (*mutate)(json&);
    };
    const Case cases[] = {
        {"unknown-type", "a storage type the engine cannot name", "cannot name", [](json& doc) {
             doc["families"]["scratch"]["groups"][0]["types"] = {"F32", "Q9_Z"};
         }},
        {"no-names", "a group that claims no tensor names", "no tensor names", [](json& doc) {
             doc["families"]["scratch"]["groups"][0]["exact"] = json::array();
         }},
        {"no-types", "a group that declares no types", "no accepted types", [](json& doc) {
             doc["families"]["scratch"]["groups"][0]["types"] = json::array();
         }},
        {"no-mirrors", "a group that cites no engine source", "no engine source", [](json& doc) {
             doc["families"]["scratch"]["groups"][0].erase("mirrors");
         }},
        {"undeclared-variant", "a variant group nothing declares", "does not declare",
         [](json& doc) { doc["families"]["scratch"]["variant_groups"] = {"absent_group"}; }},
        {"schema-two", "a schema this binary cannot read", "schema_version", [](json& doc) {
             doc["schema_version"] = 2;
         }},
    };

    for (const auto& testCase : cases) {
        json doc = sound_table();
        testCase.mutate(doc);
        const std::string path =
            (scratch_dir() / (std::string("table-") + testCase.slug + ".json")).string();
        std::ofstream out(path, std::ios::binary);
        out << doc.dump();
        out.close();
        const GufoCompat broken(path);
        check(std::string("a table with ") + testCase.what + " is reported unusable",
              !broken.usable() && broken.error().find(testCase.expected_error) != std::string::npos);
    }

    {
        const std::string path = (scratch_dir() / "table-not-json.json").string();
        std::ofstream out(path, std::ios::binary);
        out << "GGUF this is not a table";
        out.close();
        check("a table that is not JSON is reported unusable", !GufoCompat(path).usable());
    }

    // The refusal to judge has to be loud and then step aside: a broken table that
    // still decided verdicts would hide models behind a parse error.
    const GufoCompat broken(good + ".missing");
    check("a table that is missing is reported unusable", !broken.usable());
    const GufoCompatVerdict verdict =
        broken.verdict("scratch", inventory({{"fc.weight", "Q4_K"}}));
    check("an unusable table reports accept so the caller falls back to the flat check",
          verdict.accept && verdict.reason == "manifest unusable");
    check("an unusable table resolves no family", broken.resolve_family("me/scratch-GGUF").empty());
}

// ── Shipped pairings, cross-checked against the registries ──────────────

// The gate holds only what a verdict needs, so provenance is read here as data
// rather than through GufoCompat: it is a claim about the registry, and keeping it
// out of the engine of record is what stops a stale pairing row from being able to
// change a verdict.
json load_shipped_manifest() {
    std::ifstream in(lemon::utils::get_resource_path("resources/gufo_compat.json"));
    if (!in) return json();
    const json doc = json::parse(in, nullptr, false);
    return doc.is_discarded() ? json() : doc;
}

struct RegistryRow {
    std::string source;   // which registry, for the failure message
    std::string name;
    std::string main;
    std::string draft;    // empty when the entry declares no draft
    std::vector<std::string> labels;
};

struct PairingRecord {
    std::string main;
    std::string draft;
    std::string evidence;
    std::string verified;
    std::map<std::string, std::string> artifacts;   // checkpoint string → file name
};

struct PairingData {
    std::vector<RegistryRow> chat;             // gufo rows the table can speak about
    std::vector<RegistryRow> directory_rows;   // gufo rows it cannot
    std::vector<PairingRecord> records;
    bool ok = false;
};

// A checkpoint names a GGUF when it carries a variant alias or a file extension.
// `Qwen/Qwen3-ASR-1.7B` names a directory the engine loads whole, and no tensor
// group has anything to say about a directory, so those rows stay out of the
// pairing table.
bool gguf_shaped(const std::string& checkpoint) {
    if (checkpoint.find(':') != std::string::npos) return true;
    return checkpoint.size() > 5 && checkpoint.compare(checkpoint.size() - 5, 5, ".gguf") == 0;
}

std::string repository_of(const std::string& checkpoint) {
    const size_t colon = checkpoint.find(':');
    return colon == std::string::npos ? checkpoint : checkpoint.substr(0, colon);
}

std::vector<RegistryRow> gufo_rows(const fs::path& path) {
    std::vector<RegistryRow> rows;
    std::ifstream in(path);
    if (!in) return rows;
    const json doc = json::parse(in, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) return rows;
    for (const auto& item : doc.items()) {
        const json& row = item.value();
        if (!row.is_object() || !(row.contains("recipe") && row["recipe"].is_string() &&
                                  row["recipe"].get<std::string>() == "gufo")) {
            continue;
        }
        RegistryRow out;
        out.source = path.filename().string();
        out.name = item.key();
        // Read every field defensively: a registry row that is malformed is a
        // finding for the coverage checks below, not a reason to abort the suite.
        const json checkpoints = row.contains("checkpoints") ? row["checkpoints"] : json();
        if (checkpoints.is_object()) {
            if (checkpoints.contains("main") && checkpoints["main"].is_string()) {
                out.main = checkpoints["main"].get<std::string>();
            }
            if (checkpoints.contains("draft") && checkpoints["draft"].is_string()) {
                out.draft = checkpoints["draft"].get<std::string>();
            }
        }
        if (out.main.empty() && row.contains("checkpoint") && row["checkpoint"].is_string()) {
            out.main = row["checkpoint"].get<std::string>();
        }
        if (row.contains("labels") && row["labels"].is_array()) {
            for (const auto& label : row["labels"]) {
                if (label.is_string()) out.labels.push_back(label.get<std::string>());
            }
        }
        rows.push_back(std::move(out));
    }
    return rows;
}

PairingData load_pairings() {
    PairingData data;
    const json manifest = load_shipped_manifest();
    if (!manifest.is_object() || !manifest.contains("pairings")) return data;
#ifdef GUFO_MODEL_CATALOG
    const std::string catalog = GUFO_MODEL_CATALOG;
#else
    const std::string catalog = "src/cpp/resources/server_models.json";
#endif
#ifdef GUFO_MODEL_SEED
    const std::string seed = GUFO_MODEL_SEED;
#else
    const std::string seed = "docker/user_models.json";
#endif
    for (const std::string& path : {catalog, seed}) {
        for (const RegistryRow& row : gufo_rows(path)) {
            (gguf_shaped(row.main) ? data.chat : data.directory_rows).push_back(row);
        }
    }
    for (const auto& entry : manifest["pairings"]) {
        if (!entry.is_object()) return data;   // a malformed record voids the suite
        PairingRecord record;
        record.main = entry.value("main", std::string());
        record.draft = entry.value("draft", std::string());
        record.evidence = entry.value("evidence", std::string());
        record.verified = entry.value("verified", std::string());
        const json artifacts = entry.value("artifacts", json::object());
        for (auto it = artifacts.begin(); it != artifacts.end(); ++it) {
            if (it->is_string()) record.artifacts[it.key()] = it->get<std::string>();
        }
        data.records.push_back(std::move(record));
    }
    data.ok = !data.records.empty() && !data.chat.empty();
    return data;
}

// A pairing is the combination, so a record matches an entry only when the draft is
// present in both and spelled the same in both. This is the check that keeps a
// measured main from vouching for a draft nobody measured with it.
const PairingRecord* find_pairing(const PairingData& data, const RegistryRow& row) {
    for (const auto& record : data.records) {
        if (record.main != row.main) continue;
        if (record.draft.empty() != row.draft.empty()) continue;
        if (!row.draft.empty() && record.draft != row.draft) continue;
        return &record;
    }
    return nullptr;
}

// Which file a checkpoint string refers to. A `.gguf` suffix is its own answer; a
// variant alias is only answerable through the record, which is where the artifact
// that was actually read is named.
std::string artifact_file(const PairingRecord& record, const std::string& checkpoint) {
    const auto it = record.artifacts.find(checkpoint);
    if (it != record.artifacts.end()) return it->second;
    const size_t colon = checkpoint.find(':');
    const std::string tail =
        colon == std::string::npos ? checkpoint : checkpoint.substr(colon + 1);
    if (tail.size() > 5 && tail.compare(tail.size() - 5, 5, ".gguf") == 0) return tail;
    return {};
}

void test_pairing_conformance() {
    const PairingData data = load_pairings();
    check("pairings and the gufo registry rows that need them are readable", data.ok);
    if (!data.ok) return;

    // The rows outside the table must be the ones a tensor group cannot describe:
    // if a chat row ever lands there, it escaped the check below.
    bool directory_is_other_modalities = !data.directory_rows.empty();
    if (data.directory_rows.empty()) {
        std::printf("     no gufo row is directory-shaped, so nothing was excluded from the table\n");
    }
    for (const auto& row : data.directory_rows) {
        for (const auto& label : row.labels) {
            if (label != "transcription" && label != "tts" && label != "image" &&
                label != "embedding" && label != "reranking") {
                directory_is_other_modalities = false;
                std::printf("     %s/%s is directory-shaped but labelled %s\n", row.source.c_str(),
                            row.name.c_str(), label.c_str());
            }
        }
    }
    check("the gufo rows outside the pairing table are the directory-shaped modalities",
          directory_is_other_modalities);

    std::string unrecorded, wrong_partner;
    for (const auto& row : data.chat) {
        if (find_pairing(data, row)) continue;
        bool main_recorded_alone = false;
        for (const auto& record : data.records) {
            if (record.main == row.main) main_recorded_alone = true;
        }
        std::printf("     %s/%s main %s draft %s: %s\n", row.source.c_str(), row.name.c_str(),
                    row.main.c_str(), row.draft.empty() ? "(none)" : row.draft.c_str(),
                    main_recorded_alone ? "its partner is not what was recorded"
                                        : "no pairing recorded");
        if (main_recorded_alone) {
            wrong_partner = row.name;
        } else if (unrecorded.empty()) {
            unrecorded = row.name;
        }
    }
    check("every shipped gufo entry matches a recorded pairing", unrecorded.empty());
    check("a recorded main does not vouch for a draft it was not recorded with",
          wrong_partner.empty());

    bool every_record_ships = true;
    for (const auto& record : data.records) {
        bool shipped = false;
        for (const auto& row : data.chat) {
            if (record.main == row.main && record.draft == row.draft) shipped = true;
        }
        if (!shipped) {
            every_record_ships = false;
            std::printf("     pairing %s + %s names no shipped entry\n", record.main.c_str(),
                        record.draft.empty() ? "(no draft)" : record.draft.c_str());
        }
    }
    check("no pairing is recorded for an entry nothing ships", every_record_ships);

    const std::set<std::string> tiers = {"engine-loaded", "inventory-measured",
                                         "catalog-declared"};
    bool tier_known = true;
    bool date_shaped = true;
    bool artifacts_belong = true;
    bool measurement_names_files = true;
    for (const auto& record : data.records) {
        if (!tiers.count(record.evidence)) {
            tier_known = false;
            std::printf("     pairing %s claims evidence \"%s\"\n", record.main.c_str(),
                        record.evidence.c_str());
        }
        if (record.verified.size() != 10 || record.verified[4] != '-' || record.verified[7] != '-') {
            date_shaped = false;
        }
        for (const auto& artifact : record.artifacts) {
            if (artifact.first != record.main && artifact.first != record.draft) {
                artifacts_belong = false;
            }
        }
        if (record.evidence == "inventory-measured") {
            if (!record.main.empty() && !record.artifacts.count(record.main)) {
                measurement_names_files = false;
            }
            if (!record.draft.empty() && !record.artifacts.count(record.draft)) {
                measurement_names_files = false;
            }
        }
    }
    // A record that says it measured something has to name the file it measured, or
    // the tier means nothing a reader could re-check.
    check("every pairing names a known evidence tier", tier_known);
    check("every pairing carries the date its evidence was gathered", date_shaped);
    check("artifacts are named only for checkpoints the pairing itself holds", artifacts_belong);
    check("a measurement names the artifact files it read", measurement_names_files);
    check("at least one shipped pairing carries a measurement rather than a declaration",
          std::any_of(data.records.begin(), data.records.end(),
                      [](const PairingRecord& r) { return r.evidence != "catalog-declared"; }));

    // Two spellings of the same download are two claims: an alias is resolved to a
    // file only by looking at the snapshot, and pairing provenance must not do that
    // quietly in either direction.
    check("a pairing keyed by a variant alias does not also cover that alias's file name",
          std::none_of(data.records.begin(), data.records.end(), [](const PairingRecord& r) {
              return r.main == "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_XL.gguf";
          }));
    // The combination that started this change stays unrecorded on purpose. Adding
    // it would turn a refusal the engine makes into a row that says it ships.
    check("the refused MTP Q4_K_M draft is not recorded as a shipped pairing",
          std::none_of(data.records.begin(), data.records.end(), [](const PairingRecord& r) {
              return r.draft.find("mtp-") != std::string::npos &&
                     r.draft.find("Q4_K_M") != std::string::npos;
          }));

    const GufoCompat& table = GufoCompat::current();
    bool pairing_repos_declared = true;
    for (const auto& record : data.records) {
        for (const std::string* checkpoint : {&record.main, &record.draft}) {
            if (checkpoint->empty()) continue;
            if (table.resolve_family(repository_of(*checkpoint)).empty()) {
                pairing_repos_declared = false;
                std::printf("     pairing checkpoint %s resolves to no family\n",
                            checkpoint->c_str());
            }
        }
    }
    check("every pairing checkpoint names a repository the table declares a family for",
          pairing_repos_declared);
}

// ── Drift: re-read what the table says it copied ─────────────────────

std::vector<std::string> read_lines(const fs::path& path, bool& ok) {
    std::ifstream in(path);
    ok = static_cast<bool>(in);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

// "src/models/qwen/weights.cpp:78-87 accepted at :163, :263-266" → the file plus
// every cited range, in order, the first of which is the expression itself.
bool parse_citation(const std::string& citation, std::string& path,
                    std::vector<std::pair<int, int>>& ranges) {
    const size_t colon = citation.find(':');
    if (colon == std::string::npos) return false;
    path = citation.substr(0, colon);
    const std::string rest = citation.substr(colon + 1);
    size_t at = 0;
    while (at < rest.size()) {
        while (at < rest.size() && !std::isdigit(static_cast<unsigned char>(rest[at]))) ++at;
        if (at >= rest.size()) break;
        const int first = std::atoi(rest.c_str() + at);
        int last = first;
        while (at < rest.size() && std::isdigit(static_cast<unsigned char>(rest[at]))) ++at;
        if (at < rest.size() && rest[at] == '-') {
            ++at;
            last = std::atoi(rest.c_str() + at);
            while (at < rest.size() && std::isdigit(static_cast<unsigned char>(rest[at]))) ++at;
        }
        ranges.emplace_back(first, last);
    }
    return !ranges.empty();
}

// Storage types named inside one cited range, in the engine's own spelling minus
// the enum prefix.  Anything the table does not list shows up as extra, which is
// how a citation that drifted onto a wider expression is caught.
std::set<std::string> engine_types_in(const std::vector<std::string>& lines, int from, int to) {
    static const std::string needle = "GgmlType::k";
    std::set<std::string> found;
    for (int index = from - 1; index <= to - 1 && index < static_cast<int>(lines.size()); ++index) {
        if (index < 0) continue;
        const std::string& line = lines[static_cast<size_t>(index)];
        size_t at = line.find(needle);
        while (at != std::string::npos) {
            size_t end = at + needle.size();
            while (end < line.size() &&
                   (std::isalnum(static_cast<unsigned char>(line[end])) || line[end] == '_')) {
                ++end;
            }
            found.insert(line.substr(at + needle.size(), end - at - needle.size()));
            at = line.find(needle, end);
        }
    }
    return found;
}

// Where the downloaded artifacts live, if anywhere: --hub wins, then HF_HUB_CACHE.
// No home directory is guessed, because a wrong guess here would report "those
// files are fine" about a cache that holds nothing.
fs::path hub_cache(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--hub") return fs::path(argv[i + 1]);
    }
    const char* from_env = std::getenv("HF_HUB_CACHE");
    return (from_env && *from_env) ? fs::path(from_env) : fs::path();
}

// The layout the downloader writes: models--<org>--<repo>/snapshots/<rev>/<file>.
fs::path find_in_hub(const fs::path& hub, const std::string& repository,
                     const std::string& file) {
    const size_t slash = repository.find('/');
    if (file.empty() || slash == std::string::npos) return {};
    std::error_code ec;
    const fs::path snapshots = hub / ("models--" + repository.substr(0, slash) + "--" +
                                      repository.substr(slash + 1)) / "snapshots";
    if (!fs::is_directory(snapshots, ec)) return {};
    for (fs::directory_iterator it(snapshots, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code ignored;
        const fs::path candidate = it->path() / file;
        if (fs::is_regular_file(candidate, ignored)) return candidate;
    }
    return {};
}

// The pairing table says a combination loads. This says whether the table still
// agrees, for the entries whose files happen to be on this machine: entries whose
// artifacts were never downloaded are named as absent, never counted as a pass.
int judge_shipped_pairings(int argc, char** argv) {
    using lemon::backends::gufo::verdict_entry;
    const fs::path hub = hub_cache(argc, argv);
    std::error_code ec;
    if (hub.empty() || !fs::is_directory(hub, ec)) {
        std::printf("not measured (hub): pass --hub <dir> or set HF_HUB_CACHE to re-judge the "
                    "shipped pairings on their files\n");
        return 0;
    }
    const PairingData data = load_pairings();
    if (!data.ok) {
        std::printf("FAIL pairings are not readable for the live check\n");
        return 1;
    }

    int judged = 0;
    int absent = 0;
    int refused = 0;
    for (const auto& row : data.chat) {
        const PairingRecord* record = find_pairing(data, row);
        if (record == nullptr) {
            ++absent;   // already reported, by name, by the conformance half
            continue;
        }
        const std::string main_file = artifact_file(*record, row.main);
        const fs::path main_path = find_in_hub(hub, repository_of(row.main), main_file);
        const fs::path draft_path = row.draft.empty()
                                        ? fs::path()
                                        : find_in_hub(hub, repository_of(row.draft),
                                                      artifact_file(*record, row.draft));
        if (!fs::is_regular_file(main_path, ec) ||
            (!row.draft.empty() && !fs::is_regular_file(draft_path, ec))) {
            ++absent;
            // An alias has no file name until a pairing records one, so fall back to
            // the checkpoint string the registry actually uses.
            std::printf("     not downloaded: %s (%s%s)\n", row.name.c_str(),
                        main_file.empty() ? row.main.c_str() : main_file.c_str(),
                        row.draft.empty() ? "" : ", with its draft");
            continue;
        }
        const GufoCompatVerdict verdict =
            verdict_entry(main_path.string(), repository_of(row.main), draft_path.string(),
                          repository_of(row.draft), row.labels);
        ++judged;
        if (!verdict.accept) {
            ++refused;
            std::printf("FAIL %s is refused by the table: %s\n", row.name.c_str(),
                        verdict.explanation().c_str());
            continue;
        }
        std::printf("PASS %s is servable from its files%s\n", row.name.c_str(),
                    verdict.reason.empty() ? "" : (" — " + verdict.reason).c_str());
    }
    std::printf("%s pairings: judged %d of %zu shipped entries from %s, %d not downloaded\n",
                refused ? "FAIL" : "PASS", judged, data.chat.size(), hub.string().c_str(), absent);
    return refused;
}

int run_drift(int argc, char** argv) {
    const char* from_env = std::getenv("GUFO_SOURCE_DIR");
#ifdef GUFO_SOURCE_DIR_DEFAULT
    const fs::path root = (from_env && *from_env) ? fs::path(from_env)
                                                  : fs::path(GUFO_SOURCE_DIR_DEFAULT);
#else
    const fs::path root = (from_env && *from_env) ? fs::path(from_env) : fs::path();
#endif
    if (root.empty() || !fs::is_directory(root / "src")) {
        std::printf("SKIP: no gufo checkout at %s — set GUFO_SOURCE_DIR to re-read the cited ranges\n",
                    root.empty() ? "(unset)" : root.string().c_str());
        return 0;
    }

    const GufoCompat& manifest = GufoCompat::current();
    if (!manifest.usable()) {
        std::printf("FAIL manifest unusable, cannot check its citations: %s\n",
                    manifest.error().c_str());
        return 1;
    }

    std::map<std::string, std::vector<std::string>> files;
    std::vector<std::string> missing;
    for (const auto& family : manifest.families()) {
        for (const auto& group : family.groups) {
            std::string path;
            std::vector<std::pair<int, int>> ranges;
            if (!parse_citation(group.mirrors, path, ranges)) {
                std::printf("FAIL %s/%s cites nothing readable: %s\n", family.name.c_str(),
                            group.name.c_str(), group.mirrors.c_str());
                return 1;
            }
            const std::string key = (root / path).string();
            if (files.find(path) == files.end()) {
                bool ok = false;
                files[path] = read_lines(root / path, ok);
                if (!ok) missing.push_back(path);
            }
        }
    }
    // A citation whose file is absent means this checkout is not the tree the table
    // was read from. That is a mismatch to report, not a table defect.
    if (!missing.empty()) {
        std::printf("SKIP: %s has no %s — the checkout does not match the pinned engine\n",
                    root.string().c_str(), missing.front().c_str());
        return 0;
    }

    size_t groups = 0;
    size_t mismatches = 0;
    for (const auto& family : manifest.families()) {
        for (const auto& group : family.groups) {
            std::string path;
            std::vector<std::pair<int, int>> ranges;
            static_cast<void>(parse_citation(group.mirrors, path, ranges));
            const std::vector<std::string>& lines = files[path];
            // The table stores ids; the engine's source names types by enum, so the
            // comparison happens in names.
            std::set<std::string> declared;
            for (const uint32_t id : group.types) declared.insert(lemon::ggml_type_name(id));
            ++groups;

            const std::set<std::string> mirrored =
                engine_types_in(lines, ranges.front().first, ranges.front().second);
            if (mirrored != declared) {
                ++mismatches;
                std::printf("FAIL %s/%s cites %s:%d-%d, which accepts", family.name.c_str(),
                            group.name.c_str(), path.c_str(), ranges.front().first,
                            ranges.front().second);
                for (const auto& name : mirrored) {
                    if (!declared.count(name)) std::printf(" +%s", name.c_str());
                }
                for (const auto& name : declared) {
                    if (!mirrored.count(name)) std::printf(" -%s", name.c_str());
                }
                std::printf("\n");
            }
            // The remaining ranges are where the same expression is applied to more
            // names — or the call sites that name the tensors. A range wider than the
            // group is the hazard: the table would accept a type that site refuses.
            for (size_t index = 1; index < ranges.size(); ++index) {
                const std::set<std::string> extra =
                    engine_types_in(lines, ranges[index].first, ranges[index].second);
                for (const auto& name : extra) {
                    if (declared.count(name)) continue;
                    ++mismatches;
                    std::printf("FAIL %s/%s cites %s:%d-%d as the same expression, and it adds %s\n",
                                family.name.c_str(), group.name.c_str(), path.c_str(),
                                ranges[index].first, ranges[index].second, name.c_str());
                }
            }
        }
    }
    std::printf("%s drift: %zu groups re-read from %s\n", mismatches ? "FAIL" : "PASS", groups,
                root.string().c_str());
    return static_cast<int>(mismatches) + judge_shipped_pairings(argc, argv);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--drift") return run_drift(argc, argv);

    test_shipped_manifest();
    test_precedence();
    test_group_verdicts();
    test_family_resolution();
    test_entry_verdicts();
    test_broken_table();
    test_pairing_conformance();

    std::printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
