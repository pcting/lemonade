#include "lemon/backends/gufo/gufo_compat.h"

#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"
#include <lemon/utils/aixlog.hpp>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lemon {
namespace backends {
namespace gufo {

namespace {

// Only schema 1 is defined; a newer table has to be refused as a whole rather
// than read with assumptions, which would put a parse gap in charge of a verdict.
constexpr int kSupportedSchema = 1;

// Mirror of gufo's FormatOf() block geometry (src/models/qwen38_flash_next/
// weights.cpp:19-41), used for the family's block_divides_cols rule.  Zero means
// the loader has no format for the type, which Binder::Get rejects the same way
// a wrong type is rejected (:83-88).
uint32_t block_size_of(uint32_t type) {
    switch (type) {
        case 0: case 1: case 30: return 1;       // F32, F16, BF16
        case 7: case 8: case 20: return 32;      // Q5_1, Q8_0, IQ4_NL
        case 12: case 13: case 14: return 256;   // Q4_K, Q5_K, Q6_K
        default: return 0;
    }
}

// '*' matches any run of characters, dots included; every other character is
// literal.  Deliberately narrower than fnmatch: a manifest must not gain meaning
// from a character class this matcher and the drift check would read differently.
bool glob_match(const std::string& pattern, const std::string& text) {
    size_t p = 0;
    size_t t = 0;
    size_t star = std::string::npos;
    size_t resume = 0;
    while (t < text.size()) {
        if (p < pattern.size() && pattern[p] == text[t]) {
            ++p;
            ++t;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            resume = t;
        } else if (star != std::string::npos) {
            p = star + 1;
            t = ++resume;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}

std::vector<std::string> string_list(const nlohmann::json& value) {
    std::vector<std::string> out;
    for (const auto& entry : value) out.push_back(entry.get<std::string>());
    return out;
}

// A storage type the manifest names that no gufo release can name is a table
// that drifted from the engine, and a rule built on it would never match, so it
// has to take the whole table out of service rather than judge silently.
std::vector<uint32_t> resolve_types(const nlohmann::json& value, const std::string& where) {
    std::vector<uint32_t> types;
    for (const auto& entry : value) {
        const std::string name = entry.get<std::string>();
        uint32_t id = 0;
        if (!ggml_type_id(name, id)) {
            throw std::runtime_error(where + " names storage type \"" + name +
                                     "\", which this engine cannot name");
        }
        types.push_back(id);
    }
    if (types.empty()) throw std::runtime_error(where + " declares no accepted types");
    return types;
}

// "…-00002-of-00003.gguf" for a requested shard index, preserving zero padding.
std::string sibling_shard(const std::string& path, int index) {
    static const std::regex shard_re(R"(([0-9]+)-of-([0-9]+)\.gguf$)", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(path, match, shard_re)) return {};
    const std::string pad = match[1].str();
    std::string digits = std::to_string(index);
    if (digits.size() < pad.size()) digits.insert(0, pad.size() - digits.size(), '0');
    return path.substr(0, static_cast<size_t>(match.position(1))) + digits +
           path.substr(static_cast<size_t>(match.position(1) + pad.size()));
}

}  // namespace

GufoCompat::GufoCompat() : GufoCompat(std::string()) {}

GufoCompat::GufoCompat(const std::string& path) {
    const std::string file =
        path.empty() ? utils::get_resource_path("resources/gufo_compat.json") : path;
    try {
        nlohmann::json doc = utils::JsonUtils::load_from_file(file);
        schema_version_ = doc.at("schema_version").get<int>();
        if (schema_version_ != kSupportedSchema) {
            throw std::runtime_error("unsupported schema_version " + std::to_string(schema_version_));
        }
        engine_pin_ = doc.at("engine").at("pin").get<std::string>();
        min_supported_ = doc.at("engine").at("min_supported").get<std::string>();
        for (const auto& item : doc.at("families").items()) {
            const nlohmann::json& source = item.value();
            GufoCompatFamily family;
            family.name = item.key();
            family.repos = string_list(source.value("repos", nlohmann::json::array()));
            family.block_divides_cols = source.value("block_divides_cols", false);
            family.variant_groups =
                string_list(source.value("variant_groups", nlohmann::json::array()));
            for (const auto& entry : source.value("groups", nlohmann::json::array())) {
                GufoCompatGroup group;
                group.name = entry.value("name", std::string());
                const std::string where = "family " + family.name + " group " + group.name;
                group.exact = string_list(entry.value("exact", nlohmann::json::array()));
                group.patterns = string_list(entry.value("patterns", nlohmann::json::array()));
                if (group.name.empty()) throw std::runtime_error(where + " has no name");
                if (group.exact.empty() && group.patterns.empty()) {
                    throw std::runtime_error(where + " claims no tensor names");
                }
                group.types = resolve_types(entry.value("types", nlohmann::json::array()), where);
                group.mirrors = entry.value("mirrors", std::string());
                if (group.mirrors.empty()) {
                    throw std::runtime_error(where + " cites no engine source to mirror");
                }
                group.why = entry.value("why", std::string());
                family.groups.push_back(std::move(group));
            }
            // Named, not inline: a range-for does not extend the lifetime of a
            // temporary inside its range expression, so iterating .items() of the
            // value() result straight away would read a destroyed object.
            const nlohmann::json companions =
                source.value("companions", nlohmann::json::object());
            for (const auto& companion : companions.items()) {
                const std::string target = companion.value().value("family", std::string());
                family.companions.emplace_back(companion.key(), target);
            }
            // A variant group that names nothing is a marker nobody reads, and the
            // drift check would quietly stop re-verifying the names it covered.
            for (const auto& marked : family.variant_groups) {
                const bool declared = std::any_of(family.groups.begin(), family.groups.end(),
                                                  [&](const GufoCompatGroup& g) { return g.name == marked; });
                if (!declared) {
                    throw std::runtime_error("family " + family.name +
                                             " marks variant group " + marked + ", which it does not declare");
                }
            }
            families_.push_back(std::move(family));
        }
        usable_ = true;
    } catch (const std::exception& e) {
        usable_ = false;
        error_ = e.what();
        families_.clear();
        LOG(ERROR, "Gufo") << "gufo_compat.json is unusable (" << error_
                           << "), so checkpoint checks fall back to the flat storage-type rule"
                           << std::endl;
    }
}

const GufoCompat& GufoCompat::current() {
    static const GufoCompat instance;
    return instance;
}

std::string GufoCompat::stamp() const {
    return "gufo_compat schema " + std::to_string(schema_version_) + " / " + engine_pin_;
}

const GufoCompatFamily* GufoCompat::family_named(const std::string& name) const {
    const auto it = std::find_if(families_.begin(), families_.end(),
                                 [&](const GufoCompatFamily& f) { return f.name == name; });
    return it == families_.end() ? nullptr : &*it;
}

std::string GufoCompat::resolve_family(const std::string& repository) const {
    for (const auto& family : families_) {
        if (std::find(family.repos.begin(), family.repos.end(), repository) != family.repos.end()) {
            return family.name;
        }
    }
    return {};
}

std::string GufoCompat::companion_family(const std::string& host_family,
                                         const std::string& label,
                                         const std::string& draft_repository) const {
    const std::string own = resolve_family(draft_repository);
    if (!own.empty()) return own;
    const GufoCompatFamily* host = family_named(host_family);
    if (host) {
        for (const auto& companion : host->companions) {
            if (companion.first == label) {
                return companion.second.empty() ? host_family : companion.second;
            }
        }
    }
    // An unlabeled or unknown companion of a known host is still judged, on the
    // host's rules: it is a checkpoint of that architecture, and permissive-by-
    // default belongs to the family being unknown, not to a draft of a family we
    // do know.
    return host_family;
}

const GufoCompatGroup* GufoCompat::claim(const GufoCompatFamily& family,
                                         const std::string& tensor) const {
    for (const auto& group : family.groups) {
        if (std::find(group.exact.begin(), group.exact.end(), tensor) != group.exact.end()) {
            return &group;
        }
    }
    for (const auto& group : family.groups) {
        for (const auto& pattern : group.patterns) {
            if (glob_match(pattern, tensor)) return &group;
        }
    }
    return nullptr;
}

const GufoCompatGroup* GufoCompat::group_claiming(const std::string& family_name,
                                                 const std::string& tensor) const {
    const GufoCompatFamily* family = family_named(family_name);
    return family ? claim(*family, tensor) : nullptr;
}

GufoCompatVerdict GufoCompat::verdict(const std::string& family_name,
                                      const GgufTensorTypes& inventory) const {
    GufoCompatVerdict result;
    if (!usable_) {
        result.reason = "manifest unusable";
        return result;
    }
    const GufoCompatFamily* family = family_named(family_name);
    if (!family) {
        result.reason = "family unresolved";
        return result;
    }
    for (const auto& tensor : inventory.tensors) {
        // A name no group claims is a tensor this loader never looks up, and the
        // engine loads what it does not validate without complaint.
        const GufoCompatGroup* group = claim(*family, tensor.name);
        if (!group) continue;
        const bool accepted =
            std::find(group->types.begin(), group->types.end(), tensor.type) != group->types.end();
        const uint32_t block = family->block_divides_cols ? block_size_of(tensor.type) : 1;
        const uint64_t cols = tensor.dimensions.empty() ? 1 : tensor.dimensions[0];
        if (accepted && block != 0 && cols % block == 0) continue;

        result.accept = false;
        result.reason = accepted ? "block size does not divide the tensor's columns"
                                 : "type not accepted by this tensor group";
        result.offending_tensor = tensor.name;
        result.type_name = ggml_type_name(tensor.type);
        result.group = group->name;
        result.manifest = stamp();
        return result;
    }
    result.reason = "every declared tensor group loads";
    return result;
}

GufoCompatVerdict GufoCompat::verdict(const std::string& family_name, const std::string& path) const {
    GgufTensorTypes inventory;
    if (!gufo_read_inventory(inventory, path)) {
        // The engine reads the header first and fails there; refusing a file we
        // cannot inspect would hide a model that loads fine.
        GufoCompatVerdict result;
        result.artifact = path;
        result.reason = "header unreadable";
        return result;
    }
    GufoCompatVerdict result = verdict(family_name, inventory);
    result.artifact = path;
    return result;
}

std::string GufoCompatVerdict::explanation() const {
    if (accept) return "accept: " + reason;
    std::string out = "refuse: " + reason;
    if (!artifact.empty()) out += " | artifact " + artifact;
    if (!offending_tensor.empty()) out += " | tensor " + offending_tensor;
    if (!type_name.empty()) out += " is " + type_name;
    if (!group.empty()) out += " | group " + group;
    if (!manifest.empty()) out += " | manifest " + manifest;
    return out;
}

GufoCompatVerdict verdict_entry(const std::string& main_path,
                                const std::string& main_repository,
                                const std::string& draft_path,
                                const std::string& draft_repository,
                                const std::vector<std::string>& labels) {
    const GufoCompat& manifest = GufoCompat::current();
    if (!manifest.usable()) {
        GufoCompatVerdict result;
        result.artifact = main_path;
        result.reason = "manifest unusable";
        return result;
    }
    const std::string host = manifest.resolve_family(main_repository);
    if (host.empty()) {
        GufoCompatVerdict result;
        result.artifact = main_path;
        result.reason = "family unresolved";
        return result;
    }

    GufoCompatVerdict result = manifest.verdict(host, main_path);
    if (result.accept && !draft_path.empty()) {
        const std::string label = gufo_speculative_label(labels);
        result = manifest.verdict(manifest.companion_family(host, label, draft_repository),
                                  draft_path);
    }
    if (!result.accept) {
        LOG(WARNING, "Gufo") << result.explanation() << std::endl;
    }
    return result;
}

bool gufo_read_inventory(GgufTensorTypes& out, const std::string& path) {
    if (!read_gguf_tensor_types(out, path)) return false;
    if (out.tensor_count != 0 || out.split_count <= 1) return true;
    for (int index = 2; index <= out.split_count; ++index) {
        const std::string sibling = sibling_shard(path, index);
        if (sibling.empty()) break;
        GgufTensorTypes candidate;
        if (read_gguf_tensor_types(candidate, sibling) && candidate.tensor_count > 0) {
            out = std::move(candidate);
            return true;
        }
    }
    // A metadata stub with no readable sibling declares nothing to judge.
    return true;
}

std::string gufo_speculative_label(const std::vector<std::string>& labels) {
    if (std::find(labels.begin(), labels.end(), "mtp") != labels.end()) return "mtp";
    if (std::find(labels.begin(), labels.end(), "dspark") != labels.end()) return "dspark";
    // Matches the flag wiring's fallback: a draft on a model that labels nothing
    // is wired as dflash2, so it has to be judged as one too.
    return "dflash";
}

}  // namespace gufo
}  // namespace backends
}  // namespace lemon
