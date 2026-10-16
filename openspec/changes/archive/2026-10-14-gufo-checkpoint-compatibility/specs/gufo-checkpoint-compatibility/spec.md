## Purpose

States, independently of the inference engine's source, which checkpoint artifacts the gufo engine can load — per model family, per tensor group, and including speculative draft companions — so that the engine-twin combinations lemonade offers stop including ones the engine will refuse, and so the pairings it ships are recorded against that description in CI with the evidence behind each one stated.

## ADDED Requirements

### Requirement: Engine-pinned compatibility manifest
The gufo backend SHALL describe the engine's loadable checkpoint formats in a declarative manifest rather than in hard-coded rules. The manifest SHALL declare the engine release it mirrors, and for each model family: the repositories that belong to the family, the tensor groups it distinguishes, the storage types accepted for each group, any shape or sharding constraint, and the rules that apply to a speculative draft companion. Each group SHALL name the engine source and line range it mirrors, and SHALL correspond to a single engine rule rather than a combination of two.

#### Scenario: Groups are narrower than the engine's type vocabulary
- **WHEN** the manifest is consulted for the Qwen3.8 dense chat family and a tensor in the normalization or recurrent-state group declares Q8_0
- **THEN** the verdict is reject, even though the engine implements Q8_0 elsewhere, because that group accepts F32 only

#### Scenario: Same type, different group, different verdict
- **WHEN** two checkpoints of the same family declare the same storage type, one on a projection tensor the engine supports in that type and one on a projection tensor it does not
- **THEN** the first is accepted and the second is rejected

#### Scenario: Same tensor name, different family, different verdict
- **WHEN** a tensor name that exists in two families with different engine rules declares a type one family accepts and the other refuses
- **THEN** each family's own group decides its verdict, and neither family's pattern list is widened to accommodate the other

#### Scenario: Stale pin is detectable
- **WHEN** the compatibility conformance check runs and a family's pinned engine release differs from the release the backend declares support for
- **THEN** the check reports that family instead of passing silently

#### Scenario: Mirrored source is verifiable
- **WHEN** the conformance check runs on a machine that has the pinned engine source available
- **THEN** it compares each group's declared type list against the cited source range and reports the group when they differ; when the source is unavailable the check reports that it skipped rather than reporting success

### Requirement: Per-artifact verdict
For a checkpoint artifact on disk, the gufo backend SHALL resolve the artifact's family from its repository and return a verdict of accept or reject by judging every tensor the artifact declares against the first group in declared order whose name rule matches that tensor, exact names taking precedence over patterns. A tensor whose name matches no declared group of a known family SHALL be treated permissively, leaving the engine's own load error as the backstop, and an artifact whose repository belongs to no declared family SHALL likewise be accepted. The absence of any tensor SHALL never produce a refusal, whether or not the group containing its name is one the engine loads only under a configuration variant; the manifest SHALL record those variant groups so the mirroring check knows which names to re-verify. An artifact whose header cannot be read SHALL be treated permissively. Where an artifact is split and the first shard declares no tensors, the shard that declares the tensors SHALL be the one judged.

#### Scenario: Projection type the engine lacks
- **WHEN** a Qwen3.8 dense chat checkpoint declares Q4_0 on its projection tensors
- **THEN** the verdict is reject, although Q4_0 is a type the engine can name

#### Scenario: Unknown tensor group
- **WHEN** a checkpoint of a known family declares a tensor whose name matches none of that family's declared groups
- **THEN** the verdict remains accept and the reason records that a group was unmatched

#### Scenario: Repository outside every declared family
- **WHEN** a checkpoint registered by a user names a repository that no family in the manifest claims
- **THEN** the verdict is accept, so an unmirrored publisher is never hidden by a gap in the manifest

#### Scenario: Tensor of a variant-only group absent
- **WHEN** an artifact declares none of the tensors of a group the engine loads only under a configuration variant
- **THEN** the verdict is unaffected, because absence is never a refusal for any group

#### Scenario: Tensor name claimed by two group rules
- **WHEN** one tensor name matches the rule of an earlier declared group and also the rule of a later one
- **THEN** the earlier declared group decides the verdict, and that precedence is asserted over a real tensor inventory so reordering the rules cannot silently move a name

#### Scenario: Metadata-stub first shard
- **WHEN** a split artifact's first shard declares zero tensors and its following shard declares them
- **THEN** the following shard's tensors decide the verdict

#### Scenario: Unreadable header
- **WHEN** the artifact's header cannot be parsed
- **THEN** the verdict is accept, so an unreadable file never hides a model that works

### Requirement: Draft companion judged with its own rules
When a model entry declares a speculative draft companion, the verdict SHALL be computed for the draft as well as the main checkpoint, using the family's draft rules. A draft that fails SHALL refuse the entry as a whole, because the combination cannot load even when the main checkpoint is fine.

#### Scenario: Main accepted, draft rejected
- **WHEN** a Qwen3.8-Flash-Next main checkpoint passes and its paired MTP draft declares Q4_K on a mixer tensor
- **THEN** the entry's verdict is reject, naming the draft artifact and the offending tensor

#### Scenario: Loadable pairing accepted
- **WHEN** the same main checkpoint is paired with a draft whose mixer tensors are Q8_0
- **THEN** the entry's verdict is accept

#### Scenario: No draft declared
- **WHEN** a model entry declares no draft companion
- **THEN** only the main checkpoint is judged

#### Scenario: Draft published by a third party
- **WHEN** a draft companion comes from a repository the manifest declares as a family in its own right
- **THEN** the draft is judged against its own tensor names under that family's rules rather than the host family's, because the two sets differ in both directions

#### Scenario: Draft from an undeclared repository
- **WHEN** a draft companion comes from a repository the manifest does not declare, for a host family that declares that companion label
- **THEN** the draft is judged against its own tensor names under the host family's rules, and is not accepted merely because its repository is unknown

### Requirement: Verdicts are explainable
A reject verdict SHALL identify the artifact, the offending tensor name, the storage type found, the group that produced the decision, and the manifest version used, and SHALL record that explanation in the server log so a refusal can be diagnosed without attempting a load. An accept verdict SHALL NOT produce log output for each judged artifact.

#### Scenario: Refusal is diagnosable from logs
- **WHEN** an entry is refused because of a draft companion's tensor format
- **THEN** one log entry names the draft path, the tensor, `Q4_K`, the group, and the manifest version

#### Scenario: Accepted artifacts stay quiet
- **WHEN** a registry rebuild judges many servable artifacts that all pass
- **THEN** no per-artifact log line is emitted for them

### Requirement: Shipped pairings are recorded with provenance
The manifest SHALL record every `main`/`draft` pairing that an entry in the built-in catalog or the container seed declares, keyed by repository together with the registry checkpoint string exactly as written in the registry — whether that string is a variant alias or a file name. Each record SHALL state what its claim rests on: that an engine process loaded the artifact, that the artifact was read from disk and every tensor it declares was judged by the groups claiming it, or that no artifact was reachable when the record was written, in which case the record asserts only the pairing the registry intends to ship. Records cover checkpoints that name a GGUF; a checkpoint naming a model directory is served whole and carries no tensor inventory, and which rows fall outside the table SHALL be asserted rather than assumed. Every gufo entry the catalog or the seed declares SHALL have to match a recorded pairing, checked without downloading weights.

#### Scenario: Unverified pairing fails the check
- **WHEN** a catalog or seeded entry declares a draft whose registry checkpoint string differs from every recorded pairing
- **THEN** the conformance check fails naming that entry, and the entry is not silently accepted because its main checkpoint alone would pass

#### Scenario: Alias and file name are distinct keys
- **WHEN** a pairing was verified through a variant alias and an entry names the underlying file instead
- **THEN** the check treats them as different claims and reports the mismatch rather than resolving one to the other

#### Scenario: A pairing nobody could test is disclosed rather than hidden
- **WHEN** a recorded pairing states that no artifact was reachable when it was written
- **THEN** entries still have to match it verbatim, and the check names it as untested instead of omitting it or counting it as verified

#### Scenario: Recorded pairings are re-judged where their files exist
- **WHEN** the check runs with access to a cache holding the artifacts a pairing names
- **THEN** that entry is judged through the manifest's rules and a refusal fails the check naming the entry, while an entry whose artifacts are absent is reported as not downloaded rather than as passing

### Requirement: Scope confined to the gufo recipe
The manifest SHALL govern verdicts for entries whose recipe is `gufo` only. Entries of other recipes, and every other backend's model resolution, download, and load behavior, SHALL be unaffected by its presence or by any change to it.

#### Scenario: Other engines untouched
- **WHEN** the manifest rejects an artifact for a gufo entry
- **THEN** the same checkpoint remains fully available to its original non-gufo entry
