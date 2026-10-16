## MODIFIED Requirements

### Requirement: Per-artifact verdict

For a checkpoint artifact on disk, the gufo backend SHALL resolve the artifact's family from its repository and return a verdict of accept or reject by judging every tensor the artifact declares against the first group in declared order whose name rule matches that tensor, exact names taking precedence over patterns. A tensor whose name matches no declared group of a known family SHALL be treated permissively, leaving the engine's own load error as the backstop, and an artifact whose repository belongs to no declared family SHALL likewise be accepted. The absence of any tensor SHALL never produce a refusal, whether or not the group containing its name is one the engine loads only under a configuration variant; the manifest SHALL record those variant groups so the mirroring check knows which names to re-verify. An artifact whose header cannot be read SHALL be treated permissively, and that covers a header which parses yet declares more than the file can hold: such an artifact SHALL be accepted as unreadable and SHALL NOT interrupt the registry build that asked for the verdict. Where an artifact is split and the first shard declares no tensors, the shard that declares the tensors SHALL be the one judged.

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

#### Scenario: Header declaring what the file cannot hold
- **WHEN** a file's header parses but declares far more tensors than the file has bytes to describe
- **THEN** the verdict is accept with the header treated as unreadable, and the registry build that asked is not interrupted
