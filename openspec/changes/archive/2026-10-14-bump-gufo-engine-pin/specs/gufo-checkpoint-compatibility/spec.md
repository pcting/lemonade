## MODIFIED Requirements

### Requirement: Engine-pinned compatibility manifest

The gufo backend SHALL describe the engine's loadable checkpoint formats in a declarative manifest rather than in hard-coded rules. The manifest SHALL declare both the engine release it mirrors and the oldest release whose rules it still matches, that floor being justified by the mirror re-diff rather than assumed, and for each model family: the repositories that belong to the family, the tensor groups it distinguishes, the storage types accepted for each group, any shape or sharding constraint, and the rules that apply to a speculative draft companion. Each group SHALL name the engine source and line range it mirrors, and SHALL correspond to a single engine rule rather than a combination of two. The mirrored release SHALL be the release the shipped image bundles, and the declared floor SHALL be the backend's runtime minimum version, so the two are checked against separate anchors and either can move without silently moving the other.

#### Scenario: Groups are narrower than the engine's type vocabulary
- **WHEN** the manifest is consulted for the Qwen3.8 dense chat family and a tensor in the normalization or recurrent-state group declares Q8_0
- **THEN** the verdict is reject, even though the engine implements Q8_0 elsewhere, because that group accepts F32 only

#### Scenario: Same type, different group, different verdict
- **WHEN** two checkpoints of the same family declare the same storage type, one on a projection tensor the engine supports in that type and one on a projection tensor it does not
- **THEN** the first is accepted and the second is rejected

#### Scenario: Same tensor name, different family, different verdict
- **WHEN** a tensor name that exists in two families with different engine rules declares a type one family accepts and the other refuses
- **THEN** each family's own group decides its verdict, and neither family's pattern list is widened to accommodate the other

#### Scenario: Stale mirror is detectable
- **WHEN** the compatibility conformance check runs and the manifest's mirrored engine release differs from the release the shipped image bundles
- **THEN** the check reports that instead of passing silently

#### Scenario: Mirrored source is verifiable
- **WHEN** the conformance check runs on a machine that has the pinned engine source available
- **THEN** it compares each group's declared type list against the cited source range and reports the group when they differ; when the source is unavailable the check reports that it skipped rather than reporting success

#### Scenario: Mirror and floor are anchored separately
- **WHEN** the manifest's mirrored release moves to a newer engine whose mirrored call sites are unchanged
- **THEN** the declared floor stays where the evidence puts it, and the check passes because the floor still equals the backend's runtime minimum version while the mirror equals the bundled release

#### Scenario: Floor is backed by the re-diff
- **WHEN** the manifest declares the oldest release whose rules it still matches
- **THEN** that declaration rests on the mirror re-diff of each cited file rather than on the mirrored release's own version number

## ADDED Requirements

### Requirement: An engine pin change re-opens engine-loaded evidence

When the mirrored engine release changes, every record whose evidence is that an engine process loaded the artifact SHALL be re-verified against the release now mirrored before the change lands, and each such record SHALL name the release that produced its evidence. Records resting on a read of the artifact, or on the registry's stated intent, carry over unchanged because neither claim was about an engine build. A record that has not been re-verified SHALL NOT continue to state that an engine loaded it.

#### Scenario: Engine-loaded record is re-earned on a bump
- **WHEN** the mirrored release moves and a record's evidence is that an engine loaded the artifact
- **THEN** the artifact is loaded by the newly mirrored release before the change lands, and the record names that release

#### Scenario: Measured and declared records carry over
- **WHEN** the mirrored release moves and a record's evidence is a read of the artifact or the registry's stated intent
- **THEN** the record is unchanged

#### Scenario: An unverified claim is not left standing
- **WHEN** a bump lands without re-loading an artifact whose record says an engine loaded it
- **THEN** that record no longer states that an engine loaded it, rather than naming a release that no longer ships
