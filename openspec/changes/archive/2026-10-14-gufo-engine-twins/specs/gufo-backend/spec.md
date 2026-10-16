# gufo-backend

## ADDED Requirements

### Requirement: Client model-option exposure

A gufo-recipe model SHALL expose its configurable options in client model-option surfaces: context size and gufo custom arguments. `gufo_backend` SHALL remain a round-trippable configuration value settled through the server API and applied at load time, but because gufo has a single supported backend, client surfaces SHALL NOT present a device-backend selector for it (design D7/D8). Values chosen in a client SHALL round-trip through the model configuration to the subprocess command line without being dropped or rewritten, and the presented options SHALL appear for gufo-recipe models the same way other recipes' options appear for theirs.

#### Scenario: Options are visible
- **WHEN** a client opens the model options for a gufo-recipe model
- **THEN** the context size and gufo arguments options are presented (not an empty option panel), and no device-backend selector is shown

#### Scenario: Backend fixed, not selectable
- **WHEN** a gufo-recipe model's options are inspected
- **THEN** the effective backend is gufo's single supported value (`rocm`) and no other value is offered in the client

#### Scenario: Options round-trip
- **WHEN** a client saves a gufo custom argument (for example `--sessions 4`) on a gufo model
- **THEN** the value persists in the model's recipe options and appears on the gufo subprocess command line at load time

#### Scenario: Reserved values still rejected
- **WHEN** a saved argument would override a Lemonade-reserved flag
- **THEN** the load is refused with the reserved-flag error, unchanged from the server-side contract
