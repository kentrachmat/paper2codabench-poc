# Competition Bundle Validator

A lightweight Python validator for Codabench competition bundles.

This script verifies the structure of a competition bundle folder before upload. It checks required fields, validates file references, enforces structural constraints, and performs consistency checks across tasks, phases, pages, and leaderboards.

------------------------------------------------------------
What It Validates
------------------------------------------------------------

The validator performs the following checks:

- Presence of competition.yaml
- Required top-level fields:
  - title
  - image
  - tasks
  - phases
  - leaderboards
  - pages
  - terms
  - docker_image

Image
- Must be JPG, JPEG, or PNG
- File must exist in the bundle

Pages
- At least one page required
- Each page must have:
  - title
  - file
- Page files must exist
- Page files must not be empty

Tasks
- Unique task indexes
- scoring_program is required
- ingestion_program, input_data, reference_data are optional
- If optional files are provided, they must exist

Terms
- Terms file must exist
- Terms file must not be empty

Phases
- At least one phase required
- Sequential date validation
- Referenced files (public_data, starting_kit) must exist if provided

Leaderboards
- At least one leaderboard required
- Unique leaderboard indexes
- Required fields:
  - index
  - title
  - key
  - submission_rule
- At least one column required per leaderboard
- Columns must contain:
  - index
  - title
  - key
- Column indexes must be unique within each leaderboard


------------------------------------------------------------
Usage
------------------------------------------------------------

Run from the repository root:

```
python3 competition_bundle_validator.py <bundle_path>
```

Example:

```
python3 competition_bundle_validator.py ./my_competition_bundle
```

If the bundle is valid:
```
[*] Validating bundle at: ./my_competition_bundle
[+] Bundle is valid!
```

If validation fails:

```
[-] Validation Error: <error_message>
```
