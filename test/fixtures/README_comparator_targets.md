# comparator_targets.toml

`comparator_targets.toml` is a machine-readable index of serialized fixtures in this directory.
It is a fixture index only, not comparator evidence. It flips no covered row.

Each `[[target]]` row names the fixture (`id`, `path`), the estimator target read off the fixture's
own header (`target`), the kind of evidence (`evidence_type = "julia_target"`), the comparator
status (`comparator_status = "not asserted"`) and a `claim_boundary` saying what the fixture does
not prove. "Not asserted" means the index makes no claim about whether R consumes the fixture or
whether an external comparator agrees; neither can be verified from this repository.

A fixture appearing here does not mean it is covered, matched or validated against anything. Do not
cite a row of this index as comparator coverage.

`test/test_comparator_targets.jl` checks that the file parses, that every row has all required keys
with non-empty values, and that every `path` exists. Add a row only for a fixture already on main.

Pattern borrowed from HSquared.jl's `test/fixtures/comparator_targets.toml`.
