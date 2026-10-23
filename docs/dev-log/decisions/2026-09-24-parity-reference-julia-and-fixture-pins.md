# Decision: Julia reference version for #323 evidence, NB2 fixture pin, Student gradient, Stage 0 L11 sign

Status: **ACCEPTED** by Shinichi in chat on 2026-09-24 ("OWED 2: A, OWED 4: A"), choosing
between options Claude drafted for OWED items 2 and 4 of
`docs/dev-log/handover/2026-09-24-claude-handover-closeout.md`.

## Why a decision was needed

Track A (Totoro, Julia 1.10.12) and the advisory Frozen R CI job (Julia 1.13.0) ran the same
frozen gllvmTMB pin `b4d5fee64` and disagreed on every holdout
(`docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md`):

| Holdout | Totoro, Julia 1.10.12 | CI, Julia 1.13.0 |
|---|---|---|
| NATIVE-06 (NB2) | data-hash guard refused; R check not reached | R side fails, `r_gradient_max` 2.43e-3 |
| NATIVE-10 (Student, Cell 9) | passes, Δ logLik 2.0e-8 | fails, R optimizer code 1, Δ logLik 2.86e-3 |
| NATIVE-12 (truncated NB2) | R side fails, `r_gradient_max` 5.90e-4 | passes |

The Stage 0 `loading_profile` fixture also pins L11 = −0.8, while R's MASK-B-PINS case pins
+0.8.

## OWED 2, option A

1. **Julia 1.10 LTS is the reference version for #323 evidence.** It is CI's primary version
   and the version Totoro runs. Results on 1.13 or other newer versions stay advisory.
2. **The NB2 fixture (NATIVE-06) is stored as data, not regenerated from a seed.** Every Julia
   version then fits the same numbers.
3. **The Student cell (NATIVE-10, Parity Cell 9) records `r_gradient_max`**, how close R's
   optimizer got to a flat point. It is recorded only; it is not a pass/fail gate.

Measured while writing this record (Mac Studio, same seed and code as
`test/parity/test_negbin_parity.jl`, SHA-256 of `vec(Float64.(Y))`):

| Julia | NB2 data hash | Matches the pin in `nb2_health.jl` |
|---|---|---|
| 1.10.0, 1.10.12 | `af68c91fe8b1e4e2…` | no |
| 1.12.6, 1.13.0 | `7abde2731134afe6…` | yes |

The handover's guess that the seeded random stream differs between Julia versions is
therefore confirmed: the pinned data was drawn on Julia 1.12 or later. The stored data will be
that pinned draw, so the existing data-hash pin does not change.

## OWED 4, option A

Keep both signs. The Stage 0 fixture keeps L11 = −0.8, and a comment on
`loading_profile_fixture_mask_b_pins()` now names R's +0.8 and where it is recorded. Nothing
was broken: each fixture's tests check against their own value, and switching the Stage 0 sign
would touch 13 references in two test files.

## Implementation

- OWED 4: the fixture comment lands with this record.
- OWED 2 items 2 and 3: a separate branch, followed by a local re-run of NATIVE-06 and
  NATIVE-10 on Julia 1.10.12 against the frozen oracle.

## Not decided here

- A Totoro re-run of the holdouts. That still needs its own time estimate (D-139).
- Promoting any ledger row, or any `Project.toml` change.
