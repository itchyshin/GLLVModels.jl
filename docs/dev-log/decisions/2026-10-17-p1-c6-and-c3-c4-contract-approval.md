# P1 C6 and C3/C4 contract approval

Date: 2026-10-17

Shinichi approved the recommended C6 and C3/C4 contract choices in chat. This
records the scope of that approval; it does not claim any measurement, row
admission, merge, or completion.

## C6: scoped R twin rule

Use `SCOPED_R_TWIN` only when the evidence identifies the exact pinned R
counterpart, the same public route and estimand, and an executable receipt for
the stated P1 scope, with exclusions stated explicitly. Names alone do not
establish a twin. Unmeasured or mismatched scopes stay held. The 37-name C6
inventory is not approved wholesale. The later, name-bounded approvals below
apply only to the ten listed Decision 2 scopes and 17 listed Decision 3–4
extras; Decision 5 remains held. This approval does not sign the other C6
names, admit aliases beyond the measured scopes, or alter the scoreboard.

## C3/C4: bounded diagnostic contract

Proceed with bounded diagnostics and pilot measurements under the existing
contract. Preserve all current tolerances. C3 uses the signed `1e-5` gradient
rule. C4 comparisons use returned-fit quantities. A diagnostic or pilot result
does not itself bind a row; row admission still requires its canonical gates.
Private urbanisation data remain out of scope unless separately located and
authorised. Failed fits, blockers, and all-attempt denominators remain visible.

### Exact four-fit method approved on 2026-10-17

Shinichi approved implementation and four diagnostics under this exact
contract. This approval does not admit a scoreboard row, authorize polishing
or slices, or permit a merge.

- Capture only the returned fits for `ordinal`, `spider`, `beetle`, and `fungi`.
  Reconstruct the objective at the returned parameter vector using the same
  data, parameter map, normalization, and inner-mode controls. Set
  `reconstruction_atol=0.0`; any nonzero discrepancy is retained as
  `RECONSTRUCTION_FAILED`. Do not round, subtract constants, or widen the bound.
- For ordinal, use the existing central finite-difference gradient with
  `h=1e-5`, inner `maxiter=100`, and `tol=1e-9`. For spider and beetle, use the
  existing analytic grouped-NB2 gradient. For fungi, use the existing analytic
  covariate-family gradient. Keep inner controls at `maxiter=100` and
  `tol=1e-9`. A C4 gradient above `1e-5` is diagnostic and does not reject
  returned-fit binding evidence. The signed N9 gradient bound remains `1e-5`.
- Keep convergence, iteration count, reported log-likelihood, input identity,
  returned coordinates, reconstruction result, and gradient result together.
  Stop before CI, prediction, point replacement, or `DONE`. These captures are
  diagnostic only and do not bind a row; canonical receipt gates remain
  required for row admission.
- Run the four fits sequentially on Totoro, with one Julia thread and
  `OPENBLAS_NUM_THREADS=1`, using fresh output directories. The estimate is
  10–30 minutes per fit and two hours total. Stop each run at 30 minutes, and
  do not start when current usage would exceed the 150-core cap. Private
  urbanisation data and the `urban` fit are excluded.

No fit had run under this contract when it was approved. Failed fits and
blockers remain visible; approval of the diagnostic method is not evidence of a
successful capture.

## C6: individual scope approvals, 2026-10-17

Shinichi explicitly approved the ten scoped counterpart proposals in Decision 2
of `docs/dev-log/plans/2026-10-07-p1-c6-signature-packet.md`, covering eleven
Julia export names because `StudentT` and `StudentTFamily` are one object. The
approved scopes are exactly those in the packet, including `fit_gllvm` only for
the measured NB2 route and named outputs. The Truncated-NB2 bridge refusal is
accepted only as boundary evidence. This approval does not broaden any route,
admit an alias outside its listed scope, or change any scoreboard row.

Shinichi also approved all 17 proposed Julia extras: the 13 names in Decision 3
and the four names in Decision 4 (`GllvmModel`, `StatsAPI`,
`random_balanced_tree`, and `shrinkage_factor`). Their approved disposition is
`KEPT_AS_JULIA_EXTRA`, under the packet's cited basis. The ten held alias/CHECK
scopes in Decision 5 remain unresolved. These approvals do not change any
scoreboard row or claim package-wide parity.

## Authority boundary

This approval does not authorize changed numerical tolerances, row admissions,
scope expansion, private-data access, P1 public claims, merges, releases, or
changes to the gllvmTMB reference. The C3/C4 diagnostics may test the exact
approved scope and report outcomes; any change to model identity, admission,
or classification requires a separate explicit decision.

## Maintainer follow-up: retain the remaining held scopes

On 2026-10-17, Shinichi directed that the nine remaining names stay held until
their separate fit or behavioural routes are measured: `NBFit`, `Ordinal`,
`TweedieFit`, `ZIB`, `ZINegBin`, `fit_animal_latent_gllvm`,
`fit_kernel_latent_gllvm`, `gllvm`, and `link_residual`. The unapproved remainder
of `fit_gllvm` also stays held outside its signed NB2 route. This is an explicit
`HOLD_PENDING_EVIDENCE` disposition for those nine names: they remain held until
route-specific evidence supports a later maintainer decision. It does not classify
them as Julia extras, admit a scoreboard row, or establish parity. C6 remains open
until the checker and assembler recognize this narrow hold disposition and the
checkpoint's negative controls prove that it cannot be used for any other name or
to admit a scoreboard row.
