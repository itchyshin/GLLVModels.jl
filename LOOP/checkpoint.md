# Checkpoint: honest 0.7 R↔Julia true parity (`/goal` armed in Cursor)

**Status 2026-09-25:** item 4 below ("T4 realistic-size second-order: Totoro
grid; D-139 ack before spend") is stale on two counts. First, the T4 P6 grid
it names (`p in {20,50}`, `n in {500,2000}`, Gaussian/Poisson/NB2) already
closed with 12 of 12 cells passing on 2026-09-05 (PR #297; receipts
`docs/dev-log/core070/t4-p6-*-receipt-2026-09-05.json`). Second, the D-139
ack this checkpoint is waiting on was given 2026-09-24 for the separate #323
frozen-reference smoke (Totoro Track A ran; receipts
`docs/dev-log/core070/totoro-323-track-a-20260924/`, after-task
`docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md`). Also
landed since this checkpoint's "Rehydrate (2026-09-17)" line: PR #478
(2026-09-25, NB2 restart at the Poisson boundary). The rest of this file is
left as-is; see `docs/dev-log/handover/2026-09-24-claude-handover-closeout.md`
for a more current picture.

GOAL: see [`LOOP/GOAL.md`](GOAL.md). STATE: DestB DONE; true-parity vs frozen 0.7.0; Project.toml = 0.3.0.

CreateGoal: armed in parent chat (2026-09-14). Do not mark complete.

**LANE (2026-09-15):** Mac Studio owns true-parity. START HERE: [`docs/dev-log/handover/2026-09-15-mac-studio-true-parity-handover.md`](../docs/dev-log/handover/2026-09-15-mac-studio-true-parity-handover.md) (canonical plan: [`docs/dev-log/plans/2026-09-05-true-parity-ultra-plan.md`](../docs/dev-log/plans/2026-09-05-true-parity-ultra-plan.md)).

ARC IN PROGRESS: true-parity tranche. Board: [`docs/dev-log/2026-09-14-true-parity-pending-board.md`](../docs/dev-log/2026-09-14-true-parity-pending-board.md).
Morning wake: [`docs/dev-log/handover/2026-09-16-morning-wake-briefing.md`](../docs/dev-log/handover/2026-09-16-morning-wake-briefing.md).
Paste packet (canonical): [`docs/dev-log/owed/2026-09-16-post-402-paste-packet.md`](../docs/dev-log/owed/2026-09-16-post-402-paste-packet.md).

Rehydrate (2026-09-17): GLLVM.jl `origin/main` @ **`e590eb9ec`** ([#418](https://github.com/itchyshin/GLLVM.jl/pull/418) paste-ready status; #402 runbooks @ `08ca9e487`; #401 @ `c33745302`; #357 bridge @ `5ee6dc596`; #391 @ `c4dba35c4`); gllvmTMB origin/main @ `02b46cfc8`.

CLOUD STOP: Ungated cloud queue **exhausted** after **#391** + docs through **#418**, **#401** @ `c33745302`, **#402** runbooks @ `08ca9e487`. No further ungated cloud engine slices. No mergeable leftover ours PRs. Next requires Shinichi pastes only.

NEXT (ranked):

1. **Shinichi pastes only** (exact strings):
   - `accept delta dispersion A`
   - `G0 Stage 1`
   - `S4 probe yes`
   - `ack Totoro D-139 #323 Track A`
2. **#357** bridge logLik receipts **MERGED** @ `5ee6dc596` (lognormal + truncated-Poisson live Δ).
4. T4 realistic-size second-order: Totoro grid; D-139 ack before spend (Mac / paste).
5. Project.toml stays 0.3.0.

IN FLIGHT: Skipped #363/#314 CONFLICTING DRAFT. **#384** CLOSED superseded. **#357** MERGED. Cloud: **paste-gated STOP**.

DONE this tranche: (prior SO tranche through #378); **#385 through #397**; **#391** Tweedie estimated-power SO @ `c4dba35c4` -> **PARTIAL**; **#357** bridge logLik receipts @ `5ee6dc596`; **#401** node24 CI @ `c33745302`; **#402** paste-gated runbooks @ `08ca9e487`; docs tip chain through **#418** @ `e590eb9ec`.

DISPOSED: #323 waive; matched-θ C; §2 Hessian A; arcG Julia-only ACCOUNTED (#358); Rose dual-PR HOLD (#384/#391) → **#391 winner**.

OPEN GATES: QS4; Stage 1; version bump forbidden; Delta dispersion paste: all wait on Shinichi paste.

HOLD OUTS (SO): GP-1 / Student-t free ν / Delta species dispersion / BB shared-φ **φ pairing** / Λ raw / Tweedie **jointly-optimised** power still OUT; Ordinal+Lognormal+Trunc*+Multinomial FE+Student-t fixed-ν+BB shared-φ+Tweedie shared+estimated-power **cells** PARTIAL (native Wald + paired/EOO toy cells; option A plug-in); lognormal + truncated-Poisson **bridge logLik** live Δ on main (#357).

RESUME: Mac owns programme. Tip `e590eb9ec` (#418 paste-ready status merged). Cloud ungated exhausted. Remainder: Shinichi pastes only. No Stage 1 / S4 / Totoro / Delta without paste; no Project.toml bump; no gllvmTMB engine surgery. Goal **not** complete.
