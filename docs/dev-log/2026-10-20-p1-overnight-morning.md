# P1 overnight, for the 5 a.m. read

The parity goal is not met. C2, C3, C4, and X2 stay NOT_MET on this candidate. Nothing was merged. origin/main was not updated.

HEAD in `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` is `5cfc6754212588222779ccfd66ad7000375cf8d5`. `git rev-parse HEAD` returned that commit before this note was written. The commit was left there.

Six checks finished. Receipts, `GATES.md`, and tolerances were left unchanged. No source file was edited.

Temporal, in `docs/dev-log/2026-10-20-p1-f-temporal-1fo.md`, is STILL_OPEN. The tests already in the tree miss OU latent scores. They also miss a first-order paired latent-score check for both AR1 and OU. `test/test_temporal_lowrank.jl` compares the marginal likelihood and its gradient with a dense block, for AR1 and for OU, and it never calls `getLV`. `test/test_temporal_helpers.jl` checks `getLV` on one AR1 latent fit with `unique = true`, and that file has no OU fit. The F-TEMPORAL-1FO box was left unticked. No Julia process was started for this check.

Fungi, in `docs/dev-log/2026-10-20-p1-overnight-fungi.md`, is NO_PROVEN_BUG on RD-FUNGI-BINOMIAL. The log-likelihoods agree to 6.151276465971023e-08. The gaps that remain sit where R stopped, at a maximum absolute gradient of 0.00386035726088885. The receipt stays FAIL. The row is not admitted.

Ordinal, in `docs/dev-log/2026-10-20-p1-overnight-ordinal.md`, is NO_PROVEN_BUG on ORDINAL-LOGIT-RSZ. Julia's likelihood at R's covariance matches R's log-likelihood to 2.1864252630621195e-9. The loading-product gap on (t01, t15) stays 0.00012989127759771568, above the bar of 0.0001. The receipt stays FAIL. The row is not admitted.

Identity, in `docs/dev-log/2026-10-20-p1-overnight-identity-df8.md`, is DF8_PASS on the existing unique model. The test is `test/parity/test_gaussian_original_required.jl`. `native_dof` is 8. The run passed 31 checks: 12 in the native pair, 10 in the wide formula pair, and 9 in the reordered long formula. The largest likelihood gap is 7.958078640513122e-13. The df 5 bridge was not used. The stored receipts and `GATES.md` were left unchanged, so the scoreboard row is still open. In the case files, `CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL` and `CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE` remain FAIL, and `CORE070-FAMILY-00-IDENTITY-PUBLIC-R-BRIDGE` remains NOT_EXECUTED. The scoreboard row stays `partial_case_not_executed`. The receipt was not flipped.

The identity run printed PIN_REFUSED. `CORE070_SOURCE_PIN.toml` is present at `/Users/z3437171/local-scratch/a3cov-oracle/build/library/gllvmTMB/CORE070_SOURCE_PIN.toml`. The identity note says that marker matched the P1 pin on reference commit, namespace hash, source-tree hash, and archive hash. The required-mode loader then stopped because `find.package("gllvmTMB")` still pointed at the user library. The fit loaded gllvmTMB 0.7.1 from that oracle library. The run's receipt files went to `/tmp/family00-identity-df8-20261020` and were not copied into the repo.

Spider, in `docs/dev-log/2026-10-20-p1-overnight-spider-beetle.md`, is NO_PROVEN_BUG on RD-SPIDER-NB2. The grouped-NB2 Laplace objective at R's stored parameters matches R's log-likelihood to 8.050e-10. The two evaluations differ by 1.458e-6. Both numbers come from the same objective, once at Julia's stored point and once at R's stored point, and the difference matches the receipt gap of 1.457e-6. The receipt stays FAIL. The row is not admitted.

Beetle, in the same note, is BLOCKED_BY_R_PDHESS on RD-BEETLE-NB2. R's `pdHess` is false on the receipt, so the row stays unbound. The pathology split was not run. A local `beetle_wide.csv` has sha256 `5ea37a22fab9ca2fbb599e18e8fa28ed7477675f73db8add3b6dc0df0419f545`. The receipt hash is `1d7603ddceac61ea7179ed9519cb87bc5b7557ec1dd30f9267bd6224cd952c9b`. Those hashes differ, so that likelihood was not cross-evaluated. The row was not marked passed.

https://github.com/itchyshin/GLLVModels.jl/issues/986 remains the later resume point for the thirteen rows. It was not reopened, and it was not edited.

One Julia thread was used. No full suite was launched.

A second pass found no legal move. Identity stays unbound because the pin has no version field, the other twelve rows stay closed, temporal stays open, and the goal is not met.
