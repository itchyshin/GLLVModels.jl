# Covariance receipts at gllvmTMB P1: how to re-run them

These receipts back `../../case-map-covariance.json` (the 17 covariance rows of the P1 carry scan).
Every file here is written by `tools/core070_covariance_p1_receipts.py` from raw run directories;
nothing is edited by hand. This note lists what a re-run needs. It does not change any gate.

## Inputs

- **P1 oracle library.** Built with `tools/core070_build_oracle.py` at `GLLVM_PARITY_PIN=P1`:
  `prepare --repo <gllvmTMB clone> --destination <oracle>/source`, then
  `build --archive <oracle>/source/gllvmTMB-core070.tar --source-receipt <oracle>/source/source.json
  --destination <oracle>/build --r-binary <R>`, then `verify --destination <oracle>/build`.
  The library is `<oracle>/build/library`; it holds `gllvmTMB/CORE070_SOURCE_PIN.toml`.
- **Oracle receipts.** `oracle/build.json` and `oracle/source.json` are the receipts of that build.
  At `GLLVM_PARITY_PIN=P1` the required runner reads them from here (P0 still reads the untracked
  `.unlazy/core070-aghq/` paths). Before use it checks `reference_commit`, `source_tree_sha256`
  and `archive_sha256` (and `namespace_sha256` for `source.json`) against
  `tools/core070_oracle_pins.toml`, and checks that `build.json`'s `installed_tree_sha256` is the
  library it just validated. A mismatch is an error.
- **A clean commit.** Run every batch and the receipt tool from the same commit with no modified
  tracked files. The receipt tool records that commit as `glvmodels_commit`, refuses a dirty tree
  (unless `--allow-dirty`, which is then recorded), and refuses if any harness file in the runparity
  run's execution inventory differs from HEAD.

## Environment

```
GLLVM_PARITY_PIN=P1
CORE070_PARITY_REQUIRED=1
GLLVM_PARITY_R_SOURCE_PIN=<oracle>/build/library/gllvmTMB/CORE070_SOURCE_PIN.toml
R_HOME=$(R RHOME)
R_LIBS=<oracle>/build/library:<the library holding TMB, Matrix, jsonlite>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=4
```

`R_LIBS` must list the oracle library first and also the library that holds `TMB`; setting
`GLLVM_PARITY_R_LIBS` alone does not put the oracle first for `find.package`.

## Batches (run from the worktree root)

1. Default-control mode fits (the baseline the tight-control run checks against). It exits
   nonzero with three R-gradient failures (ORD-DEP, ANIMAL-DEP, KERNEL-DEP), as at P0; its
   `result.toml` is kept as `mode-fits-default-control/result.toml`.
   `julia --project=test/parity tools/core070_covariance_mode_fits.jl <out>/mode-fits-default`
2. Copy that run's `*.rds` and `result.toml` into `./baseline/` in the worktree root. The
   tight-control mode fits inside runparity stop with "retained baseline required" without it.
   Remove `baseline/` afterwards.
3. Runparity, 18 cases. Case subsets must be whole fixture groups (a Gaussian covariance fixture
   group is refused if only part of it is requested); these 18 ids are four complete groups.
   `GLLVM_PARITY_TESTS=1 GLLVM_PARITY_RECEIPT_DIR=<out>/runparity
   CORE070_PARITY_CASE_IDS=<the 18 ids, comma-separated> julia --project=test/parity test/parity/runparity.jl`
4. Wave6 batch (whole 10-case batch; it reads FAIL on the postfit nobs case, see the case map).
   `Rscript --vanilla tools/core070_wave6_conversion_batch.R <oracle>/build/library <out>/wave6`
5. R-only grammar batch. `CORE070_READBACK_DIR` is the extracted P1 archive
   (`tar -xf <oracle>/source/gllvmTMB-core070.tar -C <out>/readback`).
   `CORE070_READBACK_DIR=<out>/readback Rscript --vanilla tools/core070_covariance_batch.R <oracle>/build/library <out>/cov-batch`
6. Public R bridge boundary.
   `Rscript tools/core070_covariance_bridge_boundary.R <out>/bridge/bridge-boundary-p1.tsv`

Steps 4 and 5 stop unless the library's `CORE070_SOURCE_PIN.toml` matches the P1 entry of
`tools/core070_oracle_pins.toml`, and record the marker in their receipts; their verifiers check it.

## Receipts

```
python3 tools/core070_covariance_p1_receipts.py --runparity <out>/runparity \
  --default-modes <out>/mode-fits-default --wave6 <out>/wave6 --cov-batch <out>/cov-batch \
  --bridge-tsv <out>/bridge/bridge-boundary-p1.tsv --oracle-dir <oracle> --runtimes <runtimes.json>
```

The tool runs the wave6 and covariance-batch verifiers itself and writes their output as
`verify.log`. A row binds as `numeric` only if its batch verifier passed, or if
`--numeric-exceptions` supplies an exception signed by the maintainer.

## Why `oracle/source.json` is tracked (1.25 MB)

It is reproducible: `prepare` at the pinned commit regenerates it byte for byte
(sha256 `3ec6631ba855624d5b0e9e66772d2e3b3049270c17047835333e7663d6eb2ad8`). It stays tracked
because it is the source receipt the P1 required runner validates, copies into every run and
hashes into `run.toml`. Dropping it would put back the untracked, hand-staged path that the
runner used to depend on.
