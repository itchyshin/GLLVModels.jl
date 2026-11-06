# bootstrap_Sigma_r n_cores via Julia threads (2026-11-02)

Branch claude/bsig-ncores-20261102, cut from claude/bsig-driver-fixes-20261102 (PR #1076 still open). No row status, scoreboard or receipt change.

## Design
`n_cores = k > 1` runs the already-simulated batch on `k` tasks pulling indices from an atomic counter; each replicate writes its own slot; counters are reduced afterwards in replicate order. `1 <= n_cores <= Threads.nthreads()` else `ArgumentError`. `InterruptException` from a worker is rethrown as such.

## Verification (Totoro, Julia 1.12.6, `-t 4`, OPENBLAS_NUM_THREADS=1, load average about 250-300 on 384 cores)
- Red: the new testset against the old driver failed/errored (n_cores = 2 refused) 2026-11-02.
- Green: whole `test/test_bootstrap_sigma_r.jl` at `-t 4`, 898 pass / 898 total (348 + 184 + 289 + 39 + 38 across the five top-level testsets), 10m46s wall, output saved untruncated on Totoro (~/bsig-ncores-final.log).
- Mutants (n_cores testset plus its helpers, `-t 4`): unmutated control passes 36/36 (before N10 was added). 12 of 12 mutants killed: reversed slot, non-atomic counter, swallowed worker error, bound +1, lower bound 0, spawn nthreads tasks instead of n_cores, always serial, wrong n_thrown, failed ignoring non-converged, progress dropped when threaded, interrupt wrapped, slot-1-only write. The non-atomic counter survived the first 36 assertions and was killed 3 of 3 times only after adding N10 (3000 fast replicates); that kill is probabilistic by nature of a race.
- Speedup, one real fit (p = 30, n = 200, K = 2, has_diag, n_boot = 99, what = Sigma + communality, seed 1; machine heavily loaded by other users, so wall times are noisy and only indicative):
  n_cores = 1: 254.1 s (2.57 s per replicate); n_cores = 2: 160.3 s (1.62 s; 1.58x); n_cores = 4: 123.4 s (1.25 s; 2.06x). Draws identical across all three (`isequal`).
  The 1.2 s per replicate quoted for an unloaded machine was not reproduced here (2.57 s at n_cores = 1 under load).

## Not verified
- Scaling on an unloaded machine; behaviour with user-supplied `_refit` that is not thread-safe (documented as the caller's duty).
- The threaded legs of the test file only run when Julia has at least 2 (or 4) threads; the default `Pkg.test()` is single-threaded, where they are skipped (refusal tests still run).

## Update after race review (head after this commit)
- Fixed: an interrupt on the calling task now sets `stop[]` and waits for in-flight refits (no orphaned workers); the fitter's warn-once set in `src/fit.jl` (`_UNIDENTIFIED_WARNED`) is lock-guarded (the one src/fit.jl change). Tests N11 (interrupt scheduled on the main task, workers must stop) and N12 (warn-once exactly once, 300 released-together trials plus driver runs).
- Merged origin/main (no conflicts). Whole file on Totoro `-t 4`: 913 pass / 913 (348 + 184 + 289 + 39 + 47), 9m40s, log ~/bsig-ncores-final2.log.
- Mutants: 16 of 16 killed, unmutated control 47/47. New: no stop on interrupt (M13), no lock (M14), no wait for in-flight (M15), stop flag not checked (M16).
- N12 hammers at n_cores = 4 (the cap of 4 cores); the reviewer's n_cores = 8 case was not run.
