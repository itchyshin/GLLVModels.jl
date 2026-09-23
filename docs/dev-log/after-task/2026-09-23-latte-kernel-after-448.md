# After-task: Latte-kernel after #448 (2026-09-23)

Active lenses: Shannon, Ada, Noether, Curie (perspectives). Spawned subagents: none.

## Scope

G0 next-after-448: merge #448, then keyword-gated kernel-first Latte arc
(diag precision / identical log-link). NM-alone forbidden. Totoro certify.

## Outcome

1. **#448 merged** as `35790b91d` (Frozen-R advisory only).
2. Branch `cursor/latte-kernel-20260923` (worktree
   `~/local-scratch/lanes/GLLVM.jl-latte-kernel-20260923b`).
3. **S3 kernel** (default OFF):
   - `diag_precision` / `reuse_identical_hf_ho` on `joint_grouped_laplace_loglik`
   - `diag_precision_kernel` on `fit_grouped_nongaussian` (passes both)
   - mid-loop `_GroupedDiagFactor`; final `.factor` stays CHOLMOD for S8
   - Poisson+LogLink may share Hf≡Ho; NB2+LogLink correctly keeps reuse cold
4. **S4 identity PASS** (rtol 1e-8): 26/26; Δll ≤ 3e-16 on Poisson / NB2 /
   phylo-correlated W; phylo GLM dense anchor intact.
5. **Timing:** not quoted (identity held; no public speed claim).
6. **NM:** `nelder_mead::Bool=true` unchanged.

## Checks

- Laptop: `test/test_latte_kernel_identity.jl` → 26/26 PASS
- Totoro: `test/test_latte_kernel_identity.jl` → **26/26 PASS**

## Rose

No README/NEWS speed claim. Keyword default OFF. Oracle scalar path not
conflated with joint CHOLMOD.

## Next

Draft PR; Shinichi decides whether to flip `diag_precision_kernel` default ON
after reviewing identity + (optional) retime.
