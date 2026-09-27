# Review of PR #546, iSDM build (head e2b53b25b), 2026-09-27: NON-BLOCKING

Fresh P1 install reproduced every recorded number (door logLik -134.938415169836; xobj -134.93841516967774; gradients; theta_diag_B of the default fit). Tests 158/158, admission 41/41, paired 213 + 3 broken. Kernel: convergence rule a faithful copy of the merged _mixed_laplace_mode; cloglog copy branch-for-branch with nested ForwardDiff weight; one-step implicit gradient passes FD; adversarial inputs (separation, 1e6 counts, one-row cells, detection offsets, K = 0, eta 800 sentinel) behave. Refusals match R's order and cli text; Julia-only fences documented. Scope clean.

1. MEDIUM. Polished R optimum did not converge (code 1) on predict (no movement) and srcform_pois: disclose; assert polished_convergence where 0.
2. MEDIUM. @test_broken on three door b_fix rows can mask regressions: use a loose absolute bound (max gap 2.1e-5) instead.
3. LOW. Asymmetric predict branch disappears with 2.
4. LOW. Factor levels by Julia byte order vs R locale collation: document.
5. LOW. Separated detection arms converge with drifting intercepts, as in R.

Deviation 1 (unique = TRUE, R's latent() default): theta_diag_B adds s_B(t, s) ~ N(0, exp(theta_d[t])^2) per trait and unit, integrated by Laplace (p extra dimensions per cell). R's public-door tests use the default but assert nothing about theta_diag_B. On both fixtures it runs to the boundary because p = 2, K = 1 is not identified (Anderson-Rubin needs p >= 2K + 1). Port cost 1 to 2 days via Lambda_aug = [Lambda diag(exp(theta_d))], z_aug ~ N(0, I_{K+p}), with a p >= 3 fixture. Reviewer recommendation: refuse now with a message printing the exact formula, port as follow-up "ISDM-PSI".
