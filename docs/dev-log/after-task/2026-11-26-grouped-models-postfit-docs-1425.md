# After-task: grouped-models postfit docs (#1425)

**Goal:** Align `docs/src/grouped-models.md` with live postfit calling conventions so readers do not copy one-argument `predict(fit)` onto ungrouped Laplace family fits.

**Change:** Docs-only prose clarifying that `GroupedGaussianFit` / `GroupedNonGaussianFit` retain training `Y` and support in-sample one-argument postfit, while ungrouped Laplace routes require `predict(fit, Y)`, `fitted(fit, Y)`, and `residuals(fit, Y)`.

**Tests:** `test/test_issue_1425_grouped_models_docs.jl` — 19/19 pass in 30.8s (`julia --project=. --startup-file=no -e 'include(...)'`, Julia 1.13.4, cloud agent).

**Rose:** Doc claim matches engine; no API change; grouped examples unchanged and verified runnable.

**refs #1425**