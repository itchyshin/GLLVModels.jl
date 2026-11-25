# Live test (GLLVModels.jl #1272): real RowEffectFit / RowRandomFit through the R bridge.
# Needs R JuliaConnectoR + Julia with this repo's project instantiated.
# Run from the repository root: Rscript r/test_row_dispersion_live.R
if (!requireNamespace("JuliaConnectoR", quietly = TRUE)) { cat("skip: JuliaConnectoR missing\n"); quit(status = 0) }
Sys.setenv(GLLVM_JL_PATH = normalizePath("."))
source("r/gllvmtmb_julia.R")
set.seed(1); n <- 30; p <- 5
y <- list(negative.binomial = matrix(rpois(n * p, 3), n, p),
          beta = matrix(rbeta(n * p, 2, 3), n, p),
          Gamma = matrix(rgamma(n * p, 2, 1), n, p))
for (fam in names(y)) for (re in c("fixed", "random")) for (df in list(NULL, ~1)) {
  f <- gllvm_julia(y[[fam]], family = fam, num.lv = 1L, row.eff = re, disp.formula = df)
  stopifnot(identical(f$dispersion$name, "phi"), length(f$dispersion$value) == 1L,
            is.finite(f$dispersion$value), f$dispersion$value > 0)
  # the converted value must equal the fit's own `dispersion` (phi = 1/r for NB2)
  d <- as.numeric(f$jl_fit$dispersion)
  want <- if (fam == "negative.binomial") 1 / d else d
  stopifnot(isTRUE(all.equal(f$dispersion$value, want)))
}
cat("ok\n")
