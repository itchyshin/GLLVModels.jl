## Regenerates ordinal_logit_p1_data.csv and the r_reference block of
## ordinal_logit_p1.toml. NOT run by CI or by any Julia test -- it is
## provenance for how the fixture was produced, against gllvmTMB pinned at
## commit 9539352f66f2db2cc26b1c393e67212a359b60c9 (version 0.7.1, "P1" in the
## true-parity export ledger). Re-running this script requires gllvmTMB
## installed at that exact commit (e.g. into a scratch library via
## `git worktree add --detach <dir> 9539352f66f2db2cc26b1c393e67212a359b60c9`
## then `R CMD INSTALL --library=<lib> <dir>`); point .libPaths() at that
## library before sourcing this file, or install into the default library.
##
## Do not hand-edit ordinal_logit_p1.toml's [r_reference] block; regenerate it
## from this script's output instead.
library(gllvmTMB)

cat("gllvmTMB version:", as.character(packageVersion("gllvmTMB")), "\n")
cat("R version:", R.version.string, "\n")

set.seed(20260926L)
p <- 3L
n_unit <- 150L
alpha_true  <- c(0.2, -0.1, 0.15)
lambda_true <- c(1.4, -1.1, 1.2)
taus_true   <- c(0, 0.7, 1.4)     # K = 4 categories -> 2 free cutpoints (tau_1=0 fixed)
trait_names <- c("t1", "t2", "t3")

f <- stats::rnorm(n_unit, 0, 1)
rows <- vector("list", n_unit * p)
k <- 1L
for (i in seq_len(n_unit)) {
  for (t in seq_len(p)) {
    ystar <- alpha_true[t] + lambda_true[t] * f[i] + stats::rlogis(1L, 0, 1)
    y <- 1L + sum(ystar > taus_true)
    rows[[k]] <- data.frame(unit = i, trait = trait_names[t], value = y)
    k <- k + 1L
  }
}
df <- do.call(rbind, rows)
df$unit  <- factor(df$unit, levels = seq_len(n_unit))
df$trait <- factor(df$trait, levels = trait_names)

## Save the raw dataset (the fixture Julia will replay) BEFORE fitting so the
## dataset itself is reproducible independent of any fitting-side change.
write.csv(df, "ordinal_logit_p1_data.csv", row.names = FALSE)

## unique = FALSE: the rotation-invariant loadings-only fit (Sigma = Lambda
## Lambda^T, no separate per-trait Psi). This is the model structurally
## comparable to GLLVModels.jl's Ordinal() pertrait fitter, which has no Psi
## term (eta = beta + Lambda * z, z ~ N(0, I_K)); the default `unique = TRUE`
## Psi-augmented latent() has no Julia twin and is NOT what this fixture
## compares. In practice this changes nothing for ordinal_logit(): family_id
## 20 is already in gllvmTMB's `auto_unique_off_family` gate (Psi is
## unidentifiable against the fixed logistic residual variance
## sigma_d^2 = pi^2/3), so `unique = TRUE` and `unique = FALSE` give the
## identical fit here -- the flag only silences an informational warning.
fit <- gllvmTMB(
  value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
  data = df, unit = "unit", family = ordinal_logit(),
  control = gllvmTMBcontrol(se = FALSE)
)

cat("convergence:", fit$opt$convergence, "\n")
stopifnot(identical(fit$opt$convergence, 0L))
stopifnot(fit$tmb_data$family_id_vec[1] == 20L)

ll <- as.numeric(logLik(fit))
cuts <- extract_cutpoints(fit, quiet = TRUE)
Lambda_B <- as.numeric(fit$report$Lambda_B)
sigma_d2 <- unname(gllvmTMB:::link_residual_per_trait(fit))

fmt <- function(x) sprintf("%.17g", x)
cat("logLik", fmt(ll), "\n")
cat("convergence", fit$opt$convergence, "\n")
for (i in seq_len(nrow(cuts))) {
  cat("cutpoint", cuts$trait[i], cuts$cutpoint_index[i], fmt(cuts$tau_estimate[i]), "\n")
}
cat("Lambda_B", paste(vapply(Lambda_B, fmt, character(1)), collapse = " "), "\n")
cat("sigma_d2", paste(vapply(sigma_d2, fmt, character(1)), collapse = " "), "\n")
cat("DONE\n")
