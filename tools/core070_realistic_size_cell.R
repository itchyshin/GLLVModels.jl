#!/usr/bin/env Rscript
# core070_realistic_size_cell.R -- T4 realistic-size grid, R side.
# Reads the SAME CSV Julia's core070_realistic_size_cell.jl wrote to
# ./data/<family>_p<p>_n<n>_K<K>.csv, fits with se=TRUE (TMB sdreport),
# and writes summary + term tables to ./out/, same layout as the
# se-prerun-01 r_fit.R this is generalized from.
#
# Usage: Rscript core070_realistic_size_cell.R <family> <p> <n> <K> <seed>
suppressMessages(library(gllvmTMB))

args <- commandArgs(trailingOnly = TRUE)
fam  <- args[1]
p    <- as.integer(args[2])
n    <- as.integer(args[3])
K    <- as.integer(args[4])
seed <- as.integer(args[5])  # not used to regenerate data; Julia's CSV is authoritative

tag <- sprintf("%s_p%d_n%d_K%d", fam, p, n, K)
dat_path <- file.path("data", paste0(tag, ".csv"))
if (!file.exists(dat_path)) {
    stop(sprintf("data file not found: %s (run the Julia cell first to generate it)", dat_path))
}
Y <- as.matrix(read.table(dat_path, sep = ",", header = FALSE))
stopifnot(nrow(Y) == p, ncol(Y) == n)

trait_names <- paste0("t", seq_len(p))
df_long <- data.frame(
    site  = factor(rep(seq_len(n), each = p)),
    trait = factor(rep(trait_names, times = n), levels = trait_names),
    value = as.vector(Y)
)

fam_obj <- switch(fam,
    gaussian = stats::gaussian(),
    poisson  = stats::poisson(),
    nb2      = gllvmTMB::nbinom2(),
    binomial = stats::binomial(),
    stop("unknown family: ", fam)
)

t0 <- Sys.time()
fit_r <- gllvmTMB(
    value ~ 0 + trait + latent(0 + trait | site, d = K, unique = FALSE),
    data = df_long,
    unit = "site",
    trait = "trait",
    family = fam_obj,
    control = gllvmTMBcontrol(n_init = 1L, se = TRUE)
)
wall_fit <- as.numeric(Sys.time() - t0, units = "secs")

r_logL <- as.numeric(stats::logLik(fit_r))
r_conv <- identical(as.integer(fit_r$opt$convergence), 0L)
has_sd <- !is.null(fit_r$sd_report)

dir.create("out", showWarnings = FALSE)
summ_path <- file.path("out", paste0(tag, "_r_summary.txt"))
writeLines(c(
    sprintf("family=%s", fam),
    sprintf("p=%d n=%d K=%d seed=%d", p, n, K, seed),
    sprintf("logLik=%.10f", r_logL),
    sprintf("converged=%s", r_conv),
    sprintf("has_sd_report=%s", has_sd),
    sprintf("wall_fit_sec=%.4f", wall_fit)
), summ_path)

if (!has_sd) {
    cat("NO SD REPORT for", tag, "\n")
    quit(status = 0)
}

t1 <- Sys.time()
sdr <- fit_r$sd_report
pf  <- sdr$par.fixed
cv  <- sdr$cov.fixed
se_raw <- sqrt(diag(cv))
nm <- names(pf)
is_log <- grepl("^log_", nm)
z <- qnorm(0.975)
lo <- ifelse(is_log, exp(pf - z * se_raw), pf - z * se_raw)
hi <- ifelse(is_log, exp(pf + z * se_raw), pf + z * se_raw)
wall_confint <- as.numeric(Sys.time() - t1, units = "secs")
cat(sprintf("wall_confint_sec=%.4f\n", wall_confint), file = summ_path, append = TRUE)

# cond(H) via cov.fixed's own inverse condition number (cov.fixed = H^-1 on
# the fixed block TMB extracted); report cond(solve(cv)) as the Hessian
# condition number, matching Julia's cond(Hs) on the same conceptual block.
condH <- tryCatch(kappa(solve(cv)), error = function(e) NA_real_)
cat(sprintf("cond_H=%.6g\n", condH), file = summ_path, append = TRUE)

full_tbl <- data.frame(term = nm, term_index = seq_along(nm), estimate_raw = pf,
                        se_raw = se_raw, lower = lo, upper = hi)
write.csv(full_tbl, file.path("out", paste0(tag, "_r_allterms.csv")), row.names = FALSE)

beta_idx <- which(nm == "b_fix")
if (length(beta_idx) == p) {
    beta_tbl <- data.frame(term = trait_names, estimate = pf[beta_idx], se = se_raw[beta_idx],
                            lower = pf[beta_idx] - z * se_raw[beta_idx],
                            upper = pf[beta_idx] + z * se_raw[beta_idx])
    write.csv(beta_tbl, file.path("out", paste0(tag, "_r_fixed.csv")), row.names = FALSE)
    write.table(cv[beta_idx, beta_idx], file.path("out", paste0(tag, "_r_vcov_beta.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)
}

cat("DONE", tag, "\n")
