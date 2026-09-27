## Does a loading ridge rescue d recovery on binary data? gllvmTMB, Laplace fits.
## For each dataset: select_lv without ridge, and with aghq_ridge = 2 (Laplace + ridge).
## Records the guarded choice and per-d status/logLik/max loading row norm.
devtools::load_all("/Users/z3437171/local-scratch/lanes/gllvmTMB-auto-d-20260926", quiet = TRUE)
out <- "/Users/z3437171/local-scratch/lanes/GLLVM.jl-auto-d-20260926/LOOP/lanes/auto-d-20260926/ridge"
to_long <- function(Y) {
  p <- nrow(Y); n <- ncol(Y)
  df <- data.frame(unit = factor(rep(seq_len(n), each = p)), trait = factor(rep(seq_len(p), n)), value = as.vector(Y))
  df
}
sim <- function(n, p, K, seed) {
  set.seed(seed)
  L <- matrix(0.8 * rnorm(p * K), p, K); U <- matrix(rnorm(K * n), K, n)
  eta <- L %*% U
  matrix(rbinom(p * n, 1, plogis(eta)), p, n)
}
rows <- list()
reps <- as.integer(commandArgs(TRUE)[1]); if (is.na(reps)) reps <- 10
for (cell in list(c(60, 10, 1), c(60, 10, 2), c(120, 10, 2), c(120, 20, 2))) {
  for (r in seq_len(reps)) {
    Y <- sim(cell[1], cell[2], cell[3], 1000 * cell[3] + 10 * cell[1] + cell[2] + r)
    df <- to_long(Y)
    for (ridge in c(Inf, 2)) {
      ctrl <- if (is.finite(ridge)) gllvmTMBcontrol(se = FALSE, aghq_ridge = ridge) else gllvmTMBcontrol(se = FALSE)
      t0 <- Sys.time()
      sel <- tryCatch(suppressMessages(suppressWarnings(select_lv(
        value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
        data = df, unit = "unit", trait = "trait", family = stats::binomial(),
        d_max = 4, criterion = "bic", control = ctrl))), error = function(e) e)
      if (inherits(sel, "error")) {
        rows[[length(rows) + 1]] <- data.frame(n = cell[1], p = cell[2], K_true = cell[3], rep = r, ridge = ridge,
          selected = NA, statuses = paste("ERROR", conditionMessage(sel)), secs = as.numeric(Sys.time() - t0, units = "secs"))
      } else {
        rows[[length(rows) + 1]] <- data.frame(n = cell[1], p = cell[2], K_true = cell[3], rep = r, ridge = ridge,
          selected = sel$selected_d, statuses = paste(sel$table$status, collapse = "/"),
          secs = as.numeric(Sys.time() - t0, units = "secs"))
      }
      write.csv(do.call(rbind, rows), file.path(out, "ridge_binary.csv"), row.names = FALSE)
    }
  }
}
cat("DONE\n")
