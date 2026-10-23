## Fit gllvmTMB select_lv on the exact Julia binary-ridge datasets (matched-data check).
## Usage: Rscript r_on_julia_data.R <n> <p> <K> <out.csv>
suppressMessages(devtools::load_all("~/local-scratch/lanes/gllvmTMB-auto-d-20260926", quiet = TRUE))
a <- commandArgs(TRUE); n <- as.integer(a[1]); p <- as.integer(a[2]); K <- as.integer(a[3]); out <- a[4]
rows <- list()
for (r in 1:10) {
  Y <- as.matrix(read.csv(file.path(dirname(out), sprintf("Y_n%d_p%d_K%d_r%d.csv", n, p, K, r)), header = FALSE))
  Y <- matrix(as.integer(Y == TRUE | Y == "true"), nrow(Y), ncol(Y))
  df <- data.frame(unit = factor(rep(seq_len(ncol(Y)), each = nrow(Y))),
                   trait = factor(rep(seq_len(nrow(Y)), ncol(Y))), value = as.vector(Y))
  for (ridge in c(Inf, 2)) {
    t0 <- Sys.time()
    sel <- tryCatch(suppressMessages(suppressWarnings(select_lv(
      value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE),
      data = df, unit = "unit", trait = "trait", family = stats::binomial(),
      d_max = 4, criterion = "bic", binary_ridge = ridge))), error = function(e) e)
    secs <- as.numeric(Sys.time() - t0, units = "secs")
    if (inherits(sel, "error")) {
      rows[[length(rows) + 1]] <- data.frame(n = n, p = p, K_true = K, rep = r, ridge = ridge,
        sel_bic = NA, sel_bic_sites = NA, statuses = paste("ERROR", conditionMessage(sel)), secs = secs)
    } else {
      tb <- sel$table; okr <- tb$status %in% c("ok", "warm_start")
      pick <- function(col) if (any(okr)) tb$d[okr][which.min(tb[[col]][okr])] else NA
      rows[[length(rows) + 1]] <- data.frame(n = n, p = p, K_true = K, rep = r, ridge = ridge,
        sel_bic = pick("bic"), sel_bic_sites = pick("bic_sites"),
        statuses = paste(tb$status, collapse = "/"), secs = secs)
    }
    write.csv(do.call(rbind, rows), out, row.names = FALSE)
  }
}
