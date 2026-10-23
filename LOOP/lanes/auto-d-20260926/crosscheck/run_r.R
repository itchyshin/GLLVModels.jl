## Cross-check: fit gllvmTMB::select_lv() on the same data Julia GLLVModels
## select_lv() was run on (Y_poisson.csv, Y_binomial.csv, Y_gaussian.csv,
## species x sites layout, written by run_julia.jl in this same directory).

devtools::load_all("/Users/z3437171/local-scratch/lanes/gllvmTMB-auto-d-20260926", quiet = TRUE)

outdir <- "/Users/z3437171/local-scratch/lanes/GLLVM.jl-auto-d-20260926/LOOP/lanes/auto-d-20260926/crosscheck"

to_long <- function(Y) {
  ## Y: p (species/trait) x n (sites/unit) matrix
  p <- nrow(Y); n <- ncol(Y)
  traits <- paste0("sp", seq_len(p))
  units <- paste0("site", seq_len(n))
  df <- do.call(rbind, lapply(seq_len(n), function(j) {
    data.frame(unit = units[j], trait = traits, value = Y[, j])
  }))
  df$unit <- factor(df$unit, levels = units)
  df$trait <- factor(df$trait, levels = traits)
  df
}

ctrl <- gllvmTMBcontrol(optimizer = "optim", optArgs = list(method = "BFGS"), se = FALSE)

run_one <- function(fam_name, family_obj) {
  raw <- read.csv(file.path(outdir, sprintf("Y_%s.csv", fam_name)), header = FALSE,
                   colClasses = "character")
  vals <- if (fam_name == "binomial") as.numeric(as.logical(as.matrix(raw))) else as.numeric(as.matrix(raw))
  Y <- matrix(vals, nrow = nrow(raw))
  df <- to_long(Y)
  sel <- select_lv(
    value ~ 0 + trait + latent(0 + trait | unit, d = 1),
    data = df, unit = "unit", trait = "trait", family = family_obj,
    d_max = 4, control = ctrl
  )
  tab <- sel$table
  tab$dataset <- fam_name
  tab$selected_d <- sel$selected_d
  tab
}

results <- rbind(
  run_one("poisson", stats::poisson()),
  run_one("binomial", stats::binomial()),
  run_one("gaussian", stats::gaussian())
)

write.csv(results, file.path(outdir, "r_results.csv"), row.names = FALSE)
print(results)
cat("DONE\n")
