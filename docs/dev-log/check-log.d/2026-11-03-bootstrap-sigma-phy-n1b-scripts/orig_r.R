# N1b: original R fit (BFGS reltol 1e-14 as N1) + summaries
args <- commandArgs(trailingOnly = TRUE); dd <- args[[1]]; od <- args[[2]]; ds <- args[[3]]
.libPaths(c("/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library", .libPaths()))
suppressMessages(library(gllvmTMB, lib.loc = "/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library"))
dir.create(od, showWarnings = FALSE, recursive = TRUE)
tree <- ape::read.tree(file.path(dd, "tree.nwk"))
df <- read.csv(file.path(dd, "long.csv"), stringsAsFactors = FALSE)
traits <- sort(unique(df$trait)); tips <- tree$tip.label
df$species <- factor(df$species, levels = tips); df$trait <- factor(df$trait, levels = traits)
ctrl <- gllvmTMBcontrol(n_init = 1L, optimizer = "optim",
  optArgs = list(method = "BFGS", control = list(maxit = 2000L, reltol = 1e-14)))
fit <- suppressWarnings(gllvmTMB(
  value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree) + phylo_unique(species, tree = tree),
  data = df, trait = "trait", unit = "species", cluster = "species",
  family = gaussian(), REML = FALSE, engine = "tmb", control = ctrl))
S_un <- extract_Sigma(fit, level = "phy", part = "unique")
s <- gllvmTMB:::.extract_summaries(fit, level = "phy", what = c("Sigma","R","communality"), link_residual = "auto")
flat <- c(as.numeric(s$Sigma_phy), as.numeric(s$R_phy), as.numeric(s$communality_phy))
writeLines(c(paste0("convergence=", fit$opt$convergence), paste0("nll=", sprintf("%.15g", fit$opt$objective)),
  paste0("unique=", paste(sprintf("%.15g", S_un$s), collapse=",")),
  paste0("flat=", paste(sprintf("%.17g", flat), collapse=","))), file.path(od, paste0("origR_", ds, ".txt")))
cat("convergence", fit$opt$convergence, "unique", S_un$s, "\n")
