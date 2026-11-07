args <- commandArgs(trailingOnly = TRUE)
ds <- args[[1]]; root <- args[[2]]; od <- args[[3]]
.libPaths(c("/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library", .libPaths()))
suppressMessages(library(gllvmTMB, lib.loc = "/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library"))
dd <- file.path(root, "data", paste0("d", ds))
tree <- ape::read.tree(file.path(dd, "tree.nwk"))
df <- read.csv(file.path(dd, "long.csv"), stringsAsFactors = FALSE)
traits <- sort(unique(df$trait)); tips <- tree$tip.label
df$species <- factor(df$species, levels = tips); df$trait <- factor(df$trait, levels = traits)
spn <- sort(unique(as.character(df$species)))
df$j <- (match(as.character(df$species), spn) - 1L) * 2L + df$replicate
df$i <- match(as.character(df$trait), traits)
fy <- as.matrix(read.csv(file.path(od, paste0("allY_", ds, ".csv")), header = FALSE))
nb <- nrow(fy)
M <- matrix(NA_real_, nb, 55); info <- data.frame(b = fy[,2], conv = NA_integer_, nll = NA_real_)
for (k in seq_len(nb)) {
  if (k %% 50 == 0) message("rep ", k, " ", format(Sys.time()))
  Yb <- matrix(fy[k, -(1:2)], nrow = 5)
  dat <- df; dat$value <- Yb[cbind(dat$i, dat$j)]
  out <- tryCatch(suppressMessages(suppressWarnings(gllvmTMB(
    value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree) + phylo_unique(species, tree = tree),
    data = dat, trait = "trait", unit = "species", cluster = "species", family = gaussian(),
    REML = FALSE, engine = "tmb", silent = FALSE, control = gllvmTMBcontrol(se = FALSE)))), error = function(e) NULL)
  if (is.null(out)) next
  info$conv[k] <- out$opt$convergence; info$nll[k] <- out$opt$objective
  if (isTRUE(out$opt$convergence == 0L)) {
    s <- tryCatch(gllvmTMB:::.extract_summaries(out, level = "phy", what = c("Sigma","R","communality"), link_residual = "auto"), error = function(e) NULL)
    if (!is.null(s)) M[k, ] <- c(as.numeric(s$Sigma_phy), as.numeric(s$R_phy), as.numeric(s$communality_phy))
  }
}
write.csv(M, file.path(od, paste0("pairedR_", ds, ".csv")), row.names = FALSE)
write.csv(info, file.path(od, paste0("pairedR_info_", ds, ".csv")), row.names = FALSE)
message("DONE")
