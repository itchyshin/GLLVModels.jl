# N1b Section 5: R simulator moments via simulate(fit, nsim) (the call bootstrap_Sigma makes), same statistics as s5_j.jl
args <- commandArgs(trailingOnly = TRUE); dd <- args[[1]]; od <- args[[2]]; ds <- args[[3]]; nsim <- as.integer(args[[4]])
.libPaths(c("/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library", .libPaths()))
suppressMessages(library(gllvmTMB, lib.loc = "/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library"))
tree <- ape::read.tree(file.path(dd, "tree.nwk"))
df <- read.csv(file.path(dd, "long.csv"), stringsAsFactors = FALSE)
traits <- sort(unique(df$trait)); tips <- tree$tip.label
df$species <- factor(df$species, levels = tips); df$trait <- factor(df$trait, levels = traits)
fit <- suppressWarnings(gllvmTMB(
  value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree) + phylo_unique(species, tree = tree),
  data = df, trait = "trait", unit = "species", cluster = "species", family = gaussian(),
  REML = FALSE, engine = "tmb", control = gllvmTMBcontrol(n_init = 1L, optimizer = "optim",
    optArgs = list(method = "BFGS", control = list(maxit = 2000L, reltol = 1e-14)))))
dat <- fit$data
stopifnot(nrow(dat) == nrow(df))
sim <- simulate(fit, nsim = nsim, seed = 424242)
stopifnot(nrow(sim) == nrow(dat), ncol(sim) == nsim)
cat("sim dim", dim(sim), " data cols", names(dat), "\n")
# fitted mean per trait from the data-order eta mean (fixed effects only): use trait means of the original Y minus nothing; use fit's report
beta <- tapply(as.numeric(fit$report$eta), dat$trait, mean)  # eta includes fitted RE modes; replaced below by fixed-effect estimate
fx <- fit$tmb_obj$env$parList()$b_fix
cat("b_fix", fx, "\n")
sp <- as.integer(factor(as.character(dat$species), levels = sort(unique(as.character(dat$species)))))
ti <- as.integer(factor(as.character(dat$trait), levels = traits))
rep_ <- dat$replicate
d <- 5L; lt <- do.call(rbind, lapply(1:d, function(j) cbind(i = j:d, j = j)))
nspec <- length(unique(sp))
idx <- function(s, r) { k <- which(sp == s & rep_ == r); k[order(ti[k])] }
I1 <- t(vapply(seq_len(nspec), function(s) idx(s, 1), integer(d)))  # nspec x d
I2 <- t(vapply(seq_len(nspec), function(s) idx(s, 2), integer(d)))
mu <- fx[ti]
out <- matrix(NA_real_, nsim, 30)
for (b in seq_len(nsim)) {
  r <- sim[, b] - mu
  R1 <- matrix(r[I1], nspec, d); R2 <- matrix(r[I2], nspec, d)
  Rall <- rbind(R1, R2)
  w <- crossprod(Rall) / nrow(Rall)
  cc <- (crossprod(R1, R2) + crossprod(R2, R1)) / (2 * nspec)
  out[b, ] <- c(w[lt], cc[lt])
}
write.table(out, file.path(od, paste0("s5R_", ds, ".csv")), sep = ",", row.names = FALSE, col.names = FALSE)
cat("DONE s5 R\n")
