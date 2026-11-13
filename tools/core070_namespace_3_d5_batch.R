# namespace-3 (D5) batch, R half: the paired joint phylogenetic-plus-species check behind the
# namespace/export/extract_phylo_signal numeric case.
#
# usage: Rscript tools/core070_namespace_3_d5_batch.R <outdir>
# The seed is fixed by the contract (docs/dev-log/core070/true-parity-latest/
# namespace-3-d5-batch-contract-p1.json, seed 4401) and is not an argument: one data draw, no reseeding.
# gllvmTMB is read from the frozen P1 oracle library named by GLLVMTMB_ORACLE_LIB (never edited).
#
# Design (Case A): 8 species x 3 sites x 3 traits (72 values), a fixed Newick tree, one bare
# phylogenetic factor (d = 1) plus a species-level diagonal, residual SD 0.6 shared across traits.
# Truth is drawn on the correlation scale (tree rescaled to root-to-tip height 1): lambda =
# (0.70, -0.45, 0.50), species variances (0.12, 0.22, 0.18).
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: Rscript tools/core070_namespace_3_d5_batch.R <outdir>")
od <- args[[1]]; dir.create(od, showWarnings = FALSE, recursive = TRUE)
seed <- 4401L

lib <- Sys.getenv("GLLVMTMB_ORACLE_LIB", "")
if (!nzchar(lib)) stop("GLLVMTMB_ORACLE_LIB must name the frozen P1 oracle library")
.libPaths(c(lib, .libPaths()))
suppressMessages(library(gllvmTMB, lib.loc = lib))

nwk <- "(((s1:2,s2:2):1,(s3:1,s4:1):2):1,((s5:1.5,s6:1.5):1,(s7:1,s8:1):1.5):1.5);"
tree <- ape::read.tree(text = nwk)
tips <- tree$tip.label
set.seed(seed)
C <- ape::vcv(tree, corr = TRUE)[tips, tips]
lam <- c(0.70, -0.45, 0.50); u_sd <- sqrt(c(0.12, 0.22, 0.18)); res_sd <- 0.6
g <- as.numeric(t(chol(C)) %*% rnorm(length(tips)))   # one phylogenetic factor over species
U <- sapply(u_sd, function(s) rnorm(length(tips), 0, s))   # species-level diagonal, 8 x 3
dat <- expand.grid(species = tips, site = paste0("site", 1:3), trait = paste0("trait", 1:3),
                   stringsAsFactors = FALSE)
sp <- match(dat$species, tips); tr <- as.integer(sub("trait", "", dat$trait))
dat$value <- lam[tr] * g[sp] + U[cbind(sp, tr)] + rnorm(nrow(dat), 0, res_sd)

write.csv(dat, file.path(od, "frozen_A.csv"), row.names = FALSE, quote = FALSE)
writeLines(nwk, file.path(od, "tree.nwk"))

dat$species <- factor(dat$species, levels = tips); dat$trait <- factor(dat$trait)
t0 <- Sys.time()
fit <- gllvmTMB(value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree) +
                  indep(0 + trait | species),
                data = dat, trait = "trait", unit = "site", cluster = "species",
                family = gaussian(), REML = FALSE, engine = "tmb")
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
sig <- extract_phylo_signal(fit)

nm <- names(fit$opt$par)
if (is.null(nm)) stop("fit$opt$par has no names; cannot save parameters by name")
idx <- as.integer(ave(seq_along(nm), nm, FUN = seq_along))
write.csv(data.frame(name = nm, index = idx, value = as.numeric(fit$opt$par)),
          file.path(od, "R_params_A.csv"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(trait = as.character(sig$trait), H2 = sig$H2, C2_non = sig$C2_non,
                     Psi = sig$Psi, V_eta = sig$V_eta),
          file.path(od, "R_signal_A.csv"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(key = c("seed", "logLik", "convergence", "elapsed_s", "n_par"),
                     value = c(seed, format(-fit$opt$objective, digits = 17), fit$opt$convergence, el, length(nm))),
          file.path(od, "R_summary_A.csv"), row.names = FALSE, quote = FALSE)
desc <- utils::packageDescription("gllvmTMB", lib.loc = lib)
write.csv(data.frame(key = c("r_version", "gllvmtmb_version", "gllvmtmb_remote_sha", "omp_threads", "openblas_threads"),
                     value = c(as.character(getRversion()), desc$Version,
                               if (is.null(desc$RemoteSha)) "absent" else desc$RemoteSha,
                               Sys.getenv("OMP_NUM_THREADS"), Sys.getenv("OPENBLAS_NUM_THREADS"))),
          file.path(od, "R_session.csv"), row.names = FALSE, quote = FALSE)
cat(sprintf("R fit %.2f s, convergence %s, logLik %.10f, seed %d\n",
            el, fit$opt$convergence, -fit$opt$objective, seed))
