#!/usr/bin/env Rscript
# Bind the 14 public-fit AGHQ policy rows (t8-aghq-bind-next-slice.md).
# Every row uses exported gllvmTMB() / gllvmTMBcontrol() — no ::: helpers.
#
# Usage:
#   Rscript --vanilla tools/core070_aghq_public_policy_bind.R \
#     <gllvmTMB-source-tree> <receipt-json-out>
#
# Example:
#   Rscript --vanilla tools/core070_aghq_public_policy_bind.R \
#     ../gllvmTMB-gllvm-twin-20260904 \
#     docs/dev-log/core070/aghq-public-policy-bind-receipt-2026-09-04.json

args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)

# Pin switch (D-294/D-295): GLLVM_PARITY_PIN unset or "P0" keeps the original
# 2026-09-04 behaviour below (devtools::load_all on a gllvmTMB source tree,
# receipt written to <receipt-json-out>). "P1" runs the same 14 rows against
# the installed P1 oracle library instead:
#
#   GLLVM_PARITY_PIN=P1 Rscript --vanilla tools/core070_aghq_public_policy_bind.R \
#     <frozen-library> <destination>
#
# At P1 the row list is checked against the P1 contract written by
# tools/core070_aghq_p1_contract.py, the library must carry the
# CORE070_SOURCE_PIN.toml marker for P1 (tools/core070_source_pin.R), and
# <destination> (a directory that must not exist) is created only after those
# checks pass; it receives receipt.json. Any other pin value stops here.
parity_pin <- toupper(trimws(Sys.getenv("GLLVM_PARITY_PIN", "P0")))
if (!parity_pin %in% c("P0", "P1")) stop("GLLVM_PARITY_PIN must be P0 or P1, got '", parity_pin, "'")
p1_reference <- "9539352f66f2db2cc26b1c393e67212a359b60c9"
p1_contract_rel <- "docs/dev-log/core070/true-parity-latest/aghq-public-policy-contract-p1.json"

sha256_file <- function(path) {
  cmd <- if (nzchar(Sys.which("sha256sum"))) "sha256sum" else "shasum"
  argv <- if (identical(cmd, "sha256sum")) shQuote(path) else c("-a", "256", shQuote(path))
  line <- system2(cmd, argv, stdout = TRUE, stderr = TRUE)
  stopifnot(is.null(attr(line, "status")), length(line) >= 1L)
  sub("[[:space:]].*$", "", line[[1L]])
}

if (identical(parity_pin, "P1")) {
  frozen_library <- normalizePath(args[[1]], mustWork = TRUE)
  output_dir <- args[[2]]
  stopifnot(!file.exists(output_dir))
  receipt_path <- file.path(output_dir, "receipt.json")
  .libPaths(c(frozen_library, .libPaths()))
  suppressPackageStartupMessages({
    library(gllvmTMB)
    library(jsonlite)
  })
  stopifnot(normalizePath(find.package("gllvmTMB")) ==
            normalizePath(file.path(frozen_library, "gllvmTMB")))
  root <- normalizePath(".")
  contract_path <- file.path(root, p1_contract_rel)
  contract <- jsonlite::read_json(contract_path, simplifyVector = FALSE)
  stopifnot(identical(contract$reference_commit, p1_reference),
            identical(contract$status, "P1_AGHQ_PUBLIC_POLICY_CONTRACT"),
            length(contract$cases) == 14L)
  source(file.path(root, "tools/core070_source_pin.R"))
  source_pin <- core070_source_pin(root, frozen_library, parity_pin, p1_reference)
  # No host path in the receipt: the library is identified by its marker record.
  source_pin$marker_path <- "gllvmTMB/CORE070_SOURCE_PIN.toml"
  # Create the destination only after the pin checks pass, so a refused run
  # leaves nothing behind.
  dir.create(output_dir, recursive = TRUE)
} else {
  pkg_root <- normalizePath(args[[1]], mustWork = TRUE)
  receipt_path <- args[[2]]
  stopifnot(!file.exists(receipt_path))

  suppressPackageStartupMessages({
    if (!requireNamespace("devtools", quietly = TRUE)) {
      stop("devtools required to load gllvmTMB source tree", call. = FALSE)
    }
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("jsonlite required for receipt output", call. = FALSE)
    }
    devtools::load_all(pkg_root, quiet = TRUE)
  })
}

base_ctrl <- function(aghq) {
  gllvmTMBcontrol(
    n_init = 1L, init_jitter = 0, se = FALSE,
    aghq = aghq, aghq_ridge = Inf
  )
}

make_long <- function(p_traits, n_sites, seed, fam = "binomial") {
  set.seed(seed)
  tr <- paste0("t", seq_len(p_traits))
  site <- factor(rep(paste0("s", seq_len(n_sites)), each = p_traits))
  trait <- factor(rep(tr, n_sites), levels = tr)
  eta <- stats::rnorm(length(site), 0, 0.3)
  y <- switch(fam,
    binomial = stats::rbinom(length(eta), 1L, stats::plogis(eta)),
    poisson = stats::rpois(length(eta), exp(pmin(eta, 3))),
    gaussian = eta + stats::rnorm(length(eta), sd = 0.3),
    nb2 = stats::rnbinom(length(eta), mu = exp(pmin(eta, 2)), size = 2),
    ordinal = {
      ystar <- eta + stats::rnorm(length(eta))
      cut(ystar, breaks = c(-Inf, -0.5, 0.5, Inf), labels = FALSE)
    },
    delta = {
      pp <- stats::plogis(eta)
      pres <- stats::rbinom(length(eta), 1L, pp)
      pos <- stats::rgamma(length(eta), shape = 1, scale = exp(pmin(eta, 2)))
      pres * pos
    },
    tweedie = {
      stopifnot(requireNamespace("mgcv", quietly = TRUE))
      mgcv::rTweedie(exp(pmin(eta, 2)), p = 1.5, phi = 1)
    },
    stop("unknown fam: ", fam, call. = FALSE)
  )
  list(
    df = data.frame(site = site, trait = trait, y = y),
    fam = fam,
    p_traits = p_traits,
    n_sites = n_sites,
    seed = seed
  )
}

family_obj <- function(fam) {
  switch(fam,
    binomial = stats::binomial(),
    poisson = stats::poisson(),
    gaussian = stats::gaussian(),
    nb2 = gllvmTMB::nbinom2(),
    ordinal = gllvmTMB::ordinal_probit(),
    delta = gllvmTMB::delta_gamma(),
    tweedie = gllvmTMB::tweedie()
  )
}

public_fit <- function(fixture, aghq) {
  fam <- family_obj(fixture$fam)
  fit <- suppressWarnings(gllvmTMB::gllvmTMB(
    y ~ 0 + trait + latent(0 + trait | site, d = 1, unique = FALSE),
    data = fixture$df,
    family = fam,
    unit = "site",
    control = base_ctrl(aghq)
  ))
  list(
    used = isTRUE(fit$aghq$used),
    k = fit$aghq$k,
    reason = fit$aghq$reason,
    convergence = fit$opt$convergence,
    objective = unname(fit$opt$objective)
  )
}

assert_row <- function(id, ok, detail, obs) {
  list(
    row_id = id,
    pass = isTRUE(ok),
    detail = detail,
    observed = obs
  )
}

cases <- list()

# --- AUTO-K family rows (p=5, n=30, seed=42, aghq="auto") -------------------
auto_k_specs <- list(
  "AGHQ-AUTO-K-BINOMIAL" = list(fam = "binomial", expected_k = 5L),
  "AGHQ-AUTO-K-POISSON"  = list(fam = "poisson",  expected_k = 5L),
  "AGHQ-AUTO-K-GAUSSIAN" = list(fam = "gaussian", expected_k = 5L),
  "AGHQ-AUTO-K-NB2"      = list(fam = "nb2",      expected_k = 5L),
  "AGHQ-AUTO-K-ORDINAL"  = list(fam = "ordinal",  expected_k = 9L),
  "AGHQ-AUTO-K-DELTA"    = list(fam = "delta",    expected_k = 5L),
  "AGHQ-AUTO-K-TWEEDIE"  = list(fam = "tweedie",  expected_k = 9L)
)

for (id in names(auto_k_specs)) {
  spec <- auto_k_specs[[id]]
  fx <- make_long(5L, 30L, 42L, spec$fam)
  obs <- tryCatch(
    public_fit(fx, "auto"),
    error = function(e) list(error = conditionMessage(e))
  )
  if (!is.null(obs$error)) {
    cases[[id]] <- assert_row(id, FALSE, obs$error, obs)
  } else {
    ok <- obs$used && identical(as.integer(obs$k), spec$expected_k) &&
      is.finite(obs$objective)
    note <- if (identical(id, "AGHQ-AUTO-K-DELTA")) {
      "Public fit k=5: delta_gamma() family label is 'binomial Gamma', not 'delta_gamma'; frozen helper oracle used family='delta_gamma' -> k=9."
    } else {
      NA_character_
    }
    cases[[id]] <- c(
      assert_row(id, ok, if (ok) "public auto fit" else "assertion failed", obs),
      list(
        public_call = sprintf(
          "gllvmTMB(..., family=%s(), latent(..., unique=FALSE), control=gllvmTMBcontrol(aghq='auto', aghq_ridge=Inf))",
          spec$fam
        ),
        fixture = fx[c("p_traits", "n_sites", "seed", "fam")],
        expected_k = spec$expected_k,
        helper_oracle_note = note
      )
    )
  }
}

# --- DEFAULT-OFF (no fit) ---------------------------------------------------
default_aghq <- formals(gllvmTMBcontrol)$aghq
cases[["AGHQ-DEFAULT-OFF"]] <- c(
  assert_row(
    "AGHQ-DEFAULT-OFF",
    identical(default_aghq, FALSE),
    "formals(gllvmTMBcontrol)$aghq",
    list(default_aghq = default_aghq)
  ),
  list(public_call = "formals(gllvmTMBcontrol)$aghq")
)

# --- POLICY rows on binomial p=5 fixture ------------------------------------
fx5 <- make_long(5L, 30L, 42L, "binomial")

obs_off <- public_fit(fx5, FALSE)
cases[["AGHQ-POLICY-OFF"]] <- c(
  assert_row("AGHQ-POLICY-OFF", !obs_off$used, "aghq=FALSE -> Laplace", obs_off),
  list(public_call = "gllvmTMB(..., control=gllvmTMBcontrol(aghq=FALSE))", fixture = fx5[c("p_traits", "n_sites", "seed")])
)

obs_ex <- public_fit(fx5, 3L)
cases[["AGHQ-POLICY-EXPLICIT"]] <- c(
  assert_row(
    "AGHQ-POLICY-EXPLICIT",
    obs_ex$used && identical(as.integer(obs_ex$k), 3L),
    "aghq=3L explicit",
    obs_ex
  ),
  list(public_call = "gllvmTMB(..., control=gllvmTMBcontrol(aghq=3L))", fixture = fx5[c("p_traits", "n_sites", "seed")])
)

# --- p=20 explicit bypass cutoff ----------------------------------------------
fx20 <- make_long(20L, 40L, 120L, "binomial")
obs_bypass <- public_fit(fx20, 9L)
cases[["AGHQ-POLICY-EXPLICIT-BYPASS-CUTOFF"]] <- c(
  assert_row(
    "AGHQ-POLICY-EXPLICIT-BYPASS-CUTOFF",
    obs_bypass$used && identical(as.integer(obs_bypass$k), 9L),
    "explicit aghq=9L at n_traits=20 bypasses auto cutoff",
    obs_bypass
  ),
  list(public_call = "gllvmTMB(..., n_traits=20, control=gllvmTMBcontrol(aghq=9L))", fixture = fx20[c("p_traits", "n_sites", "seed")])
)

# --- p=20 auto enforce cutoff -----------------------------------------------
obs_cut20 <- public_fit(fx20, "auto")
reason_cut <- obs_cut20$reason
cases[["AGHQ-POLICY-AUTO-ENFORCE-CUTOFF"]] <- c(
  assert_row(
    "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF",
    !obs_cut20$used && is.character(reason_cut) && grepl("cutoff", reason_cut, fixed = TRUE),
    "aghq='auto' declines at n_traits=20",
    obs_cut20
  ),
  list(public_call = "gllvmTMB(..., n_traits=20, control=gllvmTMBcontrol(aghq='auto'))", fixture = fx20[c("p_traits", "n_sites", "seed")])
)

# --- p=19 auto stays on -------------------------------------------------------
fx19 <- make_long(19L, 35L, 1L, "binomial")
obs19 <- public_fit(fx19, "auto")
cases[["AGHQ-POLICY-TRAITS19"]] <- c(
  assert_row(
    "AGHQ-POLICY-TRAITS19",
    obs19$used && is.finite(obs19$objective),
    "aghq='auto' stays on at n_traits=19",
    obs19
  ),
  list(public_call = "gllvmTMB(..., n_traits=19, control=gllvmTMBcontrol(aghq='auto'))", fixture = fx19[c("p_traits", "n_sites", "seed")])
)

# --- p=20 boundary duplicate --------------------------------------------------
cases[["AGHQ-POLICY-TRAITS20"]] <- c(
  assert_row(
    "AGHQ-POLICY-TRAITS20",
    !obs_cut20$used && is.character(reason_cut) && grepl("cutoff", reason_cut, fixed = TRUE),
    "boundary duplicate: n_traits=20 auto decline",
    obs_cut20
  ),
  list(public_call = "same fixture as AGHQ-POLICY-AUTO-ENFORCE-CUTOFF", fixture = fx20[c("p_traits", "n_sites", "seed")])
)

all_pass <- all(vapply(cases, function(x) isTRUE(x$pass), logical(1L)))
expected_ids <- c(
  names(auto_k_specs), "AGHQ-DEFAULT-OFF", "AGHQ-POLICY-OFF", "AGHQ-POLICY-EXPLICIT",
  "AGHQ-POLICY-EXPLICIT-BYPASS-CUTOFF", "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF",
  "AGHQ-POLICY-TRAITS19", "AGHQ-POLICY-TRAITS20"
)
stopifnot(identical(sort(names(cases)), sort(expected_ids)))

if (identical(parity_pin, "P1")) {
  contract_ids <- vapply(contract$cases, `[[`, "", "row_id")
  stopifnot(identical(sort(contract_ids), sort(expected_ids)))
  r_engine <- list(
    reference_commit = p1_reference,
    gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
    r_version = R.version.string,
    load_method = "library(gllvmTMB) from the installed P1 oracle library (identified by source_pin)"
  )
} else {
  r_ref <- tryCatch(
    system2("git", c("-C", pkg_root, "rev-parse", "HEAD"), stdout = TRUE),
    error = function(e) NA_character_
  )
  if (length(r_ref)) r_ref <- r_ref[[1L]]
  r_engine <- list(
    source_tree = pkg_root,
    git_head = r_ref,
    gllvmTMB_version = as.character(utils::packageVersion("gllvmTMB")),
    r_version = R.version.string,
    load_method = "devtools::load_all(source_tree)"
  )
}

receipt <- list(
  schema = "core070-aghq-public-policy-bind/v1",
  status = if (all_pass) "PASS" else "FAIL",
  scope = "T8_AGHQ_PUBLIC_POLICY_BIND_14",
  bound_row_ids = expected_ids,
  bound_count = sum(vapply(cases, function(x) isTRUE(x$pass), logical(1L))),
  expected_count = 14L,
  cases = cases,
  r_engine = r_engine,
  model_contract = "Stage1a: latent(..., unique=FALSE), single z_B block, aghq_ridge=Inf",
  oracle_note = if (identical(parity_pin, "P1")) {
    "Installed P1 oracle library; no source tree loaded."
  } else {
    "Frozen engine oracle b4d5fee6; policy reads use R twin HEAD above."
  },
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC")
)
if (identical(parity_pin, "P1")) {
  receipt$parity_pin <- parity_pin
  receipt$reference_commit <- p1_reference
  receipt$contract <- p1_contract_rel
  receipt$contract_sha256 <- sha256_file(contract_path)
  receipt$gllvmTMB_version <- r_engine$gllvmTMB_version
  receipt$source_pin <- source_pin
}

dir.create(dirname(receipt_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  receipt, receipt_path,
  auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = 17
)
receipt_sha <- sha256_file(receipt_path)
receipt$receipt_sha256 <- receipt_sha
jsonlite::write_json(
  receipt, receipt_path,
  auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = 17
)

cat("CORE070_AGHQ_PUBLIC_POLICY_BIND_", receipt$status,
    " bound=", receipt$bound_count, "/", receipt$expected_count,
    " sha256=", receipt_sha, "\n", sep = "")
quit(status = if (all_pass) 0L else 1L)
