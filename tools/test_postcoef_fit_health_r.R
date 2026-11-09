source('tools/core070_postcoef_fit_health.R')
fit <- list(tmb_obj = list(), opt = list(par = c(a = 1, b = 2), objective = 3, convergence = 0L),
            sd_report = list(pdHess = TRUE), aghq = list(ridge_tau = Inf, penalised = FALSE))
grad <- function(obj, par, tau) c(1e-7, -2e-7)
objective <- function(obj, par, tau) 3
x <- postcoef_r_fit_health('no_fixed_effects', fit, grad, objective, fixed_prior_state = 'none')
stopifnot(identical(x$parameters, c(1, 2)), identical(x$free_indices, 1:2),
          x$case == 'no_fixed_effects', identical(x$pd_hessian, TRUE),
          x$max_abs_gradient == 2e-7, identical(x$objective_verified, TRUE),
          identical(x$penalty_configuration_recorded, TRUE), identical(x$optimizer_convergence_code, 0), is.null(x$error))
fit_nonconverged <- fit; fit_nonconverged$opt$convergence <- 1L
stopifnot(identical(postcoef_r_fit_health('no_fixed_effects', fit_nonconverged, grad, objective, 'none')$optimizer_convergence_code, 1))
y <- postcoef_r_fit_health('no_fixed_effects', fit, function(...) c(NA_real_, 1), objective, 'none')
stopifnot(is.null(y$max_abs_gradient), !is.null(y$error))
y <- postcoef_r_fit_health('no_fixed_effects', fit, grad, function(...) 4, 'none')
stopifnot(identical(y$objective_verified, FALSE))
y <- postcoef_r_fit_health('no_fixed_effects', fit, grad, objective, NULL)
stopifnot(identical(y$penalty_configuration_recorded, FALSE),
          identical(y$max_abs_gradient, 2e-7), !is.null(y$error))
fit$sd_report$pdHess <- FALSE
y <- postcoef_r_fit_health('no_fixed_effects', fit, grad, objective, 'none')
stopifnot(identical(y$pd_hessian, FALSE))
fit$aghq$penalised <- NULL
y <- postcoef_r_fit_health('no_fixed_effects', fit, grad, objective, 'none')
stopifnot(identical(y$penalty_configuration_recorded, FALSE))
fit$aghq$penalised <- FALSE
seen <- NULL
components <- function(obj, opt, aghq) {
  seen <<- list(opt = opt, aghq = aghq)
  list(optimization_nll = 3)
}
z <- postcoef_r_health_from_fit('trait_intercepts', fit, grad, components, 'none')
stopifnot(identical(z$objective_verified, TRUE), identical(seen$opt, fit$opt),
          identical(seen$aghq, fit$aghq), z$max_abs_gradient == 2e-7)
cat('POSTCOEF_R_HEALTH_SYNTHETIC_PASS\n')

prior_call <- quote(gllvmTMB(f, data = df_long, unit = "site", trait = "trait",
                            family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE)))
prior_meta <- list(pin = 'P1', reference_commit = '9539352f66f2db2cc26b1c393e67212a359b60c9',
                   source_tree_sha256 = '86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db',
                   version = '0.7.1', oracle_verified = TRUE, call = prior_call)
stopifnot(identical(postcoef_historical_prior_state(prior_meta), 'none'))
canonical <- prior_meta; canonical$call <- quote(gllvmTMB(formula=f, data=df_long, trait='trait', unit='site', family=gaussian(), control=gllvmTMBcontrol(n_init=1L, se=TRUE)))
stopifnot(identical(postcoef_historical_prior_state(canonical), 'none'))
for (name in c('pin','reference_commit','source_tree_sha256','version','oracle_verified','call')) {
  bad <- prior_meta; bad[[name]] <- NULL
  stopifnot(is.null(postcoef_historical_prior_state(bad)))
}
for (bad_call in list(quote(gllvmTMB(f, data=df_long, prior='active')),
  quote(gllvmTMB(f, data=df_long, unit='site', trait='trait', family=gaussian(), control=gllvmTMBcontrol(n_init=1L, se=FALSE))),
  quote(gllvmTMB(f, data=df_long, unit='site', trait='trait', family=gaussian(), control=gllvmTMBcontrol(n_init=1L, se=TRUE), engine='julia')))) {
  bad <- prior_meta; bad$call <- bad_call
  stopifnot(is.null(postcoef_historical_prior_state(bad)))
}

cat('POSTCOEF_R_PRIOR_PROVENANCE_CONTROLS_PASS\n')
mock <- function(formula, data, trait, unit, family, control) match.call()
f <- NULL; df_long <- NULL
gllvmTMBcontrol <- function(...) NULL
gaussian <- function(...) NULL
gllvmTMB <- mock
observed <- gllvmTMB(f, data = df_long, unit = "site", trait = "trait",
                    family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE))
expected <- 'gllvmTMB(formula = f, data = df_long, trait = "trait", unit = "site", family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE))'
stopifnot(identical(paste(deparse(observed, width.cutoff=500L), collapse=''), expected))
actual_meta <- prior_meta; actual_meta$call <- observed
stopifnot(identical(postcoef_historical_prior_state(actual_meta), 'none'))
call_expr <- quote(gllvmTMB(f, data = df_long, unit = 'site', trait = 'trait',
                          family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE)))
eval_observed <- eval(call_expr, envir=environment())
normalised <- match.call(definition = gllvmTMB, call = call_expr)
stopifnot(identical(eval_observed, normalised))
call_meta <- prior_meta; call_meta$call <- normalised
stopifnot(identical(postcoef_historical_prior_state(call_meta), 'none'))
cat('POSTCOEF_R_MATCH_CALL_CONTROL_PASS\n')
