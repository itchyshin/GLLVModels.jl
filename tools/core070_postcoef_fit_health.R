# Draft diagnostic collector. It does not fit a model or admit a parity row.
# gradient_eval and objective_eval must evaluate the exact fitted objective,
# including every active fixed-effect and loading prior. Their route and
# provenance require separate verification by the calling capture.
postcoef_r_fit_health <- function(case, fit, gradient_eval, objective_eval, fixed_prior_state) {
  out <- list(case = case, pd_hessian = NULL, max_abs_gradient = NULL,
              objective_verified = FALSE, penalty_configuration_recorded = FALSE,
              error = NULL, parameters = NULL, gradient = NULL, free_indices = NULL, optimizer_objective = NULL,
              recomputed_objective = NULL, ridge_tau = NULL,
              ridge_penalised = NULL, fixed_prior_state = fixed_prior_state,
              optimizer_convergence_code = NULL,
              # Honest absence: only sdreport's pdHess flag is retained, so no
              # Hessian matrix, Hessian coordinate or error radius is recorded and
              # R cannot be certified. Gradient is at fit$opt$par (TMB optimizer
              # coordinates); no gradient error radius is recorded.
              hessian_available = FALSE, gradient_coordinate = 'tmb_opt_par')
  error <- function(message) { out$error <<- message; out }
  if (!is.list(fit) || !is.list(fit$opt) || is.null(fit$tmb_obj) ||
      !is.numeric(fit$opt$par) || !length(fit$opt$par) ||
      any(!is.finite(fit$opt$par))) return(error('fit parameters unavailable or non-finite'))
  out$parameters <- as.numeric(fit$opt$par)
  code <- fit$opt$convergence
  if (is.numeric(code) && length(code) == 1L && is.finite(code))
    out$optimizer_convergence_code <- as.numeric(code)
  tau <- fit$aghq$ridge_tau
  penalised <- fit$aghq$penalised
  ridge_ok <- is.numeric(tau) && length(tau) == 1L && !is.na(tau) && tau > 0 &&
    is.logical(penalised) && length(penalised) == 1L && !is.na(penalised) &&
    ((isTRUE(penalised) && is.finite(tau)) || (identical(penalised, FALSE) && is.infinite(tau)))
  config_ok <- ridge_ok &&
    is.character(fixed_prior_state) && length(fixed_prior_state) == 1L &&
    !is.na(fixed_prior_state) && fixed_prior_state %in% c('none', 'active')
  out$penalty_configuration_recorded <- isTRUE(config_ok)
  if (is.numeric(tau) && length(tau) == 1L && !is.na(tau))
    out$ridge_tau <- if (is.infinite(tau)) 'Inf' else tau
  if (is.logical(penalised) && length(penalised) == 1L && !is.na(penalised))
    out$ridge_penalised <- penalised
  pd <- fit$sd_report$pdHess
  if (is.logical(pd) && length(pd) == 1L && !is.na(pd)) out$pd_hessian <- pd
  observed <- fit$opt$objective
  if (is.numeric(observed) && length(observed) == 1L && is.finite(observed))
    out$optimizer_objective <- as.numeric(observed)
  if (!ridge_ok) return(error('loading-ridge configuration incomplete'))
  computed <- tryCatch(objective_eval(fit$tmb_obj, fit$opt$par, tau), error = function(e) NULL)
  if (is.numeric(computed) && length(computed) == 1L && is.finite(computed)) {
    out$recomputed_objective <- as.numeric(computed)
    out$objective_verified <- !is.null(out$optimizer_objective) &&
      identical(out$recomputed_objective, out$optimizer_objective)
  }
  g <- tryCatch(gradient_eval(fit$tmb_obj, fit$opt$par, tau), error = function(e) NULL)
  if (!is.numeric(g) || length(g) != length(fit$opt$par) || any(!is.finite(g)))
    return(error('gradient unavailable, malformed or non-finite'))
  out$gradient <- as.numeric(g)
  out$free_indices <- seq_along(g)
  out$max_abs_gradient <- max(abs(g))
  if (!config_ok) out$error <- 'fixed-effect prior configuration incomplete'
  if (isTRUE(penalised)) out$error <- 'penalised Hessian not established by sdreport'
  if (is.null(out$pd_hessian)) out$error <- 'Hessian status unavailable'
  out
}

# components_eval must be the source-pinned objective decomposition, and
# gradient_eval must be its matching penalty-aware gradient. A caller must
# independently establish fixed_prior_state from the actual fitted model.
postcoef_r_health_from_fit <- function(case, fit, gradient_eval, components_eval, fixed_prior_state) {
  objective_eval <- function(obj, par, tau) {
    stopifnot(identical(obj, fit$tmb_obj), identical(par, fit$opt$par),
              identical(tau, fit$aghq$ridge_tau))
    components <- components_eval(obj, fit$opt, fit$aghq)
    components$optimization_nll
  }
  postcoef_r_fit_health(case, fit, gradient_eval, objective_eval, fixed_prior_state)
}

# Narrow source-reviewed historical route only. The caller must bind the
# oracle_verified assertion and call to the authenticated capture record.
postcoef_historical_prior_state <- function(meta) {
  if (!is.list(meta) || !identical(meta$pin, 'P1') ||
      !identical(meta$reference_commit, '9539352f66f2db2cc26b1c393e67212a359b60c9') ||
      !identical(meta$source_tree_sha256, '86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db') ||
      !identical(meta$version, '0.7.1') || !identical(meta$oracle_verified, TRUE) ||
      !is.call(meta$call)) return(NULL)
  call <- as.list(meta$call)
  argnames <- names(call)
  if (is.null(argnames) || length(call) != 7L || !identical(as.character(call[[1L]]), 'gllvmTMB')) return(NULL)
  argnames[2L] <- if (identical(argnames[2L], '')) 'formula' else argnames[2L]
  if (!setequal(argnames[-1L], c('formula', 'data', 'unit', 'trait', 'family', 'control')) ||
      anyDuplicated(argnames[-1L])) return(NULL)
  if (!identical(call[[2L]], as.name('f')) ||
      !identical(call[['data']], as.name('df_long')) ||
      !identical(call[['unit']], 'site') || !identical(call[['trait']], 'trait') ||
      !identical(call[['family']], quote(gaussian())) ||
      !identical(call[['control']], quote(gllvmTMBcontrol(n_init = 1L, se = TRUE)))) return(NULL)
  'none'
}
