"""Classify retained fit-health fields without admitting a parity receipt.

The caller must separately authenticate provenance and establish that each
record describes the objective and free-parameter block actually fitted.
"""
import math
import sys
import re

ENGINES = ('R', 'Julia')
CASES = ('no_fixed_effects', 'trait_intercepts', 'trait_intercepts_plus_covariate')
GRADIENT_LIMIT = 1e-5  # Signed 2026-10-21 rule: strictly below this value.


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def hessian_pd(matrix, indices, dimension):
    """Validate the retained symmetric matrix and Cholesky its free block."""
    if (not isinstance(matrix, list) or len(matrix) != dimension or
        any(not isinstance(row, list) or len(row) != dimension or
            not all(finite_number(x) for x in row) for row in matrix)):
        return None
    if any(matrix[i][j] != matrix[j][i] for i in range(dimension) for j in range(i)):
        return None
    block = [[matrix[i-1][j-1] for j in indices] for i in indices]
    n = len(block)
    lower = [[0.0] * n for _ in range(n)]
    try:
        for i in range(n):
            for j in range(i+1):
                value = block[i][j] - sum(lower[i][k]*lower[j][k] for k in range(j))
                if not math.isfinite(value):
                    return None
                if i == j:
                    if value <= 0:
                        return False
                    lower[i][j] = math.sqrt(value)
                else:
                    lower[i][j] = value / lower[j][j]
    except (OverflowError, ZeroDivisionError):
        return None
    return True


def hessian_roundoff_valid(fit, dimension):
    raw, sym = fit.get('hessian_raw'), fit.get('hessian')
    def valid(m):
        return (isinstance(m, list) and len(m) == dimension and
                all(isinstance(r, list) and len(r) == dimension and all(finite_number(x) for x in r) for r in m))
    if not valid(raw) or not valid(sym):
        return False
    asym = max(abs(raw[i][j]-raw[j][i]) for i in range(dimension) for j in range(dimension))
    bound = 32 * sys.float_info.epsilon * dimension * max(1.0, max(abs(x) for row in raw for x in row))
    return (finite_number(asym) and finite_number(bound) and asym <= bound and
            finite_number(fit.get('hessian_asymmetry')) and finite_number(fit.get('hessian_asymmetry_limit')) and
            fit.get('hessian_asymmetry') == asym and fit.get('hessian_asymmetry_limit') == bound and
            all(sym[i][j] == (raw[i][j] + raw[j][i])/2 for i in range(dimension) for j in range(dimension)))


def historical_r_prior_bound(provenance, build_sha256):
    """Validate a claim against an independently authenticated build hash."""
    return (isinstance(provenance, dict) and
            set(provenance) == {'pin', 'reference_commit', 'source_tree_sha256', 'version',
                                'oracle_build_sha256', 'call_verified', 'call_text'} and
            provenance['pin'] == 'P1' and
            provenance['reference_commit'] == '9539352f66f2db2cc26b1c393e67212a359b60c9' and
            provenance['source_tree_sha256'] == '86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db' and
            provenance['version'] == '0.7.1' and provenance['call_verified'] is True and
            provenance['call_text'] == 'gllvmTMB(formula = f, data = df_long, trait = "trait", unit = "site", family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE))' and
            isinstance(build_sha256, str) and re.fullmatch(r'[0-9a-f]{64}', build_sha256) is not None and
            provenance['oracle_build_sha256'] == build_sha256)


def julia_prior_bound(provenance, authenticated_tree, case):
    """Accept only the audited exact Gaussian profile route and source tree."""
    return (isinstance(provenance, dict) and
            set(provenance) == {'source_tree', 'case', 'route', 'rank', 'integration', 'prior'} and
            authenticated_tree == 'df63c52066625049087a321bb9b010f8d834ddcf' and
            provenance['source_tree'] == authenticated_tree and case in CASES and
            provenance['case'] == case and provenance['route'] == 'gaussian_profile_default' and
            type(provenance['rank']) is int and provenance['rank'] == 1 and
            provenance['integration'] == 'exact' and provenance['prior'] == 'none')


def classify(evidence):
    """Classify six recorded diagnostics; missing evidence never becomes a pass."""
    incomplete = False
    failed = False
    if not isinstance(evidence, dict) or set(evidence) != set(ENGINES):
        incomplete = True
    for engine in ENGINES:
        fits = evidence.get(engine) if isinstance(evidence, dict) else None
        if not isinstance(fits, list) or len(fits) != len(CASES):
            incomplete = True
        if not isinstance(fits, list):
            continue
        cases = [fit.get('case') for fit in fits if isinstance(fit, dict)]
        if (len(cases) != len(CASES) or any(type(case) is not str for case in cases) or
                len(set(cases)) != len(CASES) or set(cases) != set(CASES)):
            incomplete = True
        for fit in fits:
            if not isinstance(fit, dict):
                incomplete = True
                continue
            if (fit.get('objective_verified') is not True or
                    fit.get('penalty_configuration_recorded') is not True or
                    'error' not in fit or
                    fit.get('error') is not None):
                incomplete = True
            pd = fit.get('pd_hessian')
            if type(pd) is not bool:
                incomplete = True
            elif not pd and not (engine == 'R' and fit.get('ridge_penalised') is True):
                failed = True
            grad = fit.get('max_abs_gradient')
            if not finite_number(grad) or grad < 0:
                incomplete = True
            elif engine == 'R' and grad >= GRADIENT_LIMIT:
                failed = True
            vector, indices = fit.get('gradient'), fit.get('free_indices')
            parameters = fit.get('parameters')
            if not (isinstance(parameters, list) and isinstance(vector, list) and
                    len(parameters) == len(vector) and all(finite_number(x) for x in parameters)):
                incomplete = True
            valid_vector = isinstance(vector, list) and bool(vector) and all(finite_number(x) for x in vector)
            valid_indices = (valid_vector and isinstance(indices, list) and bool(indices) and
                             all(type(i) is int and 1 <= i <= len(vector) for i in indices) and
                             len(set(indices)) == len(indices))
            if engine == 'R' and valid_indices and set(indices) != set(range(1, len(vector) + 1)):
                valid_indices = False
            if not valid_indices:
                incomplete = True
            else:
                actual = max(abs(vector[i - 1]) for i in indices)
                if engine == 'R' and actual >= GRADIENT_LIMIT:
                    failed = True
                if not finite_number(grad) or actual != grad:
                    incomplete = True
            if engine == 'Julia':
                reconstruction = fit.get('reconstruction')
                if not isinstance(reconstruction, dict):
                    incomplete = True
                else:
                    recovered = reconstruction.get('recovered_parameters')
                    recovered_loglik = reconstruction.get('recovered_loglik')
                    fitted_objective = fit.get('optimizer_objective')
                    if (reconstruction.get('parameters_match') is not True or
                        reconstruction.get('loglik_matches') is not True or
                        'error' not in reconstruction or reconstruction['error'] is not None or
                        not isinstance(recovered, list) or not all(finite_number(x) for x in recovered) or
                        not isinstance(parameters, list) or recovered != parameters or
                        not finite_number(recovered_loglik) or not finite_number(fitted_objective) or
                        recovered_loglik != -fitted_objective):
                        incomplete = True
                opt_record = fit.get('optimizer')
                if not isinstance(opt_record, dict):
                    incomplete = True
                else:
                    params, og = opt_record.get('optimizer_parameters'), opt_record.get('optimizer_gradient')
                    om = opt_record.get('optimizer_max_abs_gradient')
                    valid = (isinstance(params, list) and bool(params) and all(finite_number(x) for x in params) and
                             isinstance(og, list) and len(og) == len(params) and all(finite_number(x) for x in og))
                    if not valid or not finite_number(om) or om < 0:
                        incomplete = True
                    else:
                        actual_og = max(abs(x) for x in og)
                        if actual_og >= GRADIENT_LIMIT or om >= GRADIENT_LIMIT:
                            failed = True
                        if actual_og != om:
                            incomplete = True
                    ov, mv = opt_record.get('optimizer_profile_value'), opt_record.get('optimizer_minimum')
                    if (opt_record.get('optimizer_profile_verified') is not True or
                        'optimizer_error' not in opt_record or opt_record['optimizer_error'] is not None or
                        not finite_number(ov) or not finite_number(mv) or ov != mv):
                        incomplete = True
            if engine == 'Julia' and valid_indices:
                if not hessian_roundoff_valid(fit, len(vector)):
                    incomplete = True
                matrix_pd = hessian_pd(fit.get('hessian'), indices, len(vector))
                if matrix_pd is None:
                    incomplete = True
                elif not matrix_pd:
                    failed = True
                if matrix_pd is None or type(pd) is not bool or matrix_pd != pd:
                    incomplete = True
            opt, recomputed = fit.get('optimizer_objective'), fit.get('recomputed_objective')
            if not finite_number(opt) or not finite_number(recomputed):
                incomplete = True
            elif engine == 'Julia':
                difference = abs(opt - recomputed)
                bound = 8 * sys.float_info.epsilon * max(1.0, abs(opt), abs(recomputed))
                if (not finite_number(difference) or not finite_number(bound) or difference > bound or
                    not finite_number(fit.get('objective_difference')) or
                    not finite_number(fit.get('objective_roundoff_limit')) or
                    fit.get('objective_difference') != difference or
                    fit.get('objective_roundoff_limit') != bound):
                    incomplete = True
            elif opt != recomputed:
                incomplete = True
            if engine == 'R':
                prior, ridge, tau = fit.get('fixed_prior_state'), fit.get('ridge_penalised'), fit.get('ridge_tau')
                # sdreport receives the unwrapped TMB objective, not the R-level ridge.
                if ridge is True:
                    incomplete = True
                if (prior != 'none' or type(ridge) is not bool or
                    (ridge and (not finite_number(tau) or tau <= 0)) or
                    (ridge is False and tau != 'Inf')):
                    incomplete = True
            elif fit.get('penalty_state') != 'none':
                incomplete = True
    if failed and incomplete:
        return 'FIT_HEALTH_FAIL_WITH_INCOMPLETE'
    if failed:
        return 'FIT_HEALTH_FAIL'
    if incomplete:
        return 'FIT_HEALTH_INCOMPLETE'
    return 'FIT_HEALTH_PASS'
