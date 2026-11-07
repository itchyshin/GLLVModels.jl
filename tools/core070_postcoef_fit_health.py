"""Classify retained fit-health fields without admitting a parity receipt.

The caller must separately authenticate provenance and establish that each
record describes the objective and free-parameter block actually fitted.

Numerical admission is held: ``classify`` never returns ``FIT_HEALTH_PASS``
while ``NUMERICAL_ADMISSION_IMPLEMENTED`` is False. The Float64 Cholesky in
``hessian_pd`` is a screen, not a certificate: a fixed binary64 matrix with a
negative exact determinant passes it. A would-be pass is therefore reported as
``FIT_HEALTH_INCOMPLETE`` until a reviewed numerical rule (and a maintainer
ruling on Hessian coordinates) replaces it.

Outcomes, in precedence order:
  FIT_HEALTH_INTEGRITY_REJECTED   a recorded summary contradicts the recorded
                                  vector, matrix or derived field it summarises
  FIT_HEALTH_FAIL_WITH_INCOMPLETE a confirmed failure plus incomplete evidence
  FIT_HEALTH_FAIL                 a confirmed failure
  FIT_HEALTH_INCOMPLETE           anything not confirmed, including every
                                  bare flag or scalar and every would-be pass
A confirmed failure is only an optimizer-coordinate gradient vector, bound to
a valid unique case with verified objective and no recorded error, whose
maximum absolute free-coordinate entry is at least ``GRADIENT_LIMIT`` and
agrees exactly with its recorded scalar. Bare ``pd_hessian`` flags, bare
scalars and a Float64-Cholesky-false matrix never yield a failure.
``FIT_HEALTH_FAIL_WITH_INCOMPLETE`` is retained because existing tests use it;
it still denotes a confirmed failure, never an incomplete-only state.
"""
import math
import sys
import re

ENGINES = ('R', 'Julia')
CASES = ('no_fixed_effects', 'trait_intercepts', 'trait_intercepts_plus_covariate')
GRADIENT_LIMIT = 1e-5  # Signed 2026-10-21 rule: strictly below this value.
NUMERICAL_ADMISSION_IMPLEMENTED = False  # Hold: no Hessian rule is certified.


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def hessian_pd(matrix, indices, dimension):
    """Validate the retained symmetric matrix and Cholesky its free block.

    Float64 screen only: True does not certify positive definiteness of the
    recorded binary64 matrix, and False does not certify a failure.
    """
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


def hessian_record_status(fit, dimension):
    """Return 'valid', 'incomplete' or 'integrity' for the recorded Hessian pair.

    'integrity' means well-formed records contradict each other: the stored
    symmetric matrix is not the symmetrised raw matrix, or the stored
    asymmetry or its limit differs from the recomputed value.
    """
    raw, sym = fit.get('hessian_raw'), fit.get('hessian')
    def valid(m):
        return (isinstance(m, list) and len(m) == dimension and
                all(isinstance(r, list) and len(r) == dimension and all(finite_number(x) for x in r) for r in m))
    if not valid(raw) or not valid(sym):
        return 'incomplete'
    asym = max(abs(raw[i][j]-raw[j][i]) for i in range(dimension) for j in range(dimension))
    bound = 32 * sys.float_info.epsilon * dimension * max(1.0, max(abs(x) for row in raw for x in row))
    if not (finite_number(asym) and finite_number(bound)):
        return 'incomplete'
    if not all(sym[i][j] == (raw[i][j] + raw[j][i])/2 for i in range(dimension) for j in range(dimension)):
        return 'integrity'
    recorded_asym, recorded_bound = fit.get('hessian_asymmetry'), fit.get('hessian_asymmetry_limit')
    if not finite_number(recorded_asym) or not finite_number(recorded_bound):
        return 'incomplete'
    if recorded_asym != asym or recorded_bound != bound:
        return 'integrity'
    return 'valid' if asym <= bound else 'incomplete'


def hessian_roundoff_valid(fit, dimension):
    return hessian_record_status(fit, dimension) == 'valid'


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


def _classify_recorded(evidence):
    """Classify six recorded diagnostics before the numerical-admission hold."""
    incomplete = False
    failed = False
    integrity = False
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
        case_set_valid = (isinstance(fits, list) and len(fits) == len(CASES) and
                          len(cases) == len(CASES) and all(type(case) is str for case in cases) and
                          len(set(cases)) == len(CASES) and set(cases) == set(CASES))
        for fit in fits:
            if not isinstance(fit, dict):
                incomplete = True
                continue
            # A failure can be confirmed only on a record bound to its case and
            # verified objective; otherwise it is merely incomplete.
            bound = (case_set_valid and fit.get('case') in CASES and
                     fit.get('objective_verified') is True and
                     fit.get('penalty_configuration_recorded') is True and
                     'error' in fit and fit.get('error') is None)
            if not bound:
                incomplete = True
            pd = fit.get('pd_hessian')
            if type(pd) is not bool or not pd:
                # A bare flag (R sdreport pdHess or Julia flag) is never a failure.
                incomplete = True
            grad = fit.get('max_abs_gradient')
            if not finite_number(grad) or grad < 0:
                incomplete = True
            # A bare scalar alone is never a failure; it only summarises the
            # vector below, which must confirm it exactly.
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
                if not finite_number(grad) or grad < 0:
                    incomplete = True  # malformed scalar: never a failure
                elif actual != grad:
                    integrity = True
                elif (engine == 'R' and actual >= GRADIENT_LIMIT and bound and
                      isinstance(parameters, list) and len(parameters) == len(vector)):
                    # R gradients are evaluated at fit$opt$par: optimizer coordinates.
                    failed = True
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
                    ov, mv = opt_record.get('optimizer_profile_value'), opt_record.get('optimizer_minimum')
                    profile_bound = (opt_record.get('optimizer_profile_verified') is True and
                                     'optimizer_error' in opt_record and opt_record['optimizer_error'] is None and
                                     finite_number(ov) and finite_number(mv) and ov == mv)
                    if not profile_bound:
                        incomplete = True
                    valid = (isinstance(params, list) and bool(params) and all(finite_number(x) for x in params) and
                             isinstance(og, list) and len(og) == len(params) and all(finite_number(x) for x in og))
                    if not valid or not finite_number(om) or om < 0:
                        incomplete = True
                    else:
                        actual_og = max(abs(x) for x in og)
                        if actual_og != om:
                            integrity = True
                        elif actual_og >= GRADIENT_LIMIT:
                            if bound and profile_bound:
                                failed = True
                            else:
                                incomplete = True
            if engine == 'Julia' and valid_indices:
                status = hessian_record_status(fit, len(vector))
                if status == 'integrity':
                    integrity = True
                elif status != 'valid':
                    incomplete = True
                matrix_pd = hessian_pd(fit.get('hessian'), indices, len(vector))
                # Float64 Cholesky is a screen: neither True nor False is a
                # certificate, so a false result or a flag mismatch is incomplete.
                if matrix_pd is not True or type(pd) is not bool or matrix_pd != pd:
                    incomplete = True
            opt, recomputed = fit.get('optimizer_objective'), fit.get('recomputed_objective')
            if not finite_number(opt) or not finite_number(recomputed):
                incomplete = True
            elif engine == 'Julia':
                difference = abs(opt - recomputed)
                limit = 8 * sys.float_info.epsilon * max(1.0, abs(opt), abs(recomputed))
                recorded_difference = fit.get('objective_difference')
                recorded_limit = fit.get('objective_roundoff_limit')
                if (not finite_number(difference) or not finite_number(limit) or difference > limit or
                    not finite_number(recorded_difference) or not finite_number(recorded_limit)):
                    incomplete = True
                elif recorded_difference != difference or recorded_limit != limit:
                    integrity = True
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
    if integrity:
        return 'FIT_HEALTH_INTEGRITY_REJECTED'
    if failed and incomplete:
        return 'FIT_HEALTH_FAIL_WITH_INCOMPLETE'
    if failed:
        return 'FIT_HEALTH_FAIL'
    if incomplete:
        return 'FIT_HEALTH_INCOMPLETE'
    return 'FIT_HEALTH_PASS'


def classify(evidence):
    """Classify six recorded diagnostics; no record is admitted while held.

    Missing evidence never becomes a pass, and while numerical admission is
    unimplemented a would-be pass is reported as FIT_HEALTH_INCOMPLETE.
    """
    verdict = _classify_recorded(evidence)
    if verdict == 'FIT_HEALTH_PASS' and not NUMERICAL_ADMISSION_IMPLEMENTED:
        return 'FIT_HEALTH_INCOMPLETE'
    return verdict
