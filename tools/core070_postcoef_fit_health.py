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

What is and is not bound here. R ``parameters`` is any finite vector of the
gradient's length, and nothing ties a case name to its parameter count except
the consistency checks of ``_evidence_binding`` (enabled certifier only): the
three Julia lambda maps share p and K, free-coefficient counts start at 0 and
never decrease along CASES, and R's parameter count differs from Julia's
optimizer-point length by one constant. Those are necessary conditions, not
authentication: the caller must still bind each record to its case, data and
objective. Likewise ``lambda_constraint_map`` is a record field that the
receipt verifier must compare with the case's authenticated input
specification. Every derivative error radius is a declaration with
``radius_status == 'declared_unproven'``; a certified pass carries that status
(``classify(...).radius_status``) and real-record admission stays blocked on an
independent derivation of the radii for the Laplace objective.
"""
import math
import sys
import re
from fractions import Fraction

ENGINES = ('R', 'Julia')
CASES = ('no_fixed_effects', 'trait_intercepts', 'trait_intercepts_plus_covariate')
GRADIENT_LIMIT = 1e-5  # Signed 2026-10-21 rule: strictly below this value.
NUMERICAL_ADMISSION_IMPLEMENTED = False  # Hold: certifier below awaits independent review.
HESSIAN_COORDINATE = 'packed_free_index'  # Maintainer ruling 2026-11-07.
CERT_MAX_FREE_DIMENSION = 32   # Refuse (INCOMPLETE) above this free-block size.
CERT_MAX_WORK = 2000000        # Refuse (INCOMPLETE) above this many deterministic work units.
CERT_GRADIENT_LIMIT = Fraction(1, 100000)  # Exact decimal 1e-5, strict upper bound.
CERT_EPS = Fraction(1, 2 ** 52)  # Representation unit for the radius floors below.
RADIUS_STATUS = 'declared_unproven'  # The only admissible status: no derivation of the radii exists.


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


# ---------------------------------------------------------------------------
# Numerical-admission certifier (diagnostic; enabled only by the hold flag).
#
# Object certified: the RECORDED symmetric packed Hessian restricted to the
# recorded free indices ("packed free-index" coordinate). Nothing is
# transferred to another parameterisation: away from a stationary point the
# Hessian is not invariant (H_phi = J'HJ + sum_i g_i * Hess(theta_i)), so the
# record must declare ``hessian_coordinate == 'packed_free_index'``.
#
# Method. Exact binary64 values are lifted to Fractions. Exact LDL^T without
# pivoting is a Sylvester test: all pivots > 0 iff the block is positive
# definite; the first pivot <= 0 proves it is not. With the declared entrywise
# error radius r (plus the recorded symmetrisation asymmetry), any true
# Hessian H satisfies |H - A| <= r entrywise on the k-by-k block, so by Weyl
# and ||E||_2 <= ||E||_F <= k*r:   lambda_min(A) - k*r <= lambda_min(H)
#                                  <= lambda_min(A) + k*r.
# Hence  A - k*r*I positive definite  => H certified positive definite;
#        A + k*r*I not positive definite => H certified not positive definite
#        (indefinite or singular); anything between is INCOMPLETE.
# Gradient. Two separate bounds, each strictly below 1e-5 with its own declared
# radius and in its own coordinates: the optimizer-coordinate gradient (the
# signed rule; it may certify a failure) and the packed gradient at the point
# where the Hessian was evaluated (it may only block a pass). Their maxima are
# different quantities and are not compared for equality.
# Cost. Exact work is metered in deterministic units (see _Work), never by
# wall-clock, so a verdict depends on the record alone.
# Not covered: the justification of r itself, Hessian accuracy beyond r, and
# the gradient error radii; all radii are caller declarations
# (``radius_status == 'declared_unproven'``).
# ---------------------------------------------------------------------------
CERT_PASS, CERT_FAIL, CERT_INCOMPLETE = 'CERT_PASS', 'CERT_FAIL', 'CERT_INCOMPLETE'
LAMBDA_MAP_LAYOUT = 'beta_free,sigma_eps,lambda_b_packed'


class _CertRefusal(Exception):
    pass


class _Work:
    """Deterministic work meter: one unit per Fraction multiply-add plus its operand size.

    Replaces a wall-clock cap so a verdict depends on the record alone. Cost of
    one multiply-add is 1 + (bits of the operands) // 64, with the bits of a
    Fraction counted as numerator plus denominator bit lengths.
    """

    def __init__(self, cap):
        self.cap, self.used = cap, 0

    def charge(self, *values):
        self.used += 1 + sum(v.numerator.bit_length() + v.denominator.bit_length() for v in values) // 64
        if self.used > self.cap:
            raise _CertRefusal('work cap exceeded')


def _exact_pd(block, shift, work):
    """Exact test that block - shift*I is positive definite (Fractions)."""
    k = len(block)
    a = [[block[i][j] - (shift if i == j else 0) for j in range(k)] for i in range(k)]
    lower = [[Fraction(0)] * k for _ in range(k)]
    diag = []
    for j in range(k):
        pivot = a[j][j]
        for s in range(j):
            work.charge(lower[j][s], diag[s])
            pivot -= lower[j][s] ** 2 * diag[s]
        if pivot <= 0:
            return False
        diag.append(pivot)
        for i in range(j + 1, k):
            entry = a[i][j]
            for s in range(j):
                work.charge(lower[i][s], lower[j][s], diag[s])
                entry -= lower[i][s] * lower[j][s] * diag[s]
            work.charge(entry, pivot)
            lower[i][j] = entry / pivot
    return True


def certify_hessian(matrix, free_indices, coordinate, error_radius, asymmetry=0.0,
                    pinned_indices=None, max_dimension=None, max_work=None,
                    parameter_count=None):
    """Certify the recorded packed free-index Hessian; return a verdict dict.

    verdict is CERT_PASS (certified positive definite with margin exceeding
    the declared error bound), CERT_FAIL (certified not positive definite)
    or CERT_INCOMPLETE (cannot be certified, with a reason). The echoed
    ``free_indices`` and ``excluded_indices`` show which block was decided.

    ``pinned_indices`` and ``parameter_count`` (the length of the fit's own
    recorded parameters vector) are mandatory: free and pinned indices must
    partition exactly 1..parameter_count, and the matrix must have that size.
    ``error_radius`` is a caller declaration, not provenance; the floor
    k * 2**-52 * max|H_ij| over the free block (k = block size) is only a
    necessary sanity bound against sub-representation radii.
    """
    max_dimension = CERT_MAX_FREE_DIMENSION if max_dimension is None else max_dimension
    max_work = CERT_MAX_WORK if max_work is None else max_work
    out = {'verdict': CERT_INCOMPLETE, 'reason': None, 'coordinate': coordinate,
           'free_indices': None, 'excluded_indices': None, 'error_bound': None, 'work_units': 0}

    def refuse(reason):
        out['reason'] = reason
        return out
    if coordinate != HESSIAN_COORDINATE:
        return refuse('Hessian coordinate is not the declared packed free-index coordinate')
    if not isinstance(matrix, list) or not matrix:
        return refuse('matrix missing or malformed')
    n = len(matrix)
    if (any(not isinstance(row, list) or len(row) != n for row in matrix) or
            not all(finite_number(x) for row in matrix for x in row)):
        return refuse('matrix missing, non-square or non-finite')
    if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(i)):
        return refuse('matrix is not exactly symmetric')
    if (not isinstance(free_indices, list) or not free_indices or
            any(type(i) is not int or not 1 <= i <= n for i in free_indices) or
            len(set(free_indices)) != len(free_indices)):
        return refuse('free indices missing or invalid')
    if type(parameter_count) is not int or parameter_count != n:
        return refuse('recorded parameter count missing or differs from the Hessian size')
    if (not isinstance(pinned_indices, list) or any(type(i) is not int for i in pinned_indices) or
            len(set(pinned_indices)) != len(pinned_indices) or
            set(pinned_indices) & set(free_indices) or
            set(pinned_indices) | set(free_indices) != set(range(1, parameter_count + 1))):
        return refuse('free and pinned indices are not recorded as an exact partition of the parameters')
    excluded = sorted(pinned_indices)
    out['free_indices'], out['excluded_indices'] = list(free_indices), excluded
    if not finite_number(error_radius) or error_radius <= 0:
        return refuse('derivative error radius not recorded or not positive')
    if not finite_number(asymmetry) or asymmetry < 0:
        return refuse('asymmetry not recorded')
    k = len(free_indices)
    peak = max(abs(Fraction(matrix[i - 1][j - 1])) for i in free_indices for j in free_indices)
    if Fraction(error_radius) < k * CERT_EPS * peak:
        return refuse('declared radius is below the representation floor k*2^-52*max|H|')
    if k > max_dimension:
        return refuse('free block of %d exceeds the size cap of %d' % (k, max_dimension))
    bound = k * (Fraction(error_radius) + Fraction(asymmetry))
    out['error_bound'] = bound
    block = [[Fraction(matrix[i - 1][j - 1]) for j in free_indices] for i in free_indices]
    work = _Work(max_work)
    try:
        if _exact_pd(block, bound, work):
            out['verdict'] = CERT_PASS
        elif not _exact_pd(block, -bound, work):
            out['verdict'] = CERT_FAIL
        else:
            out['work_units'] = work.used
            return refuse('definiteness not decided: margin does not exceed the declared error bound')
    except _CertRefusal as exc:
        out['work_units'] = work.used
        return refuse(str(exc))
    out['work_units'] = work.used
    return out


def certify_gradient(max_abs_gradient, error_radius):
    """Stated gradient bound: PASS only if g + e < 1e-5 (optimizer coordinates)."""
    if not finite_number(max_abs_gradient) or max_abs_gradient < 0:
        return CERT_INCOMPLETE
    if not finite_number(error_radius) or error_radius <= 0:
        return CERT_INCOMPLETE
    g, e = Fraction(max_abs_gradient), Fraction(error_radius)
    if e < CERT_EPS * g:  # representation floor; the radius itself stays a declaration
        return CERT_INCOMPLETE
    if g + e < CERT_GRADIENT_LIMIT:
        return CERT_PASS
    if g - e >= CERT_GRADIENT_LIMIT:
        return CERT_FAIL
    return CERT_INCOMPLETE


def _finite_vector(value):
    return isinstance(value, list) and bool(value) and all(finite_number(x) for x in value)


def _lambda_pin_binding(fit):
    """Recompute the pinned packed indices from the fit's own ``lambda_constraint`` map.

    The record must carry ``lambda_constraint_map`` = {layout, p, K, q_free,
    entries} (entries: p rows of K cells, null = free, number = pinned value).
    Packed layout (src/confint_profile.jl ``_profile_all_term_names`` and
    src/packing.jl ``_lower_index``): q_free betas, one log sigma, then the
    packed loadings (diagonal 1..K, strict lower triangle column by column);
    the structural zeros above the diagonal are not in theta. Returns None when
    the declared ``pinned_indices`` equal the map-derived set, ``parameters``
    has the layout's length and sits at every pinned value, else a reason.
    The map itself is a record field: the receipt verifier must still compare
    it with the case's authenticated input specification.
    """
    m, pinned, theta = fit.get('lambda_constraint_map'), fit.get('pinned_indices'), fit.get('parameters')
    if not isinstance(m, dict):
        return 'record carries no lambda_constraint_map to bind the pinned indices to'
    if set(m) != {'layout', 'p', 'K', 'q_free', 'entries'} or m['layout'] != LAMBDA_MAP_LAYOUT:
        return 'lambda_constraint_map has the wrong keys or layout'
    p, K, q_free, entries = m['p'], m['K'], m['q_free'], m['entries']
    if (any(type(v) is not int for v in (p, K, q_free)) or p < 1 or not 1 <= K <= p or q_free < 0 or
            not isinstance(entries, list) or len(entries) != p or
            any(not isinstance(row, list) or len(row) != K for row in entries) or
            any(not (v is None or finite_number(v)) for row in entries for v in row)):
        return 'lambda_constraint_map is malformed'
    if not isinstance(theta, list) or len(theta) != q_free + 1 + p * K - K * (K - 1) // 2:
        return 'parameters length does not match the lambda_constraint_map layout'
    derived = {}
    for i in range(1, p + 1):
        for k in range(1, K + 1):
            if i <= K and k > i or entries[i - 1][k - 1] is None:
                continue
            position = k if i == k else K + (k - 1) * p - k * (k - 1) // 2 + i - k
            derived[q_free + 1 + position] = entries[i - 1][k - 1]
    if not isinstance(pinned, list) or sorted(derived) != sorted(pinned):
        return 'declared pinned indices differ from those derived from the lambda_constraint_map'
    if any(theta[j - 1] != v for j, v in derived.items()):
        return 'parameters are not at the pinned values of the lambda_constraint_map'
    return None


def _optimizer_binding(fit):
    """Bind the optimizer-coordinate block to the fit; return a reason or None.

    The block must be well formed (point and gradient finite vectors of equal
    length, scalar equal to the vector maximum, profile verified), its minimum
    must agree with the fit's own objective within the objective roundoff rule,
    and the reconstruction must have been recovered from exactly this point.
    """
    opt, rec = fit.get('optimizer'), fit.get('reconstruction')
    if not isinstance(opt, dict) or not isinstance(rec, dict):
        return 'optimizer or reconstruction block missing'
    point, gradient = opt.get('optimizer_parameters'), opt.get('optimizer_gradient')
    if not _finite_vector(point) or not _finite_vector(gradient) or len(point) != len(gradient):
        return 'optimizer point or gradient missing, non-finite or of unequal length'
    if opt.get('optimizer_max_abs_gradient') != max(abs(x) for x in gradient):
        return 'optimizer gradient scalar is not the maximum of the optimizer gradient vector'
    value, minimum = opt.get('optimizer_profile_value'), opt.get('optimizer_minimum')
    if (opt.get('optimizer_profile_verified') is not True or opt.get('optimizer_error', True) is not None or
            not finite_number(value) or not finite_number(minimum) or value != minimum):
        return 'optimizer profile value is not verified against the optimizer minimum'
    objective = fit.get('optimizer_objective')
    if not finite_number(objective) or abs(minimum - objective) > (
            8 * sys.float_info.epsilon * max(1.0, abs(minimum), abs(objective))):
        return 'optimizer minimum disagrees with the fitted objective'
    if rec.get('optimizer_parameters') != point:
        return 'reconstruction was not recovered from the recorded optimizer point'
    if opt.get('radius_status') != RADIUS_STATUS or fit.get('radius_status') != RADIUS_STATUS:
        return 'derivative error radii are not recorded as declared_unproven'
    return None


def _certify_fit(fit, engine):
    result = {'verdict': CERT_INCOMPLETE, 'reason': None, 'hessian': None, 'gradient': None,
              'radius_status': None}
    if engine != 'Julia' or not isinstance(fit, dict):
        result['reason'] = 'no recorded Hessian for this engine'
        return result
    if fit.get('penalty_state') != 'none':
        result['reason'] = 'penalised or prior model: the no-prior diagnostic does not certify it'
        return result
    vector, parameters, free = fit.get('gradient'), fit.get('parameters'), fit.get('free_indices')
    if not _finite_vector(vector) or not _finite_vector(parameters) or len(vector) != len(parameters):
        result['reason'] = 'packed gradient vector missing, non-finite or not the parameter dimension'
        return result
    dimension = len(vector)
    if (not isinstance(free, list) or not free or any(type(i) is not int or not 1 <= i <= dimension for i in free) or
            len(set(free)) != len(free)):
        result['reason'] = 'free indices missing or invalid'
        return result
    for reason in (_lambda_pin_binding(fit), _optimizer_binding(fit)):
        if reason:
            result['reason'] = reason
            return result
    for key in ('gradient_error_radius', 'packed_gradient_error_radius'):
        # Even a certified Hessian failure needs both gradient radii in well-formed form.
        if not finite_number(fit.get(key)) or fit[key] <= 0:
            result['reason'] = 'gradient error radius not recorded or not positive'
            return result
    h = certify_hessian(fit.get('hessian'), free, fit.get('hessian_coordinate'),
                        fit.get('hessian_error_radius'), fit.get('hessian_asymmetry'),
                        fit.get('pinned_indices'), parameter_count=dimension)
    packed_max = max(abs(vector[i - 1]) for i in free)
    if fit.get('max_abs_gradient') != packed_max:
        result['reason'] = 'recorded gradient scalar is not the maximum over the free indices'
        return result
    # Two separate bounds, each with its own declared radius, in its own coordinates:
    # the optimizer-coordinate gradient (the signed rule; may also certify a failure)
    # and the packed gradient at the point where the Hessian was evaluated (may only
    # block a pass: the signed limit is not defined in packed coordinates).
    g_opt = certify_gradient(fit['optimizer']['optimizer_max_abs_gradient'], fit.get('gradient_error_radius'))
    g_packed = certify_gradient(packed_max, fit.get('packed_gradient_error_radius'))
    g = CERT_PASS if g_opt == CERT_PASS and g_packed == CERT_PASS else (
        CERT_FAIL if g_opt == CERT_FAIL else CERT_INCOMPLETE)
    result['hessian'], result['gradient'] = h, g
    if h['verdict'] == CERT_FAIL or g == CERT_FAIL:
        result['verdict'] = CERT_FAIL
    elif h['verdict'] == CERT_PASS and g == CERT_PASS:
        result['verdict'] = CERT_PASS
        result['radius_status'] = RADIUS_STATUS
    else:
        result['reason'] = h['reason'] or 'gradient bound not certified'
    return result


def certify_fit(fit, engine='Julia'):
    """Certificate for one recorded Julia fit: curvature and two gradient bounds.

    PASS needs the packed gradient vector, the lambda-constraint binding of the
    pinned indices, the bound optimizer block, and both gradient bounds; its
    ``radius_status`` echoes ``declared_unproven`` because the error radii are
    declarations. Never raises: an unusable record is CERT_INCOMPLETE.
    """
    try:
        return _certify_fit(fit, engine)
    except Exception:  # hostile or malformed objects never reach a verdict
        return {'verdict': CERT_INCOMPLETE, 'reason': 'record could not be evaluated safely',
                'hessian': None, 'gradient': None, 'radius_status': None}


def _r_block_valid(evidence):
    """The R evidence block is well formed for this evidence (not missing or garbage).

    Structural only: R's own pd flag or gradient size may still be a failure
    elsewhere, but unusable R evidence cannot support confirming a Julia failure.
    """
    fits = evidence.get('R') if isinstance(evidence, dict) else None
    if not isinstance(fits, list) or len(fits) != len(CASES) or not all(isinstance(f, dict) for f in fits):
        return False
    cases = [f.get('case') for f in fits]
    if any(type(c) is not str for c in cases) or set(cases) != set(CASES) or len(set(cases)) != len(CASES):
        return False
    for fit in fits:
        vector, parameters, indices = fit.get('gradient'), fit.get('parameters'), fit.get('free_indices')
        if not (fit.get('objective_verified') is True and fit.get('penalty_configuration_recorded') is True and
                'error' in fit and fit['error'] is None):
            return False
        if not (isinstance(vector, list) and vector and all(finite_number(x) for x in vector) and
                isinstance(parameters, list) and len(parameters) == len(vector) and
                all(finite_number(x) for x in parameters)):
            return False
        if not (isinstance(indices, list) and all(type(i) is int for i in indices) and
                set(indices) == set(range(1, len(vector) + 1)) and len(set(indices)) == len(indices)):
            return False
        grad = fit.get('max_abs_gradient')
        if not finite_number(grad) or grad < 0 or grad != max(abs(x) for x in vector):
            return False
        a, b = fit.get('optimizer_objective'), fit.get('recomputed_objective')
        if not finite_number(a) or not finite_number(b) or a != b or type(fit.get('pd_hessian')) is not bool:
            return False
        if (fit.get('fixed_prior_state') != 'none' or fit.get('ridge_penalised') is not False or
                fit.get('ridge_tau') != 'Inf'):
            return False
    return True


def _record_checks_valid(fit, r_valid):
    """A certified failure is confirmed only on a record that passed its own checks.

    Mirrors the gradient-failure path (bound case, verified profile) and adds
    the Hessian pair (raw present, raw-versus-symmetrised valid), the parameter
    and log-likelihood reconstruction, the recomputed objective within the
    record's own recorded roundoff check, and a valid R evidence block
    (``r_valid``, from ``_r_block_valid``).
    """
    if r_valid is not True:
        return False
    vector = fit.get('gradient')
    if not isinstance(vector, list) or not vector:
        return False
    if hessian_record_status(fit, len(vector)) != 'valid':
        return False
    reconstruction, parameters = fit.get('reconstruction'), fit.get('parameters')
    if (not isinstance(reconstruction, dict) or reconstruction.get('parameters_match') is not True or
            reconstruction.get('recovered_parameters') != parameters):
        return False
    objective, recomputed = fit.get('optimizer_objective'), fit.get('recomputed_objective')
    recovered_loglik = reconstruction.get('recovered_loglik')
    if (reconstruction.get('loglik_matches') is not True or 'error' not in reconstruction or
            reconstruction['error'] is not None or not finite_number(objective) or
            not finite_number(recomputed) or not finite_number(recovered_loglik) or
            recovered_loglik != -objective):
        return False
    difference = abs(objective - recomputed)
    limit = 8 * sys.float_info.epsilon * max(1.0, abs(objective), abs(recomputed))
    recorded_difference, recorded_limit = fit.get('objective_difference'), fit.get('objective_roundoff_limit')
    if (not finite_number(difference) or not finite_number(limit) or difference > limit or
            not finite_number(recorded_difference) or not finite_number(recorded_limit) or
            recorded_difference != difference or recorded_limit != limit):
        return False
    opt = fit.get('optimizer')
    if not isinstance(opt, dict):
        return False
    ov, mv = opt.get('optimizer_profile_value'), opt.get('optimizer_minimum')
    return (opt.get('optimizer_profile_verified') is True and 'optimizer_error' in opt and
            opt['optimizer_error'] is None and finite_number(ov) and finite_number(mv) and ov == mv)


def _evidence_binding(evidence):
    """Case and dimension binding across the six records; a reason or None.

    Necessary consistency only (the caller still authenticates provenance):
    the three Julia lambda maps share p and K (one response matrix), the number
    of free coefficients never decreases along CASES and is 0 without fixed
    effects, and R's optimizer-parameter count differs from Julia's by one
    constant across the cases (the same model, parameterised alike).
    """
    maps, offsets = [], []
    julia = {f.get('case'): f for f in evidence['Julia']}
    r_fits = {f.get('case'): f for f in evidence['R']}
    for case in CASES:
        m = julia[case].get('lambda_constraint_map')
        point = (julia[case].get('optimizer') or {}).get('optimizer_parameters')
        r_parameters = r_fits[case].get('parameters')
        if not isinstance(m, dict) or not isinstance(point, list) or not isinstance(r_parameters, list):
            return 'case binding evidence missing'
        maps.append(m)
        offsets.append(len(r_parameters) - len(point))
    if len({(m.get('p'), m.get('K')) for m in maps}) != 1:
        return 'the three cases do not share the response dimension and rank'
    q = [m.get('q_free') for m in maps]
    if any(type(v) is not int for v in q) or q[0] != 0 or q != sorted(q):
        return 'free-coefficient counts are inconsistent with the case order'
    if len(set(offsets)) != 1:
        return 'R and Julia parameter counts are not bound to the same cases'
    return None


def certify_evidence(evidence):
    """Per-fit certificates; ``failure_confirmable`` needs a valid unique case set."""
    certs, confirmable = [], True
    r_valid = _r_block_valid(evidence)
    fits = evidence.get('Julia') if isinstance(evidence, dict) else None
    if not isinstance(fits, list) or len(fits) != len(CASES):
        return [], False
    cases = [f.get('case') if isinstance(f, dict) else None for f in fits]
    if any(type(case) is not str for case in cases):  # unhashable or non-string labels: never reach set()
        return [], False
    if set(cases) != set(CASES) or len(set(cases)) != len(CASES):
        confirmable = False
    for fit in fits:
        cert = certify_fit(fit)
        bound = (isinstance(fit, dict) and fit.get('objective_verified') is True and
                 fit.get('penalty_configuration_recorded') is True and
                 'error' in fit and fit.get('error') is None)
        if not bound:
            confirmable = False
        cert['confirmable'] = bool(bound and _record_checks_valid(fit, r_valid))
        certs.append(cert)
    binding = _evidence_binding(evidence) if confirmable and r_valid is True else None
    if binding:
        # An unbound case set cannot support a pass or a confirmed failure.
        for cert in certs:
            cert['confirmable'] = False
            if cert['verdict'] == CERT_PASS:
                cert.update(verdict=CERT_INCOMPLETE, reason=binding, radius_status=None)
        confirmable = False
    return certs, confirmable


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


class FitHealthVerdict(str):
    """A verdict string that also names the status of the error radii behind it.

    Equal to the plain verdict string; ``radius_status`` is ``declared_unproven``
    on a certified pass (the radii are declarations, not derivations) and None
    otherwise.
    """
    radius_status = None


def _classify(evidence):
    verdict = _classify_recorded(evidence)
    if not NUMERICAL_ADMISSION_IMPLEMENTED:
        return 'FIT_HEALTH_INCOMPLETE' if verdict == 'FIT_HEALTH_PASS' else verdict
    if verdict == 'FIT_HEALTH_INTEGRITY_REJECTED':
        return verdict
    certs, confirmable = certify_evidence(evidence)
    certified_failure = confirmable and any(c['verdict'] == CERT_FAIL and c['confirmable'] for c in certs)
    all_certified = len(certs) == len(CASES) and all(c['verdict'] == CERT_PASS for c in certs)
    if verdict == 'FIT_HEALTH_PASS':
        if certified_failure:
            return 'FIT_HEALTH_FAIL'
        if not all_certified:
            return 'FIT_HEALTH_INCOMPLETE'
        passed = FitHealthVerdict(verdict)
        passed.radius_status = RADIUS_STATUS
        return passed
    if certified_failure:
        return 'FIT_HEALTH_FAIL' if verdict == 'FIT_HEALTH_FAIL' else 'FIT_HEALTH_FAIL_WITH_INCOMPLETE'
    return verdict


def classify(evidence):
    """Classify six recorded diagnostics; never raises.

    While ``NUMERICAL_ADMISSION_IMPLEMENTED`` is False every would-be pass is
    FIT_HEALTH_INCOMPLETE and the certifier is not consulted. When enabled, a
    pass additionally needs every Julia fit certified (curvature with margin
    over the declared error bound, plus the stated gradient bounds), and a
    certified indefinite or singular Hessian on valid evidence is a failure.
    Anything not certified stays INCOMPLETE. A certified pass carries
    ``radius_status == 'declared_unproven'``: the radii are declarations.
    An object that raises while being inspected yields FIT_HEALTH_INCOMPLETE,
    never an exception and never a pass.
    """
    try:
        return _classify(evidence)
    except Exception:
        return 'FIT_HEALTH_INCOMPLETE'
