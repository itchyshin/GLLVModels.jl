"""Exact / rigorously enclosed derivatives for the POST-COEF K = 1 Gaussian route (PROTOTYPE).

Scratch prototype, not wired into ``core070_postcoef_fit_health``. It implements
section b2 of radii-derivation-20261108.md for the only route the writer certifies
(``penalty_state == 'none'``, K = 1, no diagonal, no W tier, no phylogeny,
closed-form Gaussian marginal):

* ``exact_optimizer_gradient``: the gradient of ``gaussian_profile_nll(...;
  profile_beta=false)`` at the recorded binary64 optimizer point, exactly, in
  Fractions (the derivative is rational in the inputs; no exp/log survives).
* ``packed_enclosure``: a rigorous interval enclosure of the packed gradient and
  Hessian of ``gaussian_nll_packed`` at the recorded binary64 packed point. The
  only non-rational quantity is ``s = exp(2*l)``; it is enclosed by a Taylor
  series with a Lagrange remainder, and everything else is evaluated in
  interval arithmetic over Fractions with outward (directed) dyadic rounding.
* ``computed_radii``: the radii a certifier would then use, computed from the
  record's own values instead of declared.

Trusted base: Python ``int``/``Fraction`` and the formulas below. That the
formulas equal what the Julia functions compute is a code-reading claim
(caveat i of the derivation), checked here only numerically against an
independent high-precision transcription of the Julia functions (tests).

Hardening (wave 15, after the wave-14 red team): inputs are copied once to tuples of exact
binary64 floats and only the copy is used (plain list/tuple only, no subclasses); |2l| <= 708;
q must be an int; computed_radii binds the packed point to the optimizer point exactly
(check_binding) and meters work from the shape (estimate_work / work_budget).

Outside the module, by design (cannot be closed without authenticated inputs): that Y and X are
the case's real data (byte-level digest against the authenticated input, which also fixes sign
flips and observation order, both invisible to the likelihood at q = 0); that p, n, q, K and the
constraint map match the authenticated spec; that sigma_eps^2 equals Q/(n p) for the real data;
that the recorded values came from the real fit; the receipt chain and flag flip.

Layout follows the Julia engine: ``Y`` is p x n (``Y[t][s]``), ``X`` is
p x n x q (``X[t][s][j]``) or None; packed ``theta = [beta (q); l = log sigma_eps;
lambda (p)]``; optimizer ``x = [beta (q); L (p)]``.
"""
from fractions import Fraction
import math

PREC_BITS = 256          # Mantissa bits kept by outward rounding.
# |2l| above this is refused. exp(x) overflows binary64 for x > 709.78 and is subnormal below
# -708.40, so 708 keeps s = exp(2l) a finite normal binary64 value, i.e. a point the Julia
# engine can actually produce (l = log sigma_eps, |l| <= 354).
EXP_ARG_LIMIT = 708
STATUS_COMPUTED = 'derived_enclosure'
# Deterministic, shape-only cost bound (see estimate_work). About 15-16 microseconds per unit on
# the Mac in pure Python (measured: p=12,n=200 ~ 2.9 s, p=20,n=200 ~ 21 s). Over budget is refused
# before any arithmetic; a caller may pass a larger work_budget explicitly.
DEFAULT_WORK_BUDGET = 2_000_000


class ExactInputError(ValueError):
    """Input is outside the K = 1 closed-form route or not exact binary64 data."""


# ---------------------------------------------------------------------------
# Outward-rounded rational intervals
# ---------------------------------------------------------------------------
def _floor_dyadic(v, bits=PREC_BITS):
    """Largest dyadic rational <= v with about ``bits`` significant bits."""
    if v == 0:
        return Fraction(0)
    e = bits - v.numerator.bit_length() + v.denominator.bit_length()
    if e >= 0:
        return Fraction((v.numerator << e) // v.denominator, 1 << e)
    return Fraction(v.numerator // (v.denominator << -e) << -e)


def _ceil_dyadic(v, bits=PREC_BITS):
    return -_floor_dyadic(-v, bits)


class Iv:
    """Closed interval [lo, hi] with Fraction endpoints; every result rounds outward."""

    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        hi = lo if hi is None else hi
        if lo > hi:
            raise ValueError('empty interval')
        self.lo, self.hi = Fraction(lo), Fraction(hi)

    @staticmethod
    def _out(lo, hi):
        return Iv(_floor_dyadic(lo), _ceil_dyadic(hi))

    @staticmethod
    def lift(v):
        return v if isinstance(v, Iv) else Iv(v)

    def __add__(self, o):
        o = Iv.lift(o)
        return Iv._out(self.lo + o.lo, self.hi + o.hi)
    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-Iv.lift(o))

    def __rsub__(self, o):
        return Iv.lift(o) - self

    def __mul__(self, o):
        o = Iv.lift(o)
        c = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Iv._out(min(c), max(c))
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Iv.lift(o)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError('interval divisor contains zero')
        return self * Iv._out(1 / o.hi, 1 / o.lo)

    def __rtruediv__(self, o):
        return Iv.lift(o) / self

    def contains(self, v):
        return self.lo <= v <= self.hi

    def radius_about(self, v):
        """max |e - v| over e in the interval: a rigorous bound on |true - v|."""
        v = Fraction(v)
        return max(abs(self.lo - v), abs(self.hi - v))


def exp_enclosure(t, bits=PREC_BITS):
    """Rigorous Iv containing exp(t) for an exact rational t (Taylor + Lagrange remainder).

    Argument reduction t = u * 2^k with |u| <= 1/2. On |u| <= 1/2 the remainder of
    the N-term series is at most |u|^N / N! * e^{1/2} < 2 |u|^N / N!. The result is
    squared k times; the enclosure stays positive so squaring is monotone.
    """
    t = Fraction(t)
    if abs(t) > EXP_ARG_LIMIT:
        raise ExactInputError('exp argument outside the supported range')
    k = 0
    while abs(t) / (1 << k) > Fraction(1, 2):
        k += 1
    u = t / (1 << k)
    target = Fraction(1, 1 << (bits + 16))
    total, term, i = Fraction(0), Fraction(1), 0
    while True:
        total += term
        i += 1
        term = term * u / i
        bound = 2 * abs(term)  # |u|^i / i! * e^{1/2} bound for the tail starting at index i
        if bound < target:
            break
    enc = Iv(_floor_dyadic(total - bound, bits), _ceil_dyadic(total + bound, bits))
    if enc.lo <= 0:
        raise ArithmeticError('exp enclosure lost positivity')  # cannot happen for |u| <= 1/2
    for _ in range(k):
        enc = Iv(_floor_dyadic(enc.lo * enc.lo, bits), _ceil_dyadic(enc.hi * enc.hi, bits))
    return enc


# ---------------------------------------------------------------------------
# Input validation: copy once to tuples of floats, validate the COPY, use only the copy
# ---------------------------------------------------------------------------
def _is_binary64(v):
    return type(v) is float and math.isfinite(v)


def _freeze_seq(v, what):
    """Copy an exact ``list`` or ``tuple`` to a tuple once. Subclasses are refused: their
    __iter__/__getitem__ could answer differently on later reads (time-of-check/time-of-use)."""
    if type(v) is not list and type(v) is not tuple:
        raise ExactInputError('%s must be a plain list or tuple, not %s' % (what, type(v).__name__))
    return tuple(v)


def _freeze_floats(v, length, what):
    t = _freeze_seq(v, what)
    if len(t) != length or not all(_is_binary64(x) for x in t):
        raise ExactInputError('%s must be %d finite binary64 values' % (what, length))
    return t


def _check_q(q):
    if type(q) is not int or q < 0:
        raise ExactInputError('q must be a non-negative int')
    return q


def estimate_work(p, n, q):
    """Deterministic cost bound in abstract units from the shape alone (not a runtime counter).

    n p^2 (1+q+q^2) exact moment products + 4 (p+1) p^3 + 3 (p+1)^2 p^2 interval products."""
    return n * p * p * (1 + q + q * q) + 4 * (p + 1) * p ** 3 + 3 * (p + 1) ** 2 * p ** 2


def _check_work(p, n, q, work_budget):
    if type(work_budget) is not int or work_budget < 0:
        raise ExactInputError('work_budget must be a non-negative int')
    if estimate_work(p, n, q) > work_budget:
        raise ExactInputError('estimated work %d exceeds work_budget %d' %
                              (estimate_work(p, n, q), work_budget))


def _freeze_data(Y, X, q):
    """Return (Yt, Xt, p, n): tuples of tuples of binary64 floats, validated on the copy."""
    _check_q(q)
    rows = _freeze_seq(Y, 'Y')
    if not rows:
        raise ExactInputError('Y must be a non-empty p x n matrix')
    Yt = tuple(_freeze_seq(r, 'Y row') for r in rows)
    p, n = len(Yt), len(Yt[0])
    if n == 0 or any(len(r) != n for r in Yt) or not all(_is_binary64(v) for r in Yt for v in r):
        raise ExactInputError('Y must be rectangular and hold finite binary64 values')
    if q == 0:
        if X is not None:
            raise ExactInputError('q = 0 but X supplied')
        return Yt, None, p, n
    if X is None:
        raise ExactInputError('X must be p x n x q')
    Xt = tuple(tuple(_freeze_seq(c, 'X cell') for c in _freeze_seq(row, 'X row'))
               for row in _freeze_seq(X, 'X'))
    if (len(Xt) != p or any(len(row) != n for row in Xt) or
            any(len(c) != q for row in Xt for c in row) or
            not all(_is_binary64(v) for row in Xt for c in row for v in c)):
        raise ExactInputError('X must be p x n x q with finite binary64 values')
    return Yt, Xt, p, n


class _Moments:
    """Exact Fraction moments of the data at a given beta (no dependence on s or lambda)."""

    def __init__(self, Y, X, beta):
        p, n, q = len(Y), len(Y[0]), len(beta)
        b = [Fraction(v) for v in beta]
        Xf = None if X is None else [[[Fraction(v) for v in c] for c in row] for row in X]
        r = [[Fraction(Y[t][s]) - (sum(Xf[t][s][j] * b[j] for j in range(q)) if q else 0)
              for s in range(n)] for t in range(p)]
        self.p, self.n, self.q = p, n, q
        # R2[a][b] = sum_s r[a,s] r[b,s]
        self.R2 = [[sum(r[a][s] * r[c][s] for s in range(n)) for c in range(p)] for a in range(p)]
        # Xr[j][a][b] = sum_s X[a,s,j] r[b,s]        (sum_s x_sj r_s')
        self.Xr = [[[sum(Xf[a][s][j] * r[c][s] for s in range(n)) for c in range(p)] for a in range(p)]
                   for j in range(q)]
        # XX[j][k][a][b] = sum_s X[a,s,j] X[b,s,k]   (sum_s x_sj x_sk')
        self.XX = [[[[sum(Xf[a][s][j] * Xf[c][s][k] for s in range(n)) for c in range(p)]
                     for a in range(p)] for k in range(q)] for j in range(q)]


def _matmul(A, B):
    m, l, w = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][j] for t in range(1, l)), A[i][0] * B[0][j]) for j in range(w)]
            for i in range(m)]


def _tr_prod(A, B):
    """tr(A B) = sum_ab A[a][b] B[b][a]."""
    p = len(A)
    out = A[0][0] * B[0][0]
    for a in range(p):
        for b in range(p):
            if a or b:
                out = out + A[a][b] * B[b][a]
    return out


def _transpose(A):
    return [list(r) for r in zip(*A)]


# ---------------------------------------------------------------------------
# R2: exact optimizer-coordinate gradient (profile NLL, profile_beta = false)
# ---------------------------------------------------------------------------
def exact_optimizer_gradient(Y, X, x, q, work_budget=DEFAULT_WORK_BUDGET):
    """Exact gradient of gaussian_profile_nll(x; profile_beta=false) for K = 1.

    f = 1/2 [N log 2pi + N + N log(Q/N) + n log a], a = 1 + L'L,
    A^-1 = I - L L'/a, Q = sum_s r_s' A^-1 r_s, N = n p.
    d/dbeta_j = -(N/Q) sum_s x_sj' A^-1 r_s ;  d/dL_i = -(N/Q) (A^-1 R2 A^-1 L)_i + n L_i / a.
    Returns a list of Fractions (exact).
    """
    Y, X, p, n = _freeze_data(Y, X, q)
    x = _freeze_floats(x, q + p, 'optimizer point')
    _check_work(p, n, q, work_budget)
    return _exact_optimizer_gradient(Y, X, x, q, p, n)


def _exact_optimizer_gradient(Y, X, x, q, p, n):
    L = [Fraction(v) for v in x[q:]]
    mom = _Moments(Y, X, x[:q])
    a = 1 + sum(v * v for v in L)
    Ainv = [[(1 if i == j else 0) - L[i] * L[j] / a for j in range(p)] for i in range(p)]
    Q = _tr_prod(Ainv, mom.R2)
    if Q <= 0:
        raise ExactInputError('profile quadratic form is not positive')
    N = n * p
    g = [-(Fraction(N) / Q) * _tr_prod(Ainv, _transpose(mom.Xr[j])) for j in range(q)]
    AinvL = [v / a for v in L]                      # A^-1 L = L / a
    R2AinvL = [sum(mom.R2[i][j] * AinvL[j] for j in range(p)) for i in range(p)]
    w = [sum(Ainv[i][j] * R2AinvL[j] for j in range(p)) for i in range(p)]
    g += [-(Fraction(N) / Q) * w[i] + n * L[i] / a for i in range(p)]
    return g


# ---------------------------------------------------------------------------
# R1 / R3: rigorous enclosure of the packed gradient and Hessian
# ---------------------------------------------------------------------------
def packed_enclosure(Y, X, theta, q, work_budget=DEFAULT_WORK_BUDGET):
    """Rigorous Iv enclosures (g, H) of the gradient and Hessian of gaussian_nll_packed.

    s = exp(2 l), c = s + lambda'lambda, P = (I - lambda lambda'/c)/s, r_s = y_s - X_s beta.
    Sigma_l = 2 s I, Sigma_ll = 4 s I, Sigma_li = e_i lambda' + lambda e_i',
    Sigma_lilj = e_i e_j' + e_j e_i', Sigma_{l,li} = 0. With A_psi = P Sigma_psi,
    W = P R2, G = W P:
      g_beta_j = -<P, Xr_j>
      g_psi    = 1/2 [n tr(A_psi) - tr(A_psi W)]
      H_bb_jk  = <P, XX_jk>
      H_bpsi_j = tr(A_psi P Xr_j')
      H_psiphi = 1/2 n [tr(P Sigma_psiphi) - tr(A_phi A_psi)] + tr(A_phi A_psi W)
                 - 1/2 tr(Sigma_psiphi G)
    Returns (g, H): a list of Iv and a list of lists of Iv, packed order.
    """
    Y, X, p, n = _freeze_data(Y, X, q)
    theta = _freeze_floats(theta, q + 1 + p, 'packed parameters')
    _check_work(p, n, q, work_budget)
    return _packed_enclosure(Y, X, theta, q, p, n)


def _packed_enclosure(Y, X, theta, q, p, n):
    mom = _Moments(Y, X, theta[:q])
    lam = [Fraction(v) for v in theta[q + 1:]]
    s = exp_enclosure(2 * Fraction(theta[q]))
    lam2 = sum(v * v for v in lam)
    c = s + lam2
    P = [[((1 if i == j else 0) - lam[i] * lam[j] / c) / s for j in range(p)] for i in range(p)]
    W = _matmul(P, mom.R2)
    G = _matmul(W, P)
    zero = Iv(0)
    # Sigma_psi as explicit matrices (Iv entries); psi = 0 is l, psi = 1 + i is lambda_i.
    sig = [[[2 * s if i == j else zero for j in range(p)] for i in range(p)]]
    for t in range(p):
        sig.append([[Iv((lam[j] if i == t else 0) + (lam[i] if j == t else 0)) for j in range(p)]
                    for i in range(p)])
    A = [_matmul(P, S_) for S_ in sig]
    AW = [_matmul(Ai, W) for Ai in A]
    trP, trG = sum(P[i][i] for i in range(p)), sum(G[i][i] for i in range(p))
    m = p + 1

    def tr_P_sig2(a, b):           # tr(P Sigma_ab)
        if a == 0 and b == 0:
            return 4 * s * trP
        if a == 0 or b == 0:
            return zero
        return 2 * P[b - 1][a - 1]

    def tr_sig2_G(a, b):           # tr(Sigma_ab G)
        if a == 0 and b == 0:
            return 4 * s * trG
        if a == 0 or b == 0:
            return zero
        return G[a - 1][b - 1] + G[b - 1][a - 1]

    k = q + m
    g = [None] * k
    H = [[None] * k for _ in range(k)]
    for j in range(q):
        g[j] = -_tr_prod(P, _transpose(mom.Xr[j]))
        for l_ in range(q):
            H[j][l_] = _tr_prod(P, _transpose(mom.XX[j][l_]))
    PXrT = [_matmul(P, _transpose(mom.Xr[j])) for j in range(q)]
    for a in range(m):
        trA = sum(A[a][i][i] for i in range(p))
        g[q + a] = Fraction(1, 2) * (n * trA - sum(AW[a][i][i] for i in range(p)))
        for j in range(q):
            H[j][q + a] = H[q + a][j] = _tr_prod(A[a], PXrT[j])
        for b in range(m):
            H[q + a][q + b] = (Fraction(n, 2) * (tr_P_sig2(a, b) - _tr_prod(A[b], A[a]))
                               + _tr_prod(A[b], AW[a]) - Fraction(1, 2) * tr_sig2_G(a, b))
    return g, H


# ---------------------------------------------------------------------------
# Binding of the packed point to the optimizer point (no data needed)
# ---------------------------------------------------------------------------
def _rn_preimage(v):
    """Closed real interval containing every real that rounds to the binary64 ``v``."""
    prev, nxt = math.nextafter(v, -math.inf), math.nextafter(v, math.inf)
    fv = Fraction(v)
    lo = (fv + Fraction(prev)) / 2 if math.isfinite(prev) else fv - (Fraction(nxt) - fv) / 2
    hi = (fv + Fraction(nxt)) / 2 if math.isfinite(nxt) else fv + (fv - Fraction(prev)) / 2
    return lo, hi


MAX_DECISION_BITS = 4096


def _in_log_cell(sigma, llo, lhi):
    """True iff exp(llo) < sigma < exp(lhi) (equality is impossible: exp of a non-zero rational is
    irrational). Precision is raised until the enclosures decide, which is needed when the cell is
    tiny (l = 0 has a subnormal-width cell); undecided at the cap -> False (fail closed)."""
    sg, bits = Fraction(sigma), PREC_BITS
    while bits <= MAX_DECISION_BITS:
        e_lo, e_hi = exp_enclosure(llo, bits), exp_enclosure(lhi, bits)
        if sg < e_lo.lo or sg > e_hi.hi:
            return False
        if e_lo.hi <= sg <= e_hi.lo:
            return True
        bits *= 2
    return False


def check_binding(theta, x, q):
    """Verify, without data, that packed ``theta`` is what the writer derives from optimizer ``x``.

    Writer (src/profile.jl profile_recover): sigma_eps = sqrt(quad/np); lambda_j = fl(sigma_eps * L_j);
    packed slot q+1 = fl(log sigma_eps); betas copied. The check is exact (no tolerance): betas are
    equal; there is a binary64 sigma > 0 with fl(sigma * L_j) = lambda_j for every j (L_j = 0 needs
    the same-signed zero; L_j != 0 needs a non-zero loading, so a genuine underflow to zero is
    refused, matching the live certifier's rule) and RN(log sigma) = l, the latter decided by a rigorous exp enclosure that must
    place sigma strictly inside the rounding cell of l (precision is raised until decided; still
    undecided at 4096 bits is refused).
    Returns that sigma. Raises ExactInputError otherwise. Candidate set capped at 64.

    NOT checked here (needs authenticated Y, X): that sigma^2 = Q/(n p) for the real data.
    """
    _check_q(q)
    x = _freeze_seq(x, 'optimizer point')
    p = len(x) - q
    if p < 1:
        raise ExactInputError('optimizer point too short')
    x = _freeze_floats(x, q + p, 'optimizer point')
    theta = _freeze_floats(theta, q + 1 + p, 'packed parameters')
    if any(theta[j] != x[j] for j in range(q)):
        raise ExactInputError('binding: beta differs between packed and optimizer points')
    ell, lam, L = theta[q], theta[q + 1:], x[q:]
    if abs(Fraction(ell)) > EXP_ARG_LIMIT:
        raise ExactInputError('binding: log sigma_eps outside the supported range')
    llo, lhi = _rn_preimage(ell)
    e_lo, e_hi = exp_enclosure(llo), exp_enclosure(lhi)
    lower, upper = e_lo.lo, e_hi.hi                       # outer real interval for sigma from l
    for lj, Lj in zip(lam, L):
        if Lj == 0.0:
            if lj != 0.0 or math.copysign(1.0, lj) != math.copysign(1.0, Lj):
                raise ExactInputError('binding: zero loading does not match zero L (or sign differs)')
            continue
        if lj == 0.0 or (lj > 0.0) != (Lj > 0.0):
            raise ExactInputError('binding: non-zero L needs a non-zero loading of the same sign')
        plo, phi = _rn_preimage(lj)
        r1, r2 = plo / Fraction(Lj), phi / Fraction(Lj)
        lower, upper = max(lower, min(r1, r2)), min(upper, max(r1, r2))
    if lower > upper:
        raise ExactInputError('binding: no sigma_eps reproduces the packed point from the optimizer point')
    cand = math.nextafter(float(lower), -math.inf)
    for _ in range(64):
        if Fraction(cand) > upper:
            break
        if (cand > 0.0 and
                all(cand * Lj == lj and math.copysign(1.0, cand * Lj) == math.copysign(1.0, lj)
                    for lj, Lj in zip(lam, L)) and _in_log_cell(cand, llo, lhi)):
            return cand
        cand = math.nextafter(cand, math.inf)
    else:
        raise ExactInputError('binding: candidate sigma_eps set too large')
    raise ExactInputError('binding: no binary64 sigma_eps reproduces the packed point')


# ---------------------------------------------------------------------------
# Computed radii
# ---------------------------------------------------------------------------
def computed_radii(Y, X, q, theta, x, hessian_raw, packed_gradient, optimizer_gradient,
                   work_budget=DEFAULT_WORK_BUDGET):
    """Radii from the record's own inputs, replacing declared ones (prototype).

    Every input is copied once to tuples of binary64 floats and only the copy is used.
    The packed point must be bound to the optimizer point (check_binding) before any arithmetic.
    The caller must still authenticate Y and X (see the module docstring, "Outside the module").

    hessian_radius / packed_gradient_radius: rigorous upper bounds on
    max |recorded - true| (entrywise / infinity norm) at theta.
    optimizer_gradient_radius: the exact error max |recorded - true| at x.
    exact_optimizer_max_abs_gradient: max |true gradient| at x, exact; a certifier
    could test it directly against 1e-5 with no radius at all.
    """
    Y, X, p, n = _freeze_data(Y, X, q)
    k = q + 1 + p
    theta = _freeze_floats(theta, k, 'packed parameters')
    x = _freeze_floats(x, k - 1, 'optimizer point')
    hrows = _freeze_seq(hessian_raw, 'recorded Hessian')
    if len(hrows) != k:
        raise ExactInputError('recorded Hessian has the wrong size')
    hessian_raw = tuple(_freeze_floats(r, k, 'recorded Hessian row') for r in hrows)
    packed_gradient = _freeze_floats(packed_gradient, k, 'recorded packed gradient')
    optimizer_gradient = _freeze_floats(optimizer_gradient, k - 1, 'recorded optimizer gradient')
    _check_work(p, n, q, work_budget)
    check_binding(theta, x, q)
    g_enc, H_enc = _packed_enclosure(Y, X, theta, q, p, n)
    g_opt = _exact_optimizer_gradient(Y, X, x, q, p, n)
    return {
        'radius_status': STATUS_COMPUTED,
        'hessian_radius': max(H_enc[i][j].radius_about(hessian_raw[i][j]) for i in range(k) for j in range(k)),
        'packed_gradient_radius': max(g_enc[i].radius_about(packed_gradient[i]) for i in range(k)),
        'optimizer_gradient_radius': max(abs(Fraction(optimizer_gradient[i]) - g_opt[i]) for i in range(k - 1)),
        'exact_optimizer_max_abs_gradient': max(abs(v) for v in g_opt),
        'packed_gradient_enclosure_max_abs': max(max(abs(e.lo), abs(e.hi)) for e in g_enc),
    }


def round_up_binary64(v):
    """Smallest binary64 >= the Fraction v (for handing a radius to a float-typed API)."""
    f = float(v)
    while Fraction(f) < v:
        f = math.nextafter(f, math.inf)
    return f
