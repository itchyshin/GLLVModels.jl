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

Layout follows the Julia engine: ``Y`` is p x n (``Y[t][s]``), ``X`` is
p x n x q (``X[t][s][j]``) or None; packed ``theta = [beta (q); l = log sigma_eps;
lambda (p)]``; optimizer ``x = [beta (q); L (p)]``.
"""
from fractions import Fraction
import math

PREC_BITS = 256          # Mantissa bits kept by outward rounding.
EXP_ARG_LIMIT = 1400     # |2l| above this is refused (exp would overflow binary64 anyway).
STATUS_COMPUTED = 'derived_enclosure'


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


def exp_enclosure(t):
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
    target = Fraction(1, 1 << (PREC_BITS + 16))
    total, term, i = Fraction(0), Fraction(1), 0
    while True:
        total += term
        i += 1
        term = term * u / i
        bound = 2 * abs(term)  # |u|^i / i! * e^{1/2} bound for the tail starting at index i
        if bound < target:
            break
    enc = Iv(_floor_dyadic(total - bound), _ceil_dyadic(total + bound))
    if enc.lo <= 0:
        raise ArithmeticError('exp enclosure lost positivity')  # cannot happen for |u| <= 1/2
    for _ in range(k):
        enc = Iv(_floor_dyadic(enc.lo * enc.lo), _ceil_dyadic(enc.hi * enc.hi))
    return enc


# ---------------------------------------------------------------------------
# Input validation and exact data moments
# ---------------------------------------------------------------------------
def _is_binary64(v):
    return type(v) is float and math.isfinite(v)


def _check_inputs(Y, X, q):
    if not isinstance(Y, list) or not Y or any(not isinstance(r, list) for r in Y):
        raise ExactInputError('Y must be a non-empty p x n list of lists')
    p, n = len(Y), len(Y[0])
    if n == 0 or any(len(r) != n for r in Y) or not all(_is_binary64(v) for r in Y for v in r):
        raise ExactInputError('Y must be rectangular and hold finite binary64 values')
    if q == 0:
        if X is not None:
            raise ExactInputError('q = 0 but X supplied')
        return p, n
    if (not isinstance(X, list) or len(X) != p or
            any(not isinstance(row, list) or len(row) != n for row in X) or
            any(not isinstance(c, list) or len(c) != q for row in X for c in row) or
            not all(_is_binary64(v) for row in X for c in row for v in c)):
        raise ExactInputError('X must be p x n x q with finite binary64 values')
    return p, n


def _check_vector(v, length, what):
    if not isinstance(v, list) or len(v) != length or not all(_is_binary64(x) for x in v):
        raise ExactInputError('%s must be %d finite binary64 values' % (what, length))


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
def exact_optimizer_gradient(Y, X, x, q):
    """Exact gradient of gaussian_profile_nll(x; profile_beta=false) for K = 1.

    f = 1/2 [N log 2pi + N + N log(Q/N) + n log a], a = 1 + L'L,
    A^-1 = I - L L'/a, Q = sum_s r_s' A^-1 r_s, N = n p.
    d/dbeta_j = -(N/Q) sum_s x_sj' A^-1 r_s ;  d/dL_i = -(N/Q) (A^-1 R2 A^-1 L)_i + n L_i / a.
    Returns a list of Fractions (exact).
    """
    p, n = _check_inputs(Y, X, q)
    _check_vector(x, q + p, 'optimizer point')
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
def packed_enclosure(Y, X, theta, q):
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
    p, n = _check_inputs(Y, X, q)
    _check_vector(theta, q + 1 + p, 'packed parameters')
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
# Computed radii
# ---------------------------------------------------------------------------
def computed_radii(Y, X, q, theta, x, hessian_raw, packed_gradient, optimizer_gradient):
    """Radii from the record's own inputs, replacing declared ones (prototype).

    hessian_radius / packed_gradient_radius: rigorous upper bounds on
    max |recorded - true| (entrywise / infinity norm) at theta.
    optimizer_gradient_radius: the exact error max |recorded - true| at x.
    exact_optimizer_max_abs_gradient: max |true gradient| at x, exact; a certifier
    could test it directly against 1e-5 with no radius at all.
    """
    k = q + 1 + len(Y)
    if (not isinstance(hessian_raw, list) or len(hessian_raw) != k or
            any(not isinstance(r, list) for r in hessian_raw)):
        raise ExactInputError('recorded Hessian has the wrong size')
    for row in hessian_raw:
        _check_vector(row, k, 'recorded Hessian row')
    _check_vector(packed_gradient, k, 'recorded packed gradient')
    _check_vector(optimizer_gradient, k - 1, 'recorded optimizer gradient')
    g_enc, H_enc = packed_enclosure(Y, X, theta, q)
    g_opt = exact_optimizer_gradient(Y, X, x, q)
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
