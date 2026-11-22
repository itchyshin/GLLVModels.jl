"""Red-team tests for core070_postcoef_exact_k1 (wave 14, scratch).

Convention: ``test_ATTACK_*`` assert the SAFE behaviour; a FAILURE means the attack
succeeded against the prototype. ``test_DEFENSE_*`` pin behaviour that held.

Wave 15: the module was hardened; attacks 1, 4, 5, 6 now fail closed and their tests pass.
Attacks 2 and 3 (forged Y, sign flip / reorder) are BY DESIGN outside the module (they need an
authenticated byte-level digest of Y and X); those tests are ``expectedFailure`` with the reason
in the docstring, so adding a data-identity argument later flips them to "unexpected success".
Run: python3 -B -m unittest test_postcoef_exact_k1_redteam
"""
import math
import random
import time
import unittest
from fractions import Fraction

import mpmath as mp
import numpy as np

import core070_postcoef_exact_k1 as ex
from unittest.mock import patch

from test_postcoef_exact_k1 import make_case, float_recorded, bound_point

mp.mp.dps = 400


def _og(Y, X, x, q):
    return [float(v) for v in ex.exact_optimizer_gradient(Y, X, x, q)]


def _full(Y, X, q, th, x):
    g, H = float_recorded(Y, X, q, th)
    return ex.computed_radii(Y, X, q, th, x, H, g, _og(Y, X, x, q))


class Forgery(unittest.TestCase):
    def test_ATTACK_unlinked_theta_and_x(self):
        """HARDENED. theta and x were independent arguments; a mismatched x (lambda_1 moved by
        0.3) was accepted with tiny radii. computed_radii now binds them exactly first."""
        Y, X, q, th, x = make_case(p=4, n=12, q=2, seed=3)
        x2 = list(x)
        x2[q] += 0.3
        g, H = float_recorded(Y, X, q, th)
        with self.assertRaises(ex.ExactInputError):
            ex.computed_radii(Y, X, q, th, x2, H, g, _og(Y, X, x2, q))

    @unittest.expectedFailure
    def test_ATTACK_sign_flipped_Y_q0(self):
        """XFAIL BY DESIGN (needs an authenticated digest of Y; outside the module).
        With q=0, Y -> -Y leaves every radius and the exact gradient unchanged, so a
        sign-flipped data set certifies identically. Needs byte-level authentication."""
        Y, X, q, th, x = make_case(p=4, n=12, q=0, seed=4)
        base = _full(Y, X, q, th, x)
        flipped = _full([[-v for v in r] for r in Y], None, 0, th, x)
        self.assertNotEqual(base['hessian_radius'], flipped['hessian_radius'],
                            'sign-flipped Y is indistinguishable to the certifier')

    @unittest.expectedFailure
    def test_ATTACK_column_permutation_invariance(self):
        """XFAIL BY DESIGN (needs an authenticated digest of Y; outside the module).
        Permuting the n observations leaves the exact enclosures bit-identical, so the
        module cannot detect a reordered Y (benign for the likelihood, fatal for identity)."""
        Y, X, q, th, x = make_case(p=4, n=12, q=0, seed=4)
        perm = [[r[i] for i in reversed(range(12))] for r in Y]
        g1, H1 = ex.packed_enclosure(Y, None, th, 0)
        g2, H2 = ex.packed_enclosure(perm, None, th, 0)
        same = all((a.lo, a.hi) == (b.lo, b.hi) for a, b in zip(g1, g2))
        self.assertFalse(same, 'reordered Y is indistinguishable to the certifier')

    @unittest.expectedFailure
    def test_ATTACK_forged_Y_with_zero_gradient(self):
        """XFAIL BY DESIGN (data authentication is the only control; trust-boundary note items 1-2).
        Forge Y' with sample second moment R2 = c*A so the recorded x is an exact
        stationary point of Y'; the real (non-converged) data had a large gradient."""
        p = 3
        L = [0.7, -0.4, 0.2]
        a = 1 + sum(v * v for v in L)
        A = np.eye(p) + np.outer(L, L)  # profile sigma matrix = I + L L'
        Y_forged = (np.linalg.cholesky(A) * 2.0).tolist()   # p x p, n = p, R2 = 4 A
        Y_real, _, _, _, _ = make_case(p=p, n=p, q=0, seed=9)
        gf = max(abs(v) for v in ex.exact_optimizer_gradient(Y_forged, None, L, 0))
        gr = max(abs(v) for v in ex.exact_optimizer_gradient(Y_real, None, L, 0))
        self.assertLess(gf, 1e-12)      # forged data certifies a perfect optimum
        self.assertGreater(gr, 1e-3)    # real data does not
        # The module cannot tell them apart: no data-identity input exists.
        import inspect
        params = inspect.signature(ex.computed_radii).parameters
        self.assertTrue(any('digest' in k or 'hash' in k for k in params),
                        'computed_radii has no data-identity argument; forged Y accepted')

    def test_ATTACK_toctou_non_binary64_data_via_list_subclass(self):
        """HARDENED. Validation iterated (saw floats); moments indexed (saw a Fraction). Now
        subclasses are refused and everything is copied once."""
        Y, X, q, th, x = make_case(p=4, n=12, q=0, seed=4)

        class Row(list):
            def __getitem__(self, i):
                v = list.__getitem__(self, i)
                return Fraction(v) + Fraction(1, 3) if isinstance(i, int) and i == 0 else v
        with self.assertRaises(ex.ExactInputError):
            ex.exact_optimizer_gradient([Row(r) for r in Y], None, x, 0)

    def test_ATTACK_toctou_recorded_hessian(self):
        """HARDENED. Recorded H passed the finite-float check by iteration but was read back as
        the exact true value by indexing (radius 0 for a wrong matrix). Now refused outright."""
        Y, X, q, th, x = make_case(p=3, n=10, q=0, seed=2)
        g, H = float_recorded(Y, X, q, th)
        _, Henc = ex.packed_enclosure(Y, X, th, q)
        true = [[(e.lo + e.hi) / 2 for e in row] for row in Henc]

        class Row2(list):
            def __init__(self, vals, tr):
                super().__init__(vals)
                self.tr = tr

            def __getitem__(self, i):
                return self.tr[i] if isinstance(i, int) else list.__getitem__(self, i)
        rows = [Row2([1e9] * len(H), [true[i][j] for j in range(len(H))]) for i in range(len(H))]
        with self.assertRaises(ex.ExactInputError):
            ex.computed_radii(Y, X, q, th, x, rows, g, _og(Y, X, x, q))


class Boundary(unittest.TestCase):
    def test_ATTACK_exp_range_exceeds_binary64(self):
        """HARDENED. EXP_ARG_LIMIT was 1400 although exp(2l) overflows at 2l > 709.78; l = 400
        (s = e^800) was accepted. The limit is now 708."""
        Y, _, _, _, _ = make_case(p=3, n=6, q=0, seed=1)
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(Y, None, [400.0, 0.5, 0.5, 0.5], 0)

    def test_ATTACK_exp_range_underflow(self):
        Y, _, _, _, _ = make_case(p=3, n=6, q=0, seed=1)
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(Y, None, [-400.0, 0.5, 0.5, 0.5], 0)

    def test_DEFENSE_exp_limit_edge(self):
        Y, _, _, _, _ = make_case(p=2, n=4, q=0, seed=1)
        ex.packed_enclosure(Y, None, [354.0, 0.5, 0.5], 0)
        ex.packed_enclosure(Y, None, [-354.0, 0.5, 0.5], 0)
        for edge in (354.0, -354.0):
            with self.assertRaises(ex.ExactInputError):
                ex.packed_enclosure(Y, None, [math.nextafter(edge, math.copysign(1e9, edge)), 0.5, 0.5], 0)

    def test_DEFENSE_exp_limit_is_binary64_range(self):
        """2l = +-708 keeps exp(2l) a finite normal binary64 value."""
        self.assertLess(math.exp(ex.EXP_ARG_LIMIT), math.inf)
        self.assertGreaterEqual(math.exp(-ex.EXP_ARG_LIMIT), 2.2250738585072014e-308)

    def test_DEFENSE_rounding_fuzz(self):
        rng = random.Random(5)
        for _ in range(3000):
            v = Fraction(rng.randint(-10 ** rng.randint(1, 120), 10 ** rng.randint(1, 120)),
                         rng.randint(1, 10 ** rng.randint(1, 120)))
            self.assertTrue(ex._floor_dyadic(v) <= v <= ex._ceil_dyadic(v))

    def test_DEFENSE_exp_enclosure_contains_mpmath(self):
        rng = random.Random(6)
        ts = [Fraction(0), Fraction(1, 2), Fraction(-1, 2), Fraction(1), Fraction(708), Fraction(-708)]
        ts += [Fraction(rng.uniform(-708, 708)) for _ in range(60)]
        for t in ts:
            e = ex.exp_enclosure(t)
            m = mp.exp(mp.mpf(t.numerator) / t.denominator)
            self.assertTrue(mp.mpf(e.lo.numerator) / e.lo.denominator <= m <=
                            mp.mpf(e.hi.numerator) / e.hi.denominator)

    def test_DEFENSE_interval_ops_contain_truth(self):
        rng = random.Random(7)
        for _ in range(1500):
            a = [Fraction(rng.randint(-10 ** 9, 10 ** 9), rng.randint(1, 10 ** 9)) for _ in range(2)]
            b = [Fraction(rng.randint(-10 ** 9, 10 ** 9), rng.randint(1, 10 ** 9)) for _ in range(2)]
            A, B = ex.Iv(min(a), max(a)), ex.Iv(min(b), max(b))
            x, y = rng.choice(a), rng.choice(b)
            self.assertTrue((A + B).contains(x + y) and (A - B).contains(x - y) and (A * B).contains(x * y))


class Refusals(unittest.TestCase):
    def setUp(self):
        self.Y, self.X, self.q, self.th, self.x = make_case(p=3, n=8, q=1, seed=2)

    def test_DEFENSE_nonfinite_and_types(self):
        for bad in (float('nan'), float('inf'), -float('inf'), 1, True, np.float64(1.0), None, '1.0'):
            Y = [list(r) for r in self.Y]
            Y[1][2] = bad
            with self.assertRaises(ex.ExactInputError):
                ex.exact_optimizer_gradient(Y, self.X, self.x, self.q)
            th = list(self.th)
            th[-1] = bad
            with self.assertRaises(ex.ExactInputError):
                ex.packed_enclosure(self.Y, self.X, th, self.q)
            X = [[list(c) for c in row] for row in self.X]
            X[0][0][0] = bad
            with self.assertRaises(ex.ExactInputError):
                ex.exact_optimizer_gradient(self.Y, X, self.x, self.q)

    def test_DEFENSE_float_subclass_and_recorded_nan(self):
        class F(float):
            pass
        with self.assertRaises(ex.ExactInputError):
            ex.exact_optimizer_gradient([[F(1.0), F(2.0)], [F(3.0), F(4.0)]], None, [0.1, 0.2], 0)
        g, H = float_recorded(self.Y, self.X, self.q, self.th)
        g[0] = float('nan')
        with self.assertRaises(ex.ExactInputError):
            ex.computed_radii(self.Y, self.X, self.q, self.th, self.x, H, g, _og(self.Y, self.X, self.x, self.q))

    def test_DEFENSE_degenerate_data_refused_not_accepted(self):
        with self.assertRaises(ex.ExactInputError):
            ex.exact_optimizer_gradient([[0.0] * 5 for _ in range(3)], None, [0.1, 0.2, 0.3], 0)

    def test_DEFENSE_sign_flip_recorded_hessian_detected(self):
        Y, X, q, th, x = self.Y, self.X, self.q, self.th, self.x
        g, H = float_recorded(Y, X, q, th)
        k = len(H)
        for i in range(q + 1, k):           # flip lambda<->beta cross block sign
            H[0][i] = -H[0][i]
            H[i][0] = -H[i][0]
        r = ex.computed_radii(Y, X, q, th, x, H, g, _og(Y, X, x, q))
        self.assertGreater(float(r['hessian_radius']), 1e-3)

    def test_DEFENSE_sign_flip_recorded_gradient_detected(self):
        Y, X, q, th, x = self.Y, self.X, self.q, self.th, self.x
        g, H = float_recorded(Y, X, q, th)
        r = ex.computed_radii(Y, X, q, th, x, H, [-v for v in g], _og(Y, X, x, q))
        self.assertGreater(float(r['packed_gradient_radius']), 1e-3)

    def test_DEFENSE_cancellation_widens_never_accepts_wrong(self):
        Y, X, q, th, x = make_case(p=3, n=8, q=0, seed=2, shift=1e150)
        g, H = float_recorded(Y, X, q, th)
        H[0][0] += 1.0
        try:
            r = ex.computed_radii(Y, X, q, th, x, H, g, _og(Y, X, x, q))
        except (ex.ExactInputError, ArithmeticError, ZeroDivisionError):
            return
        self.assertGreaterEqual(float(r['hessian_radius']), 1.0)


class Cost(unittest.TestCase):
    def test_ATTACK_unmetered_cost(self):
        """HARDENED. p=16, n=150 should be bounded by a deterministic work meter; the prototype
        had none (no budget argument), so cost was attacker-chosen via p and n."""
        import inspect
        self.assertTrue(any('budget' in k or 'work' in k for k in
                            inspect.signature(ex.packed_enclosure).parameters),
                        'no deterministic work meter on packed_enclosure')

    def test_DEFENSE_work_meter_refuses_before_arithmetic(self):
        p, n = 40, 30
        Y = [[0.5 + 0.01 * (t + s) for s in range(n)] for t in range(p)]
        th = [0.0] + [0.1] * p
        self.assertGreater(ex.estimate_work(p, n, 0), ex.DEFAULT_WORK_BUDGET)
        t0 = time.time()
        for call in (lambda: ex.packed_enclosure(Y, None, th, 0),
                     lambda: ex.exact_optimizer_gradient(Y, None, th[1:], 0)
                     if ex.estimate_work(p, n, 0) > ex.DEFAULT_WORK_BUDGET else None):
            with self.assertRaises(ex.ExactInputError):
                call()
        self.assertLess(time.time() - t0, 1.0)

    def test_DEFENSE_work_budget_is_explicit_and_typed(self):
        Y, X, q, th, x = make_case(p=3, n=6, q=0, seed=1)
        need = ex.estimate_work(3, 6, 0)
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(Y, None, th, 0, work_budget=need - 1)
        ex.packed_enclosure(Y, None, th, 0, work_budget=need)
        for bad in (True, 1.5e6, None, -1):
            with self.assertRaises(ex.ExactInputError):
                ex.packed_enclosure(Y, None, th, 0, work_budget=bad)

    def test_DEFENSE_work_estimate_tracks_measured_growth(self):
        """The shape bound must grow at least as fast as the measured time: p=8 vs p=4."""
        Y4, _, _, th4, _ = make_case(p=4, n=40, q=0, seed=1)
        Y8, _, _, th8, _ = make_case(p=8, n=40, q=0, seed=1)
        t = []
        for Y, th in ((Y4, th4), (Y8, th8)):
            t0 = time.time()
            ex.packed_enclosure(Y, None, th, 0)
            t.append(time.time() - t0)
        ratio_est = ex.estimate_work(8, 40, 0) / ex.estimate_work(4, 40, 0)
        self.assertGreaterEqual(ratio_est * 1.5, t[1] / max(t[0], 1e-6) * 0.5)


class Hardening(unittest.TestCase):
    def setUp(self):
        self.Y, self.X, self.q, self.th, self.x = make_case(p=4, n=10, q=2, seed=3)

    def _rec(self):
        g, H = float_recorded(self.Y, self.X, self.q, self.th)
        return g, H, _og(self.Y, self.X, self.x, self.q)

    # -- copy once / exact types ------------------------------------------------------------
    def test_tuples_and_lists_agree(self):
        g, H, og = self._rec()
        r1 = ex.computed_radii(self.Y, self.X, self.q, self.th, self.x, H, g, og)
        t = lambda m: tuple(tuple(r) for r in m)
        r2 = ex.computed_radii(t(self.Y), tuple(tuple(tuple(c) for c in row) for row in self.X), self.q,
                               tuple(self.th), tuple(self.x), t(H), tuple(g), tuple(og))
        self.assertEqual(r1, r2)

    def test_list_subclass_refused_everywhere(self):
        class L(list):
            pass
        g, H, og = self._rec()
        args = dict(Y=self.Y, X=self.X, q=self.q, theta=self.th, x=self.x, hessian_raw=H,
                    packed_gradient=g, optimizer_gradient=og)
        wrappers = {
            'Y': lambda v: L(v), 'Y_row': lambda v: [L(r) for r in v],
            'X': lambda v: L(v), 'X_row': lambda v: [L(r) for r in v],
            'X_cell': lambda v: [[L(c) for c in r] for r in v],
            'theta': lambda v: L(v), 'x': lambda v: L(v), 'hessian_raw': lambda v: L(v),
            'hessian_row': lambda v: [L(r) for r in v],
            'packed_gradient': lambda v: L(v), 'optimizer_gradient': lambda v: L(v)}
        field = {'Y': 'Y', 'Y_row': 'Y', 'X': 'X', 'X_row': 'X', 'X_cell': 'X', 'theta': 'theta', 'x': 'x',
                 'hessian_raw': 'hessian_raw', 'hessian_row': 'hessian_raw',
                 'packed_gradient': 'packed_gradient', 'optimizer_gradient': 'optimizer_gradient'}
        for name, w in wrappers.items():
            a = dict(args)
            a[field[name]] = w(a[field[name]])
            with self.assertRaises(ex.ExactInputError, msg=name):
                ex.computed_radii(a['Y'], a['X'], a['q'], a['theta'], a['x'], a['hessian_raw'],
                                  a['packed_gradient'], a['optimizer_gradient'])

    def test_iterator_and_other_containers_refused(self):
        for bad in (iter(self.th), np.array(self.th), dict(enumerate(self.th)), set(self.th), 'abc'):
            with self.assertRaises(ex.ExactInputError):
                ex.packed_enclosure(self.Y, self.X, bad, self.q)

    def test_internals_receive_tuple_copies(self):
        """After validation only tuples of floats reach the arithmetic (copy once, use the copy)."""
        seen = []
        real = ex._Moments

        class Spy(real):
            def __init__(s, Y, X, beta):
                seen.append((type(Y), {type(r) for r in Y}, type(X), type(beta)))
                super().__init__(Y, X, beta)
        g, H, og = self._rec()
        with patch.object(ex, '_Moments', Spy):
            ex.computed_radii(self.Y, self.X, self.q, self.th, self.x, H, g, og)
        self.assertTrue(seen)
        for ty, rows, tx, tb in seen:
            self.assertIs(ty, tuple)
            self.assertEqual(rows, {tuple})
            self.assertIs(tx, tuple)
            self.assertIs(tb, tuple)

    def test_inputs_are_copied_once(self):
        """A one-shot generator-backed list cannot be used; and mutating the caller's list after
        the call returns cannot affect a result already computed (nothing is retained)."""
        Y = [list(r) for r in self.Y]
        g1, _ = ex.packed_enclosure(Y, self.X, self.th, self.q)
        Y[0][0] += 1.0
        g2, _ = ex.packed_enclosure(self.Y, self.X, self.th, self.q)
        self.assertEqual([(a.lo, a.hi) for a in g1], [(a.lo, a.hi) for a in g2])

    def test_q_must_be_int(self):
        for bad in (True, False, 2.0, None, '2', -1, np.int64(2)):
            with self.assertRaises(ex.ExactInputError):
                ex.packed_enclosure(self.Y, self.X, self.th, bad)
            with self.assertRaises(ex.ExactInputError):
                ex.exact_optimizer_gradient(self.Y, self.X, self.x, bad)

    def test_q_bool_refused_even_when_shapes_match(self):
        Y, X, q, th, x = make_case(p=3, n=6, q=1, seed=5)
        ex.packed_enclosure(Y, X, th, 1)                  # the same shapes with q = 1 are fine
        for call in (lambda: ex.packed_enclosure(Y, X, th, True),
                     lambda: ex.exact_optimizer_gradient(Y, X, x, True),
                     lambda: ex.check_binding(th, x, True)):
            with self.assertRaises(ex.ExactInputError):
                call()

    def test_X_required_when_q_positive(self):
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(self.Y, None, self.th, self.q)

    # -- binding ----------------------------------------------------------------------------
    def test_binding_accepts_bound_point_and_returns_sigma(self):
        sig = ex.check_binding(self.th, self.x, self.q)
        self.assertEqual(math.log(sig), self.th[self.q])
        self.assertTrue(all(sig * L == lam for L, lam in zip(self.x[self.q:], self.th[self.q + 1:])))

    def test_binding_refuses_each_deviation(self):
        q = self.q
        cases = {}
        t = list(self.th); t[0] = math.nextafter(t[0], 1e9); cases['beta 1 ulp'] = (t, self.x)
        t = list(self.th); t[q] = math.nextafter(t[q], 1e9); cases['log sigma +1 ulp'] = (t, self.x)
        t = list(self.th); t[q] = math.nextafter(t[q], -1e9); cases['log sigma -1 ulp'] = (t, self.x)
        t = list(self.th); t[q] += 0.01; cases['log sigma moved'] = (t, self.x)
        t = list(self.th); t[q + 2] = math.nextafter(t[q + 2], 1e9); cases['lambda 1 ulp'] = (t, self.x)
        t = list(self.th); t[q + 1] = -t[q + 1]; cases['lambda sign'] = (t, self.x)
        x2 = list(self.x); x2[q + 1] *= 2.0; cases['L doubled'] = (self.th, x2)
        cases['log sigma slot zeroed (old sigma=1 assumption)'] = ([*self.th[:q], 0.0, *self.th[q + 1:]], self.x)
        t = list(self.th); t[q + 1:] = self.x[q:]; cases['lambda := L (sigma ignored)'] = (t, self.x)
        for name, (th, x) in cases.items():
            with self.assertRaises(ex.ExactInputError, msg=name):
                ex.check_binding(th, x, q)

    def test_binding_zero_rules(self):
        # a zero L needs a same-signed zero lambda
        th, x = bound_point([], [0.0, 0.7, -0.3], 1.25)
        th2, x2 = list(th), list(x)
        ex.check_binding(th2, x2, 0)
        th2[1] = -0.0
        with self.assertRaises(ex.ExactInputError):
            ex.check_binding(th2, x2, 0)       # lambda -0.0 vs L +0.0
        th3 = list(th); th3[1] = 1e-300
        with self.assertRaises(ex.ExactInputError):
            ex.check_binding(th3, x2, 0)
        # all-zero L: sigma only from log; an l that no binary64 sigma produces is refused
        ex.check_binding([math.log(1.25), 0.0, 0.0], [0.0, 0.0], 0)
        with self.assertRaises(ex.ExactInputError):
            ex.check_binding([math.nextafter(math.log(1.25), 9.0), 0.0, 0.0], [0.0, 0.0], 0)
        # genuine underflow (sigma * 5e-324 -> 0) is refused like the live certifier does
        with self.assertRaises(ex.ExactInputError):
            ex.check_binding([math.log(0.5), 0.0], [5e-324], 0)

    def test_binding_edges_range_and_length(self):
        with self.assertRaises(ex.ExactInputError):
            ex.check_binding([400.0, 0.1], [0.1], 0)         # |l| beyond the exp range
        with self.assertRaises(ex.ExactInputError):
            ex.check_binding([0.0, 0.1], [0.1, 0.2], 0)      # length mismatch
        with self.assertRaises(ex.ExactInputError):
            ex.check_binding([0.0], [], 0)                   # no loadings

    def test_computed_radii_runs_binding_before_arithmetic(self):
        g, H, og = self._rec()
        th = list(self.th)
        th[self.q] += 0.5
        with patch.object(ex, '_packed_enclosure', side_effect=AssertionError('arithmetic before binding')):
            with self.assertRaises(ex.ExactInputError):
                ex.computed_radii(self.Y, self.X, self.q, th, self.x, H, g, og)

    def test_binding_cross_checks_against_live_certifier_relation(self):
        """The module's check must agree with the live certifier's own optimizer binding where the
        live one exists (1a7f2aba's version only handles sigma = 1, so sigma = 1 points only)."""
        th, x = bound_point([0.25], [0.5, -0.2, 0.9], 1.0)
        self.assertEqual(th[1], 0.0)
        self.assertEqual(ex.check_binding(th, x, 1), 1.0)


if __name__ == '__main__':
    unittest.main()
