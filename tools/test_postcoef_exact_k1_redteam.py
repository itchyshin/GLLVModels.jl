"""Red-team tests for core070_postcoef_exact_k1 (wave 14, scratch).

Convention: ``test_ATTACK_*`` assert the SAFE behaviour; a FAILURE means the attack
succeeded against the prototype. ``test_DEFENSE_*`` pin behaviour that held.
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
from test_postcoef_exact_k1 import make_case, float_recorded

mp.mp.dps = 400


def _og(Y, X, x, q):
    return [float(v) for v in ex.exact_optimizer_gradient(Y, X, x, q)]


def _full(Y, X, q, th, x):
    g, H = float_recorded(Y, X, q, th)
    return ex.computed_radii(Y, X, q, th, x, H, g, _og(Y, X, x, q))


class Forgery(unittest.TestCase):
    def test_ATTACK_unlinked_theta_and_x(self):
        """theta and x are separate arguments; nothing ties the packed point to the
        optimizer point. A mismatched x (lambda_1 moved by 0.3) is accepted with tiny radii."""
        Y, X, q, th, x = make_case(p=4, n=12, q=2, seed=3)
        x2 = list(x)
        x2[q] += 0.3
        g, H = float_recorded(Y, X, q, th)
        r = ex.computed_radii(Y, X, q, th, x2, H, g, _og(Y, X, x2, q))
        self.assertGreater(float(r['optimizer_gradient_radius']), 1e-6,
                           'accepted an optimizer point that is not the packed point')

    def test_ATTACK_sign_flipped_Y_q0(self):
        """With q=0, Y -> -Y leaves every radius and the exact gradient unchanged, so a
        sign-flipped data set certifies identically. Needs byte-level authentication."""
        Y, X, q, th, x = make_case(p=4, n=12, q=0, seed=4)
        base = _full(Y, X, q, th, x)
        flipped = _full([[-v for v in r] for r in Y], None, 0, th, x)
        self.assertNotEqual(base['hessian_radius'], flipped['hessian_radius'],
                            'sign-flipped Y is indistinguishable to the certifier')

    def test_ATTACK_column_permutation_invariance(self):
        """Permuting the n observations leaves the exact enclosures bit-identical, so the
        module cannot detect a reordered Y (benign for the likelihood, fatal for identity)."""
        Y, X, q, th, x = make_case(p=4, n=12, q=0, seed=4)
        perm = [[r[i] for i in reversed(range(12))] for r in Y]
        g1, H1 = ex.packed_enclosure(Y, None, th, 0)
        g2, H2 = ex.packed_enclosure(perm, None, th, 0)
        same = all((a.lo, a.hi) == (b.lo, b.hi) for a, b in zip(g1, g2))
        self.assertFalse(same, 'reordered Y is indistinguishable to the certifier')

    def test_ATTACK_forged_Y_with_zero_gradient(self):
        """Forge Y' with sample second moment R2 = c*A so the recorded x is an exact
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
        """_check_inputs iterates (sees floats); _Moments indexes (sees a Fraction)."""
        Y, X, q, th, x = make_case(p=4, n=12, q=0, seed=4)

        class Row(list):
            def __getitem__(self, i):
                v = list.__getitem__(self, i)
                return Fraction(v) + Fraction(1, 3) if isinstance(i, int) and i == 0 else v
        with self.assertRaises(ex.ExactInputError):
            ex.exact_optimizer_gradient([Row(r) for r in Y], None, x, 0)

    def test_ATTACK_toctou_recorded_hessian(self):
        """Recorded H passes the finite-float check by iteration but is read back as the
        exact true value by indexing, yielding radius 0 for a wrong recorded matrix."""
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
        try:
            r = ex.computed_radii(Y, X, q, th, x, rows, g, _og(Y, X, x, q))
        except Exception:
            return  # refused: fine
        self.assertGreater(float(r['hessian_radius']), 1.0, 'wrong recorded Hessian certified with ~0 radius')


class Boundary(unittest.TestCase):
    def test_ATTACK_exp_range_exceeds_binary64(self):
        """EXP_ARG_LIMIT=1400 claims 'exp would overflow binary64 anyway' but exp(2l) overflows
        at 2l > 709.78; l = 400 (s = e^800) is accepted although Julia's exp(2l) is Inf there."""
        Y, _, _, _, _ = make_case(p=3, n=6, q=0, seed=1)
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(Y, None, [400.0, 0.5, 0.5, 0.5], 0)

    def test_ATTACK_exp_range_underflow(self):
        Y, _, _, _, _ = make_case(p=3, n=6, q=0, seed=1)
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(Y, None, [-400.0, 0.5, 0.5, 0.5], 0)

    def test_DEFENSE_exp_limit_edge(self):
        Y, _, _, _, _ = make_case(p=2, n=4, q=0, seed=1)
        ex.packed_enclosure(Y, None, [700.0, 0.5, 0.5], 0)
        with self.assertRaises(ex.ExactInputError):
            ex.packed_enclosure(Y, None, [math.nextafter(700.0, 1e9), 0.5, 0.5], 0)

    def test_DEFENSE_rounding_fuzz(self):
        rng = random.Random(5)
        for _ in range(3000):
            v = Fraction(rng.randint(-10 ** rng.randint(1, 120), 10 ** rng.randint(1, 120)),
                         rng.randint(1, 10 ** rng.randint(1, 120)))
            self.assertTrue(ex._floor_dyadic(v) <= v <= ex._ceil_dyadic(v))

    def test_DEFENSE_exp_enclosure_contains_mpmath(self):
        rng = random.Random(6)
        ts = [Fraction(0), Fraction(1, 2), Fraction(-1, 2), Fraction(1), Fraction(1400), Fraction(-1400)]
        ts += [Fraction(rng.uniform(-1400, 1400)) for _ in range(60)]
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
        """p=16, n=150 should be bounded by a deterministic work meter; the prototype has none
        (no budget argument), so cost is attacker-chosen via p and n."""
        import inspect
        self.assertTrue(any('budget' in k or 'work' in k for k in
                            inspect.signature(ex.packed_enclosure).parameters),
                        'no deterministic work meter on packed_enclosure')


if __name__ == '__main__':
    unittest.main()
