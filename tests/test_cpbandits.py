import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cpbandits import Environment, NoiseSchedule, PolicyConfig, laplace_radius, simulate  # noqa: E402

ENV = Environment(3000, 1000, 2000, [[0.7, 0.4], [0.1, 0.4], [0.75, 0.45]])


class TestEnvironment(unittest.TestCase):
    def test_arm2_constant_at_tau1(self):
        with self.assertRaises(AssertionError):
            Environment(100, 10, 20, [[0.5, 0.4], [0.2, 0.3], [0.1, 0.1]])

    def test_phase_boundaries(self):
        self.assertEqual([ENV.phase(t) for t in (1, 1000, 1001, 2000, 2001, 3000)], [0, 0, 1, 1, 2, 2])


class TestRadius(unittest.TestCase):
    def test_noise_free_limit(self):
        n = np.array([1, 10, 100.0]); L = 5.0
        r = laplace_radius(n, n, 0.0, L, time_varying=False)
        np.testing.assert_allclose(r, np.sqrt(2 * L / n))

    def test_constant_scale_matches_eq18(self):
        n, b, L = 50.0, 1.5, math.log(2 / 0.05)
        r = laplace_radius(n, n * (1 + 2 * b * b), b, L, time_varying=False)
        self.assertAlmostEqual(r, math.sqrt(2 * (1 + 2 * b * b) * L / n) + b * L / n)


class TestSchedule(unittest.TestCase):
    def test_budget_is_respected(self):
        blocks = [(1, 100), (101, 150), (151, 300)]
        sch = NoiseSchedule.blocks(300, blocks, [40.0, 30.0, 30.0], c=1.0)
        self.assertAlmostEqual(sch.total_cost(), 100.0)
        self.assertEqual(sch.K, 3)
        self.assertAlmostEqual(sch.b()[0], 100 / 40)


class TestSimulator(unittest.TestCase):
    def test_deterministic(self):
        a = simulate(ENV, PolicyConfig("sw_bias", window=100), R=5, seed=3)
        b = simulate(ENV, PolicyConfig("sw_bias", window=100), R=5, seed=3)
        np.testing.assert_array_equal(a.regret, b.regret)

    def test_window_statistics_match_brute_force(self):
        """Ring-buffer mean and stale fractions equal a direct recomputation from the pull history."""
        env = Environment(400, 100, 250, [[0.7, 0.4], [0.1, 0.4], [0.75, 0.45]])
        w = 37
        res = simulate(env, PolicyConfig("sw", window=w), R=1, seed=11, record_estimates=True,
                       forced_policy="round_robin")
        # rebuild with an explicit loop using the same random stream
        rng = np.random.default_rng(11)
        hist = {0: [], 1: []}
        for t in range(1, env.T + 1):
            a = (t - 1) % 2
            k = env.phase(t)
            x = float(rng.random(1)[0] < env.means[k, a])
            hist[a].append((x, k))
            for i in (0, 1):
                win = hist[i][-w:]
                if not win:
                    continue
                est = np.mean([v for v, _ in win])
                self.assertAlmostEqual(float(res.extras["est"][t - 1, 0, i]), est, places=5)
                a1 = sum(1 for _, ph in win if ph == 0) / len(win)
                self.assertAlmostEqual(float(res.extras["alpha"][t - 1, 0, i, 0]), a1, places=6)

    def test_plain_window_never_revisits_stale_arm(self):
        """After tau2 arm 1 becomes best, but the plain sliding window keeps a stale estimate."""
        res = simulate(ENV, PolicyConfig("sw", window=200), R=20, seed=5)
        self.assertGreater(res.phase_regret(ENV)[:, 2].mean(), 0.25 * 1000)
        res_b = simulate(ENV, PolicyConfig("sw_bias", window=200), R=20, seed=5)
        self.assertLess(res_b.phase_regret(ENV)[:, 2].mean(), 0.5 * res.phase_regret(ENV)[:, 2].mean())

    def test_reset_keeps_arm2_at_tau1(self):
        # with an arm-aware reset arm 2 is not re-initialized at tau1, so it is never forced
        env = Environment(300, 100, 200, [[0.9, 0.1], [0.8, 0.1], [0.8, 0.1]])
        r1 = simulate(env, PolicyConfig("reset", window=300), R=10, seed=2)
        r2 = simulate(env, PolicyConfig("reset_all", window=300), R=10, seed=2)
        self.assertLessEqual(r1.phase_regret(env)[:, 1].mean(), r2.phase_regret(env)[:, 1].mean())


if __name__ == "__main__":
    unittest.main()
