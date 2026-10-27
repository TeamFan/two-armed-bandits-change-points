"""Two-armed Bernoulli bandits with two known change points and Laplace observation noise.

The simulator runs R independent replications in parallel (vectorized over runs) and
implements the policies analysed in the report:

* phase 1 (t <= tau1): UCB with radius r_H(n) = sqrt(2 L_H / n), L_H = log(4H/delta_H);
* after tau1: a sliding-window estimator over the last ``w`` *pulls* of each arm, with
  index  mu_SW + r(m),  r(m) = sqrt(2 L_T / m),  L_T = log(4T/delta),  in one of the
  variants below.

Variants after tau1
-------------------
``sw``          the mechanism of part (c): same window across tau1 and tau2, no reset.
``sw_bias``     sliding window plus the stale-fraction bonus  J * alpha_i(t), where
                alpha_i(t) is the fraction of arm i's window collected before the last
                change of arm i's mean.  Arm 2 is unaffected at tau1, so its bonus is zero
                in phase 2 (the asymmetry discussed in (c)(iii)).  J bounds the jump; the
                default J = 1 is the reward range, i.e. the jump-independent bias bound.
``reset``       discard the window of every arm whose mean changes (arm 1 at tau1, both
                arms at tau2), keep the other arm.
``reset_all``   discard both windows at both change points (ignores the asymmetry).

With Laplace noise, observation y_t = x_t + Z_t, Z_t ~ Laplace(0, b_t), and the radius
is the sub-gamma bound of part (d): (19) for a constant scale and (20) for a schedule
with K distinct scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

PHASE_CHANGED = ((True, False), (True, True))  # which arms change at tau1 and tau2


@dataclass
class Environment:
    """Bernoulli means per phase: means[k, i] for phase k in {0, 1, 2}, arm i in {0, 1}."""

    T: int
    tau1: int
    tau2: int
    means: np.ndarray

    def __post_init__(self):
        self.means = np.asarray(self.means, dtype=float)
        assert self.means.shape == (3, 2)
        assert 0 < self.tau1 < self.tau2 < self.T
        assert self.means[0, 1] == self.means[1, 1], "only arm 1 changes at tau1"
        assert np.all((self.means >= 0) & (self.means <= 1))

    def phase(self, t: int) -> int:
        return 0 if t <= self.tau1 else (1 if t <= self.tau2 else 2)

    def gaps(self) -> np.ndarray:
        return np.abs(self.means[:, 0] - self.means[:, 1])


@dataclass
class NoiseSchedule:
    """Piecewise-constant Laplace scales: b_t = levels[level_of_round[t-1]] (b = 0: no noise)."""

    levels: np.ndarray
    level_of_round: np.ndarray  # int array, length T

    @staticmethod
    def none(T: int) -> "NoiseSchedule":
        return NoiseSchedule(np.array([0.0]), np.zeros(T, dtype=np.int8))

    @staticmethod
    def constant(T: int, b: float) -> "NoiseSchedule":
        return NoiseSchedule(np.array([float(b)]), np.zeros(T, dtype=np.int8))

    @staticmethod
    def blocks(T: int, blocks: list[tuple[int, int]], budgets: list[float], c: float = 1.0) -> "NoiseSchedule":
        """Equation (24): rounds of block l (1-based [start, end]) get eps_t = E_l / h_l, b_t = c h_l / E_l.

        Blocks with the same scale share a level, so the schedule has K <= len(blocks) levels.
        """
        b_round = np.empty(T)
        covered = np.zeros(T, dtype=bool)
        for (s, e), E_l in zip(blocks, budgets):
            h = e - s + 1
            b_round[s - 1:e] = c * h / E_l
            assert not covered[s - 1:e].any(), "blocks overlap"
            covered[s - 1:e] = True
        assert covered.all(), "blocks must partition 1..T"
        levels, inv = np.unique(np.round(b_round, 12), return_inverse=True)
        return NoiseSchedule(levels, inv.astype(np.int8))

    @property
    def K(self) -> int:
        return int(len(self.levels))

    @property
    def is_noisy(self) -> bool:
        return bool(np.any(self.levels > 0))

    def b(self) -> np.ndarray:
        return self.levels[self.level_of_round]

    def total_cost(self, c: float = 1.0) -> float:
        b = self.b()
        return float(np.sum(c / b)) if np.all(b > 0) else math.inf


@dataclass
class PolicyConfig:
    variant: str = "sw"            # sw | sw_bias | reset | reset_all
    window: int = 1000             # w (number of pulls per arm)
    delta: float | None = None     # confidence level after tau1 (default 1/T)
    jump_bound: float = 1.0        # J in the stale-fraction bonus (sw_bias)
    radius_scale: float = 1.0      # multiplies every confidence radius (1 = theory)
    noise_aware: bool = True       # use the Laplace radius (19)/(20) when there is noise

    def __post_init__(self):
        assert self.variant in ("sw", "sw_bias", "reset", "reset_all")
        assert self.window >= 1


@dataclass
class RunResult:
    regret: np.ndarray            # (R, T) per-round pseudo-regret
    pulls: np.ndarray             # (R, 2) total pulls per arm
    phase1_pulls: np.ndarray      # (R, 2) pulls per arm in phase 1
    extras: dict = field(default_factory=dict)

    def phase_regret(self, env: Environment) -> np.ndarray:
        """(R, 3) cumulative pseudo-regret in each phase."""
        c = self.regret
        return np.stack([c[:, :env.tau1].sum(1), c[:, env.tau1:env.tau2].sum(1), c[:, env.tau2:].sum(1)], 1)


def laplace_radius(n, V, bmax, L, time_varying: bool):
    """Confidence radius of a mean of n noisy observations.

    Constant scale, eq. (19):        (sqrt(2 V L) + b L) / n  with V = n (1 + 2 b^2).
    K-level schedule, eq. (20):      (2 sqrt(V L) + b_max L) / n.
    Without noise both reduce to sqrt(2 L / n) (resp. 2 sqrt(L / n)); the noise-free
    case always uses sqrt(2 L / n), the radius of parts (b) and (c).
    """
    n = np.maximum(n, 1)
    if time_varying:
        return (2.0 * np.sqrt(V * L) + bmax * L) / n
    return (np.sqrt(2.0 * V * L) + bmax * L) / n


def simulate(env: Environment, cfg: PolicyConfig, R: int, seed: int,
             noise: NoiseSchedule | None = None, record_estimates: bool = False,
             forced_policy: str | None = None) -> RunResult:
    """Run R replications.  ``forced_policy='round_robin'`` alternates arms (used to test the
    estimation bounds of part (c) independently of the selection rule)."""
    T, t1, t2 = env.T, env.tau1, env.tau2
    noise = noise or NoiseSchedule.none(T)
    rng = np.random.default_rng(seed)
    W = int(min(cfg.window, T))
    ar = np.arange(R)
    b_levels = noise.levels
    b_round = noise.b()
    noisy = noise.is_noisy and cfg.noise_aware
    time_varying = noise.K > 1
    delta = cfg.delta if cfg.delta is not None else 1.0 / T
    delta_H = float(t1) ** -2
    L_H = math.log(4 * t1 / delta_H)
    L_T = math.log(4 * T / delta)
    if noisy and time_varying:  # L_* of eq. (20)
        D_T = 1 + math.ceil(math.log2(T))
        L_T = math.log(4 * T * noise.K * D_T / delta)
        L_H = math.log(4 * t1 * noise.K * D_T / delta_H)
    kappa = cfg.radius_scale

    # cumulative (phase-1) statistics
    cum_n = np.zeros((R, 2)); cum_s = np.zeros((R, 2)); cum_V = np.zeros((R, 2)); cum_bmax = np.zeros((R, 2))
    # sliding-window ring buffers over pulls
    buf_y = np.zeros((R, 2, W)); buf_ph = np.zeros((R, 2, W), dtype=np.int8); buf_lv = np.zeros((R, 2, W), dtype=np.int8)
    ptr = np.zeros((R, 2), dtype=np.int64); m = np.zeros((R, 2), dtype=np.int64)
    s = np.zeros((R, 2)); sb2 = np.zeros((R, 2))
    phcnt = np.zeros((R, 2, 3), dtype=np.int64); lvcnt = np.zeros((R, 2, noise.K), dtype=np.int64)

    regret = np.zeros((R, T), dtype=np.float32)
    pulls = np.zeros((R, 2), dtype=np.int64)
    est = np.full((T, R, 2), np.nan, dtype=np.float32) if record_estimates else None
    win_m = np.zeros((T, R, 2), dtype=np.int32) if record_estimates else None
    win_bias = np.zeros((T, R, 2), dtype=np.float32) if record_estimates else None
    win_alpha = np.zeros((T, R, 2, 2), dtype=np.float32) if record_estimates else None
    phase1_pulls = None

    def window_bmax():
        if noise.K == 1:
            return np.full((R, 2), b_levels[0])
        has = lvcnt > 0
        idx = np.where(has.any(2), (noise.K - 1) - np.argmax(has[:, :, ::-1], axis=2), 0)
        return b_levels[idx]

    def reset_arms(arms):
        for i in arms:
            m[:, i] = 0; ptr[:, i] = 0; s[:, i] = 0.0; sb2[:, i] = 0.0
            phcnt[:, i, :] = 0; lvcnt[:, i, :] = 0

    for t in range(1, T + 1):
        k = 0 if t <= t1 else (1 if t <= t2 else 2)
        if t == t1 + 1 and cfg.variant in ("reset", "reset_all"):
            reset_arms([0] if cfg.variant == "reset" else [0, 1])
        if t == t2 + 1 and cfg.variant in ("reset", "reset_all"):
            reset_arms([0, 1])

        # ---------------- index computation ----------------
        if forced_policy == "round_robin":
            a = np.full(R, (t - 1) % 2)
        elif t <= 2:
            a = np.full(R, t - 1)
        else:
            if k == 0:
                n_, mean_ = cum_n, cum_s / np.maximum(cum_n, 1)
                if noisy:
                    rad = laplace_radius(n_, cum_V, cum_bmax, L_H, time_varying)
                else:
                    rad = np.sqrt(2 * L_H / np.maximum(n_, 1))
            else:
                n_, mean_ = m, s / np.maximum(m, 1)
                if noisy:
                    rad = laplace_radius(m, m + 2 * sb2, window_bmax(), L_T, time_varying)
                else:
                    rad = np.sqrt(2 * L_T / np.maximum(m, 1))
            idx = mean_ + kappa * rad
            if k >= 1 and cfg.variant == "sw_bias":
                # fraction of window collected before the last change of each arm's mean
                stale1 = phcnt[:, 0, :k].sum(1)            # arm 1 changes at tau1 and tau2
                stale2 = phcnt[:, 1, :2].sum(1) if k == 2 else 0 * stale1  # arm 2 only at tau2
                alpha = np.stack([stale1, stale2], 1) / np.maximum(m, 1)
                idx = idx + cfg.jump_bound * alpha
            idx = np.where(n_ == 0, np.inf, idx)
            a = np.argmax(idx, axis=1)  # ties -> arm 1 (index 0)

        # ---------------- reward and observation ----------------
        mu = env.means[k]
        x = (rng.random(R) < mu[a]).astype(float)
        lv = int(noise.level_of_round[t - 1]); b_t = b_round[t - 1]
        y = x + (rng.laplace(0.0, b_t, R) if b_t > 0 else 0.0)
        regret[:, t - 1] = mu.max() - mu[a]
        pulls[ar, a] += 1

        # cumulative stats
        cum_n[ar, a] += 1; cum_s[ar, a] += y; cum_V[ar, a] += 1 + 2 * b_t ** 2
        cum_bmax[ar, a] = np.maximum(cum_bmax[ar, a], b_t)

        # ring buffer update (drop oldest if full)
        p = ptr[ar, a]
        full = m[ar, a] == W
        if full.any():
            rf, af, pf = ar[full], a[full], p[full]
            s[rf, af] -= buf_y[rf, af, pf]
            sb2[rf, af] -= b_levels[buf_lv[rf, af, pf]] ** 2
            np.subtract.at(phcnt, (rf, af, buf_ph[rf, af, pf].astype(np.int64)), 1)
            np.subtract.at(lvcnt, (rf, af, buf_lv[rf, af, pf].astype(np.int64)), 1)
        buf_y[ar, a, p] = y; buf_ph[ar, a, p] = k; buf_lv[ar, a, p] = lv
        s[ar, a] += y; sb2[ar, a] += b_t ** 2
        phcnt[ar, a, k] += 1; lvcnt[ar, a, lv] += 1
        ptr[ar, a] = (p + 1) % W
        m[ar, a] = np.minimum(m[ar, a] + 1, W)

        if t == t1:
            phase1_pulls = pulls.copy()
        if record_estimates:
            # window estimate available at the *start* of round t+1
            est[t - 1] = np.where(m > 0, s / np.maximum(m, 1), np.nan)
            win_m[t - 1] = m
            kn = 0 if t + 1 <= t1 else (1 if t + 1 <= t2 else 2)
            mix = (phcnt * env.means.T[None, :, :]).sum(2) / np.maximum(m, 1)  # mean of window means
            win_bias[t - 1] = mix - env.means[kn][None, :]
            win_alpha[t - 1, :, :, 0] = phcnt[:, :, 0] / np.maximum(m, 1)
            win_alpha[t - 1, :, :, 1] = phcnt[:, :, :2].sum(2) / np.maximum(m, 1)

    extras = {}
    if record_estimates:
        extras = dict(est=est, m=win_m, bias=win_bias, alpha=win_alpha, L_T=L_T)
    return RunResult(regret=regret, pulls=pulls, phase1_pulls=phase1_pulls, extras=extras)


def predicted_window(gap: float, L: float, b: float = 0.0) -> float:
    """Sufficient full-window size: (14) without noise, max of the two conditions behind (23) with noise."""
    if b == 0:
        return 32 * L / gap ** 2
    return max(128 * (1 + 2 * b * b) * L / gap ** 2, 8 * b * L / gap)
