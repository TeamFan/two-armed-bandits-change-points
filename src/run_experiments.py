"""Experiments for the report, one stage per part of the exercise.

    python src/run_experiments.py                 # all stages (about 3 minutes)
    python src/run_experiments.py --stage window  # one stage
    python src/run_experiments.py --quick         # fewer replications (smoke test)

Stages: hoeffding (a), ucb (b), estimation (c)(i)-(ii), window (c)(iii),
noise_coverage, noise_window, budget (d).  Results go to results/*.csv and
results/figures/*.{pdf,png}.  Every stage is seeded and deterministic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cpbandits import Environment, NoiseSchedule, PolicyConfig, simulate, predicted_window  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
BASE_SEED = 20260929

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"sw": "#c0392b", "sw_bias": "#1f6fb2", "reset": "#2e8b57", "reset_all": "#8e44ad"}
LABELS = {"sw": "SW-UCB (no reset)", "sw_bias": "bias-aware SW-UCB", "reset": "arm-aware reset",
          "reset_all": "reset both arms"}

T, TAU1, TAU2 = 30_000, 10_000, 20_000
SCENARIOS = {
    # name: (means per phase, short description)
    "S1": ([[0.70, 0.40], [0.10, 0.40], [0.75, 0.45]], "comparable gaps (0.3, 0.3, 0.3); ranking flips twice"),
    "S2": ([[0.70, 0.40], [0.30, 0.40], [0.90, 0.30]], "different gaps (0.3, 0.1, 0.6); ranking flips twice"),
    "S3": ([[0.60, 0.40], [0.10, 0.40], [0.50, 0.80]], "common upward shift at tau2; ranking kept"),
}
WINDOWS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]


def seed(tag: str) -> int:
    return BASE_SEED + zlib.crc32(tag.encode()) % 1_000_000


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def env_of(name: str) -> Environment:
    return Environment(T, TAU1, TAU2, SCENARIOS[name][0])


# --------------------------------------------------------------------------- (a)
def stage_hoeffding(R):
    n_rep = 200_000 if R >= 100 else 20_000
    rng = np.random.default_rng(seed("hoeffding"))
    rows = []
    ts = np.linspace(0.0, 1.5, 31)
    for dist in ("gaussian", "rademacher", "bernoulli(1/2)"):
        for n in (5, 20, 100):
            if dist == "gaussian":
                mean = rng.standard_normal((n_rep, n)).mean(1)
            elif dist == "rademacher":
                mean = (2 * rng.integers(0, 2, (n_rep, n)) - 1).mean(1)
            else:  # rewards in [0, 1] are 1/2-sub-Gaussian, hence 1-sub-Gaussian
                mean = rng.integers(0, 2, (n_rep, n)).mean(1) - 0.5
            dev = np.abs(mean)
            for t in ts:
                rows.append(dict(dist=dist, n=n, t=t, empirical=float((dev >= t).mean()),
                                 bound=min(1.0, 2 * math.exp(-n * t * t / 2))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "hoeffding.csv"), index=False)
    assert (df.empirical <= df.bound + 3 * np.sqrt(df.bound * (1 - df.bound) / n_rep) + 1e-12).all()
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
    for ax, n in zip(axes, (5, 20, 100)):
        d = df[df.n == n]
        for dist, st in zip(("gaussian", "rademacher", "bernoulli(1/2)"), ("-", "--", ":")):
            dd = d[d.dist == dist]
            ax.semilogy(dd.t, np.maximum(dd.empirical, 1 / n_rep), st, label=dist)
        b = d[d.dist == "gaussian"]
        ax.semilogy(b.t, b.bound, "k", lw=1.8, label=r"$2e^{-nt^2/2}$")
        ax.set_title(f"n = {n}"); ax.set_xlabel("t"); ax.set_ylim(1 / n_rep, 1.5)
    axes[0].set_ylabel(r"$P(|\hat\mu_n-\mu|\geq t)$"); axes[0].legend(fontsize=7)
    save(fig, "hoeffding")
    return dict(mc_samples=n_rep)


# --------------------------------------------------------------------------- (b)
def stage_ucb(R):
    rows = []
    for H in (1_000, 3_000, 10_000, 30_000):
        for gap in (0.1, 0.2, 0.3, 0.5):
            env = Environment(H + 2, H, H + 1, [[0.5 + gap / 2, 0.5 - gap / 2]] * 3)
            res = simulate(env, PolicyConfig("sw", window=1), R=R, seed=seed(f"ucb{H}{gap}"))
            reg = res.phase_regret(env)[:, 0]
            nj = res.phase1_pulls[:, 1]
            L_H = math.log(4 * H ** 3)
            bound = min(H * gap, gap + 8 * L_H / gap)
            rows.append(dict(H=H, gap=gap, regret_mean=reg.mean(), regret_q95=np.quantile(reg, 0.95),
                             regret_max=reg.max(), bound=bound, Nj_max=nj.max(),
                             Nj_bound=1 + 8 * L_H / gap ** 2,
                             violations=int((reg > bound + 1e-9).sum()), runs=R))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "ucb_phase1.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for gap, c in zip((0.1, 0.2, 0.3, 0.5), plt.cm.viridis(np.linspace(0, 0.85, 4))):
        d = df[df.gap == gap]
        axes[0].plot(d.H, d.regret_mean, "o-", color=c, label=fr"$\Delta={gap}$")
        axes[0].plot(d.H, d.bound, "--", color=c, lw=1)
        axes[1].plot(np.log(d.H), d.regret_mean * gap, "o-", color=c, label=fr"$\Delta={gap}$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel(r"$\tau_1$")
    axes[0].set_ylabel("phase-1 regret"); axes[0].set_title("mean regret (solid) vs bound (4) (dashed)")
    axes[1].set_xlabel(r"$\log\tau_1$"); axes[1].set_ylabel(r"$\Delta\cdot R(1{:}\tau_1)$")
    axes[1].set_title(r"$\Delta R$ grows linearly in $\log\tau_1$"); axes[1].legend(fontsize=7)
    save(fig, "ucb_phase1")
    return dict(total_violations=int(df.violations.sum()))


# --------------------------------------------------------------------------- (c)(i)-(ii)
def stage_estimation(R):
    env = env_of("S1"); mu = env.means
    J1 = abs(mu[1, 0] - mu[0, 0]); J12 = abs(mu[2, 0] - mu[1, 0]); J22 = abs(mu[2, 1] - mu[1, 1])
    rows, curves = [], {}
    for w in (200, 1000, 3000):
        res = simulate(env, PolicyConfig("sw", window=w, delta=0.05), R=R, seed=seed(f"est{w}"),
                       record_estimates=True, forced_policy="round_robin")
        ex = res.extras
        est, m, bias, alpha = ex["est"], ex["m"].astype(float), ex["bias"], ex["alpha"]
        L_T = math.log(4 * T / 0.05)
        t_next = np.arange(2, T + 2)  # estimate after round t is used at round t+1
        k = np.where(t_next <= TAU1, 0, np.where(t_next <= TAU2, 1, 2))
        mu_now = mu[k]  # (T, 2)
        err = np.abs(est - mu_now[:, None, :])
        r = np.sqrt(2 * L_T / np.maximum(m, 1))
        # bias bound of (8) and (12): arm 1 alpha11 J1 (+ alpha12 J12 after tau2); arm 2 alpha22 J22 after tau2
        bb = np.zeros_like(err)
        a11, a12 = alpha[..., 0, 0], alpha[..., 0, 1]
        a22 = alpha[..., 1, 1]
        bb[:, :, 0] = np.where(k[:, None] >= 1, a11 * J1, 0) + np.where(k[:, None] == 2, a12 * J12, 0)
        bb[:, :, 1] = np.where(k[:, None] == 2, a22 * J22, 0)
        valid = (m > 0) & (t_next[:, None, None] > TAU1)
        viol_runs = ((err > r + bb + 1e-9) & valid).any(axis=(0, 2))
        excess = (np.abs(bias) - bb)[valid]  # |B_i(t)| <= bias bound of (8)/(12): must be <= 0
        rows.append(dict(window=w, runs=R, delta=0.05, runs_with_violation=int(viol_runs.sum()),
                         max_err_over_bound=float(np.nanmax(np.where(valid, err / (r + bb), np.nan))),
                         max_bias_excess=float(np.max(excess)) if excess.size else 0.0))
        sl = slice(TAU1 - 500, T)
        curves[w] = dict(t=t_next[sl], err=np.nanmean(err[sl], axis=1), r=np.nanmean(r[sl], axis=1),
                         bias=np.nanmean(np.abs(bias[sl]), axis=1), bb=np.nanmean(bb[sl], axis=1))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "estimation_bounds.csv"), index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.6), sharex=True)
    for col, w in enumerate((1000, 3000)):
        c = curves[w]
        for arm in (0, 1):
            ax = axes[arm, col]
            ax.plot(c["t"], c["err"][:, arm], color="k", lw=1, label=r"mean $|\hat\mu^{SW}_i-\mu_i(t)|$")
            ax.plot(c["t"], c["bias"][:, arm], color="#c0392b", lw=1.2, label=r"mixing bias $|B_i(t)|$")
            ax.plot(c["t"], c["r"][:, arm] + c["bb"][:, arm], color="#1f6fb2", lw=1.2, label="bound (8)/(12)")
            ax.plot(c["t"], c["r"][:, arm], ":", color="#1f6fb2", lw=1, label=r"sampling term $r(m)$")
            for tau in (TAU1, TAU2):
                ax.axvline(tau, color="grey", lw=0.8, ls="--")
            ax.set_title(f"arm {arm + 1}, w = {w}")
            ax.set_ylim(0, 1.0)
    axes[1, 0].set_xlabel("round t"); axes[1, 1].set_xlabel("round t"); axes[0, 0].legend(fontsize=7)
    save(fig, "estimation_bounds")
    return dict(rows=rows)


# --------------------------------------------------------------------------- (c)(iii)
def stage_window(R):
    rows = []
    for sc in SCENARIOS:
        env = env_of(sc)
        for variant in ("sw", "sw_bias"):
            for w in WINDOWS:
                res = simulate(env, PolicyConfig(variant, window=w), R=R, seed=seed(f"win{sc}{variant}{w}"))
                pr = res.phase_regret(env)
                rows.append(dict(scenario=sc, variant=variant, window=w, **phase_stats(pr)))
        for variant in ("reset", "reset_all"):
            res = simulate(env, PolicyConfig(variant, window=T), R=R, seed=seed(f"win{sc}{variant}"))
            rows.append(dict(scenario=sc, variant=variant, window=T, **phase_stats(res.phase_regret(env))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "window_sweep.csv"), index=False)
    L_T = math.log(4 * T * T)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for j, sc in enumerate(SCENARIOS):
        g = env_of(sc).gaps()
        for i, ph in enumerate((1, 2)):
            ax = axes[i, j]
            for variant in ("sw", "sw_bias"):
                d = df[(df.scenario == sc) & (df.variant == variant)]
                ax.errorbar(d.window, np.maximum(d[f"p{ph}_mean"], 1.0), yerr=d[f"p{ph}_se"] * 1.96, fmt="o-",
                            ms=3, color=COLORS[variant], label=LABELS[variant], capsize=2)
            for variant in ("reset", "reset_all"):
                v = df[(df.scenario == sc) & (df.variant == variant)][f"p{ph}_mean"].iloc[0]
                ax.axhline(v, color=COLORS[variant], ls="--", lw=1, label=LABELS[variant])
            wstar = predicted_window(g[ph], L_T)
            if wstar <= 2 * T:
                ax.axvline(wstar, color="grey", ls=":", lw=1)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.7, 4000)
            ax.set_title(f"{sc}, phase {ph + 1} (gap {g[ph]:.2f})", fontsize=9)
        axes[1, j].set_xlabel("window w (pulls per arm)")
    axes[0, 0].set_ylabel("regret in phase 2"); axes[1, 0].set_ylabel("regret in phase 3 (values < 1 shown at 1)")
    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    save(fig, "window_sweep")
    return {}


def phase_stats(pr):
    n = pr.shape[0]
    out = {}
    for ph in range(3):
        out[f"p{ph}_mean"] = float(pr[:, ph].mean())
        out[f"p{ph}_se"] = float(pr[:, ph].std(ddof=1) / math.sqrt(n))
    out["total_mean"] = float(pr.sum(1).mean())
    return out


# --------------------------------------------------------------------------- (d)
def stage_noise_coverage(R):
    """Miscoverage of three radii for means of n Bernoulli(1/2) rewards plus Laplace(0, b) noise."""
    n_rep = 400_000 if R >= 100 else 40_000
    rng = np.random.default_rng(seed("noisecov"))
    rows = []
    for b in (0.25, 1.0, 2.0):
        for n in (1, 2, 5, 10, 20, 50, 100):
            dev = np.empty(n_rep)
            for c0 in range(0, n_rep, 50_000):
                c1 = min(n_rep, c0 + 50_000)
                x = rng.integers(0, 2, (c1 - c0, n)) - 0.5 + rng.laplace(0, b, (c1 - c0, n))
                dev[c0:c1] = np.abs(x.mean(1))
            for delta in (0.05, 0.001):
                ell = math.log(2 / delta)
                r_naive = math.sqrt(2 * ell / n)                      # (1): ignores the noise
                r_var = math.sqrt(2 * (1 + 2 * b * b) * ell / n)      # variance inflation only
                r_full = r_var + b * ell / n                          # (18)
                rows.append(dict(b=b, n=n, delta=delta, reps=n_rep,
                                 miss_naive=float((dev > r_naive).mean()),
                                 miss_variance_only=float((dev > r_var).mean()),
                                 miss_full=float((dev > r_full).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "noise_coverage.csv"), index=False)
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.6), sharex=True)
    for r_, delta in enumerate((0.05, 0.001)):
        for ax, b in zip(axes[r_], (0.25, 1.0, 2.0)):
            d = df[(df.b == b) & (df.delta == delta)]
            floor = 0.5 / n_rep / delta
            ax.plot(d.n, np.maximum(d.miss_naive / delta, floor), "o-", color="#c0392b", ms=3, label="noise-free radius")
            ax.plot(d.n, np.maximum(d.miss_variance_only / delta, floor), "s-", color="#e67e22", ms=3,
                    label="variance term only")
            ax.plot(d.n, np.maximum(d.miss_full / delta, floor), "^-", color="#1f6fb2", ms=3, label="radius (18)")
            ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(floor * 0.8, 20)
            ax.set_title(rf"$b={b}$, $\delta={delta}$", fontsize=9)
        axes[r_, 0].set_ylabel(r"miscoverage / $\delta$")
    for ax in axes[1]:
        ax.set_xlabel("n")
    axes[0, 0].legend(fontsize=7)
    save(fig, "noise_coverage")
    return dict(mc_samples=n_rep)


def stage_noise_window(R):
    env = env_of("S1")
    rows = []
    for b in (0.0, 0.25, 0.5, 1.0):
        noise = NoiseSchedule.constant(T, b) if b > 0 else None
        for w in WINDOWS:
            res = simulate(env, PolicyConfig("sw_bias", window=w), R=R, noise=noise, seed=seed(f"nw{b}{w}"))
            rows.append(dict(b=b, window=w, aware=True, **phase_stats(res.phase_regret(env))))
        if b > 0:
            for w in (500, 2000):
                res = simulate(env, PolicyConfig("sw_bias", window=w, noise_aware=False), R=R, noise=noise,
                               seed=seed(f"nwu{b}{w}"))
                rows.append(dict(b=b, window=w, aware=False, **phase_stats(res.phase_regret(env))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "noise_window.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharex=True)
    L_T = math.log(4 * T * T)
    for b, c in zip((0.0, 0.25, 0.5, 1.0), plt.cm.plasma(np.linspace(0, 0.8, 4))):
        d = df[(df.b == b) & df.aware]
        for ax, ph in zip(axes, (1, 2)):
            ax.errorbar(d.window, d[f"p{ph}_mean"], yerr=1.96 * d[f"p{ph}_se"], fmt="o-", ms=3, color=c,
                        capsize=2, label=f"b = {b}")
            best = d.window.iloc[int(np.argmin(d[f"p{ph}_mean"].values))]
            ax.plot(best, d[f"p{ph}_mean"].min(), "*", color=c, ms=10)
    for ax, ph in zip(axes, (1, 2)):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("window w")
        ax.set_title(f"phase {ph + 1}: regret of bias-aware SW-UCB")
    axes[0].set_ylabel("regret"); axes[0].legend(fontsize=7)
    save(fig, "noise_window")
    return {}


def stage_budget(R):
    env = env_of("S1")
    h = 2000
    E = 2.0 * T  # c = 1: uniform spending gives b_t = c T / E = 0.5
    blocks = [(1, TAU1), (TAU1 + 1, TAU1 + h), (TAU1 + h + 1, TAU2), (TAU2 + 1, TAU2 + h), (TAU2 + h + 1, T)]
    lengths = np.array([e - s + 1 for s, e in blocks], float)
    rec = np.array([False, True, False, True, False])
    rho0 = lengths[rec].sum() / T
    rows = []
    for rho in (rho0, 0.25, 0.4, 0.55, 0.7):
        budgets = np.where(rec, rho * E * lengths / lengths[rec].sum(), (1 - rho) * E * lengths / lengths[~rec].sum())
        noise = NoiseSchedule.blocks(T, blocks, list(budgets))
        assert abs(noise.total_cost() - E) < 1e-6 * E
        b_rec, b_stab = noise.b()[TAU1], noise.b()[0]
        for w in (300, 1000, 3000):
            res = simulate(env, PolicyConfig("sw_bias", window=w), R=R, noise=noise, seed=seed(f"bud{rho:.3f}{w}"))
            rows.append(dict(rho=rho, uniform=bool(abs(rho - rho0) < 1e-12), b_recovery=b_rec, b_stable=b_stab,
                             window=w, **phase_stats(res.phase_regret(env))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "budget_allocation.csv"), index=False)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for w, c in zip((300, 1000, 3000), ("#1f6fb2", "#2e8b57", "#c0392b")):
        d = df[df.window == w]
        tot = (d.p1_mean + d.p2_mean).values
        se = np.sqrt(d.p1_se ** 2 + d.p2_se ** 2).values
        ref = tot[d.uniform.values][0]
        ax.errorbar(d.rho, tot / ref, yerr=1.96 * se / ref, fmt="o-", ms=3, color=c, capsize=2,
                    label=f"w = {w} (uniform: {ref:.0f})")
    ax.axvline(rho0, color="grey", ls=":", lw=1)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel(r"share $\rho$ of the budget spent in the two recovery blocks")
    ax.set_ylabel(r"regret after $\tau_1$ / uniform"); ax.legend(fontsize=7)
    save(fig, "budget_allocation")
    return dict(E=E, h=h, rho_uniform=rho0)


STAGES = {"hoeffding": stage_hoeffding, "ucb": stage_ucb, "estimation": stage_estimation,
          "window": stage_window, "noise_coverage": stage_noise_coverage, "noise_window": stage_noise_window,
          "budget": stage_budget}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    ap.add_argument("--quick", action="store_true", help="few replications (smoke test)")
    args = ap.parse_args()
    R = 20 if args.quick else 200
    os.makedirs(FIG, exist_ok=True)
    meta_path = os.path.join(RES, "run_info.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name in (STAGES if args.stage == "all" else [args.stage]):
        t0 = time.time()
        info = STAGES[name](R)
        meta[name] = dict(replications=R, seconds=round(time.time() - t0, 1), **(info or {}))
        print(f"{name:15s} done in {time.time() - t0:6.1f} s")
    meta["config"] = dict(T=T, tau1=TAU1, tau2=TAU2, base_seed=BASE_SEED, numpy=np.__version__,
                          scenarios={k: v[0] for k, v in SCENARIOS.items()})
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
