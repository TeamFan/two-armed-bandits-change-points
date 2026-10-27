# Two-Armed Bandits with Change Points and Noisy Observations

A two-armed stochastic bandit with two known change points, studied in theory and
in simulation:

* **(a)** a Hoeffding bound for means of sub-Gaussian samples, with proof;
* **(b)** a UCB rule for the first phase with a high-probability regret bound
  `R(τ₁) = O(log τ₁ / Δ)`;
* **(c)** error bounds for a sliding-window estimator whose window mixes samples
  from up to three phases (only arm 1 changes at τ₁, both arms may change at τ₂),
  and whether one window size can serve both change points;
* **(d)** confidence bounds when every observation carries Laplace noise with a
  per-round cost ε_t = c / b_t, and how to allocate a fixed budget.

The report is [`report/main.pdf`](report/main.pdf): the written solution
(Sections 1–5) followed by a numerical study (Section 6) that checks every bound
and turns the discussion of (c)(iii) and (d) into measured trade-offs.

| Folder | Content |
|---|---|
| [`src/cpbandits.py`](src/cpbandits.py) | Vectorized simulator: environment, phase-1 UCB, sliding-window variants, Laplace noise schedules and radii |
| [`src/run_experiments.py`](src/run_experiments.py) | One stage per part of the exercise; writes `results/` |
| [`results/`](results) | CSV results and figures used by the report |
| [`report/`](report) | LaTeX source and compiled PDF |
| [`tests/`](tests) | Unit tests (window statistics against a brute-force recomputation, radii, budget schedules) |

## Running

```bash
pip install -r requirements.txt
python -m unittest discover -s tests       # < 1 s
python src/run_experiments.py              # all stages, about 3 minutes
python src/run_experiments.py --stage window
cd report && tectonic main.tex             # or latexmk -pdf main.tex
```

All experiments use Bernoulli rewards, T = 30,000 rounds, τ₁ = 10,000,
τ₂ = 20,000 and 200 seeded replications per configuration.

## Policies after τ₁

| Policy | Rule |
|---|---|
| SW-UCB | window mean over the last `w` pulls of each arm + `sqrt(2 L_T / m)`; no reset (the mechanism of part (c)) |
| bias-aware SW-UCB | same index + the fraction of the arm's window collected before *its own* last change (times the jump bound 1); arm 2 gets no bonus at τ₁ |
| arm-aware reset | restart the arms whose mean changes (arm 1 at τ₁, both at τ₂) |
| reset both arms | restart both arms at both change points |

## Main findings

* **Bounds.** The Hoeffding bound, the phase-1 regret bound (0 violations in
  3,200 runs; largest regret 47 % of the bound), the window error bounds across
  both change points (0 violations) and the Laplace radius hold in every run.
  The Laplace radius needs its linear term: with the variance term only,
  miscoverage reaches 3.1 δ at δ = 10⁻³.
* **The window alone does not refresh stale arms.** When the arm that was rarely
  pulled in phase 2 becomes best after τ₂, plain SW-UCB keeps its stale estimate
  and pays almost the full gap for the rest of the horizon (S1: about 2,900 in
  phase 3). With full windows the two confidence radii are equal and cancel, so
  the index does not explore. A bonus equal to the stale fraction of the window
  fixes this (S1: 14).
* **One window or two.** With comparable post-change gaps one small window is
  best in both phases. With gaps 0.1 and 0.6 the best windows are 200 and 50,
  and a common window costs 43–55 in one of the phases. The sufficient window
  `32 L_T / g²` of the analysis is 10–100 times larger than the best one.
* **A common shift of both arms is harmless** to a policy that does not explore:
  plain SW-UCB has zero regret in that phase, while the bias-aware policy pays
  for unnecessary exploration.
* **Noise moves the best window up**: 50, 50, 100 and 200 pulls for Laplace
  scales 0, 0.25, 0.5 and 1. Spending about 25 % of a fixed budget in the
  2,000 rounds after each change (13 % under uniform spending) lowers regret by
  2–2.5 %; stronger concentration helps only long windows.

| Phase-wise best regret | S1 phase 2 | S1 phase 3 | S2 phase 2 | S2 phase 3 | S3 phase 3 |
|---|---|---|---|---|---|
| SW-UCB | 13.9 | 586.2 | 27.4 | 8.5 | 0.0 |
| bias-aware SW-UCB | 14.8 | 14.0 | 31.9 | 18.1 | 13.3 |
| arm-aware reset | 98.8 | 99.2 | 154.2 | 60.0 | 96.7 |

(Best window chosen per phase; mean over 200 runs. See Table 3 of the report.)
