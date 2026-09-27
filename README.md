# RL Load Balancing under Partial Observability

Tabular reinforcement learning vs. classical and optimal dispatchers for load
balancing when server monitoring is **delayed and/or noisy** (a POMDP),
extending Chawla (2024) [arXiv:2409.04896], which assumes perfect, real-time
server state.

The project adds an **exact per-server Bayesian belief filter** (the dispatcher's
own dispatch log makes the POMDP belief factorise into one small distribution per
server) and a **confidence-gated hybrid** dispatcher (learned Q-policy when the
belief is confident, Shortest Expected Delay on the belief otherwise).

## Setup

```bash
pip install numpy pandas matplotlib pyyaml pytest
python -m pytest -q          # 27 tests
```

## System model (short)

- Discrete time slots; at most one job arrives per slot with probability λ
  (steady, bursty two-state Markov-modulated, or cyclical traffic).
- 5 heterogeneous servers, service probabilities μ = (0.9, 0.7, 0.5, 0.4, 0.3),
  FIFO queues of capacity 4; a job sent to a full queue is dropped.
- Reward = −(total jobs waiting) − 5 × (job dropped).
- State = the 5 queue lengths → 5^5 = 3,125 states.
- Monitoring modes: `full`, `noise_low/high` (±1 with p = 0.1/0.3),
  `delay_2/5/10` (k-slot-old snapshot), `noise_delay` (k = 5, p = 0.15).

## Configurations

| ID  | Dispatcher | What it observes |
|-----|------------|------------------|
| C1  | Round Robin | ignored |
| C2  | Least Connections | raw observation |
| C3  | Shortest Expected Delay (SED) | raw observation |
| C4  | Value Iteration (exact optimum; steady traffic, full observability) | true state |
| C5  | Q-learning | true state |
| C6  | Q-learning (Chawla-style under realistic monitoring) | raw observation |
| C8  | Q-learning | dispatch-corrected estimate |
| C11 | Q-learning | **belief filter** estimate |
| C12 | SED | **belief filter** estimate |
| C13 | **Hybrid**: Q-learning if belief is confident, else SED on the belief | belief filter |

(IDs C7, C9 and C10 were used by earlier prototype variants — a traffic-forecast
feature and two corrector controls — that were removed because they did not
contribute to the results.)

## Experiments

```bash
python scripts/run_grid.py --config configs/full_grid.yaml                                  # normal load
python scripts/run_grid.py --config configs/mu_mismatch.yaml --out-dir results_mu_mismatch  # wrong server speeds
python scripts/run_grid.py --config configs/heavy_load.yaml  --out-dir results_heavy        # heavy load
python scripts/make_plots.py  --results-dir results
python scripts/make_report.py --results-dir results
```

`configs/quick.yaml` is a fast smoke run. Every method is evaluated on the same
random seeds (seed + 10,000). Each results folder contains `raw/results.csv`,
`summary.csv`, `plots/` and `REPORT.md`.

- **mu_mismatch**: the simulator uses `mu_true = (0.3, 0.7, 0.5, 0.4, 0.9)` while
  SED, Value Iteration and the estimators keep the configured `mu`.
- **heavy_load**: every μ halved (≈ 57% utilisation).

## Key results (5 seeds, mean response time in slots over delayed monitoring modes)

| Experiment | Chawla-style Q-learning (C6) | SED on raw data (C3) | Best new method |
|---|---|---|---|
| Normal load | 1.77 | 1.47 | **1.29** — hybrid (C13) |
| Wrong server speeds | 1.81 | 3.93 | **1.52** — Q + belief (C11) |
| Heavy load | 7.35 | 5.31 | **3.83** — SED + belief (C12) |

With perfect monitoring, SED, Value Iteration and Q-learning all reach ≈ 1.14
(normal load). The belief filter relies on the configured server speeds, so SED +
belief and the hybrid degrade when those speeds are wrong.

## Repository layout

```
lb_rl/env/            simulator: servers, traffic, environment
lb_rl/observation/    monitoring models, dispatch corrector, belief filter
lb_rl/agents/         Round Robin, LC, SED, Value/Policy Iteration, Q-learning, hybrid
lb_rl/models/         exact transition model for Value/Policy Iteration
lb_rl/eval/           rollout loop and metrics
scripts/              experiment runners, plots, report, milestone checks
configs/              experiment configurations
tests/                unit tests
docs/                 project proposal and workflow documents
```

## Scope notes

- Q-learning hyper-parameters (α, γ) are tuned once per traffic pattern with full
  observability and reused for all partial-observability configurations.
- Single dispatcher, discrete-time Bernoulli arrivals, identical job sizes.
