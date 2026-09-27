"""Per-run metrics accumulation."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from lb_rl.env.environment import StepInfo


@dataclass
class MetricsTracker:
    time_bound: Optional[int] = None
    response_times: List[int] = field(default_factory=list)
    drops: int = 0
    arrivals: int = 0
    queue_history: List[np.ndarray] = field(default_factory=list)

    def record(self, info: StepInfo, state: np.ndarray) -> None:
        if info.arrived:
            self.arrivals += 1
        if info.dropped:
            self.drops += 1
        self.response_times.extend(info.response_times)
        self.queue_history.append(np.asarray(state))

    def summary(self) -> dict:
        rt = np.array(self.response_times, dtype=np.float64)
        queues = np.array(self.queue_history) if self.queue_history else np.zeros((0, 0))
        avg_util_per_server = queues.mean(axis=0) if queues.size else np.zeros(0)
        completions = len(self.response_times)
        within_bound = (
            int((rt <= self.time_bound).sum()) if (self.time_bound is not None and rt.size) else None
        )
        return {
            "avg_response_time": float(rt.mean()) if rt.size else float("nan"),
            "drops": self.drops,
            "drop_rate": (self.drops / self.arrivals) if self.arrivals else 0.0,
            "arrivals": self.arrivals,
            "completions": completions,
            "load_balance_variance": float(np.var(avg_util_per_server)) if avg_util_per_server.size else float("nan"),
            "completion_rate_within_bound": (within_bound / completions) if (within_bound is not None and completions) else None,
        }


def convergence_step(rewards: List[float], window: int = 200, tol: float = 0.05) -> Optional[int]:
    """First training step at which the trailing moving-average reward is
    within `tol` (relative) of the final moving-average reward. None if the
    run was too short to compute a moving average."""
    if len(rewards) < window:
        return None
    rewards = np.asarray(rewards, dtype=np.float64)
    kernel = np.ones(window) / window
    moving_avg = np.convolve(rewards, kernel, mode="valid")
    final = moving_avg[-1]
    if final == 0:
        return None
    within = np.abs((moving_avg - final) / abs(final)) < tol
    idx = np.flatnonzero(within)
    return int(idx[0]) + window - 1 if idx.size else None
