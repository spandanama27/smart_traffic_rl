"""
Traffic Performance Metrics Module.
Calculates academic-grade KPIs for traffic engineering evaluation:
- Average Vehicle Waiting Time (seconds)
- Total Throughput (vehicles cleared)
- Average & Maximum Queue Length
- Congestion Index (proportion of traffic in stationary state)
- Directional Delay Fairness (Jain's Fairness Index)
"""
from typing import Dict, List
import numpy as np


def compute_summary_metrics(
    step_records: List[Dict[str, any]],
    total_passed: int,
    completed_wait_times: List[float],
) -> Dict[str, float]:
    """
    Computes aggregated performance indicators from simulation telemetry.
    """
    if not step_records:
        return {
            "avg_waiting_time": 0.0,
            "total_throughput": 0,
            "avg_queue_length": 0.0,
            "max_queue_length": 0,
            "avg_congestion_index": 0.0,
            "fairness_index": 1.0,
        }

    # Queue lengths over time
    total_queues = [r["total_queue"] for r in step_records]
    congestions = [r["congestion_index"] for r in step_records]
    avg_waits = [r["avg_waiting_time"] for r in step_records]

    avg_queue = float(np.mean(total_queues)) if total_queues else 0.0
    max_queue = int(np.max(total_queues)) if total_queues else 0
    avg_congestion = float(np.mean(congestions)) if congestions else 0.0
    final_avg_wait = float(avg_waits[-1]) if avg_waits else 0.0

    # Calculate Directional Fairness using Jain's Fairness Index:
    # J(x) = (sum(x_i))^2 / (n * sum(x_i^2))
    # where x_i is average delay per approach
    last_wait_dict = step_records[-1].get("wait_times", {})
    dir_waits = list(last_wait_dict.values())
    if dir_waits and sum(dir_waits) > 0:
        sum_w = sum(dir_waits)
        sum_sq = sum(w ** 2 for w in dir_waits)
        fairness = float((sum_w ** 2) / (len(dir_waits) * sum_sq + 1e-6))
    else:
        fairness = 1.0

    return {
        "avg_waiting_time": round(final_avg_wait, 2),
        "total_throughput": int(total_passed),
        "avg_queue_length": round(avg_queue, 2),
        "max_queue_length": int(max_queue),
        "avg_congestion_index": round(avg_congestion, 3),
        "fairness_index": round(fairness, 3),
    }


def compute_percentage_improvement(baseline_val: float, test_val: float, lower_is_better: bool = True) -> float:
    """
    Computes percentage improvement of test_val relative to baseline_val.
    """
    if abs(baseline_val) < 1e-6:
        return 0.0

    if lower_is_better:
        return round(((baseline_val - test_val) / baseline_val) * 100.0, 2)
    else:
        return round(((test_val - baseline_val) / baseline_val) * 100.0, 2)
