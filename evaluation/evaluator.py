"""
Performance Evaluation Engine.
Performs rigorous, repeatable head-to-head simulations comparing:
1. Traditional Fixed-Time Traffic Signal
2. Q-Learning Intelligent Controller
3. Deep Q-Network (DQN) Intelligent Controller

Uses identical stochastic random seeds for fairness so all agents face identical vehicle arrival sequences.
"""
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from utils.config import Config
from environment.traffic_env import TrafficEnvironment
from agents.base_agent import BaseAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.fixed_time_agent import FixedTimeAgent
from evaluation.metrics import compute_summary_metrics, compute_percentage_improvement


class PerformanceEvaluator:
    """Evaluates and benchmarks traffic signal controllers under controlled simulation conditions."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def evaluate_agent(
        self,
        agent: BaseAgent,
        preset_name: str = "Balanced Medium",
        num_steps: int = 300,
        seed: int = 42,
    ) -> Tuple[Dict[str, any], pd.DataFrame]:
        """
        Runs a single evaluation session for an agent under a fixed seed.

        Returns:
            summary_metrics (dict): Aggregated KPIs.
            telemetry_df (pd.DataFrame): Second-by-second simulation trace.
        """
        # Set deterministic seeds for identical vehicle arrivals
        random.seed(seed)
        np.random.seed(seed)

        rates = self.config.TRAFFIC_PRESETS.get(preset_name, self.config.DEFAULT_ARRIVAL_RATES)
        env = TrafficEnvironment(config=self.config, arrival_rates=rates, max_steps=num_steps)

        obs, info = env.reset(seed=seed)
        agent.reset()

        step_records: List[Dict[str, any]] = []
        is_q_learning = isinstance(agent, QLearningAgent)

        for step in range(num_steps):
            state = env.discretize_state(obs) if is_q_learning else obs

            # Pure exploitation mode (deterministic=True)
            action = agent.act(state, info, deterministic=True)

            next_obs, reward, terminated, truncated, step_info = env.step(action)

            # Record step telemetry
            step_record = {
                "step": step_info["step"],
                "phase": step_info["phase"],
                "total_passed": step_info["total_passed"],
                "total_queue": step_info["total_queue"],
                "total_active_vehicles": step_info["total_active_vehicles"],
                "avg_waiting_time": step_info["avg_waiting_time"],
                "congestion_index": step_info["congestion_index"],
                "q_north": step_info["queue_lengths"]["NORTH"],
                "q_south": step_info["queue_lengths"]["SOUTH"],
                "q_east": step_info["queue_lengths"]["EAST"],
                "q_west": step_info["queue_lengths"]["WEST"],
            }
            step_records.append(step_record)

            obs = next_obs
            info = step_info

            if terminated or truncated:
                break

        summary = compute_summary_metrics(
            step_records=step_records,
            total_passed=env.intersection.total_passed,
            completed_wait_times=[v.wait_time for v in env.intersection.completed_vehicles],
        )
        summary["controller"] = agent.name
        summary["preset"] = preset_name
        summary["steps"] = num_steps

        df_telemetry = pd.DataFrame(step_records)
        return summary, df_telemetry

    def compare_all(
        self,
        agents_dict: Dict[str, BaseAgent],
        preset_name: str = "Balanced Medium",
        num_steps: int = 300,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Runs comparative benchmark across all supplied agents using identical random seeds.

        Returns:
            comparison_df (pd.DataFrame): Comparative table with % improvements against baseline.
            telemetry_dict (dict): Telemetry traces per agent.
        """
        summaries = []
        telemetries = {}

        for name, agent in agents_dict.items():
            summary, df_tel = self.evaluate_agent(agent, preset_name, num_steps, seed)
            summaries.append(summary)
            telemetries[name] = df_tel

        df_summary = pd.DataFrame(summaries)

        # Calculate improvement vs Traditional Fixed-Time if present
        baseline_rows = df_summary[df_summary["controller"].str.contains("Fixed-Time", case=False)]
        if not baseline_rows.empty:
            base_wait = float(baseline_rows["avg_waiting_time"].iloc[0])
            base_queue = float(baseline_rows["avg_queue_length"].iloc[0])
            base_tp = float(baseline_rows["total_throughput"].iloc[0])

            df_summary["Wait Reduction (%)"] = df_summary["avg_waiting_time"].apply(
                lambda v: compute_percentage_improvement(base_wait, v, lower_is_better=True)
            )
            df_summary["Queue Reduction (%)"] = df_summary["avg_queue_length"].apply(
                lambda v: compute_percentage_improvement(base_queue, v, lower_is_better=True)
            )
            df_summary["Throughput Gain (%)"] = df_summary["total_throughput"].apply(
                lambda v: compute_percentage_improvement(base_tp, v, lower_is_better=False)
            )

        return df_summary, telemetries
