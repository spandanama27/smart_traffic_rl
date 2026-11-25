"""
Reinforcement Learning Trainer for Traffic Signal Optimization.
Handles episode rollouts, policy updates, metric aggregation, checkpointing,
and callback hooks for live visual dashboard integration.
"""
import os
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from utils.config import Config
from utils.logger import setup_logger
from environment.traffic_env import TrafficEnvironment
from agents.base_agent import BaseAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent


class RLTrainer:
    """Orchestrates multi-episode training of RL agents on the traffic intersection."""

    def __init__(
        self,
        agent: BaseAgent,
        env: TrafficEnvironment,
        config: Config | None = None,
        save_dir: str = "saved_models",
        logger=None,
    ):
        self.agent = agent
        self.env = env
        self.config = config or Config()
        self.save_dir = save_dir
        self.logger = logger or setup_logger("RLTrainer")

        os.makedirs(self.save_dir, exist_ok=True)

        # Performance history
        self.history: List[Dict[str, Any]] = []

    def train(
        self,
        num_episodes: int = 100,
        max_steps_per_episode: int = 300,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> pd.DataFrame:
        """
        Executes multi-episode training loop.

        Args:
            num_episodes: Number of training episodes.
            max_steps_per_episode: Duration of each episode in simulation ticks.
            progress_callback: Optional callback fn(episode, total_episodes, episode_metrics)
                               for real-time dashboard updates.

        Returns:
            pd.DataFrame of training history.
        """
        self.logger.info(f"Starting training for {self.agent.name} over {num_episodes} episodes...")
        best_reward = -float("inf")
        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_queues = []
            episode_waits = []

            is_q_learning = isinstance(self.agent, QLearningAgent)
            state = self.env.discretize_state(obs) if is_q_learning else obs

            for step in range(max_steps_per_episode):
                # Epsilon-greedy action
                action = self.agent.act(state, info, deterministic=False)

                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated

                next_state = self.env.discretize_state(next_obs) if is_q_learning else next_obs

                # Agent learning update
                self.agent.learn(state, action, reward, next_state, done)

                episode_reward += reward
                episode_queues.append(step_info["total_queue"])
                episode_waits.append(step_info["avg_waiting_time"])

                state = next_state
                obs = next_obs
                info = step_info

                if done:
                    break

            # Decay exploration rate epsilon
            if hasattr(self.agent, "decay_epsilon"):
                self.agent.decay_epsilon()

            # End of episode telemetry
            avg_queue = float(np.mean(episode_queues)) if episode_queues else 0.0
            final_avg_wait = float(episode_waits[-1]) if episode_waits else 0.0
            throughput = int(info["total_passed"])
            epsilon = getattr(self.agent, "epsilon", 0.0)

            ep_metrics = {
                "episode": episode,
                "reward": round(episode_reward, 2),
                "avg_queue": round(avg_queue, 2),
                "avg_waiting_time": round(final_avg_wait, 2),
                "throughput": throughput,
                "epsilon": round(epsilon, 4),
            }
            self.history.append(ep_metrics)

            if episode_reward > best_reward:
                best_reward = episode_reward
                self._save_checkpoint("best")

            if episode % 10 == 0 or episode == num_episodes:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Ep {episode:03d}/{num_episodes} | Reward: {episode_reward:7.1f} | "
                    f"Wait: {final_avg_wait:5.1f}s | Queue: {avg_queue:4.1f} | "
                    f"Throughput: {throughput:3d} | Epsilon: {epsilon:.3f} | Elapsed: {elapsed:.1f}s"
                )

            if progress_callback:
                progress_callback(episode, num_episodes, ep_metrics)

        # Save final model checkpoint
        self._save_checkpoint("final")

        # Save metrics DataFrame
        df_history = pd.DataFrame(self.history)
        history_csv = os.path.join(self.save_dir, f"{self.agent.name.lower().replace(' ', '_')}_history.csv")
        df_history.to_csv(history_csv, index=False)
        self.logger.info(f"Training complete. History saved to {history_csv}")

        return df_history

    def _save_checkpoint(self, tag: str):
        """Saves model checkpoint file."""
        agent_type = "q_table" if isinstance(self.agent, QLearningAgent) else "dqn_model"
        ext = "pkl" if isinstance(self.agent, QLearningAgent) else ("pt" if getattr(self.agent, "use_torch", False) else "npz")
        filename = f"{agent_type}_{tag}.{ext}"
        filepath = os.path.join(self.save_dir, filename)
        self.agent.save(filepath)
