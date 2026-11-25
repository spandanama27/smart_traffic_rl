"""
Q-Learning Agent for Intelligent Traffic Signal Control.

Key Theoretical Concepts Implemented:
1. Markov Decision Process (MDP):
   Models the intersection as <S, A, P, R, gamma> where:
   - S: Discretized queue lengths and densities for NS and EW + current active phase.
   - A: {0: Keep Green, 1: Switch Phase}.
   - R: Immediate scalar reward based on queue reduction, throughput, and delay minimization.
   - gamma: Discount factor for future rewards.

2. Temporal Difference (TD) Control & Bellman Update:
   Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

3. Epsilon-Greedy Policy:
   Balances exploration (discovering new signal timing strategies) with exploitation
   (using known best signal actions). Epsilon decays exponentially over training episodes.
"""
from collections import defaultdict
import os
import pickle
import random
from typing import Any, Dict, Tuple
import numpy as np

from agents.base_agent import BaseAgent
from utils.config import Config


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning Agent with epsilon-greedy exploration and state discretization.
    """

    def __init__(
        self,
        num_actions: int = 2,
        learning_rate: float = 0.12,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.992,
        config: Config | None = None,
    ):
        super().__init__(name="Q-Learning Agent")
        self.config = config or Config()
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table mapping discrete state tuple -> np.ndarray of shape (num_actions,)
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

        # Training telemetry
        self.total_updates = 0
        self.td_errors: list = []

    def get_q_values(self, state: Tuple) -> np.ndarray:
        """Retrieves or initializes the Q-value vector for a given state."""
        return self.q_table[state]

    def act(self, observation: Any, info: Dict[str, Any] | None = None, deterministic: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy.

        Args:
            observation: Either raw continuous obs or already discretized state tuple.
            info: Optional auxiliary metrics.
            deterministic: If True, always acts greedily (no exploration, test mode).

        Returns:
            action (int): 0 (Keep Green) or 1 (Switch Phase)
        """
        state = self._ensure_discrete_state(observation)

        # Exploration vs Exploitation
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        q_vals = self.q_table[state]
        # Break ties randomly to avoid directional bias
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(random.choice(best_actions))

    def learn(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """
        Applies the Bellman update rule:
        TD_Error = reward + gamma * max_a'(Q(s', a')) - Q(s, a)
        Q(s, a) = Q(s, a) + alpha * TD_Error
        """
        s = self._ensure_discrete_state(state)
        s_next = self._ensure_discrete_state(next_state)

        # Target calculation
        if done:
            target = reward
        else:
            best_future_q = np.max(self.q_table[s_next])
            target = reward + self.gamma * best_future_q

        # Temporal Difference (TD) Error
        current_q = self.q_table[s][action]
        td_error = target - current_q

        # Q-Table update
        self.q_table[s][action] += self.alpha * td_error

        self.total_updates += 1
        self.td_errors.append(float(abs(td_error)))

    def decay_epsilon(self):
        """Decays exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _ensure_discrete_state(self, state_or_obs: Any) -> Tuple:
        """Converts raw array observation to discrete tuple if needed."""
        if isinstance(state_or_obs, tuple):
            return state_or_obs

        # Continuous array conversion
        obs = np.asarray(state_or_obs, dtype=np.float32)
        ns_queue = obs[0] + obs[1]
        ew_queue = obs[2] + obs[3]
        ns_density = obs[4] + obs[5]
        ew_density = obs[6] + obs[7]
        active_axis = int(obs[8])

        def _bin(val: float) -> int:
            if val == 0:
                return 0
            elif val <= 3:
                return 1
            elif val <= 7:
                return 2
            else:
                return 3

        return (_bin(ns_queue), _bin(ew_queue), _bin(ns_density), _bin(ew_density), active_axis)

    def save(self, filepath: str):
        """Saves Q-table and hyperparameters to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Loads Q-table and parameters from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32), data["q_table"])
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.total_updates = data.get("total_updates", 0)

    def reset(self):
        """Resets per-episode metrics."""
        self.td_errors.clear()
