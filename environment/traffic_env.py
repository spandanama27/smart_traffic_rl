"""
Gymnasium-Compliant Traffic Intersection Environment.
Provides standardized reset(), step(), action_space, observation_space, and state discretization
for both Tabular Q-Learning and Deep Q-Networks (DQN).
"""
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from utils.config import Config, Direction, SignalPhase
from environment.traffic_intersection import TrafficIntersection


class TrafficEnvironment(gym.Env):
    """
    Gymnasium Environment modeling an intelligent four-way traffic intersection.

    State / Observation Space (Continuous Box, shape=(10,)):
        [0]: North approach queue length
        [1]: South approach queue length
        [2]: East approach queue length
        [3]: West approach queue length
        [4]: North approach vehicle count (density)
        [5]: South approach vehicle count (density)
        [6]: East approach vehicle count (density)
        [7]: West approach vehicle count (density)
        [8]: Current green phase (0.0 for NS, 1.0 for EW)
        [9]: Elapsed time in current phase (normalized)

    Action Space (Discrete(2)):
        0: Keep current green phase
        1: Switch phase (triggers yellow clearance transition)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Config | None = None,
        arrival_rates: Dict[str, float] | None = None,
        max_steps: int = 300,
    ):
        super().__init__()
        self.config = config or Config()
        self.max_steps = max_steps
        self.intersection = TrafficIntersection(self.config, arrival_rates)

        # Action space: 0 = Keep Green, 1 = Switch Phase
        self.action_space = spaces.Discrete(2)

        # Observation space: 10 continuous features
        self.observation_space = spaces.Box(
            low=0.0,
            high=150.0,
            shape=(10,),
            dtype=np.float32,
        )

        self.current_step = 0
        self._prev_total_wait = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to initial state."""
        super().reset(seed=seed)
        self.intersection.reset()
        self.current_step = 0
        self._prev_total_wait = 0.0

        obs = self._get_observation()
        info = self.intersection._calculate_metrics(phase_changed=False)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one action step in the environment.

        Args:
            action (int): 0 (Keep) or 1 (Switch)

        Returns:
            obs (np.ndarray): Next observation vector.
            reward (float): Immediate reward signal calculated by reward function.
            terminated (bool): True if episode ended naturally.
            truncated (bool): True if maximum step limit reached.
            info (dict): Comprehensive live traffic metrics.
        """
        self.current_step += 1

        # Advance intersection physics and signal controller
        metrics = self.intersection.step(action)

        # Compute RL reward
        reward = self._compute_reward(metrics, action)

        # Check termination / truncation
        truncated = self.current_step >= self.max_steps
        terminated = False

        obs = self._get_observation()
        return obs, reward, terminated, truncated, metrics

    def _get_observation(self) -> np.ndarray:
        """Extracts the 10-dimensional observation vector."""
        ql = self.intersection.step_count
        metrics = self.intersection._calculate_metrics(phase_changed=False)

        q = metrics["queue_lengths"]
        c = metrics["vehicle_counts"]
        phase = self.intersection.signal_controller.current_phase

        # Binary indicator: 0.0 for North-South active, 1.0 for East-West active
        phase_num = 0.0 if phase in (SignalPhase.NS_GREEN, SignalPhase.NS_YELLOW) else 1.0
        phase_timer_norm = min(1.0, self.intersection.signal_controller.phase_timer / self.config.MAX_GREEN_TIME)

        obs = np.array([
            float(q["NORTH"]),
            float(q["SOUTH"]),
            float(q["EAST"]),
            float(q["WEST"]),
            float(c["NORTH"]),
            float(c["SOUTH"]),
            float(c["EAST"]),
            float(c["WEST"]),
            phase_num,
            phase_timer_norm,
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, metrics: Dict[str, Any], action: int) -> float:
        """
        Calculates the scalar reward for the RL agent.

        Reward formulation:
            R = - (w_queue * total_queue)
                - (w_wait * wait_increment)
                + (w_tp * vehicles_passed)
                - penalty_for_unnecessary_switch

        Concept:
            Reinforcement Learning seeks to maximize cumulative reward.
            Penalizing queue length and waiting time encourages the agent to clear traffic,
            while rewarding throughput encourages maximizing green-wave flow.
        """
        total_queue = metrics["total_queue"]
        vehicles_passed = metrics["passed_this_step"]
        total_wait = metrics["total_waiting_time"]

        wait_increment = max(0.0, total_wait - self._prev_total_wait)
        self._prev_total_wait = total_wait

        # Core penalty terms
        reward = (
            - (self.config.W_QUEUE_LENGTH * total_queue)
            - (self.config.W_WAIT_TIME * wait_increment)
            + (self.config.W_THROUGHPUT * vehicles_passed)
        )

        # Small penalty if agent triggered a switch to discourage rapid oscillation
        if metrics.get("phase_changed", False) and action == self.config.ACTION_SWITCH_PHASE:
            reward -= self.config.W_PHASE_SWITCH_PENALTY

        return float(reward)

    def discretize_state(self, obs: np.ndarray) -> Tuple[int, int, int, int, int]:
        """
        Discretizes continuous observation space into compact discrete bins for Tabular Q-Learning.

        State representation:
            (NS_queue_bin, EW_queue_bin, NS_density_bin, EW_density_bin, active_axis)

        Bins:
            0: 0 vehicles (empty)
            1: 1-3 vehicles (light)
            2: 4-7 vehicles (moderate)
            3: 8+ vehicles (congested)
        """
        ns_queue = obs[0] + obs[1]
        ew_queue = obs[2] + obs[3]
        ns_density = obs[4] + obs[5]
        ew_density = obs[6] + obs[7]
        active_axis = int(obs[8])  # 0: NS, 1: EW

        def _to_bin(val: float) -> int:
            if val == 0:
                return 0
            elif val <= 3:
                return 1
            elif val <= 7:
                return 2
            else:
                return 3

        return (
            _to_bin(ns_queue),
            _to_bin(ew_queue),
            _to_bin(ns_density),
            _to_bin(ew_density),
            active_axis,
        )
