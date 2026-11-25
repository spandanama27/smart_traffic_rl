"""
Base Agent Interface.
Defines standard API for traffic signal control agents (Fixed-time, Q-Learning, DQN).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract base class for all traffic signal controllers."""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def act(self, observation: Any, info: Dict[str, Any] | None = None, deterministic: bool = False) -> int:
        """Selects an action given current observation."""
        pass

    def learn(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """Optional update step for learning agents."""
        pass

    def save(self, filepath: str):
        """Persists trained agent parameters."""
        pass

    def load(self, filepath: str):
        """Loads trained agent parameters."""
        pass

    def reset(self):
        """Resets internal agent state between episodes."""
        pass
