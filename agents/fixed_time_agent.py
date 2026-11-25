"""
Traditional Fixed-Time Traffic Signal Agent (Baseline).
Simulates conventional pre-timed cyclical signal controllers with static cycle lengths.
Ignores dynamic vehicle density and queue lengths.
"""
from typing import Any, Dict
from agents.base_agent import BaseAgent
from utils.config import Config


class FixedTimeAgent(BaseAgent):
    """
    Baseline controller operating on fixed-time cycles.
    Maintains green for a fixed duration before issuing a phase switch command.
    """

    def __init__(self, cycle_time: int = 25, config: Config | None = None):
        super().__init__(name="Fixed-Time Controller")
        self.config = config or Config()
        self.cycle_time = cycle_time
        self.ticks_in_phase = 0

    def act(self, observation: Any, info: Dict[str, Any] | None = None, deterministic: bool = True) -> int:
        """
        Switches phase after fixed cycle_time ticks have elapsed.
        """
        self.ticks_in_phase += 1
        if self.ticks_in_phase >= self.cycle_time:
            self.ticks_in_phase = 0
            return self.config.ACTION_SWITCH_PHASE
        return self.config.ACTION_KEEP_GREEN

    def reset(self):
        """Resets the internal cycle timer."""
        self.ticks_in_phase = 0
