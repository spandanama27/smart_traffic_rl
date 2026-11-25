"""
Traffic Signal Control Module.
Manages 4-phase traffic signal switching (NS_GREEN -> NS_YELLOW -> EW_GREEN -> EW_YELLOW).
Enforces minimum green times, safety yellow clearance intervals, and maximum green limits.
"""
from typing import Dict
from utils.config import Config, Direction, SignalPhase


class TrafficSignalController:
    """Controls the signal phases, timers, and clearance intervals."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.current_phase: SignalPhase = SignalPhase.NS_GREEN
        self.phase_timer: int = 0  # Number of simulation ticks elapsed in current phase
        self.total_switches: int = 0

    @property
    def is_yellow_active(self) -> bool:
        """Returns True if the signal is currently in a clearance yellow phase."""
        return self.current_phase in (SignalPhase.NS_YELLOW, SignalPhase.EW_YELLOW)

    @property
    def active_green_axis(self) -> str:
        """Returns 'NS' or 'EW' corresponding to current green or active axis."""
        if self.current_phase in (SignalPhase.NS_GREEN, SignalPhase.NS_YELLOW):
            return "NS"
        return "EW"

    def is_green(self, direction: Direction) -> bool:
        """Returns True if the given direction currently has a Green light."""
        if self.current_phase == SignalPhase.NS_GREEN:
            return direction in (Direction.NORTH, Direction.SOUTH)
        elif self.current_phase == SignalPhase.EW_GREEN:
            return direction in (Direction.EAST, Direction.WEST)
        return False

    def is_yellow(self, direction: Direction) -> bool:
        """Returns True if the given direction currently has a Yellow light."""
        if self.current_phase == SignalPhase.NS_YELLOW:
            return direction in (Direction.NORTH, Direction.SOUTH)
        elif self.current_phase == SignalPhase.EW_YELLOW:
            return direction in (Direction.EAST, Direction.WEST)
        return False

    def is_red(self, direction: Direction) -> bool:
        """Returns True if the given direction currently has a Red light."""
        return not (self.is_green(direction) or self.is_yellow(direction))

    def step(self, action: int) -> bool:
        """
        Advances the signal controller by one tick according to the chosen action.

        Actions:
            0 (Config.ACTION_KEEP_GREEN): Maintain current green phase.
            1 (Config.ACTION_SWITCH_PHASE): Request transition to alternate phase.

        Returns:
            phase_changed (bool): True if the phase transitioned this tick.
        """
        self.phase_timer += 1
        phase_changed = False

        # Phase transition finite state machine
        if self.current_phase == SignalPhase.NS_GREEN:
            # Can only switch if min green time elapsed, or forced if max green reached
            should_switch = (
                (action == self.config.ACTION_SWITCH_PHASE and self.phase_timer >= self.config.MIN_GREEN_TIME)
                or (self.phase_timer >= self.config.MAX_GREEN_TIME)
            )
            if should_switch:
                self.current_phase = SignalPhase.NS_YELLOW
                self.phase_timer = 0
                self.total_switches += 1
                phase_changed = True

        elif self.current_phase == SignalPhase.NS_YELLOW:
            # Yellow clearance interval is automatic and non-interruptible
            if self.phase_timer >= self.config.YELLOW_TIME:
                self.current_phase = SignalPhase.EW_GREEN
                self.phase_timer = 0
                self.total_switches += 1
                phase_changed = True

        elif self.current_phase == SignalPhase.EW_GREEN:
            should_switch = (
                (action == self.config.ACTION_SWITCH_PHASE and self.phase_timer >= self.config.MIN_GREEN_TIME)
                or (self.phase_timer >= self.config.MAX_GREEN_TIME)
            )
            if should_switch:
                self.current_phase = SignalPhase.EW_YELLOW
                self.phase_timer = 0
                self.total_switches += 1
                phase_changed = True

        elif self.current_phase == SignalPhase.EW_YELLOW:
            # Yellow clearance interval is automatic and non-interruptible
            if self.phase_timer >= self.config.YELLOW_TIME:
                self.current_phase = SignalPhase.NS_GREEN
                self.phase_timer = 0
                self.total_switches += 1
                phase_changed = True

        return phase_changed

    def get_signal_states(self) -> Dict[str, str]:
        """Returns the current light color ('GREEN', 'YELLOW', 'RED') for all 4 directions."""
        states = {}
        dir_map = {
            "NORTH": Direction.NORTH,
            "SOUTH": Direction.SOUTH,
            "EAST": Direction.EAST,
            "WEST": Direction.WEST,
        }
        for name, d in dir_map.items():
            if self.is_green(d):
                states[name] = "GREEN"
            elif self.is_yellow(d):
                states[name] = "YELLOW"
            else:
                states[name] = "RED"
        return states

    def reset(self, initial_phase: SignalPhase = SignalPhase.NS_GREEN):
        """Resets the signal controller."""
        self.current_phase = initial_phase
        self.phase_timer = 0
        self.total_switches = 0
