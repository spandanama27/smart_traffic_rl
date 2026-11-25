"""
Central Configuration module for Smart Traffic Signal Optimization.
Defines parameters for simulation physics, geometry, RL states/rewards, and colors.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple


class SignalPhase(IntEnum):
    """Signal phase states for the four-way intersection."""
    NS_GREEN = 0   # North-South Green, East-West Red
    NS_YELLOW = 1  # North-South Yellow, East-West Red
    EW_GREEN = 2   # East-West Green, North-South Red
    EW_YELLOW = 3  # East-West Yellow, North-South Red


class Direction(IntEnum):
    """Traffic flow entry directions."""
    NORTH = 0  # Inbound moving South (+Y)
    SOUTH = 1  # Inbound moving North (-Y)
    EAST = 2   # Inbound moving West (-X)
    WEST = 3   # Inbound moving East (+X)


@dataclass
class Config:
    """Master simulation and RL configuration."""

    # ------------------ Simulation Canvas & Geometry ------------------
    SCREEN_WIDTH: int = 700
    SCREEN_HEIGHT: int = 700
    ROAD_WIDTH: int = 140       # Total road width (70px per direction)
    LANE_WIDTH: int = 35        # Each direction has 2 lanes (inbound/outbound)
    CENTER_X: int = 350
    CENTER_Y: int = 350

    # Stop lines for inbound vehicles before entering the intersection box
    STOP_LINE_NORTH: float = 280.0  # Vehicles heading South (+Y)
    STOP_LINE_SOUTH: float = 420.0  # Vehicles heading North (-Y)
    STOP_LINE_EAST: float = 420.0   # Vehicles heading West (-X)
    STOP_LINE_WEST: float = 280.0   # Vehicles heading East (+X)

    # Intersection boundary
    INTERSECTION_LEFT: float = 280.0
    INTERSECTION_RIGHT: float = 420.0
    INTERSECTION_TOP: float = 280.0
    INTERSECTION_BOTTOM: float = 420.0

    # ------------------ Vehicle Kinematics ------------------
    VEHICLE_LENGTH: float = 22.0
    VEHICLE_WIDTH: float = 13.0
    MAX_SPEED: float = 14.0          # pixels per simulation tick
    ACCELERATION: float = 3.0       # pixels per tick^2
    DECELERATION: float = 6.0       # braking capability
    MIN_GAP: float = 8.0            # bumper-to-bumper minimum gap at standstill
    TIME_STEP: float = 1.0          # seconds per simulation tick

    # ------------------ Traffic Signal Timings ------------------
    MIN_GREEN_TIME: int = 5         # Minimum green ticks before switch is permitted
    MAX_GREEN_TIME: int = 50        # Maximum green ticks before forced switch
    YELLOW_TIME: int = 3           # Yellow clearance ticks
    FIXED_TIME_DURATION: int = 25  # Duration for baseline fixed-time controller

    # ------------------ Vehicle Generation Rates ------------------
    # Probability of spawning a vehicle per step in each direction [N, S, E, W]
    DEFAULT_ARRIVAL_RATES: Dict[str, float] = field(default_factory=lambda: {
        "NORTH": 0.30,
        "SOUTH": 0.30,
        "EAST": 0.25,
        "WEST": 0.25,
    })

    # Traffic scenarios
    TRAFFIC_PRESETS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Balanced Low": {"NORTH": 0.15, "SOUTH": 0.15, "EAST": 0.15, "WEST": 0.15},
        "Balanced Medium": {"NORTH": 0.30, "SOUTH": 0.30, "EAST": 0.30, "WEST": 0.30},
        "Balanced Heavy": {"NORTH": 0.50, "SOUTH": 0.50, "EAST": 0.50, "WEST": 0.50},
        "NS Heavy (Rush Hour)": {"NORTH": 0.60, "SOUTH": 0.55, "EAST": 0.15, "WEST": 0.15},
        "EW Heavy (Rush Hour)": {"NORTH": 0.15, "SOUTH": 0.15, "EAST": 0.60, "WEST": 0.55},
        "Asymmetric Dynamic": {"NORTH": 0.45, "SOUTH": 0.20, "EAST": 0.50, "WEST": 0.15},
    })

    # ------------------ RL Environment & Rewards ------------------
    EPISODE_DURATION: int = 300     # Ticks per training episode
    ACTION_KEEP_GREEN: int = 0
    ACTION_SWITCH_PHASE: int = 1

    # Reward Function Coefficients
    # R = - (w_wait * total_wait) - (w_queue * total_queue) + (w_tp * throughput) - penalty
    W_WAIT_TIME: float = 0.25
    W_QUEUE_LENGTH: float = 0.60
    W_THROUGHPUT: float = 8.0
    W_PHASE_SWITCH_PENALTY: float = 1.5  # Slight penalty to prevent rapid oscillation

    # ------------------ Q-Learning Hyperparameters ------------------
    Q_LEARNING_RATE: float = 0.12
    Q_DISCOUNT_FACTOR: float = 0.95
    Q_EPSILON_START: float = 1.0
    Q_EPSILON_MIN: float = 0.05
    Q_EPSILON_DECAY: float = 0.992
    Q_NUM_EPISODES: int = 150

    # ------------------ DQN Hyperparameters ------------------
    DQN_LEARNING_RATE: float = 0.001
    DQN_DISCOUNT_FACTOR: float = 0.95
    DQN_BATCH_SIZE: int = 64
    DQN_REPLAY_BUFFER_SIZE: int = 20000
    DQN_TARGET_UPDATE_FREQ: int = 10
    DQN_EPSILON_START: float = 1.0
    DQN_EPSILON_MIN: float = 0.05
    DQN_EPSILON_DECAY: float = 0.995
    DQN_NUM_EPISODES: int = 100

    # ------------------ UI & Rendering Colors (RGB) ------------------
    COLOR_BACKGROUND: Tuple[int, int, int] = (24, 32, 43)
    COLOR_GRASS: Tuple[int, int, int] = (30, 41, 59)
    COLOR_ROAD: Tuple[int, int, int] = (51, 65, 85)
    COLOR_ROAD_MARKING: Tuple[int, int, int] = (226, 232, 240)
    COLOR_YELLOW_LINE: Tuple[int, int, int] = (245, 158, 11)
    COLOR_STOP_LINE: Tuple[int, int, int] = (255, 255, 255)
    COLOR_INTERSECTION: Tuple[int, int, int] = (40, 53, 72)

    # Signal Colors
    COLOR_RED: Tuple[int, int, int] = (239, 68, 68)
    COLOR_YELLOW: Tuple[int, int, int] = (245, 158, 11)
    COLOR_GREEN: Tuple[int, int, int] = (16, 185, 129)
    COLOR_LIGHT_OFF: Tuple[int, int, int] = (55, 65, 81)
    COLOR_HOUSING: Tuple[int, int, int] = (15, 23, 42)

    # Vehicle Palette (Randomized for visual variety)
    VEHICLE_COLORS: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (56, 189, 248),   # Sky blue
        (251, 146, 60),   # Orange
        (168, 85, 247),   # Purple
        (52, 211, 153),   # Emerald
        (250, 204, 21),   # Yellow
        (244, 63, 94),    # Rose
        (20, 184, 166),   # Teal
        (129, 140, 248),  # Indigo
    ])
