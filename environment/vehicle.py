"""
Vehicle Module: Realistic Kinematics & Driver Behavior for Traffic Simulation.
Implements car-following logic, stop-line braking, acceleration, and queue metrics.
"""
from dataclasses import dataclass
import random
from typing import Optional, Tuple
from utils.config import Config, Direction


class Vehicle:
    """Represents an individual vehicle in the 4-way intersection simulation."""

    _id_counter: int = 0

    def __init__(
        self,
        direction: Direction,
        lane_index: int,
        config: Config | None = None,
        color: Optional[Tuple[int, int, int]] = None,
    ):
        Vehicle._id_counter += 1
        self.id = Vehicle._id_counter
        self.direction = direction
        self.lane_index = lane_index  # 0: inner lane, 1: outer lane
        self.config = config or Config()

        # Geometry & Dimensions
        self.length = self.config.VEHICLE_LENGTH
        self.width = self.config.VEHICLE_WIDTH
        self.color = color or random.choice(self.config.VEHICLE_COLORS)

        # Kinematic state
        self.speed = 0.0
        self.max_speed = self.config.MAX_SPEED
        self.accel = self.config.ACCELERATION
        self.decel = self.config.DECELERATION
        self.min_gap = self.config.MIN_GAP

        # Performance & RL metrics
        self.wait_time: float = 0.0       # Ticks vehicle was stationary (speed < 1.0)
        self.travel_time: float = 0.0     # Total ticks vehicle has existed
        self.is_queued: bool = False      # Stationary waiting behind signal or lead car
        self.has_crossed_stop_line: bool = False
        self.has_exited: bool = False

        # Compute initial position based on direction and lane
        self.x, self.y = self._get_initial_position()

    def _get_initial_position(self) -> Tuple[float, float]:
        """Calculates spawn coordinates based on direction and right-hand drive lane."""
        c = self.config
        offset = 17.5 + self.lane_index * 35.0  # Centers within the 35px lanes

        if self.direction == Direction.NORTH:
            # Spawns at top, moving South (+Y)
            return (c.CENTER_X + offset, -self.length)
        elif self.direction == Direction.SOUTH:
            # Spawns at bottom, moving North (-Y)
            return (c.CENTER_X - offset, c.SCREEN_HEIGHT + self.length)
        elif self.direction == Direction.EAST:
            # Spawns at right, moving West (-X)
            return (c.SCREEN_WIDTH + self.length, c.CENTER_Y - offset)
        elif self.direction == Direction.WEST:
            # Spawns at left, moving East (+X)
            return (-self.length, c.CENTER_Y + offset)

        return (0.0, 0.0)

    @property
    def front_coord(self) -> float:
        """Returns the coordinate of the vehicle's front bumper along its axis of motion."""
        if self.direction == Direction.NORTH:
            return self.y + self.length / 2.0
        elif self.direction == Direction.SOUTH:
            return self.y - self.length / 2.0
        elif self.direction == Direction.EAST:
            return self.x - self.length / 2.0
        elif self.direction == Direction.WEST:
            return self.x + self.length / 2.0
        return 0.0

    @property
    def rear_coord(self) -> float:
        """Returns the coordinate of the vehicle's rear bumper along its axis of motion."""
        if self.direction == Direction.NORTH:
            return self.y - self.length / 2.0
        elif self.direction == Direction.SOUTH:
            return self.y + self.length / 2.0
        elif self.direction == Direction.EAST:
            return self.x + self.length / 2.0
        elif self.direction == Direction.WEST:
            return self.x - self.length / 2.0
        return 0.0

    def get_stop_line_coord(self) -> float:
        """Returns the stop line position for this vehicle's direction."""
        c = self.config
        if self.direction == Direction.NORTH:
            return c.STOP_LINE_NORTH
        elif self.direction == Direction.SOUTH:
            return c.STOP_LINE_SOUTH
        elif self.direction == Direction.EAST:
            return c.STOP_LINE_EAST
        elif self.direction == Direction.WEST:
            return c.STOP_LINE_WEST
        return 0.0

    def update_kinematics(self, lead_vehicle: Optional["Vehicle"], is_signal_green: bool, is_signal_yellow: bool):
        """
        Updates speed and position using smooth car-following and stopping logic.
        Accounts for:
        1. Distance to lead car (safe headway)
        2. Signal state (green vs yellow/red) if stop line hasn't been crossed
        3. Braking vs acceleration curves
        4. Absolute stopping at red lights and queues
        """
        self.travel_time += 1.0
        stop_line = self.get_stop_line_coord()

        # Check if vehicle has crossed the stop line
        if not self.has_crossed_stop_line:
            crossed = False
            if self.direction == Direction.NORTH and self.front_coord >= stop_line:
                crossed = True
            elif self.direction == Direction.SOUTH and self.front_coord <= stop_line:
                crossed = True
            elif self.direction == Direction.EAST and self.front_coord <= stop_line:
                crossed = True
            elif self.direction == Direction.WEST and self.front_coord >= stop_line:
                crossed = True

            # Only officially mark crossed if signal allowed it (green or clearing yellow)
            if crossed and (is_signal_green or is_signal_yellow):
                self.has_crossed_stop_line = True

        # Calculate target distance to nearest obstacle (lead vehicle or stop line)
        target_speed = self.max_speed
        obstacle_dist = float("inf")

        # 1. Lead vehicle in the same lane
        if lead_vehicle is not None:
            if self.direction == Direction.NORTH:
                dist = lead_vehicle.rear_coord - self.front_coord
            elif self.direction == Direction.SOUTH:
                dist = self.front_coord - lead_vehicle.rear_coord
            elif self.direction == Direction.EAST:
                dist = self.front_coord - lead_vehicle.rear_coord
            elif self.direction == Direction.WEST:
                dist = lead_vehicle.rear_coord - self.front_coord
            obstacle_dist = max(0.0, dist - self.min_gap)

        # 2. Stop line check if signal is red or yellow (and vehicle hasn't crossed)
        if not self.has_crossed_stop_line:
            if self.direction == Direction.NORTH:
                dist_to_stop = stop_line - self.front_coord
            elif self.direction == Direction.SOUTH:
                dist_to_stop = self.front_coord - stop_line
            elif self.direction == Direction.EAST:
                dist_to_stop = self.front_coord - stop_line
            elif self.direction == Direction.WEST:
                dist_to_stop = stop_line - self.front_coord
            else:
                dist_to_stop = float("inf")

            if not is_signal_green:
                stopping_distance_needed = (self.speed ** 2) / (2.0 * self.decel + 1e-5)
                can_clear_yellow = is_signal_yellow and (dist_to_stop < stopping_distance_needed * 0.8)

                if not can_clear_yellow:
                    effective_stop_dist = max(0.0, dist_to_stop - 4.0)
                    obstacle_dist = min(obstacle_dist, effective_stop_dist)

        # Determine target speed
        if obstacle_dist <= 2.0:
            target_speed = 0.0
        elif obstacle_dist < float("inf"):
            # v <= sqrt(2 * a * d)
            safe_braking_speed = (2.0 * self.decel * obstacle_dist) ** 0.5
            target_speed = min(self.max_speed, safe_braking_speed)

        # Apply smooth acceleration or deceleration
        if self.speed < target_speed:
            self.speed = min(target_speed, self.speed + self.accel)
        elif self.speed > target_speed:
            self.speed = max(target_speed, self.speed - self.decel)

        # If vehicle is very close to obstacle and moving very slowly, stop it
        if obstacle_dist <= 3.0 and self.speed < 2.0:
            self.speed = 0.0

        # Calculate movement step
        move_dist = self.speed

        # Prevent overshooting stop line when stopping for red light
        if not self.has_crossed_stop_line and not is_signal_green and not (is_signal_yellow and can_clear_yellow):
            move_dist = min(move_dist, max(0.0, dist_to_stop - 2.0))

        # Prevent overshooting lead vehicle
        if lead_vehicle is not None:
            move_dist = min(move_dist, obstacle_dist)

        # Move vehicle position based on actual move_dist
        if self.direction == Direction.NORTH:
            self.y += move_dist
        elif self.direction == Direction.SOUTH:
            self.y -= move_dist
        elif self.direction == Direction.EAST:
            self.x -= move_dist
        elif self.direction == Direction.WEST:
            self.x += move_dist

        # If move_dist was forced to 0, speed is 0
        if move_dist <= 0.01:
            self.speed = 0.0

        # Track queue metrics: stationary/creeping before stop line
        if self.speed < 1.0 and not self.has_crossed_stop_line:
            self.is_queued = True
            self.wait_time += 1.0
        else:
            self.is_queued = False

        # Check exit conditions (off canvas)
        c = self.config
        margin = self.length * 2
        if (
            self.x < -margin
            or self.x > c.SCREEN_WIDTH + margin
            or self.y < -margin
            or self.y > c.SCREEN_HEIGHT + margin
        ):
            self.has_exited = True
