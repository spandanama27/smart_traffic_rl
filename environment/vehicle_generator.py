"""
Vehicle Generation Module.
Stochastically generates vehicles across 4 approaches based on Poisson/Bernoulli arrival rates.
Prevents overlapping vehicle spawns at entry points.
"""
import random
from typing import Dict, List
from utils.config import Config, Direction
from environment.vehicle import Vehicle


class VehicleGenerator:
    """Manages vehicle generation for North, South, East, and West approaches."""

    def __init__(self, config: Config | None = None, arrival_rates: Dict[str, float] | None = None):
        self.config = config or Config()
        self.arrival_rates = arrival_rates or self.config.DEFAULT_ARRIVAL_RATES.copy()
        self.total_generated: int = 0
        self.generated_per_direction: Dict[str, int] = {
            "NORTH": 0, "SOUTH": 0, "EAST": 0, "WEST": 0
        }

    def set_arrival_rates(self, rates: Dict[str, float]):
        """Updates the arrival rates per direction."""
        for key, val in rates.items():
            if key in self.arrival_rates:
                self.arrival_rates[key] = max(0.0, min(1.0, float(val)))

    def set_preset(self, preset_name: str):
        """Loads arrival rates from predefined traffic presets."""
        if preset_name in self.config.TRAFFIC_PRESETS:
            self.set_arrival_rates(self.config.TRAFFIC_PRESETS[preset_name])

    def step(self, active_vehicles: List[Vehicle]) -> List[Vehicle]:
        """
        Generates new vehicles for the current simulation step.
        Ensures safe headway at spawn points so vehicles do not overlap.
        """
        new_vehicles: List[Vehicle] = []

        # Direction mappings
        dir_map = [
            ("NORTH", Direction.NORTH),
            ("SOUTH", Direction.SOUTH),
            ("EAST", Direction.EAST),
            ("WEST", Direction.WEST),
        ]

        for dir_name, direction in dir_map:
            rate = self.arrival_rates.get(dir_name, 0.25)

            # Bernoulli arrival per tick
            if random.random() < rate:
                # Randomly pick lane 0 or 1 (2-lane approach)
                lane = random.choice([0, 1])

                # Check if spawn zone for this direction & lane is clear
                if self._is_spawn_clear(direction, lane, active_vehicles):
                    veh = Vehicle(direction=direction, lane_index=lane, config=self.config)
                    new_vehicles.append(veh)
                    self.total_generated += 1
                    self.generated_per_direction[dir_name] += 1

        return new_vehicles

    def _is_spawn_clear(self, direction: Direction, lane: int, vehicles: List[Vehicle]) -> bool:
        """Checks if there is enough space at the entrance of the road to spawn a vehicle."""
        safe_spawn_dist = self.config.VEHICLE_LENGTH * 2.0 + self.config.MIN_GAP

        for v in vehicles:
            if v.direction == direction and v.lane_index == lane:
                if direction == Direction.NORTH and v.y < safe_spawn_dist:
                    return False
                elif direction == Direction.SOUTH and (self.config.SCREEN_HEIGHT - v.y) < safe_spawn_dist:
                    return False
                elif direction == Direction.EAST and (self.config.SCREEN_WIDTH - v.x) < safe_spawn_dist:
                    return False
                elif direction == Direction.WEST and v.x < safe_spawn_dist:
                    return False

        return True

    def reset(self):
        """Resets generator counters."""
        self.total_generated = 0
        for k in self.generated_per_direction:
            self.generated_per_direction[k] = 0
