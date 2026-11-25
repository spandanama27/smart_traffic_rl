"""
Traffic Intersection Module.
Manages the four-way road network, vehicle collections, lane sorting,
kinematic updates, and live traffic metrics computation.
"""
from typing import Dict, List, Tuple
from utils.config import Config, Direction, SignalPhase
from environment.vehicle import Vehicle
from environment.vehicle_generator import VehicleGenerator
from environment.traffic_signal import TrafficSignalController


class TrafficIntersection:
    """Manages the full state and dynamics of the four-way intersection."""

    def __init__(self, config: Config | None = None, arrival_rates: Dict[str, float] | None = None):
        self.config = config or Config()
        self.signal_controller = TrafficSignalController(self.config)
        self.generator = VehicleGenerator(self.config, arrival_rates)

        self.active_vehicles: List[Vehicle] = []
        self.completed_vehicles: List[Vehicle] = []
        self.step_count: int = 0
        self.vehicles_passed_this_step: int = 0
        self.total_passed: int = 0

    def step(self, action: int) -> Dict[str, any]:
        """
        Advances the intersection simulation by one discrete tick.

        Args:
            action (int): 0 to keep green, 1 to switch phase.

        Returns:
            metrics (dict): Step statistics including queues, wait times, throughput, and congestion.
        """
        self.step_count += 1
        self.vehicles_passed_this_step = 0

        # 1. Update Traffic Light
        phase_changed = self.signal_controller.step(action)

        # 2. Spawn Vehicles
        new_vehicles = self.generator.step(self.active_vehicles)
        self.active_vehicles.extend(new_vehicles)

        # 3. Sort vehicles per approach and lane to identify lead vehicle correctly
        # In each lane, the vehicle furthest along the road is first
        lanes: Dict[Tuple[Direction, int], List[Vehicle]] = {}
        for d in Direction:
            lanes[(d, 0)] = []
            lanes[(d, 1)] = []

        for v in self.active_vehicles:
            lanes[(v.direction, v.lane_index)].append(v)

        # Sort along direction axis
        for (d, lane), veh_list in lanes.items():
            if d == Direction.NORTH:
                # Moving South (+Y): largest Y is furthest ahead
                veh_list.sort(key=lambda veh: veh.y, reverse=True)
            elif d == Direction.SOUTH:
                # Moving North (-Y): smallest Y is furthest ahead
                veh_list.sort(key=lambda veh: veh.y, reverse=False)
            elif d == Direction.EAST:
                # Moving West (-X): smallest X is furthest ahead
                veh_list.sort(key=lambda veh: veh.x, reverse=False)
            elif d == Direction.WEST:
                # Moving East (+X): largest X is furthest ahead
                veh_list.sort(key=lambda veh: veh.x, reverse=True)

        # 4. Update kinematics for each vehicle
        remaining_vehicles: List[Vehicle] = []
        for (d, lane), veh_list in lanes.items():
            is_green = self.signal_controller.is_green(d)
            is_yellow = self.signal_controller.is_yellow(d)

            for i, veh in enumerate(veh_list):
                lead = veh_list[i - 1] if i > 0 else None
                was_crossed = veh.has_crossed_stop_line
                veh.update_kinematics(lead, is_green, is_yellow)

                # Check if vehicle just crossed the stop line
                if not was_crossed and veh.has_crossed_stop_line:
                    self.vehicles_passed_this_step += 1
                    self.total_passed += 1

                if veh.has_exited:
                    self.completed_vehicles.append(veh)
                else:
                    remaining_vehicles.append(veh)

        self.active_vehicles = remaining_vehicles

        # 5. Calculate Metrics
        metrics = self._calculate_metrics(phase_changed)
        return metrics

    def _calculate_metrics(self, phase_changed: bool) -> Dict[str, any]:
        """Aggregates detailed performance statistics for the current step."""
        dir_names = {
            Direction.NORTH: "NORTH",
            Direction.SOUTH: "SOUTH",
            Direction.EAST: "EAST",
            Direction.WEST: "WEST",
        }

        queue_lengths = {name: 0 for name in dir_names.values()}
        vehicle_counts = {name: 0 for name in dir_names.values()}
        wait_times = {name: 0.0 for name in dir_names.values()}

        total_active_wait = 0.0
        total_queued = 0

        for v in self.active_vehicles:
            d_name = dir_names[v.direction]
            vehicle_counts[d_name] += 1
            if v.is_queued:
                queue_lengths[d_name] += 1
                total_queued += 1
            wait_times[d_name] += v.wait_time
            total_active_wait += v.wait_time

        # Total completed wait time
        total_completed_wait = sum(v.wait_time for v in self.completed_vehicles)
        total_all_wait = total_active_wait + total_completed_wait
        total_vehicles_seen = len(self.completed_vehicles) + len(self.active_vehicles)
        avg_wait = total_all_wait / max(1, total_vehicles_seen)

        # Congestion Index: queued vehicles / total active vehicles (0.0 to 1.0)
        congestion_index = total_queued / max(1, len(self.active_vehicles))

        return {
            "step": self.step_count,
            "phase": self.signal_controller.current_phase.name,
            "phase_timer": self.signal_controller.phase_timer,
            "phase_changed": phase_changed,
            "total_passed": self.total_passed,
            "passed_this_step": self.vehicles_passed_this_step,
            "queue_lengths": queue_lengths,
            "total_queue": total_queued,
            "vehicle_counts": vehicle_counts,
            "total_active_vehicles": len(self.active_vehicles),
            "wait_times": wait_times,
            "total_waiting_time": total_all_wait,
            "avg_waiting_time": avg_wait,
            "congestion_index": congestion_index,
        }

    def reset(self):
        """Resets the intersection state to an empty starting configuration."""
        self.signal_controller.reset()
        self.generator.reset()
        self.active_vehicles.clear()
        self.completed_vehicles.clear()
        self.step_count = 0
        self.vehicles_passed_this_step = 0
        self.total_passed = 0
