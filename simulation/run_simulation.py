"""
Standalone Interactive Desktop Pygame Simulation Runner.
Runs live 60 FPS traffic simulation with real-time HUD and interactive key controls:
- [SPACE]: Pause / Resume simulation
- [R]: Reset intersection
- [1]: Select Fixed-Time Baseline Controller
- [2]: Select Q-Learning RL Agent
- [3]: Select Deep Q-Network (DQN) RL Agent
- [T]: Manually toggle / force signal phase change
- [UP / DOWN]: Increase or decrease traffic arrival density
"""
import argparse
import os
import sys
import pygame
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config
from environment.traffic_env import TrafficEnvironment
from agents.fixed_time_agent import FixedTimeAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from simulation.renderer import IntersectionRenderer


def parse_args():
    parser = argparse.ArgumentParser(description="Run Live Traffic Simulation")
    parser.add_argument(
        "--agent",
        type=str,
        default="q_learning",
        choices=["fixed_time", "q_learning", "dqn"],
        help="Initial controller to run",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="Balanced Medium",
        help="Traffic flow preset",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # Initialize Pygame window
    pygame.init()
    pygame.display.set_caption("Smart Traffic Signal Optimization - Reinforcement Learning")
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # Load agents
    fixed_agent = FixedTimeAgent(cycle_time=25, config=config)

    ql_agent = QLearningAgent(config=config)
    ql_path = PROJECT_ROOT / "saved_models" / "q_table_best.pkl"
    if ql_path.exists():
        ql_agent.load(str(ql_path))

    dqn_agent = DQNAgent(config=config)
    dqn_path = PROJECT_ROOT / "saved_models" / "dqn_model_best.pt"
    if dqn_path.exists():
        dqn_agent.load(str(dqn_path))

    agents = {
        "fixed_time": fixed_agent,
        "q_learning": ql_agent,
        "dqn": dqn_agent,
    }
    current_agent_key = args.agent
    current_agent = agents[current_agent_key]

    # Initialize environment & renderer
    rates = config.TRAFFIC_PRESETS.get(args.preset, config.DEFAULT_ARRIVAL_RATES)
    env = TrafficEnvironment(config=config, arrival_rates=rates, max_steps=100000)
    obs, info = env.reset()

    renderer = IntersectionRenderer(config=config, is_headless=False)

    running = True
    paused = False
    step_metrics = info

    print("\n" + "=" * 65)
    print("AI SMART TRAFFIC SIGNAL SIMULATION RUNNING")
    print("=" * 65)
    print("Controls:")
    print("  [SPACE]      : Pause / Resume")
    print("  [R]          : Reset intersection")
    print("  [1]          : Switch to Fixed-Time Controller")
    print("  [2]          : Switch to Q-Learning Agent")
    print("  [3]          : Switch to DQN Agent")
    print("  [T]          : Force phase toggle")
    print("  [UP / DOWN]  : Adjust traffic density")
    print("  [ESC]        : Quit simulation")
    print("=" * 65 + "\n")

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("[Status]:", "PAUSED" if paused else "RESUMED")
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    step_metrics = info
                    print("[Action]: Simulation reset.")
                elif event.key == pygame.K_1:
                    current_agent_key = "fixed_time"
                    current_agent = fixed_agent
                    print("[Controller Changed]: Traditional Fixed-Time")
                elif event.key == pygame.K_2:
                    current_agent_key = "q_learning"
                    current_agent = ql_agent
                    print("[Controller Changed]: Q-Learning Agent")
                elif event.key == pygame.K_3:
                    current_agent_key = "dqn"
                    current_agent = dqn_agent
                    print("[Controller Changed]: Deep Q-Network (DQN) Agent")
                elif event.key == pygame.K_t:
                    # Force switch
                    obs, reward, term, trunc, step_metrics = env.step(config.ACTION_SWITCH_PHASE)
                    print("[Action]: Forced manual signal toggle.")
                elif event.key == pygame.K_UP:
                    for k in env.intersection.generator.arrival_rates:
                        env.intersection.generator.arrival_rates[k] = min(0.9, env.intersection.generator.arrival_rates[k] + 0.05)
                    print("[Traffic Density Increased]:", env.intersection.generator.arrival_rates)
                elif event.key == pygame.K_DOWN:
                    for k in env.intersection.generator.arrival_rates:
                        env.intersection.generator.arrival_rates[k] = max(0.05, env.intersection.generator.arrival_rates[k] - 0.05)
                    print("[Traffic Density Decreased]:", env.intersection.generator.arrival_rates)

        # Simulation step
        if not paused:
            is_ql = isinstance(current_agent, QLearningAgent)
            state = env.discretize_state(obs) if is_ql else obs
            action = current_agent.act(state, step_metrics, deterministic=True)
            obs, reward, term, trunc, step_metrics = env.step(action)

        # Render frame
        rendered_surface = renderer.render(
            intersection=env.intersection,
            agent_name=current_agent.name,
            metrics=step_metrics,
        )
        screen.blit(rendered_surface, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
