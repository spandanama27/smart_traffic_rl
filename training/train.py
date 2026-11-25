"""
CLI Script for Training Traffic Signal Optimization Agents.
Supports Q-Learning and Deep Q-Network (DQN) with configurable hyperparameters.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config
from utils.logger import setup_logger
from environment.traffic_env import TrafficEnvironment
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from training.trainer import RLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Traffic Signal RL Agents")
    parser.add_argument(
        "--agent",
        type=str,
        default="q_learning",
        choices=["q_learning", "dqn", "both"],
        help="RL algorithm to train",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--preset", type=str, default="Balanced Medium", help="Traffic flow preset")
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "saved_models"), help="Output directory")
    return parser.parse_args()


def train_agent(agent_type: str, episodes: int, steps: int, preset: str, save_dir: str):
    logger = setup_logger(f"Train-{agent_type.upper()}")
    config = Config()

    rates = config.TRAFFIC_PRESETS.get(preset, config.DEFAULT_ARRIVAL_RATES)
    env = TrafficEnvironment(config=config, arrival_rates=rates, max_steps=steps)

    if agent_type == "q_learning":
        agent = QLearningAgent(config=config)
    elif agent_type == "dqn":
        agent = DQNAgent(config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    trainer = RLTrainer(agent=agent, env=env, config=config, save_dir=save_dir, logger=logger)
    df_history = trainer.train(num_episodes=episodes, max_steps_per_episode=steps)
    return df_history


def main():
    args = parse_args()

    if args.agent in ("q_learning", "both"):
        print("\n" + "=" * 60)
        print(">>> Training Q-Learning Agent...")
        print("=" * 60)
        train_agent("q_learning", args.episodes, args.steps, args.preset, args.save_dir)

    if args.agent in ("dqn", "both"):
        print("\n" + "=" * 60)
        print(">>> Training Deep Q-Network (DQN) Agent...")
        print("=" * 60)
        train_agent("dqn", max(50, args.episodes // 2), args.steps, args.preset, args.save_dir)

    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
