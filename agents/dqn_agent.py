"""
Deep Q-Network (DQN) Agent for Traffic Signal Optimization.

Implements Deep Reinforcement Learning with:
1. Deep Q-Network Function Approximator (MLP)
2. Experience Replay Buffer (breaks temporal correlation of experiences)
3. Target Network (stabilizes Q-learning targets and prevents policy divergence)
4. Epsilon-greedy exploration with exponential decay
"""
from collections import deque
import os
import random
from typing import Any, Dict, List, Tuple
import numpy as np

from agents.base_agent import BaseAgent
from utils.config import Config

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class PyTorchQNetwork(nn.Module):
        """Multi-layer Perceptron (MLP) for continuous state -> Q-values mapping."""

        def __init__(self, state_dim: int = 10, action_dim: int = 2, hidden_dim: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


class ReplayBuffer:
    """Experience replay memory storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent with Experience Replay and Target Network.
    """

    def __init__(
        self,
        state_dim: int = 10,
        num_actions: int = 2,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        replay_capacity: int = 20000,
        target_update_freq: int = 10,
        config: Config | None = None,
    ):
        super().__init__(name="DQN Agent")
        self.config = config or Config()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.memory = ReplayBuffer(capacity=replay_capacity)
        self.train_step = 0
        self.losses: List[float] = []

        self.use_torch = TORCH_AVAILABLE
        if self.use_torch:
            self.device = torch.device("cpu")
            self.policy_net = PyTorchQNetwork(state_dim, num_actions, hidden_dim=64).to(self.device)
            self.target_net = PyTorchQNetwork(state_dim, num_actions, hidden_dim=64).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
        else:
            # Simple pure-NumPy weights fallback
            self.w1 = np.random.randn(state_dim, 64) * 0.1
            self.b1 = np.zeros(64)
            self.w2 = np.random.randn(64, num_actions) * 0.1
            self.b2 = np.zeros(num_actions)
            self.lr = learning_rate

    def act(self, observation: Any, info: Dict[str, Any] | None = None, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection."""
        obs = np.asarray(observation, dtype=np.float32)

        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        if self.use_torch:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                return int(np.argmax(q_values))
        else:
            h1 = np.maximum(0, np.dot(obs, self.w1) + self.b1)
            q = np.dot(h1, self.w2) + self.b2
            return int(np.argmax(q))

    def learn(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """Pushes transition into replay buffer and performs batch gradient update."""
        s = np.asarray(state, dtype=np.float32)
        s_next = np.asarray(next_state, dtype=np.float32)

        self.memory.push(s, action, reward, s_next, done)

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        if self.use_torch:
            s_t = torch.FloatTensor(states).to(self.device)
            a_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            r_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            s_next_t = torch.FloatTensor(next_states).to(self.device)
            d_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # Current Q(s, a)
            curr_q = self.policy_net(s_t).gather(1, a_t)

            # Target Q: r + gamma * max_a'(Q_target(s', a')) * (1 - done)
            with torch.no_grad():
                next_q = self.target_net(s_next_t).max(1)[0].unsqueeze(1)
                target_q = r_t + (1.0 - d_t) * self.gamma * next_q

            loss = self.criterion(curr_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.losses.append(float(loss.item()))

            # Update target network
            self.train_step += 1
            if self.train_step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # NumPy mini-batch gradient descent update
            for i in range(len(states)):
                h1 = np.maximum(0, np.dot(states[i], self.w1) + self.b1)
                q_pred = np.dot(h1, self.w2) + self.b2
                target = rewards[i]
                if not dones[i]:
                    h1_next = np.maximum(0, np.dot(next_states[i], self.w1) + self.b1)
                    q_next = np.dot(h1_next, self.w2) + self.b2
                    target += self.gamma * np.max(q_next)

                err = target - q_pred[actions[i]]
                grad_out = np.zeros(self.num_actions)
                grad_out[actions[i]] = -err

                grad_w2 = np.outer(h1, grad_out)
                grad_b2 = grad_out
                grad_h1 = np.dot(grad_out, self.w2.T) * (h1 > 0)
                grad_w1 = np.outer(states[i], grad_h1)
                grad_b1 = grad_h1

                self.w2 -= self.lr * grad_w2
                self.b2 -= self.lr * grad_b2
                self.w1 -= self.lr * grad_w1
                self.b1 -= self.lr * grad_b1

    def decay_epsilon(self):
        """Decays exploration parameter after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Saves model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.use_torch:
            torch.save({
                "policy_state_dict": self.policy_net.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self.train_step,
            }, filepath)
        else:
            np.savez(filepath, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, epsilon=self.epsilon)

    def load(self, filepath: str):
        """Loads model weights."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        if self.use_torch:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.target_net.load_state_dict(checkpoint["policy_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            self.train_step = checkpoint.get("train_step", 0)
        else:
            data = np.load(filepath)
            self.w1 = data["w1"]
            self.b1 = data["b1"]
            self.w2 = data["w2"]
            self.b2 = data["b2"]
            self.epsilon = float(data.get("epsilon", self.epsilon_min))

    def reset(self):
        """Resets per-episode metrics."""
        self.losses.clear()
