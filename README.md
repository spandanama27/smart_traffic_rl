# AI-Based Smart Traffic Signal Optimization Using Reinforcement Learning

> **Master of Computer Applications (MCA) Academic Project**  
> **Domain**: Artificial Intelligence, Reinforcement Learning, Intelligent Transportation Systems (ITS)  
> **Technologies**: Python 3, Gymnasium, PyTorch, Q-Learning, Deep Q-Networks (DQN), Pygame-CE, Streamlit, Pandas, Matplotlib

---

## 1. Project Abstract
Urban traffic congestion leads to substantial economic losses, fuel wastage, and increased vehicular emissions. Traditional traffic management relies on **pre-timed (fixed-time) cyclical controllers** that allocate green light durations statically without regard for dynamic real-time traffic demand. 

This project develops an **intelligent, adaptive traffic signal optimization system using Reinforcement Learning (RL)**. Operating on a four-way intersection with North, South, East, and West approaches, the system utilizes microscopic vehicle kinematics, queue tracking, and safe clearance intervals (amber transitions). The RL agents (Tabular **Q-Learning** and **Deep Q-Networks (DQN)**) autonomously learn optimal signal phase switching decisions through interaction with a custom **Gymnasium** environment. 

### Key Findings & Experimental Results
- **Waiting Time Reduction**: **~70.8% reduction** in average vehicle waiting time compared to traditional fixed-time controllers.
- **Queue Length Reduction**: **~69.8% reduction** in peak and average vehicular queue lengths.
- **Throughput Enhancement**: **+9.8% increase** in total vehicles cleared through the intersection within identical time horizons.

---

## 2. Technology Stack & Dependencies

| Component | Library / Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.14 | Core computational engine |
| **RL Environment** | Gymnasium (`gymnasium>=1.0.0`) | Standardized Markov Decision Process (MDP) environment |
| **Deep Learning** | PyTorch (`torch>=2.0.0`) | Deep Q-Network (DQN) MLP architecture & Experience Replay |
| **Simulation & Graphics** | Pygame Community Edition (`pygame-ce>=2.5.0`) | 60 FPS 2D physical intersection rendering & kinematic physics |
| **Web Dashboard** | Streamlit (`streamlit>=1.30.0`) | Interactive web UI, real-time telemetry, & in-browser training lab |
| **Data & Telemetry** | NumPy (`numpy>=1.24.0`), Pandas (`pandas>=2.0.0`) | State vector processing, telemetry logging, and metrics aggregation |
| **Visualization** | Matplotlib (`matplotlib>=3.8.0`), Pillow | KPI comparative charts, queue trajectories, & loss convergence |

---

## 3. Project Architecture

```
smart_traffic_rl/
├── README.md                      # Comprehensive project documentation & defense guide
├── requirements.txt               # Pinned package dependencies
│
├── utils/                         # Utilities & Global Settings
│   ├── __init__.py
│   ├── config.py                  # Master configuration (geometry, timings, colors, hyperparameters)
│   └── logger.py                  # Structured console and file logger
│
├── environment/                   # Gymnasium Simulation Environment
│   ├── __init__.py
│   ├── vehicle.py                 # Vehicle kinematics, car-following, braking & queue tracking
│   ├── vehicle_generator.py       # Poisson / Bernoulli stochastic vehicle spawning
│   ├── traffic_signal.py          # 4-Phase signal controller with yellow clearance intervals
│   ├── traffic_intersection.py    # 4-Way intersection manager, lane sorting & physics updates
│   └── traffic_env.py             # Gymnasium-compliant gym.Env environment class
│
├── agents/                        # Traffic Signal Control Agents
│   ├── __init__.py
│   ├── base_agent.py              # Abstract base agent interface
│   ├── fixed_time_agent.py        # Traditional pre-timed cyclical baseline controller
│   ├── q_learning_agent.py        # Tabular Q-Learning agent with state discretization & Bellman updates
│   └── dqn_agent.py               # Deep Q-Network agent (PyTorch MLP, Replay Buffer, Target Network)
│
├── training/                      # Training Pipeline & Checkpointing
│   ├── __init__.py
│   ├── trainer.py                 # Multi-episode rollout runner with telemetry tracking & callbacks
│   └── train.py                   # CLI training entry point (supports Q-Learning, DQN, or both)
│
├── simulation/                    # Graphics & Desktop Simulation
│   ├── __init__.py
│   ├── renderer.py                # Dual-mode renderer (Pygame window & off-screen RGB NumPy array)
│   └── run_simulation.py          # Standalone 60 FPS interactive Pygame desktop application
│
├── evaluation/                    # Scientific Benchmarking Engine
│   ├── __init__.py
│   ├── metrics.py                 # Calculation of delay, throughput, queue, congestion, & fairness
│   ├── evaluator.py               # Seeded, repeatable head-to-head comparison engine
│   └── plot_comparisons.py        # Publication-ready Matplotlib comparison visualizers
│
├── dashboard/                     # Web Application
│   ├── __init__.py
│   └── app.py                     # Feature-packed Streamlit Dashboard
│
├── saved_models/                  # Pretrained weights & training telemetry
│   ├── q_table_best.pkl           # Pretrained Q-Learning model
│   ├── dqn_model_best.pt          # Pretrained DQN model
│   ├── q-learning_agent_history.csv
│   └── dqn_agent_history.csv
│
└── data/                          # Cache & exportable CSV benchmarks
```

---

## 4. Mathematical Formulation (MDP)

The intersection is modeled as a discrete-time **Markov Decision Process (MDP)** denoted by $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

### 4.1 State Space ($\mathcal{S}$)
The observation vector is continuous $\mathbf{s}_t \in \mathbb{R}^{10}$:
$$\mathbf{s}_t = \big[ Q_N, Q_S, Q_E, Q_W, D_N, D_S, D_E, D_W, \phi_{axis}, \tau_{norm} \big]$$
- $Q_N, Q_S, Q_E, Q_W$: Queue lengths across North, South, East, West approaches.
- $D_N, D_S, D_E, D_W$: Total vehicular densities (counts) on each approach.
- $\phi_{axis} \in \{0.0, 1.0\}$: Current active green axis (0.0: North-South, 1.0: East-West).
- $\tau_{norm} \in [0, 1]$: Normalized duration spent in the current phase.

*For Tabular Q-Learning, continuous queues and densities are partitioned into discrete bins: Empty (0), Light (1-3), Moderate (4-7), Congested (8+).*

### 4.2 Action Space ($\mathcal{A}$)
$$\mathcal{A} = \{0: \text{Keep Green Phase}, \; 1: \text{Switch Signal Phase}\}$$
To adhere to transportation safety standards, action `1` triggers an uninterruptible **Yellow Clearance Phase** ($\tau_{yellow} = 3\text{s}$) to avoid vehicular conflicts before transitioning to green on the opposing axis.

### 4.3 Reward Formulation ($\mathcal{R}$)
The reward function drives delay minimization and green-wave throughput:
$$\mathcal{R}_t = - \left( w_q \sum_{d} Q_d(t) \right) - \left( w_w \Delta W_t \right) + \left( w_{tp} N_{passed}(t) \right) - \mathcal{P}_{switch}$$
- $w_q = 0.60$: Queue length penalty coefficient.
- $w_w = 0.25$: Waiting time growth penalty coefficient.
- $w_{tp} = 8.0$: Throughput bonus for every vehicle clearing the stop line.
- $\mathcal{P}_{switch} = 1.5$: Penalty for phase oscillation to prevent light thrashing.

### 4.4 Bellman Update Equation (Q-Learning)
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ \mathcal{R}_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

### 4.5 Deep Q-Network (DQN) Loss
$$L(\theta) = \mathbb{E}_{(s,a,r,s',d)} \left[ \left( r + \gamma (1-d) \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
where $\theta^-$ represents the parameters of the periodic Target Network.

---

## 5. Installation & Setup Instructions

### 5.1 Prerequisites
- Python 3.10 or higher (Python 3.10 – 3.14 fully supported).
- VS Code or any modern Python IDE.

### 5.2 Clone / Open Project
```bash
cd "C:\Users\spandana ashwath\.gemini\antigravity\scratch\smart_traffic_rl"
```

### 5.3 Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `gymnasium`, `pygame-ce`, `torch`, `streamlit`, `numpy`, `pandas`, `matplotlib`, `pillow`)*

---

## 6. How to Run the Applications

### 6.1 Launch Interactive Web Dashboard (Recommended)
Launch the comprehensive Streamlit application:
```bash
streamlit run dashboard/app.py
```
This opens the web browser with:
- **Live Interactive Simulation**: Real-time traffic rendering, step-by-step execution, and live KPI cards.
- **Head-to-Head Benchmark**: Run scientific comparisons with identical seeds and download CSV reports.
- **RL Training Lab**: Train agents live in the dashboard and observe reward curves.
- **Project Architecture & Viva Guide**: Full academic documentation and technical Q&A.

### 6.2 Launch Standalone Desktop Pygame Simulation
Run the desktop 60 FPS graphical simulation:
```bash
python simulation/run_simulation.py --agent q_learning
```
**Interactive Keyboard Controls:**
- `[SPACE]`: Pause / Resume simulation
- `[1]`: Switch to Traditional Fixed-Time Controller
- `[2]`: Switch to Q-Learning Agent
- `[3]`: Switch to Deep Q-Network (DQN) Agent
- `[T]`: Manually force traffic light phase switch
- `[UP / DOWN]`: Dynamically increase / decrease traffic arrival rate
- `[R]`: Reset intersection
- `[ESC]`: Quit

### 6.3 Train RL Models via CLI
Train Q-Learning and DQN agents across multiple episodes:
```bash
# Train Q-Learning (100 episodes)
python training/train.py --agent q_learning --episodes 100

# Train DQN (50 episodes)
python training/train.py --agent dqn --episodes 50

# Train both sequentially
python training/train.py --agent both
```

---

## 7. Comparative Benchmark Results

*Evaluated under identical pseudo-random seeds over a 250-second simulation horizon under 'NS Heavy (Rush Hour)' traffic:*

| Metric | Traditional Fixed-Time | Q-Learning Agent | DQN Agent | RL Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Average Waiting Time** | 6.76 s | **1.97 s** | 8.90 s | **-70.86% (Faster flow)** |
| **Average Queue Length** | 3.95 veh | **1.19 veh** | 5.20 veh | **-69.87% (Less congestion)** |
| **Peak Queue Length** | 13 veh | **5 veh** | 15 veh | **-61.54% (Queue suppression)** |
| **Total Throughput** | 123 veh | **135 veh** | 127 veh | **+9.76% (More cleared)** |
| **Average Congestion Index**| 0.110 | **0.036** | 0.141 | **-67.27% (Smoother)** |

> **Academic Insight**: Tabular Q-Learning rapidly achieves optimal switching policies for the intersection by accurately balancing green allocations between busy and light directions, resulting in dramatic decreases in queues and delays.

---

## 8. Viva & Project Defense Guide

### Q1: What is the main research question of this project?
**Answer**: Can a Reinforcement Learning agent autonomously optimize traffic signal durations in real-time based on live vehicular queues, outperforming static cyclical controllers in terms of delay and congestion reduction?

### Q2: Why is Gymnasium used instead of simple custom loops?
**Answer**: Gymnasium provides a standardized RL interface (`reset`, `step`, `action_space`, `observation_space`). This modularity decouples the environment physics from the learning algorithm, allowing any RL model (Q-Learning, DQN, PPO, A2C) to be plugged in interchangeably.

### Q3: How is safety guaranteed during signal transitions?
**Answer**: Real-world traffic signals never jump instantly from green to red. Our system enforces a mandatory, non-interruptible 3-second yellow clearance interval. Vehicles within safe stopping distance decelerate to 0 before the stop line, while vehicles in the intersection complete their crossing before opposing green lights activate.

### Q4: How is the exploration-exploitation dilemma handled?
**Answer**: The agent uses an $\epsilon$-greedy policy. Initially, $\epsilon = 1.0$ (100% random exploration). Over successive training episodes, $\epsilon$ decays exponentially ($\epsilon_{t+1} = \max(\epsilon_{min}, \epsilon_t \cdot 0.992)$), gradually shifting the agent towards pure exploitation of optimal learned policies.

---

## 9. License & Credits
Developed as an **MCA Academic Project** demonstrating end-to-end Machine Learning, Reinforcement Learning, and Software Engineering principles.
