# Reinforcement Learning Training Plan for Ungroup

## Executive Summary

This document outlines a comprehensive plan for training RL agents to play Ungroup using self-play on an RTX 4090 GPU system. The game's unique mechanics of temporary alliances, resource gathering, and multi-agent dynamics make it an excellent candidate for advanced RL techniques.

**Estimated Time to Strong Performance:** 24-72 hours of training
**Estimated Time to Expert-Level Play:** 1-2 weeks of continuous training
**Hardware Utilization:** ~80-95% GPU, high CPU usage for parallel environments

---

## 1. Game Analysis

### 1.1 Game Characteristics

**Game Type:** Competitive Multi-Agent Resource Gathering with Dynamic Alliances

**Core Mechanics:**
- **Objective:** First player to collect 10 of each resource type (RED, GREEN, BLUE, YELLOW) wins
- **Resources:** 4 mines, each containing one resource type (40 total resources)
- **Grouping:** Players can form temporary groups by toggling "joinable" status
- **Group Benefits:** Likely shared movement/resources (requires further code analysis)
- **Player Count:** 2-20 players (default max: 20)
- **Game Speed:** 60 FPS simulation

**Key Strategic Elements:**
1. **Resource Competition:** Limited resources create zero-sum competition
2. **Alliance Formation:** Deciding when to group vs. compete alone
3. **Betrayal Timing:** When to leave groups for individual advantage
4. **Spatial Navigation:** Efficient pathfinding to mines
5. **Opponent Modeling:** Predicting other players' intentions via their declared intent

### 1.2 Why This Game is Great for RL

✅ **Clear Reward Signal:** Win/loss + resource collection progress
✅ **Multi-Agent Complexity:** Requires modeling other agents' behavior
✅ **Strategic Depth:** Short-term tactics vs. long-term strategy
✅ **Emergent Behavior:** Grouping mechanics enable complex social dynamics
✅ **Fast Simulation:** 60 FPS = many timesteps for training
✅ **Existing Baselines:** NearestGreedy bot provides baseline opponent

---

## 2. State and Action Space Design

### 2.1 State Space (Observation)

**Option A: Feature-Based Representation (Recommended for Initial Training)**

```python
State Vector (per agent):
  # Self state (10 dimensions)
  - Position (x, y)                          # 2D
  - Velocity/Direction (vx, vy)              # 2D
  - Resources collected [R, G, B, Y]         # 4D
  - Is in group (binary)                     # 1D
  - Current intent (one-hot encoded)         # 4D (but could be 1D with int)

  # Mine state (4 mines × 5 dims = 20 dimensions)
  For each mine:
    - Relative position (dx, dy)             # 2D
    - Resource type (one-hot or int)         # 4D or 1D
    - Remaining resources (estimated)        # 1D
    - Distance to mine                       # 1D

  # Other players (top-k nearest, e.g., k=8)
  For each visible player (8 players × 9 dims = 72 dimensions):
    - Relative position (dx, dy)             # 2D
    - Relative velocity (dvx, dvy)           # 2D
    - Resources [R, G, B, Y]                 # 4D
    - Is joinable (binary)                   # 1D
    - Declared intent                        # 1D
    - Is in my group (binary)                # 1D
    - Distance                               # 1D

  # Group state (if in group) (10 dimensions)
  - Group size                               # 1D
  - Group centroid relative position         # 2D
  - Group total resources [R, G, B, Y]       # 4D
  - Average group intent                     # 4D (or 1D)

  Total: ~112-150 dimensions (depending on encoding choices)
```

**Option B: Visual Representation (For Advanced Training)**

```python
Top-down 2D image:
  - Resolution: 84x84 or 128x128 pixels
  - Channels:
    * Players (positions, colored by resources/intent)
    * Mines (positions, colored by type)
    * Groups (marked regions)
    * Resources (heat map)
  - Stack: 4 frames for temporal information

  Total: 4 × 84 × 84 = ~28k dimensions (but CNN reduces this)
```

**Recommendation:** Start with **Option A** (feature-based) for faster training, then explore Option B for transfer learning or human-like perception.

### 2.2 Action Space

**IMPORTANT:** Since this is a **continuous state space** game (positions, velocities are continuous), continuous actions are the natural choice.

**Continuous Action Space (RECOMMENDED):**

```python
Action Vector (6 dimensions):
  # Movement (continuous)
  - move_x: [-1.0, 1.0]           # Horizontal movement (-1=left, +1=right)
  - move_y: [-1.0, 1.0]           # Vertical movement (-1=up, +1=down)

  # Social actions (continuous, thresholded)
  - toggle_joinable: [-1.0, 1.0]  # >0.5 = toggle joinable
  - toggle_ungroup: [-1.0, 1.0]   # >0.5 = toggle ungroup
  - toggle_intent: [-1.0, 1.0]    # >0.5 = cycle intent
  - toggle_stop: [-1.0, 1.0]      # >0.5 = stop moving

  # All actions use tanh activation (outputs in [-1, 1])
```

**Alternative: Hybrid Action Space (Continuous + Discrete):**

```python
Continuous actions (2D):
  - move_x: [-1.0, 1.0]
  - move_y: [-1.0, 1.0]

Discrete actions (4 binary):
  - toggle_joinable: {0, 1}
  - toggle_ungroup: {0, 1}
  - toggle_intent: {0, 1}
  - toggle_stop: {0, 1}
```

**Fully Discrete Option (Simpler but less expressive):**

```cpp
// 8 discrete actions (for simpler implementation)
0: UP, 1: DOWN, 2: RIGHT, 3: LEFT, 4: STOP
5: TOGGLE_JOINABLE, 6: TOGGLE_UNGROUP, 7: TOGGLE_INTENT
```

**Recommendation:** Use **Continuous Action Space** (6D continuous) for best performance in continuous state space. PPO and SAC both handle this excellently.

### 2.3 Reward Shaping

**Sparse Reward (Baseline):**
```python
reward = {
    +1000  if win,
    -1000  if lose (someone else wins),
    0      otherwise
}
```

**Dense Reward (Recommended for Faster Learning):**
```python
reward = 0

# Resource collection rewards
for resource_type in [R, G, B, Y]:
    reward += 10 * (new_resources[type] - old_resources[type])

# Proximity rewards (small, to encourage exploration)
if moving_toward_needed_mine:
    reward += 0.1

# Group formation rewards (if beneficial)
if joined_beneficial_group:
    reward += 5
if left_detrimental_group:
    reward += 3

# Win/loss terminal rewards
if won:
    reward += 1000
if lost:
    reward -= 100  # Smaller penalty to encourage exploration

# Time penalty (encourage faster wins)
reward -= 0.01  # Small penalty per timestep
```

**Curriculum Reward:** Start with dense rewards, gradually shift to sparse as agent improves.

---

## 3. Self-Play Architecture

### 3.1 Why Self-Play?

Self-play is **essential** for this game because:
- **No expert data:** No human gameplay to imitate
- **Opponent modeling:** Agents learn to exploit and defend against evolving strategies
- **Emergent strategies:** Complex grouping behaviors emerge from competition
- **Robustness:** Trained agents handle diverse opponent strategies

### 3.2 Self-Play Framework Options

**Option 1: League Training (AlphaStar-style) - RECOMMENDED**

```
Main Exploiter Pool (40% of games):
  ├─ Current best agent (always)
  └─ Top-3 recent checkpoints

Main Agents Pool (40% of games):
  ├─ Current training agent
  └─ Last 10 checkpoints (weighted sampling)

League Exploiters (10% of games):
  └─ Specialist agents trained to beat main agents

Historical Agents (10% of games):
  └─ Random agents from past 100 checkpoints
```

**Matchmaking:**
- 70% vs. agents from similar skill level (Elo-based)
- 20% vs. stronger agents (aspirational)
- 10% vs. weaker agents (confidence building)

**Option 2: Population-Based Training (PBT)**

```
Population of 20 agents:
  - Each trains independently with different hyperparameters
  - Periodically, weak agents copy weights from strong agents
  - Hyperparameters mutated after copying
  - Ensures diversity in strategies
```

**Option 3: Simple Self-Play (Baseline)**

```
Rolling window of past N checkpoints:
  - Save checkpoint every K episodes
  - Sample opponent uniformly from last 50 checkpoints
  - Simpler but less robust than league training
```

**Recommendation:** Start with **Option 3**, transition to **Option 1** after initial convergence (~100k games).

### 3.3 Multi-Agent Training

Each episode runs with **N players** (recommended: 4-8 for faster training, scale to 20 later).

**Parallel Environment Setup:**
```python
# Vectorized environments for maximum throughput
num_parallel_envs = 64  # RTX 4090 can handle this
players_per_env = 4     # Start small, increase to 8-20

# Total concurrent agents
total_agents = 64 × 4 = 256 agents playing simultaneously
```

**Agent Assignment:**
- 50% current training agent (playing against itself + past versions)
- 30% recent checkpoints (last 10-20)
- 20% scripted bots (NearestGreedy, Random) for baseline

---

## 4. Algorithm Selection

### 4.1 Top Candidates

**For Continuous State + Continuous Action Space:**

| Algorithm | Pros | Cons | Suitability |
|-----------|------|------|-------------|
| **PPO** | Stable, proven for games, handles continuous actions excellently | Slightly less sample efficient than off-policy | ⭐⭐⭐⭐⭐ **BEST** |
| **SAC** | Sample efficient, continuous action expert, stable | Requires larger replay buffer (memory) | ⭐⭐⭐⭐⭐ |
| **TD3** | Very sample efficient, stable, deterministic | Less exploration than SAC | ⭐⭐⭐⭐ |
| **IMPALA** | Extremely scalable, continuous action support | Complex implementation | ⭐⭐⭐⭐ |
| **MAPPO** | Multi-agent specialist, continuous actions | Newer, less mature tooling | ⭐⭐⭐⭐ |
| **A2C/A3C** | Fast, simple | Less stable than PPO | ⭐⭐⭐ |

**Note:** DQN/Rainbow are excluded as they only work with discrete actions.

### 4.2 Recommended Algorithm: PPO (Proximal Policy Optimization)

**Why PPO?**
- ✅ **Proven:** Used in Dota 2 (OpenAI Five), StarCraft (AlphaStar)
- ✅ **Stable:** Clipped objective prevents destructive policy updates
- ✅ **Continuous Actions:** Outputs Gaussian distribution over action space
- ✅ **Sample Efficient:** Reuses data via multiple epochs per batch
- ✅ **Multi-Agent Ready:** Easy to scale to many agents
- ✅ **Mature Libraries:** Stable-Baselines3, RLlib, CleanRL

**PPO Hyperparameters (Starting Point for Continuous Actions):**

```python
# Core PPO parameters
learning_rate = 3e-4        # Can be scheduled (linear decay)
n_steps = 2048              # Steps per environment before update
batch_size = 64             # Minibatch size for SGD
n_epochs = 10               # Epochs per update
gamma = 0.99                # Discount factor
gae_lambda = 0.95           # GAE parameter for advantage estimation
clip_range = 0.2            # PPO clip parameter
ent_coef = 0.01             # Entropy coefficient (exploration)
vf_coef = 0.5               # Value function coefficient

# Network architecture (for continuous actions)
policy_layers = [256, 256, 128]   # Actor network (outputs mean + log_std)
value_layers = [256, 256, 128]    # Critic network
activation = "relu"               # or "tanh"

# Continuous action specific
action_std_init = 0.5       # Initial standard deviation for action distribution
```

### 4.3 Alternative: SAC (Soft Actor-Critic) - For Maximum Sample Efficiency

**Why SAC?**
- ✅ **Sample Efficient:** Off-policy algorithm, reuses all past experience
- ✅ **Continuous Action Specialist:** Designed specifically for continuous control
- ✅ **Stable:** Entropy regularization prevents premature convergence
- ✅ **No Clipping:** Uses soft value functions (less hyperparameter sensitive)

**When to use SAC over PPO:**
- If environment simulation is **slow** (SAC is 2-3× more sample efficient)
- If you have **enough RAM** for replay buffer (1M+ transitions ≈ 5-10 GB)
- If you want **smoother policies** (SAC outputs are less noisy)

**SAC Hyperparameters:**

```python
# Core SAC parameters
learning_rate = 3e-4
buffer_size = 1_000_000     # Replay buffer (needs RAM!)
batch_size = 256            # Larger than PPO
tau = 0.005                 # Soft update coefficient
gamma = 0.99
train_freq = 1              # Update every step (off-policy)
gradient_steps = 1          # Gradient steps per env step

# Network architecture
policy_layers = [256, 256]
q_network_layers = [256, 256]
activation = "relu"

# SAC specific
ent_coef = "auto"          # Automatic entropy tuning (recommended)
target_update_interval = 1
```

### 4.4 Alternative: IMPALA for Maximum Throughput

If game simulation is the bottleneck (CPU-bound), **IMPALA** (Importance Weighted Actor-Learner Architecture) decouples acting and learning:

```
Actor processes (CPU-heavy): 16-32 cores
  ├─ Run game simulations
  └─ Send trajectories to learner

Learner process (GPU-heavy): RTX 4090
  ├─ Receives trajectories in queue
  └─ Updates network with V-trace off-policy correction
```

**Benefits:** Can max out CPU and GPU independently.

---

## 5. Implementation Plan

### 5.1 Technology Stack

**Core Framework:**
```
Python 3.10+
├─ PyTorch 2.0+ (CUDA 12.x for RTX 4090)
├─ Stable-Baselines3 (SB3) or RLlib
├─ Gymnasium (OpenAI Gym successor)
└─ Weights & Biases (W&B) for experiment tracking
```

**C++ Game Interface:**
```
C++11 (existing codebase)
├─ pybind11 for Python bindings
├─ Headless mode (no SFML rendering)
└─ Vectorized step() for parallel envs
```

**Optional Accelerations:**
```
- EnvPool (for faster environment vectorization)
- JAX (for JIT compilation of environment logic)
- NVIDIA Warp (for GPU-accelerated physics)
```

### 5.2 Development Phases

**Phase 1: Infrastructure (1 week)**

1. **C++ Gym Wrapper**
   - [ ] Create `UngroupEnv` class inheriting from `gym.Env`
   - [ ] Implement headless game mode (disable SFML rendering)
   - [ ] Add Python bindings with pybind11
   - [ ] Implement `reset()`, `step()`, observation/action mapping
   - [ ] Vectorize for parallel environments

2. **Baseline Testing**
   - [ ] Verify environment works with random policy
   - [ ] Test against scripted bots (NearestGreedy)
   - [ ] Profile simulation speed (target: 1000+ FPS per env)

**Phase 2: Single-Agent RL (1 week)**

3. **PPO Training Loop**
   - [ ] Implement PPO agent with SB3
   - [ ] Set up reward shaping
   - [ ] Configure hyperparameters
   - [ ] Train against scripted bots only

4. **Evaluation & Debugging**
   - [ ] Tensorboard/W&B logging
   - [ ] Win rate vs. Random bot
   - [ ] Win rate vs. NearestGreedy bot
   - [ ] Debug observation/action space issues

**Phase 3: Self-Play (2 weeks)**

5. **Self-Play Infrastructure**
   - [ ] Checkpoint saving system (every N episodes)
   - [ ] Opponent sampling from checkpoint pool
   - [ ] Elo rating system for matchmaking
   - [ ] Multi-agent episode rollout

6. **League Training (Optional but Recommended)**
   - [ ] Implement main agent pool
   - [ ] Implement exploiter pool
   - [ ] Implement historical pool
   - [ ] Matchmaking algorithm

**Phase 4: Scaling & Optimization (1 week)**

7. **Performance Optimization**
   - [ ] Profile bottlenecks (CPU vs. GPU)
   - [ ] Increase parallel environments (target: 64-128)
   - [ ] Optimize C++ game code for headless speed
   - [ ] Implement efficient batching

8. **Hyperparameter Tuning**
   - [ ] Grid search / Optuna for key hyperparameters
   - [ ] Learning rate scheduling
   - [ ] Reward shaping adjustments
   - [ ] Network architecture experiments

**Phase 5: Advanced Techniques (2+ weeks)**

9. **Curriculum Learning**
   - [ ] Start with 2 players, increase to 4, 8, 20
   - [ ] Start with dense rewards, shift to sparse
   - [ ] Progressive difficulty (bot strength)

10. **Population-Based Training**
    - [ ] Run 10-20 agents with different hyperparameters
    - [ ] Implement PBT evolutionary operators
    - [ ] Track diversity metrics

### 5.3 Code Structure

```
ungroup_rl/
├── envs/
│   ├── __init__.py
│   ├── ungroup_env.py          # Main Gym environment
│   ├── wrappers.py              # Observation/reward wrappers
│   └── vec_env.py               # Vectorized environment
├── agents/
│   ├── __init__.py
│   ├── ppo_agent.py             # PPO implementation (or use SB3)
│   ├── network.py               # Actor-Critic networks
│   └── checkpoint_manager.py    # Save/load checkpoints
├── self_play/
│   ├── __init__.py
│   ├── league.py                # League training logic
│   ├── matchmaking.py           # Elo-based matchmaking
│   └── opponent_pool.py         # Checkpoint management
├── training/
│   ├── __init__.py
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation against baselines
│   └── config.py                # Hyperparameter configs
├── bindings/
│   ├── ungroup_binding.cpp      # pybind11 C++ bindings
│   └── headless_game.cpp        # Headless game implementation
├── scripts/
│   ├── run_training.sh          # Launch training job
│   ├── tensorboard.sh           # Start Tensorboard
│   └── evaluate_checkpoint.py   # Eval a single checkpoint
└── configs/
    ├── ppo_default.yaml         # Default PPO config
    └── league_config.yaml       # League training config
```

---

## 6. Hardware Optimization for RTX 4090

### 6.1 RTX 4090 Specifications

```
GPU: NVIDIA GeForce RTX 4090
├─ CUDA Cores: 16,384
├─ Tensor Cores: 512 (4th gen)
├─ VRAM: 24 GB GDDR6X
├─ Memory Bandwidth: 1,008 GB/s
├─ FP32 Performance: 82.6 TFLOPS
└─ TF32 Performance: 165 TFLOPS (deep learning)
```

**System Specs:**
- **RAM:** 32 GB (sufficient for most workloads)
- **CPU:** High-end AMD (Ryzen 9 7950X or similar, assuming 16 cores)
- **Storage:** SSD recommended for checkpoint I/O

### 6.2 Bottleneck Analysis

| Component | Load | Bottleneck? | Mitigation |
|-----------|------|-------------|------------|
| **GPU** | 80-95% | Unlikely | Increase batch size, bigger network |
| **CPU** | **HIGH** | **LIKELY** | More parallel envs, optimize C++ |
| **RAM** | 10-20 GB | Unlikely | Reduce replay buffer size |
| **Disk** | Low | Unlikely | Use SSD for checkpoints |

**Expected Bottleneck:** CPU running game simulations.

**Why?**
- 64 parallel environments × 4 players = 256 agents
- Each env runs physics, collisions, resource logic at 60 FPS
- This is **CPU-bound** even with optimized C++ code

**Solutions:**
1. **Optimize C++ game loop** for headless mode (no rendering)
2. **Reduce physics fidelity** if not critical (lower tickrate)
3. **Use faster languages** for environment (consider JAX rewrite for GPU physics)
4. **Distributed training** across multiple machines (overkill for RTX 4090)

### 6.3 Memory Budget

**GPU Memory (24 GB available):**

```python
# Policy network (example)
input_dim = 150 (observation)
hidden_layers = [256, 256, 128]
output_dim = 8 (actions)

# Estimated size: ~500 KB per network (actor + critic)
# Batch size: 2048 × 64 envs = 131k transitions
# Memory for batch: 150 dims × 131k × 4 bytes = 78 MB

Total GPU usage:
├─ Network weights: ~1 MB
├─ Optimizer state (Adam): ~2 MB
├─ Batch data: ~100 MB
├─ Activations/gradients: ~500 MB
└─ Overhead: ~1 GB

Expected: 2-3 GB for training
Remaining: 21 GB free (can run 8× bigger batch or networks!)
```

**Recommendation:** You have **plenty of headroom** to:
- Increase batch size to 4096+ for more stable updates
- Use larger networks (512, 512, 256 layers)
- Run multiple experiments in parallel (4-8 training jobs)

**RAM Usage (32 GB available):**

```python
# Parallel environments
64 envs × 4 players × 150 dims × 4 bytes = 153 KB (negligible)

# Replay buffer (for off-policy methods like SAC, not PPO)
# PPO doesn't need this, so minimal RAM usage

# Checkpoint storage
100 checkpoints × 10 MB each = 1 GB

Total RAM usage: ~5-10 GB
Remaining: 22 GB free (plenty)
```

### 6.4 Performance Targets

**Environment Simulation:**
```
Target: 1000 FPS per environment (headless)
With 64 parallel envs: 64,000 agent steps/sec
Daily samples: 64k × 86400 sec = 5.5 billion steps/day
```

**RL Training Throughput:**
```
PPO update frequency: every 2048 steps × 64 envs = 131k steps
Time per update: ~5-10 seconds (GPU forward/backward)
Updates per hour: 3600 / 7.5 = 480 updates
Daily updates: 11,520 updates

Total samples processed: 11,520 × 131k = 1.5 billion steps/day
```

**Expected Training Speed:**
- With optimized C++ env: **1-2 billion steps per day**
- Each game lasts ~500-2000 timesteps (estimate)
- **500k - 2M games per day**

---

## 7. Training Time Estimates

### 7.1 Sample Complexity Estimates

Based on similar multi-agent competitive games:

| Milestone | Games | Timesteps | Wall Time (RTX 4090) |
|-----------|-------|-----------|----------------------|
| **Beat Random Bot** | 100k | 50M | 12 hours |
| **Beat NearestGreedy** | 500k | 250M | 2-3 days |
| **Stable Self-Play** | 2M | 1B | 1 week |
| **Strong Strategic Play** | 10M | 5B | 3-4 weeks |
| **Expert-Level** | 50M+ | 25B+ | 2-3 months |

**Factors Affecting Training Time:**
- **Number of players:** More players = harder (8-20 players adds complexity)
- **Reward shaping:** Dense rewards train 2-5× faster than sparse
- **Hyperparameters:** Well-tuned can be 2× faster
- **Environment speed:** Optimized C++ is critical

### 7.2 Conservative Estimate (Your Hardware)

**Assumptions:**
- 64 parallel environments
- 4 players per game
- 1000 FPS per env (achievable with headless C++)
- PPO with dense rewards

**Timeline:**

```
Phase 1: Infrastructure (1 week)
├─ C++ wrapper, Python bindings
└─ Baseline testing

Phase 2: Initial Training (3-5 days)
├─ Train vs. scripted bots
├─ Achieve 80% win rate vs. NearestGreedy
└─ ~500M timesteps

Phase 3: Self-Play Ramp-Up (1-2 weeks)
├─ Implement self-play
├─ Train to stable performance
└─ ~2B timesteps

Phase 4: Advanced Self-Play (2-4 weeks)
├─ League training
├─ Discover complex strategies (grouping, betrayal)
└─ ~10B timesteps

Phase 5: Expert Polish (ongoing)
├─ Hyperparameter tuning
├─ Population diversity
└─ 25B+ timesteps
```

**Total Time to "Really Good and Interesting" Agents:**
- **Minimum (with perfect execution):** 4 weeks
- **Expected (realistic):** 6-8 weeks
- **Conservative (with debugging):** 10-12 weeks

### 7.3 Accelerated Timeline (Aggressive Optimization)

If you optimize aggressively:

1. **Rewrite environment in JAX** (GPU-accelerated physics): 10× faster simulation
2. **Use IMPALA distributed training:** 2× sample efficiency
3. **Aggressive hyperparameter tuning:** 1.5× faster convergence

**Accelerated Timeline:**
- **Strong agents:** 2-3 weeks
- **Expert agents:** 4-6 weeks

**But:** Requires significantly more engineering effort (2-3 weeks upfront).

---

## 8. Expected Emergent Behaviors

Based on the game mechanics, expect to see:

### 8.1 Early Training (0-500M steps)
- Random wandering → directed movement to nearest mine
- Ignoring other players
- No grouping (too complex to discover early)
- Simple greedy resource collection

### 8.2 Mid Training (500M - 2B steps)
- Efficient pathfinding to needed resources
- Awareness of other players (collision avoidance)
- Primitive grouping (accidental or rule-based)
- Intent signaling (cycling to needed resource)

### 8.3 Late Training (2B - 10B steps)
- **Purposeful alliance formation** with nearby players
- **Timing of betrayals** (leave group when resources secured)
- **Blocking behaviors** (prevent opponents from reaching mines)
- **Intent deception** (signal false intent to mislead)

### 8.4 Expert Level (10B+ steps)
- **Complex coordination** (groups of 3-4 players sweeping mines)
- **Counter-strategies** (exploiting predictable opponents)
- **Endgame tactics** (deny last resources to leading opponent)
- **Meta-game evolution** (rock-paper-scissors strategy cycles)

---

## 9. Monitoring and Evaluation

### 9.1 Key Metrics to Track

**Training Metrics (log every 1000 steps):**
```python
# Policy metrics
- policy_loss
- value_loss
- entropy (measure of exploration)
- explained_variance (how well value function predicts returns)
- kl_divergence (policy change magnitude)

# Reward metrics
- mean_episode_reward
- std_episode_reward
- max_episode_reward
- episode_length

# Game-specific metrics
- win_rate (vs. different opponent types)
- resources_collected_per_episode [R, G, B, Y]
- times_grouped_per_episode
- times_ungrouped_per_episode
- distance_traveled
```

**Evaluation Metrics (every 10k games):**
```python
# Head-to-head performance
- win_rate_vs_random (should be 95%+)
- win_rate_vs_nearest_greedy (target: 80%+)
- win_rate_vs_self (should be ~50% if agent stable)
- win_rate_vs_historical_checkpoints (Elo tracking)

# Strategic diversity
- unique_strategies_discovered (clustering of game trajectories)
- grouping_frequency (% of time in groups)
- betrayal_timing (when agents ungroup relative to resource collection)
```

**Visual Debugging (manual review):**
- Record agent gameplay videos every 50k games
- Watch for:
  - Efficient navigation
  - Logical grouping decisions
  - Complex multi-step strategies

### 9.2 Experiment Tracking with Weights & Biases

```python
import wandb

wandb.init(
    project="ungroup-rl",
    config={
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_envs": 64,
        "players_per_env": 4,
        # ... all hyperparameters
    }
)

# Log during training
wandb.log({
    "train/mean_reward": mean_reward,
    "train/win_rate": win_rate,
    "eval/elo_rating": elo,
})

# Log videos
wandb.log({"gameplay": wandb.Video("episode.mp4")})
```

### 9.3 Elo Rating System

Track agent strength over time:

```python
from elo import EloRating

# Initialize all agents at 1500 Elo
elo_system = EloRating(k_factor=32)

# After each game
if agent_won:
    new_elo = elo_system.update(agent_elo, opponent_elo, result=1.0)
else:
    new_elo = elo_system.update(agent_elo, opponent_elo, result=0.0)

# Plot Elo over training time
# Target: reach 2000+ Elo vs. baseline bots
```

---

## 10. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **CPU bottleneck** | Slow training | HIGH | Optimize C++, reduce physics fidelity |
| **Policy collapse** | Agent gets stuck | MEDIUM | Entropy regularization, population diversity |
| **Overfitting to self** | Exploitable | MEDIUM | League training, diverse opponents |
| **Sparse rewards** | No learning signal | LOW | Dense reward shaping |
| **Hyperparameter sensitivity** | Wasted time | MEDIUM | Grid search, Optuna tuning |
| **Mode collapse** | All agents converge to same strategy | MEDIUM | PBT, novelty search |

---

## 11. Advanced Extensions (Future Work)

### 11.1 Visual Observations
- Replace feature vector with 84×84 image (top-down view)
- Use CNN encoder (like DQN Atari)
- Benefit: More human-like perception, potential for transfer learning

### 11.2 Hierarchical RL
- High-level policy: Choose macro-strategy (e.g., "form group", "solo mine")
- Low-level policy: Execute movement/actions
- Benefit: Faster learning of complex strategies

### 11.3 Communication Channels
- Add explicit communication actions (e.g., "request group", "warn danger")
- Agents learn language-like protocols
- Benefit: Richer social dynamics

### 11.4 Opponent Modeling
- Explicitly model other agents' policies (Theory of Mind)
- Use opponent observations to predict actions
- Benefit: Better counter-strategies

### 11.5 Transfer to Larger Arenas
- Train on small 4-player games
- Transfer to 20-player games
- Benefit: Faster scaling, test generalization

---

## 12. Implementation Checklist

### Phase 1: Environment (Week 1)

- [ ] Disable SFML rendering, create headless mode
- [ ] Implement `UngroupEnv(gym.Env)` Python class
- [ ] Add pybind11 bindings for C++ game
- [ ] Define observation space (feature vector)
- [ ] Define action space (8 discrete actions)
- [ ] Implement `reset()` and `step()` methods
- [ ] Test with random policy (100k steps)
- [ ] Profile environment FPS (target: 1000+)
- [ ] Implement vectorized environments (64 parallel)
- [ ] Verify multi-agent rollouts work

### Phase 2: Baseline Agent (Week 2)

- [ ] Install Stable-Baselines3 + PyTorch
- [ ] Create PPO agent with default hyperparameters
- [ ] Implement dense reward shaping
- [ ] Train against Random bot (target: 95% win rate)
- [ ] Train against NearestGreedy bot (target: 60% win rate)
- [ ] Set up Tensorboard logging
- [ ] Log key metrics (reward, win rate, episode length)
- [ ] Debug observation/action bugs
- [ ] Save first checkpoint

### Phase 3: Self-Play (Weeks 3-4)

- [ ] Implement checkpoint saving (every 10k games)
- [ ] Create opponent pool (last 50 checkpoints)
- [ ] Implement opponent sampling (uniform random)
- [ ] Run self-play training (1M games)
- [ ] Track win rate vs. historical self
- [ ] Implement Elo rating system
- [ ] Visualize emergent strategies (record videos)
- [ ] Tune hyperparameters (learning rate, entropy)

### Phase 4: League Training (Weeks 5-6)

- [ ] Implement main agent pool
- [ ] Implement exploiter pool (copy & train vs. main)
- [ ] Implement historical pool (random past checkpoints)
- [ ] Create matchmaking algorithm (Elo-based)
- [ ] Run league training (5M games)
- [ ] Monitor Elo ratings of all pools
- [ ] Identify exploiters (agents that beat main)
- [ ] Incorporate successful exploiters into main pool

### Phase 5: Optimization (Weeks 7-8)

- [ ] Profile bottlenecks (cProfile, nvprof)
- [ ] Optimize C++ game loop (reduce allocations)
- [ ] Increase parallel envs to 128 (if CPU allows)
- [ ] Hyperparameter grid search (Optuna)
- [ ] Test larger networks (512, 512, 256)
- [ ] Implement curriculum learning (2→4→8→20 players)
- [ ] Run 10M+ games for final training
- [ ] Evaluate final agent vs. all baselines

### Phase 6: Analysis (Week 9+)

- [ ] Analyze discovered strategies (clustering)
- [ ] Identify failure modes (when agent loses badly)
- [ ] Create highlight reel of best gameplay
- [ ] Write technical report on findings
- [ ] Open-source code + trained models (optional)

---

## 13. Cost-Benefit Analysis

### 13.1 Time Investment

| Task | Engineering Time | Training Time (Wall Clock) |
|------|------------------|----------------------------|
| Environment setup | 20-40 hours | - |
| Baseline agent | 10-20 hours | 2-3 days |
| Self-play | 20-30 hours | 1-2 weeks |
| League training | 30-40 hours | 2-4 weeks |
| Optimization | 20-40 hours | 1-2 weeks |
| **Total** | **100-170 hours** | **6-10 weeks** |

**Calendar Time:** 2-3 months (with full-time focus), 4-6 months (part-time)

### 13.2 Expected Outcomes

**Success Criteria:**
1. ✅ Agent beats NearestGreedy 80%+ of the time
2. ✅ Agent discovers grouping/ungrouping strategies
3. ✅ Agent shows complex behaviors (betrayal timing, blocking)
4. ✅ Emergent meta-game (strategy diversity)

**Failure Cases:**
- ❌ CPU bottleneck prevents enough training samples
- ❌ Policy collapse (all agents converge to same strategy)
- ❌ Reward shaping doesn't capture game complexity

**Risk Assessment:** **LOW** - PPO + self-play is a proven approach for this class of game.

---

## 14. Recommended Next Steps

### Immediate (This Week):
1. **Set up development environment:**
   ```bash
   # Create Python virtual environment
   python3.10 -m venv ungroup_rl_env
   source ungroup_rl_env/bin/activate

   # Install dependencies
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install stable-baselines3[extra] gymnasium pybind11 wandb
   ```

2. **Create headless game mode:**
   - Modify `src/client/main.cpp` to accept `--headless` flag
   - Disable SFML window creation and rendering
   - Expose step function via pybind11

3. **Prototype basic Gym environment:**
   ```python
   import gymnasium as gym

   class UngroupEnv(gym.Env):
       def __init__(self):
           self.action_space = gym.spaces.Discrete(8)
           self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(150,))

       def reset(self):
           # Reset C++ game
           pass

       def step(self, action):
           # Call C++ step, return (obs, reward, done, info)
           pass
   ```

### Short-Term (Next 2 Weeks):
4. **Benchmark environment speed** (target: 1000 FPS)
5. **Train first PPO agent** against Random bot
6. **Iterate on reward shaping** until agent shows basic competence
7. **Implement checkpoint saving** and opponent pool

### Medium-Term (Next 1-2 Months):
8. **Run self-play training** to 1M games
9. **Implement league training** if initial results promising
10. **Scale to 8-20 players** per game
11. **Hyperparameter tuning** via Optuna

### Long-Term (3+ Months):
12. **Train to expert level** (10M+ games)
13. **Analyze emergent strategies**
14. **Create demo videos** and write up findings
15. **Consider open-sourcing** trained agents

---

## 15. Conclusion

**Summary:**
- **Feasibility:** HIGH - Your RTX 4090 system is **more than capable** of training strong RL agents
- **Timeline:** 6-10 weeks to achieve "really good and interesting" gameplay
- **Bottleneck:** CPU (game simulation), not GPU
- **Recommended Approach:** PPO + self-play + league training
- **Expected Results:** Emergent grouping, betrayal, and complex multi-agent strategies

**Key Success Factors:**
1. ✅ Optimize C++ game for headless mode (target: 1000 FPS)
2. ✅ Implement dense reward shaping for faster learning
3. ✅ Use self-play from the start (even simple version)
4. ✅ Monitor training with W&B and Elo ratings
5. ✅ Be patient - RL takes time, but results are worth it!

**Final Recommendation:**
**This project is highly feasible and exciting!** The game's mechanics (grouping, betrayal, resource competition) are perfect for discovering emergent multi-agent behaviors. With disciplined engineering and your powerful hardware, you should see impressive results within 2 months.

**Go for it! 🚀**

---

## Appendix A: Reference Papers

1. **PPO:** Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. **AlphaStar:** Vinyals et al., "Grandmaster level in StarCraft II using multi-agent RL" (2019)
3. **OpenAI Five:** Berner et al., "Dota 2 with Large Scale Deep RL" (2019)
4. **IMPALA:** Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" (2018)
5. **PBT:** Jaderberg et al., "Population Based Training of Neural Networks" (2017)
6. **Multi-Agent RL:** Hernandez-Leal et al., "A Survey of Multi-Agent RL" (2019)

## Appendix B: Useful Resources

- **Stable-Baselines3 Docs:** https://stable-baselines3.readthedocs.io/
- **Gymnasium Docs:** https://gymnasium.farama.org/
- **pybind11 Tutorial:** https://pybind11.readthedocs.io/
- **Weights & Biases:** https://wandb.ai/
- **CleanRL (Minimal RL implementations):** https://github.com/vwxyzjn/cleanrl
- **RLlib (Scalable RL):** https://docs.ray.io/en/latest/rllib/

## Appendix C: Hardware Notes for RTX 4090

- **CUDA Version:** Use CUDA 12.1+ for best performance
- **PyTorch:** Install with `cu121` variant
- **Mixed Precision Training:** RTX 4090 has fast FP16/TF32, use `torch.cuda.amp`
- **Thermal Management:** Ensure good cooling (4090 TDP is 450W)
- **Power Supply:** 4090 needs 850W+ PSU with 12VHPWR connector
