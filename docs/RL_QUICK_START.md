# Quick Start Guide: RL Training for Ungroup

This is a condensed guide to get started with RL training. See `RL_TRAINING_PLAN.md` for comprehensive details.

## TL;DR

- **Hardware:** RTX 4090 (24GB), 32GB RAM, high-end AMD CPU ✅ Perfect for this project
- **Time to Strong Agents:** 6-10 weeks of training + 2-4 weeks engineering
- **Algorithm:** PPO (Proximal Policy Optimization) with self-play
- **Expected Throughput:** 1-2 billion timesteps per day
- **Bottleneck:** CPU (game simulation), not GPU

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                            │
│                                                              │
│  ┌────────────┐      ┌──────────────┐     ┌──────────────┐ │
│  │   64 CPUs  │─────▶│  C++ Headless│────▶│  Observation │ │
│  │ Parallel   │      │  Game Envs   │     │    Buffer    │ │
│  │ Envs       │◀─────│  (1000 FPS)  │◀────│              │ │
│  └────────────┘      └──────────────┘     └──────┬───────┘ │
│                                                    │         │
│                                              ┌─────▼──────┐ │
│                                              │   RTX 4090 │ │
│                                              │   PPO Net  │ │
│  ┌────────────┐      ┌──────────────┐       │   Update   │ │
│  │Checkpoint  │─────▶│  Opponent    │◀──────│            │ │
│  │  Pool      │      │  Sampling    │       └─────┬──────┘ │
│  │ (50 agents)│      │              │             │         │
│  └────────────┘      └──────────────┘      ┌──────▼─────┐  │
│                                             │   Save     │  │
│                                             │ Checkpoint │  │
│                                             └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Critical Specifications

### State Space (Continuous, ~150 dimensions)
- **Self:** position, velocity, resources (4 types), group status, intent
- **Mines:** 4 mines × (position, type, resources remaining)
- **Other Players:** 8 nearest × (position, velocity, resources, intent, joinable)
- **Group:** size, centroid, total resources

### Action Space (Continuous, 6D - RECOMMENDED)
```python
# Continuous actions (smoother, more natural control)
- move_x: [-1.0, 1.0]          # Horizontal movement
- move_y: [-1.0, 1.0]          # Vertical movement
- toggle_joinable: [-1.0, 1.0] # Group joining
- toggle_ungroup: [-1.0, 1.0]  # Leave group
- toggle_intent: [-1.0, 1.0]   # Resource intent
- toggle_stop: [-1.0, 1.0]     # Stop movement
```

**Alternative:** 8 discrete actions (UP, DOWN, LEFT, RIGHT, STOP, JOINABLE, UNGROUP, INTENT) for simpler implementation

### Reward Function (Dense, Recommended)
```python
reward = 0
# Resource collection (primary signal)
reward += 10 * (new_resources - old_resources)
# Terminal conditions
reward += 1000 if won else -100 if lost
# Time penalty (encourage efficiency)
reward -= 0.01 per timestep
```

## Step-by-Step Implementation

### 1. Environment Setup (Week 1)

**Modify C++ Codebase:**
```cpp
// Add to src/server/main.cpp
#ifdef HEADLESS_MODE
// Disable SFML rendering
// Expose step() function via pybind11
#endif
```

**Create Python Binding (ungroup_rl/bindings/ungroup_binding.cpp):**
```cpp
#include <pybind11/pybind11.h>
#include "../../src/server/ServerGameController.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ungroup_game, m) {
    py::class_<UngroupGame>(m, "UngroupGame")
        .def(py::init<int>())  // num_players
        .def("reset", &UngroupGame::reset)
        .def("step", &UngroupGame::step)
        .def("get_observation", &UngroupGame::getObservation)
        .def("get_reward", &UngroupGame::getReward)
        .def("is_done", &UngroupGame::isDone);
}
```

**Create Gym Environment (ungroup_rl/envs/ungroup_env.py):**
```python
import gymnasium as gym
import numpy as np
from ungroup_game import UngroupGame  # C++ binding

class UngroupEnv(gym.Env):
    def __init__(self, num_players=4):
        super().__init__()
        self.game = UngroupGame(num_players)

        # Continuous action space (6D)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Continuous observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(150,), dtype=np.float32
        )

        # For discrete alternative, use:
        # self.action_space = gym.spaces.Discrete(8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.game.step(action)
        obs = self._get_observation()
        reward = self.game.get_reward()
        done = self.game.is_done()
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def _get_observation(self):
        return np.array(self.game.get_observation(), dtype=np.float32)
```

**Test Environment:**
```bash
python -c "
from ungroup_rl.envs import UngroupEnv
env = UngroupEnv()
obs, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, _ = env.reset()
print('Environment test passed!')
"
```

### 2. Baseline PPO Training (Week 2)

**Install Dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra] gymnasium wandb
```

**Training Script (ungroup_rl/training/train_baseline.py):**
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ungroup_rl.envs import UngroupEnv
import wandb
from wandb.integration.sb3 import WandbCallback

# Initialize W&B
wandb.init(
    project="ungroup-rl",
    config={
        "algorithm": "PPO",
        "n_envs": 64,
        "learning_rate": 3e-4,
    },
    sync_tensorboard=True,
)

# Create vectorized environment
env = make_vec_env(UngroupEnv, n_envs=64)

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=1,
    tensorboard_log=f"runs/{wandb.run.id}",
)

# Train
model.learn(
    total_timesteps=10_000_000,
    callback=WandbCallback(
        model_save_path=f"models/{wandb.run.id}",
        verbose=2,
    ),
)

# Save final model
model.save("ungroup_ppo_baseline")
```

**Run Training:**
```bash
python ungroup_rl/training/train_baseline.py
```

### 3. Self-Play Implementation (Weeks 3-4)

**Checkpoint Manager (ungroup_rl/self_play/checkpoint_manager.py):**
```python
import os
import random
from pathlib import Path

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints", max_checkpoints=50):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(self, model, step):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.zip"
        model.save(str(checkpoint_path))
        self.checkpoints.append(checkpoint_path)

        # Remove oldest if over limit
        if len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            oldest.unlink()

    def sample_opponent(self):
        if not self.checkpoints:
            return None
        return random.choice(self.checkpoints)
```

**Self-Play Training Loop (ungroup_rl/training/train_selfplay.py):**
```python
from stable_baselines3 import PPO
from ungroup_rl.self_play import CheckpointManager

checkpoint_mgr = CheckpointManager()
model = PPO("MlpPolicy", env, ...)

for iteration in range(1000):
    # Train for N steps
    model.learn(total_timesteps=100_000)

    # Save checkpoint every 10 iterations
    if iteration % 10 == 0:
        checkpoint_mgr.save(model, iteration * 100_000)

    # Update opponent pool in environment
    # (This requires modifying UngroupEnv to load opponent policies)
    opponent_path = checkpoint_mgr.sample_opponent()
    if opponent_path:
        env.set_opponent_policy(opponent_path)
```

### 4. Performance Optimization

**Profile Bottlenecks:**
```bash
python -m cProfile -o profile.stats ungroup_rl/training/train_baseline.py
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"
```

**Expected Bottleneck:** C++ game simulation (CPU-bound)

**Optimization Checklist:**
- [ ] Remove all rendering code in headless mode
- [ ] Reduce physics tickrate if not critical (60 FPS → 30 FPS)
- [ ] Use object pooling (avoid allocations in game loop)
- [ ] Compile with `-O3` optimization
- [ ] Profile with `perf` or `valgrind`

**CMake for Headless Build:**
```cmake
# Add to CMakeLists.txt
option(HEADLESS_MODE "Build headless for RL training" ON)

if(HEADLESS_MODE)
    add_definitions(-DHEADLESS_MODE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()
```

## Monitoring Training

**Key Metrics:**
- **Win Rate vs. Random:** Should reach 95%+ within 1 day
- **Win Rate vs. NearestGreedy:** Target 80%+ within 3-5 days
- **Episode Reward:** Should increase steadily
- **Entropy:** Should decay slowly (too fast = loss of exploration)

**W&B Dashboard:**
```python
# Log custom metrics
wandb.log({
    "eval/win_rate_random": win_rate_random,
    "eval/win_rate_greedy": win_rate_greedy,
    "train/resources_per_episode": np.mean(resources),
    "train/group_formations": group_count,
})
```

**Tensorboard (Alternative):**
```bash
tensorboard --logdir runs/
```

## Timeline & Milestones

| Milestone | Timesteps | Calendar Time | Success Criteria |
|-----------|-----------|---------------|------------------|
| **Environment Working** | - | Week 1 | 1000+ FPS, passes tests |
| **Beats Random** | 50M | Week 2 | 95%+ win rate |
| **Beats Greedy** | 250M | Week 3 | 70%+ win rate |
| **Self-Play Stable** | 1B | Week 5 | Win rate vs. self ~50% |
| **Strategic Play** | 5B | Week 8 | Grouping/betrayal observed |
| **Expert Level** | 25B+ | Week 12+ | Complex meta-game |

## Troubleshooting

### Issue: Low FPS (<100 FPS per env)
**Solution:**
- Profile C++ code, remove rendering
- Reduce physics complexity
- Use `-O3` compiler flag

### Issue: No learning progress
**Solution:**
- Check reward signal (are rewards non-zero?)
- Verify observation normalization
- Increase entropy coefficient (more exploration)
- Try denser reward shaping

### Issue: Policy collapse (all agents identical)
**Solution:**
- Increase opponent pool diversity
- Use population-based training
- Add noise to initial conditions

### Issue: GPU underutilized (<50%)
**Solution:**
- Increase batch size (you have 24GB VRAM!)
- Larger networks (512, 512, 256 layers)
- More parallel environments (if CPU allows)

## Hardware Utilization Targets

**Your RTX 4090 Setup:**
```
CPU Cores: 16 (AMD Ryzen 9)
├─ 64 parallel environments (4 cores each)
└─ Target: 70-90% CPU utilization

GPU: RTX 4090
├─ Batch size: 2048-4096
├─ Network: [256, 256, 128] (can go larger!)
└─ Target: 80-95% GPU utilization

RAM: 32 GB
├─ Environment states: ~5 GB
├─ Checkpoints: ~1 GB
└─ Headroom: 26 GB (plenty)

VRAM: 24 GB
├─ Model + gradients: ~2-3 GB
└─ Headroom: 21 GB (can run 4-8× bigger!)
```

## Expected Training Throughput

**Optimistic (Well-Optimized C++):**
- 64 envs × 1000 FPS = 64k steps/sec
- Daily: 5.5 billion steps
- **Time to 1B steps:** 4.5 hours
- **Time to 10B steps:** 1.8 days

**Realistic (Moderate Optimization):**
- 64 envs × 500 FPS = 32k steps/sec
- Daily: 2.7 billion steps
- **Time to 1B steps:** 9 hours
- **Time to 10B steps:** 3.7 days

**Conservative (Minimal Optimization):**
- 64 envs × 200 FPS = 12.8k steps/sec
- Daily: 1.1 billion steps
- **Time to 1B steps:** 22 hours
- **Time to 10B steps:** 9 days

## Quick Commands

**Build Headless Game:**
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DHEADLESS_MODE=ON -GNinja -S . -B build
ninja -C build
```

**Run Training:**
```bash
# Baseline training
python ungroup_rl/training/train_baseline.py

# Self-play training
python ungroup_rl/training/train_selfplay.py

# Evaluation
python ungroup_rl/training/evaluate.py --checkpoint models/checkpoint_1000000.zip
```

**Monitor:**
```bash
# W&B (in browser)
wandb login
# Visit: https://wandb.ai/your-username/ungroup-rl

# Tensorboard
tensorboard --logdir runs/ --port 6006
# Visit: http://localhost:6006
```

## Next Steps

1. ✅ Read full plan: `docs/RL_TRAINING_PLAN.md`
2. ⬜ Set up environment (Week 1)
3. ⬜ Train baseline agent (Week 2)
4. ⬜ Implement self-play (Weeks 3-4)
5. ⬜ Optimize & scale (Weeks 5-8)
6. ⬜ Analyze emergent strategies

## Resources

- **Full Plan:** `docs/RL_TRAINING_PLAN.md`
- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/
- **W&B Docs:** https://docs.wandb.ai/

---

**Ready to train some agents? Let's go! 🚀**
