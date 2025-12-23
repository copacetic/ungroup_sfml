# Cloud RL Training Guide: Fast Track to Expert Agents

## TL;DR

**Local RTX 4090:** 6-10 weeks to strong agents
**Cloud with 8× A100s:** **3-7 days** to strong agents
**Estimated Cost:** $500-2,000 depending on provider and optimization

---

## Why Cloud Training is Much Faster

**The Math:**
- Your single RTX 4090: ~2.7B steps/day (realistic)
- 8× A100 cluster: ~40-60B steps/day (15-20× faster)
- With distributed training: Can reach 1B steps in **~1 hour** vs. 9 hours locally

**What You Get:**
- ✅ **Faster time to results** (days vs. weeks)
- ✅ **Experiment in parallel** (test multiple hyperparameters simultaneously)
- ✅ **Scale to 20+ player games** easily
- ✅ **No wear on your personal hardware**

---

## Cloud Provider Options

### Option 1: Lambda Labs (BEST VALUE)

**Hardware:**
- 8× A100 (40GB) instance
- 768 GB RAM
- 64 vCPUs

**Pricing:**
- **$12.00/hour** ($288/day)
- On-demand, no commitment

**Timeline:**
- **Day 1-2:** Setup environment, baseline training → Beat scripted bots
- **Day 3-5:** Self-play training → Strategic behaviors emerge
- **Day 6-7:** League training → Expert-level play

**Total Cost:** $1,728 - $2,016 (7 days)

**Pros:**
- ✅ Cheapest per-GPU cost
- ✅ Simple pricing, no hidden fees
- ✅ Good availability of A100s
- ✅ Pre-configured ML images

**Cons:**
- ❌ Can have availability issues
- ❌ Less features than AWS/GCP

**Sign up:** https://lambdalabs.com/service/gpu-cloud

---

### Option 2: RunPod (FLEXIBLE, SPOT PRICING)

**Hardware:**
- 8× RTX 4090 (24GB) instance (cheaper alternative)
- OR 4× A100 (80GB)

**Pricing (Spot - can be interrupted):**
- 8× RTX 4090: **~$8-10/hour** ($192-240/day)
- 4× A100 80GB: **~$10-12/hour** ($240-288/day)

**Pricing (On-Demand - guaranteed):**
- 8× RTX 4090: **~$14/hour** ($336/day)
- 4× A100 80GB: **~$16/hour** ($384/day)

**Timeline:** Similar to Lambda (3-7 days)

**Total Cost (Spot):** $1,344 - $1,680 (7 days with 4090s)

**Pros:**
- ✅ **Spot pricing** can save 40-60%
- ✅ Great for experimentation (spin up/down quickly)
- ✅ Good RTX 4090 availability
- ✅ Community templates

**Cons:**
- ❌ Spot instances can be interrupted
- ❌ Slightly more complex setup

**Sign up:** https://runpod.io

---

### Option 3: AWS (Most Comprehensive)

**Hardware:**
- p4d.24xlarge: 8× A100 (40GB)
- OR p5.48xlarge: 8× H100 (80GB) - FASTEST

**Pricing:**
- p4d.24xlarge: **$32.77/hour** ($786/day) on-demand
- p4d.24xlarge: **~$10-15/hour** ($240-360/day) spot
- p5.48xlarge: **$98/hour** ($2,352/day) on-demand

**Timeline:**
- p4d (A100): 3-7 days
- p5 (H100): **2-4 days** (H100 is 2-3× faster than A100)

**Total Cost:**
- Spot p4d: **$1,680 - $2,520** (7 days)
- On-demand p4d: $5,502 (7 days, overkill)
- Spot p5 (if available): **$980 - $1,960** (4 days)

**Pros:**
- ✅ Most reliable infrastructure
- ✅ **Spot instances** save 70%
- ✅ S3 integration for checkpoints
- ✅ SageMaker for managed training (optional)
- ✅ Access to H100s (fastest available)

**Cons:**
- ❌ More expensive on-demand
- ❌ Complex pricing/setup
- ❌ Spot can be hard to get for p5

---

### Option 4: Google Cloud (TPU Alternative)

**Hardware:**
- TPU v4 Pod (8 chips)
- OR a2-ultragpu-8g: 8× A100 (80GB)

**Pricing:**
- TPU v4: **~$8-12/hour** ($192-288/day)
- a2-ultragpu-8g: **$29/hour** ($696/day) on-demand
- a2-ultragpu-8g: **~$9/hour** ($216/day) spot

**Timeline:** 3-7 days (TPUs may require code changes)

**Total Cost:** $1,344 - $2,016 (7 days)

**Pros:**
- ✅ TPUs are **excellent** value for RL (if using JAX)
- ✅ Preemptible pricing is very cheap
- ✅ Good for large-scale distributed training

**Cons:**
- ❌ TPUs require JAX/TensorFlow (no PyTorch)
- ❌ More complex to set up

---

### Option 5: Paperspace (Easy Setup)

**Hardware:**
- 8× A100 instance

**Pricing:**
- **$24/hour** ($576/day)

**Timeline:** 3-7 days

**Total Cost:** $4,032 (7 days)

**Pros:**
- ✅ Very easy setup (Jupyter notebooks)
- ✅ Good documentation
- ✅ Persistent storage

**Cons:**
- ❌ More expensive than alternatives
- ❌ Limited customization

---

## Recommended Setup: Lambda Labs or RunPod

**My Recommendation:**

**For Budget-Conscious ($1,300-1,700):**
- **RunPod 8× RTX 4090 (Spot)** @ $8-10/hour
- Run for 7 days
- Total: ~$1,344-1,680

**For Maximum Speed ($1,700-2,000):**
- **Lambda Labs 8× A100** @ $12/hour
- Run for 7 days
- Total: ~$2,016

**For Minimum Cost (Hybrid Approach, $300-500):**
- Dev/test locally on your RTX 4090 (1-2 weeks, free)
- Final training run on cloud for 2-3 days (~$300-500)
- Get 80% of the way locally, finish on cloud

---

## Distributed Training Architecture

With 8 GPUs, you can parallelize in multiple ways:

### Strategy 1: Data Parallelism (Recommended)

```python
# Each GPU runs its own set of environments
# Gradients are synchronized across GPUs

GPU 0: 64 envs × 4 players = 256 agents
GPU 1: 64 envs × 4 players = 256 agents
...
GPU 7: 64 envs × 4 players = 256 agents

Total: 512 envs × 4 players = 2,048 agents playing simultaneously!

# Throughput
512 envs × 1000 FPS = 512,000 steps/sec
Daily: 44 billion steps/day (16× your local setup)
```

**Time to 1B steps:** ~30 minutes (vs. 9 hours local)

### Strategy 2: Population-Based Training

```python
# Each GPU trains a different agent with different hyperparameters
GPU 0: Agent with learning_rate=1e-4, ent_coef=0.01
GPU 1: Agent with learning_rate=3e-4, ent_coef=0.01
GPU 2: Agent with learning_rate=1e-3, ent_coef=0.005
...
GPU 7: Agent with learning_rate=3e-4, ent_coef=0.05

# Periodically, weak agents copy from strong agents
# Hyperparameters mutate
# Result: Diverse population, best hyperparameters emerge
```

**Benefits:**
- Automatic hyperparameter tuning
- Diverse strategies
- More robust agents

### Strategy 3: Hybrid (Best)

```python
# 4 GPUs: Main agent training (data parallel)
# 2 GPUs: Exploiter agents (beat main agent)
# 2 GPUs: Hyperparameter search (different configs)

# After 24 hours, pick best agent and scale to all 8 GPUs
```

---

## Accelerated Timeline

**With 8× A100 on Lambda Labs:**

### Day 1: Setup & Baseline (6-8 hours active work)
- ✅ Provision instance, install dependencies
- ✅ Deploy headless game + Python bindings
- ✅ Launch distributed PPO training
- ✅ Train vs. scripted bots (100M steps in ~2 hours)
- 🎯 **Result:** Agent beats Random 95%+, Greedy 70%+

### Day 2: Self-Play Ramp (1B steps)
- ✅ Implement checkpoint pool, opponent sampling
- ✅ Train with self-play (1B steps in ~24 hours)
- 🎯 **Result:** Agent shows grouping behaviors, strategic mining

### Day 3-4: League Training (5B steps)
- ✅ Set up main + exploiter pools
- ✅ Train for 2 days (5B steps)
- 🎯 **Result:** Complex strategies emerge (betrayal timing, blocking)

### Day 5-6: Scaling & Refinement (10B+ steps)
- ✅ Scale to 8-20 players per game
- ✅ Fine-tune with diverse opponent pool
- ✅ Record gameplay videos
- 🎯 **Result:** Expert-level play, interesting meta-game

### Day 7: Evaluation & Packaging
- ✅ Comprehensive evaluation vs. baselines
- ✅ Elo ratings
- ✅ Export best checkpoints
- ✅ Create highlight reel

**Total Cost:** ~$2,000 (7 days @ $12/hour)

---

## Cost Optimization Strategies

### 1. Use Spot/Preemptible Instances

**Savings:** 50-70%

**How:**
- AWS Spot: $10-15/hour instead of $32/hour
- GCP Preemptible: $9/hour instead of $29/hour
- RunPod Spot: $8/hour instead of $14/hour

**Caveat:** Can be interrupted, but with proper checkpointing (save every 10 min), you can resume seamlessly.

### 2. Hybrid Local + Cloud

**Approach:**
1. Develop and debug locally (1 week, free)
2. Run 2-3 major training runs on cloud (2-3 days, $600)
3. Fine-tune locally if needed (1 week, free)

**Total Cost:** ~$600
**Total Time:** 4-5 weeks (vs. 10 weeks fully local)

### 3. Multi-Tenancy (Share with Others)

If you know other researchers/students:
- Split 8-GPU instance (4 GPUs each)
- Share cost: $6/hour each instead of $12/hour

### 4. Use Smaller Instances for Debugging

**Don't pay $12/hour while debugging!**

- Development: Single GPU instance ($1-2/hour)
- Testing: 2-4 GPU instance ($3-6/hour)
- Production: 8 GPU instance ($12/hour) only when ready

### 5. Academic Credits

If you're affiliated with a university:
- **AWS Educate:** $100-300 free credits
- **GCP Education:** $300-500 free credits
- **Azure for Students:** $100 free credits

---

## Implementation: Ray RLlib for Distributed Training

**Ray RLlib** makes distributed training trivial:

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# Initialize Ray cluster (automatically detects 8 GPUs)
ray.init()

# Configure distributed PPO
config = {
    "env": "UngroupEnv-v0",
    "num_workers": 64,              # 64 parallel workers
    "num_gpus": 8,                  # Use all 8 GPUs
    "num_envs_per_worker": 8,       # 8 envs per worker
    "train_batch_size": 32768,      # Large batch (8 GPUs can handle it)
    "sgd_minibatch_size": 4096,
    "framework": "torch",
}

# Launch training
tune.run(
    PPO,
    config=config,
    stop={"timesteps_total": 10_000_000_000},  # 10B steps
    checkpoint_freq=100,
)
```

**That's it!** Ray handles:
- ✅ Multi-GPU distribution
- ✅ Checkpointing
- ✅ Hyperparameter tuning (with Ray Tune)
- ✅ Monitoring (with TensorBoard)

**Alternative:** Use Stable-Baselines3 with custom distributed wrapper (more work, but simpler for single algorithm).

---

## Monitoring & Cost Control

### Set Up Billing Alerts

**AWS:**
```bash
aws cloudwatch put-metric-alarm --alarm-name "RL-Training-Cost" \
  --alarm-description "Alert if cost exceeds $500" \
  --metric-name EstimatedCharges \
  --threshold 500
```

**GCP:**
- Cloud Console → Billing → Budgets & Alerts

**Lambda/RunPod:**
- Check dashboard every 12 hours
- Set calendar reminders

### Auto-Shutdown on Completion

```python
# In training script
import boto3

def on_training_complete():
    # Save final checkpoint
    model.save("s3://my-bucket/final_model.zip")

    # Shutdown instance
    ec2 = boto3.client('ec2')
    instance_id = get_instance_id()
    ec2.stop_instances(InstanceIds=[instance_id])
```

### Use WandB for Remote Monitoring

```python
import wandb

wandb.init(project="ungroup-rl-cloud")

# Monitor from your laptop
# Visit: https://wandb.ai/your-username/ungroup-rl-cloud
```

No need to keep SSH session open!

---

## Quick Start Commands

### Lambda Labs

```bash
# 1. Launch 8× A100 instance via web UI
# 2. SSH into instance

# 3. Setup
git clone https://github.com/yourusername/ungroup_sfml.git
cd ungroup_sfml
pip install torch ray[rllib] stable-baselines3 gymnasium wandb

# 4. Build headless game
cmake -DCMAKE_BUILD_TYPE=Release -DHEADLESS_MODE=ON -GNinja -S . -B build
ninja -C build

# 5. Launch distributed training
python ungroup_rl/training/train_distributed.py --num-gpus 8 --num-workers 512

# 6. Monitor
wandb login
# Visit dashboard URL
```

### AWS (with Spot)

```bash
# 1. Launch Spot instance (p4d.24xlarge)
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-config.json

# 2. SSH and run same setup as Lambda Labs
```

### RunPod

```bash
# 1. Create instance via web UI (8× RTX 4090, Spot)
# 2. Use their Jupyter environment or SSH

# 3. Clone and setup (same as above)
```

---

## ROI Analysis

**Your Time Value:**
- 6-10 weeks local → ~100+ hours of your time monitoring
- 3-7 days cloud → ~20-30 hours of your time

**If your time is worth $50/hour:**
- Local: 100 hours × $50 = $5,000 opportunity cost
- Cloud: 30 hours × $50 = $1,500 opportunity cost + $2,000 compute = $3,500 total

**Cloud is cheaper when factoring in time!**

---

## Recommended Action Plan

### Option A: Fast Track (7 days, $2,000)

1. **Today:** Sign up for Lambda Labs
2. **Day 1:** Provision 8× A100, setup environment
3. **Day 2-6:** Run distributed training (monitor via WandB)
4. **Day 7:** Evaluate results, export models

**Total:** $2,016, **7 days to expert agents**

### Option B: Budget (7 days, $1,400)

1. **Today:** Sign up for RunPod
2. **Day 1:** Provision 8× RTX 4090 Spot
3. **Day 2-6:** Run distributed training
4. **Day 7:** Evaluate

**Total:** ~$1,400, **7 days to expert agents**

### Option C: Hybrid (4 weeks, $600)

1. **Week 1-2:** Develop locally, get baseline working
2. **Week 3:** Run 3-day cloud training ($600)
3. **Week 4:** Analyze results, fine-tune locally

**Total:** $600, **4 weeks to expert agents**

---

## Next Steps

**To proceed with cloud training:**

1. **Choose provider** (I recommend Lambda Labs or RunPod)
2. **Sign up** and add payment method
3. **Provision instance** (8× A100 or 8× RTX 4090)
4. **I can help you:**
   - Write the distributed training script
   - Set up Ray RLlib configuration
   - Configure auto-checkpointing
   - Set up monitoring
   - Optimize for cost

Let me know which option you prefer and I'll create the deployment scripts!

---

## FAQs

**Q: Can I pause training and resume later?**
A: Yes! Save checkpoints every 30 min to cloud storage (S3, GCS). Stop instance, resume anytime.

**Q: What if I run out of budget?**
A: Set billing alerts. You can always stop, download checkpoints, and continue later.

**Q: Can I test locally first?**
A: Absolutely! Test with 1-2 GPUs locally or on cheap single-GPU cloud ($1-2/hour), then scale up.

**Q: How do I know if it's working?**
A: Monitor win rate vs. baselines. Should see 95% vs. Random in 6-12 hours, strategic play in 2-3 days.

**Q: What if agents don't learn?**
A: Reduce risk by testing baseline agent first (1 day, $24-48). If that works, scale up.

---

**Ready to go fast? Let me know your budget and timeline preference!** 🚀
