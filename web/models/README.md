# Exported policies (ONNX, opset 17)

| file | checkpoint | move head | decide every | obs |
|---|---|---|---|---|
| `v9_150.onnx` | `rl/models/v9_150.pt` (life preset, 2.75 M samples) | legacy 9-way (0 stop, 1..8 compass) | 6 ticks | v3 layout, 323 floats |
| `v10_100.onnx` | `rl/models/v10_100.pt` (life preset, update 100, 2.6 M samples, the peak snapshot) | macro 24-way (10..17 mine, 18 own pad, 19 head's pad, 20..23 nearest bodies) | 15 ticks | v3 layout, 323 floats |

Input `obs` (batch, 323) float32 from `Game.observe(i)`; outputs `move`, `join`, `leave`, `intent` are masked logits
(masked classes are -inf). The sidecar `.json` carries nvec, decide_every, obs_dim and the training config that
`web/src/agent.js` reads. Regenerate with

    OMP_NUM_THREADS=1 python3 rl/export_onnx.py rl/models/v9_150.pt web/models/v9_150.onnx
    OMP_NUM_THREADS=1 python3 rl/export_onnx.py rl/models/v10_100.pt web/models/v10_100.onnx

which also checks the ONNX logits against torch with onnxruntime (256 random observations, tolerance 1e-4).
Try them in the browser with `web/dev/agent_demo.html` (needs jsDelivr for onnxruntime-web).
