"""
Export a v2 policy checkpoint to ONNX (opset 17) for the browser (web/src/agent.js, onnxruntime-web).

  python3 rl/export_onnx.py rl/models/v9_150.pt web/models/v9_150.onnx
  python3 rl/export_onnx.py rl/checkpoints/v10/latest.pt web/models/v10_latest.onnx --check 256

The exported graph takes `obs` (N, obs_dim) float32 and returns the four per-head logit tensors
`move`, `join`, `leave`, `intent` with the action masks of Policy.masks already applied (masked classes
carry -inf, so a sampler only has to softmax each row; the browser treats logit < -1e8 as masked).

The wrapper re-implements Policy.masks with torch.where instead of boolean-index assignment, which
the ONNX exporters cannot trace with a dynamic batch dimension; it is verified against Policy.forward
before export, and the ONNX file is verified against torch with onnxruntime when it is installed.

Requirements: torch and onnx (legacy exporter, dynamo=False); onnxscript for the dynamo exporter
fallback; onnxruntime for --check. A sidecar JSON (same stem, .json) records obs_dim, nvec,
decide_every, layout and the training config so the browser knows how to drive the model.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v2 import IDX_GROUP_N, IDX_OTHER_PRESENT, MACRO_BASE, MAX_SLOTS, MINE_DIM, load_checkpoint  # noqa: E402

OPSET = 17
NEG = float("-inf")


class OnnxPolicy(nn.Module):
    """obs (N, obs_dim) -> [move, join, leave, intent] masked logits, export friendly."""

    def __init__(self, policy):
        super().__init__()
        self.p = policy
        self.max_group = float(policy.max_group)
        self.macro = policy.nvec[0] > MACRO_BASE
        self.mine0 = policy.own_dim + MAX_SLOTS * policy.other_dim

    def forward(self, obs):
        p = self.p
        h = p.features(obs)
        n = obs[:, IDX_GROUP_N] * self.max_group
        solo = (n < 1.5).unsqueeze(1)                          # (N, 1)
        zero = torch.zeros((), dtype=obs.dtype)
        neg = torch.full((), NEG, dtype=obs.dtype)
        move = p.heads[0](h)
        join = p.heads[1](h)
        leave = p.heads[2](h)
        intent = p.heads[3](h)
        # leave: class 1 masked when solo
        leave = leave + torch.cat([torch.zeros_like(solo, dtype=obs.dtype), torch.where(solo, neg, zero)], 1)
        # intent: classes 1..4 masked while grouped
        grouped_mask = torch.where(solo, zero, neg)               # (N, 1)
        intent = intent + torch.cat([torch.zeros_like(grouped_mask), grouped_mask.expand(-1, 4)], 1)
        if self.macro:
            cols = [torch.zeros_like(grouped_mask)]                # class 0 (stop) allowed
            cols.append(torch.full_like(grouped_mask, NEG).expand(-1, MACRO_BASE - 1))  # 1..9 direct steering masked
            for m in range(8):
                if m < p.n_mines:
                    dead = (obs[:, self.mine0 + m * MINE_DIM + 7] < 0.5).unsqueeze(1)
                    cols.append(torch.where(dead, neg, zero))
                else:
                    cols.append(torch.full_like(grouped_mask, NEG))
            cols.append(torch.zeros_like(grouped_mask))            # 18 own pad
            cols.append(torch.zeros_like(grouped_mask))            # 19 head's pad
            for k in range(4):
                absent = (obs[:, p.own_dim + k * p.other_dim + IDX_OTHER_PRESENT] < 0.5).unsqueeze(1)
                cols.append(torch.where(absent, neg, zero))
            move = move + torch.cat(cols, 1)
        return move, join, leave, intent


def random_obs(policy, n, seed=0):
    """Random observations that exercise the masks: real-looking flags in the fields the masks read."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, size=(n, policy.obs_dim)).astype(np.float32)
    obs[:, IDX_GROUP_N] = rng.integers(1, policy.max_group + 1, size=n) / policy.max_group
    for k in range(MAX_SLOTS):
        obs[:, policy.own_dim + k * policy.other_dim + IDX_OTHER_PRESENT] = (rng.random(n) < 0.7).astype(np.float32)
    mine0 = policy.own_dim + MAX_SLOTS * policy.other_dim
    for m in range(policy.n_mines):
        obs[:, mine0 + m * MINE_DIM + 7] = (rng.random(n) < 0.8).astype(np.float32)
    return obs


def compare(name, a, b, tol):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    fin = np.isfinite(a) & np.isfinite(b)
    same_mask = np.array_equal(np.isfinite(a), np.isfinite(b))
    err = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0
    ok = same_mask and err <= tol
    print(f"  {name:7s} max|diff| {err:.2e} masks {'match' if same_mask else 'DIFFER'} -> {'ok' if ok else 'FAIL'}")
    return ok


def export(ck_path, out_path, check=256, tol=1e-4):
    policy, cfg, ck = load_checkpoint(ck_path)
    wrapper = OnnxPolicy(policy).eval()
    obs = torch.from_numpy(random_obs(policy, max(check, 8)))
    with torch.no_grad():
        ref = [t.numpy() for t in policy.forward(obs)]
        wrapped = [t.numpy() for t in wrapper(obs)]
    print(f"{ck_path}: obs_dim {policy.obs_dim} nvec {policy.nvec} decide_every {policy.decide_every} samples {ck.get('samples')}")
    print("wrapper vs Policy.forward (torch):")
    names = ["move", "join", "leave", "intent"]
    assert all(compare(nm, w, r, 1e-6) for nm, w, r in zip(names, wrapped, ref)), "wrapper mask mismatch"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    example = obs[:2].clone()
    exported = None
    try:
        torch.onnx.export(wrapper, (example,), out_path, opset_version=OPSET, input_names=["obs"], output_names=names,
                          dynamic_axes={"obs": {0: "batch"}, **{nm: {0: "batch"} for nm in names}}, dynamo=False)
        exported = "legacy (dynamo=False)"
    except Exception as e:  # noqa: BLE001
        print(f"legacy exporter failed: {type(e).__name__}: {e}\ntrying the dynamo exporter (needs onnxscript)")
        prog = torch.onnx.export(wrapper, (example,), opset_version=OPSET, input_names=["obs"], output_names=names,
                                 dynamic_shapes={"obs": {0: "batch"}}, dynamo=True)
        prog.save(out_path)
        exported = "dynamo"
    print(f"exported with the {exported} exporter -> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")

    meta = {"checkpoint": os.path.basename(ck_path), "obs_dim": policy.obs_dim, "nvec": list(policy.nvec),
            "decide_every": policy.decide_every, "n_mines": policy.n_mines, "max_group": policy.max_group,
            "own_dim": policy.own_dim, "other_dim": policy.other_dim, "macro": bool(policy.nvec[0] > MACRO_BASE),
            "opset": OPSET, "inputs": ["obs"], "outputs": names, "samples": ck.get("samples", 0), "git": ck.get("git"),
            "config": asdict(cfg)}
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)

    try:
        import onnx
        onnx.checker.check_model(onnx.load(out_path))
        print("onnx.checker: ok")
    except ImportError:
        print("onnx not installed: skipping checker")

    if check > 0:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed: skipping the numeric check")
            return True
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"obs": obs.numpy()})
        print(f"onnxruntime vs torch on {obs.shape[0]} random observations (tol {tol}):")
        ok = all(compare(nm, o, r, tol) for nm, o, r in zip(names, outs, ref))
        # batch-1 as the browser uses it
        o1 = sess.run(None, {"obs": obs[:1].numpy()})
        ok &= all(compare(nm + "@1", o, r[:1], tol) for nm, o, r in zip(names, o1, ref))
        print("verification", "PASSED" if ok else "FAILED")
        return ok
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--check", type=int, default=256, help="random observations for the onnxruntime check (0 = skip)")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    torch.set_num_threads(1)
    sys.exit(0 if export(a.checkpoint, a.out, a.check, a.tol) else 1)
