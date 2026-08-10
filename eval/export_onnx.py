#!/usr/bin/env python
"""Export the DDC actor from the .pt to ONNX (obs -> actions).

Uses the actor definition from the deploy export toolchain (InnerActor / CompleteActor / load_actor):
[Linear -> LayerNorm -> SiLU] x3 -> Linear -> tanh-squash, on the (obs - mean)/(std + 0.01) normalized
observation. It reads only `actor_state_dict` + `obs_normalizer_state` from the checkpoint (no IsaacSim,
no deploy framework), exports a general obs->actions ONNX, and verifies it against `fast_policy.py`.

Note: the real-robot runtime uses a *per-motion* ONNX — a specific motion's reference baked into the
graph via the Holosoma inference exporter. That baking is a deploy-side step and is not part of this
repository; this script exports the general actor ONNX (feed it the full per-step observation).

    python export_onnx.py                          # -> actor_0262000.onnx next to the .pt
    python export_onnx.py --pt <ckpt.pt> --out actor.onnx
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


class InnerActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(463, 512), nn.LayerNorm(512), nn.SiLU(),
                                 nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU())
        self.fc_mu = nn.Sequential(nn.Linear(128, 29))
        self.register_buffer("action_scale", torch.zeros(29))
        self.register_buffer("action_bias", torch.zeros(29))

    def forward(self, x):
        mu = self.fc_mu(self.net(x))
        return torch.tanh(mu) * self.action_scale + self.action_bias


class CompleteActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = InnerActor()
        self.register_buffer("mean", torch.zeros(1, 463))
        self.register_buffer("std", torch.ones(1, 463))

    def forward(self, x):
        return self.actor((x - self.mean) / (self.std + 0.01))


def load_actor(pt):
    d = torch.load(pt, map_location="cpu", weights_only=True)
    ca = CompleteActor()
    res = ca.actor.load_state_dict(d["actor_state_dict"], strict=False)
    assert not res.missing_keys, f"actor params not loaded: {res.missing_keys}"
    assert all("logstd" in k for k in res.unexpected_keys), f"unexpected extra keys: {res.unexpected_keys}"
    n = d["obs_normalizer_state"]
    ca.mean.copy_(n["_mean"].float().reshape(1, -1))
    ca.std.copy_(n["_std"].float().reshape(1, -1))
    return ca.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.join(PKG, "policy", "model_0262000.pt"))
    ap.add_argument("--out", default="")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.pt)),
                                   "actor_" + os.path.splitext(os.path.basename(args.pt))[0].split("_")[-1] + ".onnx")

    actor = load_actor(args.pt)
    dummy = torch.zeros(1, 463)
    kw = dict(input_names=["obs"], output_names=["actions"],
              dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=args.opset)
    try:
        torch.onnx.export(actor, dummy, out, dynamo=False, **kw)   # legacy exporter (no onnxscript dep)
    except TypeError:
        torch.onnx.export(actor, dummy, out, **kw)                 # older torch without the dynamo kwarg
    print(f"wrote {out}  (obs=463 -> actions=29)")

    # verify against the numpy reference (fast_policy.py) on random inputs
    sys.path.insert(0, HERE)
    from fast_policy import FastPolicy
    try:
        import onnxruntime as ort
    except Exception:
        print("(install onnxruntime to run the built-in verification)"); return
    pol = FastPolicy(args.pt, os.path.join(PKG, "policy", "robot_config.json"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(out, sess_options=so, providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0); worst = 0.0
    for _ in range(64):
        obs = rng.randn(463).astype(np.float32)
        ref = pol.infer(obs, 0)["actions"].reshape(-1)
        got = sess.run(["actions"], {"obs": obs.reshape(1, -1)})[0].reshape(-1)
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"max |ONNX - fast_policy| over 64 random obs = {worst:.2e}   "
          f"{'OK (float32 precision)' if worst < 1e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
