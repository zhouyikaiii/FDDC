#!/usr/bin/env python
"""DDC per-motion FULL-metrics shard runner (deliverable 1). Unlike run_eval.py (which keeps only the
3-tier outcome), this dumps every metrics.py field per motion so the continuous balance columns (MoS
min/mean, xCoM/CoM margins, xCoM-TTB, out-of-bounds duration) can be aggregated. Under the new metric
code, log['com'] is the TRUE whole-body CoM (obs com_rel stays proxy).

args: motion_dir tag shard nshards K outdir condition [limit]
  condition in {clean,noisy}; for K>1 the per-motion record is the mean over seeds 0..K-1 of every
  numeric field (booleans -> rates), matching the noisy-K10 aggregation.
"""
import sys, json, glob, os, re, time
import numpy as np

motion_dir, tag = sys.argv[1], sys.argv[2]
sid, nsh, K, outdir, cond = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
limit = int(sys.argv[8]) if len(sys.argv) > 8 else 0
os.environ["WBT_EVAL_MOTION_DIR"] = motion_dir
os.environ.setdefault("WBT_ORT_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import wbt_rollout as W, metrics as M
from fast_policy import FastPolicy

COND = {"clean": dict(dof_vel_noise=0.0, dof_vel_delay=0, imu_noise=False),
        "noisy": dict(dof_vel_noise=0.20, dof_vel_delay=1, imu_noise=True)}[cond]
pol = FastPolicy(os.path.join(PKG, "policy", "model_0262000.pt"),
                 os.path.join(PKG, "policy", "robot_config.json"))
os.makedirs(outdir, exist_ok=True)
of = f"{outdir}/{tag}__sh{sid}.json"
if os.path.exists(of):
    print(f"[skip] {of}"); sys.exit(0)

mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p))
              for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
if limit:
    mots = mots[:limit]
my = mots[sid::nsh]

def agg(seeds):
    if len(seeds) == 1:
        return seeds[0]
    out = {}
    for k, v in seeds[0].items():
        if isinstance(v, bool):
            out[k] = float(np.mean([float(s[k]) for s in seeds]))
        elif isinstance(v, (int, float)):
            out[k] = float(np.mean([s[k] for s in seeds]))
        else:
            out[k] = v            # motion_id / support_foot / single_window
    return out

res, t0 = {}, time.time()
for mid in my:
    seeds = [M.compute_metrics(W.run_rollout(pol, mid, seed=s, use_npz_ref=True, **COND)) for s in range(K)]
    res[mid] = agg(seeds)
json.dump({"tag": tag, "condition": cond, "K": K, "per_motion": res}, open(of, "w"))
print(f"[done] {tag} {cond} sh{sid} {len(my)}mot K={K} {time.time()-t0:.0f}s")
