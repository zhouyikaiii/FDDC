#!/usr/bin/env python
"""DDC single-leg-balance benchmark — evaluate one policy (.pt) on the held-out motions.

Reproduces the paper's headline outcome tiers (Perfect / Marginal / Failure) for the deployed
DDC checkpoint. One .pt is scored against every motion npz via the decoupled reference path
(use_npz_ref=True) — no per-motion ONNX needed.

  Perfect  = clean single-leg stance: support foot never hops, swing foot never touches down,
             no fall, and the body tracks the reference within HuB's 0.5 m 12-keypoint gate.
  Failure  = a fall.
  Marginal = stays upright but breaks the single-leg constraint (hops / touches the swing foot down).

Conditions (paper §5.1):
  clean  — deterministic, no observation noise (K=1)             -> Table 1 (DDC 98.9 / 0.0 / 1.1)
  noisy  — deployment-relevant obs noise: per-step dof-vel noise + 1-step delay + HuB IMU-OU
           orientation drift, averaged over K=10 fixed seeds     -> Table 2 (DDC 61.8)
"""
import argparse, json, os, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # package root (eval/ is under it)

CONDITIONS = {
    "clean": dict(dof_vel_noise=0.0, dof_vel_delay=0, imu_noise=False),
    "noisy": dict(dof_vel_noise=0.20, dof_vel_delay=1, imu_noise=True),
}
DEFAULT_K = {"clean": 1, "noisy": 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.join(PKG, "policy", "model_0262000.pt"))
    ap.add_argument("--robot-config", default=os.path.join(PKG, "policy", "robot_config.json"))
    ap.add_argument("--robot-xml", default=os.path.join(PKG, "robot", "g1_29dof", "g1_29dof.xml"))
    ap.add_argument("--motion-dir", default=os.path.join(PKG, "data", "data_stratified_900", "test"))
    ap.add_argument("--dataset-info", default=os.path.join(PKG, "data", "data_stratified_900", "dataset_info.json"))
    ap.add_argument("--condition", choices=list(CONDITIONS), default="clean")
    ap.add_argument("--k", type=int, default=None, help="seeds per motion (default: 1 clean / 10 noisy)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N motions (0 = all)")
    ap.add_argument("--threads", default="1", help="onnxruntime/OMP threads per process")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # cap threads + point the harness at the bundled robot XML + motions (read at import time)
    os.environ["WBT_ORT_THREADS"] = args.threads
    os.environ.setdefault("OMP_NUM_THREADS", args.threads)
    os.environ["WBT_EVAL_ROBOT_XML"] = args.robot_xml
    os.environ["WBT_EVAL_MOTION_DIR"] = args.motion_dir
    sys.path.insert(0, HERE)
    import wbt_rollout as W, metrics as M
    from fast_policy import FastPolicy

    K = args.k if args.k is not None else DEFAULT_K[args.condition]
    cond_kw = CONDITIONS[args.condition]
    cls = {}
    if os.path.exists(args.dataset_info):
        for e in json.load(open(args.dataset_info)).get("manifest", []):
            cls[e["motion"]] = e["class"]

    import glob, re
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p))
                  for p in glob.glob(os.path.join(args.motion_dir, "sample_*_mj.npz")))
    if args.limit:
        mots = mots[:args.limit]
    print(f"policy   : {os.path.basename(args.pt)}")
    print(f"condition: {args.condition}  (K={K}, {cond_kw})")
    print(f"motions  : {len(mots)} from {args.motion_dir}\n")

    pol = FastPolicy(args.pt, args.robot_config)
    per_motion, t0 = {}, time.time()
    for i, mid in enumerate(mots):
        perfect = fail = 0.0
        for s in range(K):
            m = M.compute_metrics(W.run_rollout(pol, mid, seed=s, use_npz_ref=True, **cond_kw))
            perfect += float(m["success"]); fail += float(m["fell"])
        perfect /= K; fail /= K
        per_motion[mid] = {"perfect": perfect, "fail": fail, "marginal": max(0.0, 1 - perfect - fail),
                           "class": cls.get(mid, "?")}
        print(f"  [{i+1:>3}/{len(mots)}] {mid:12s} {per_motion[mid]['class']:5s} "
              f"P={perfect:.0%} M={per_motion[mid]['marginal']:.0%} F={fail:.0%}")

    n = len(per_motion)
    P = 100 * sum(v["perfect"] for v in per_motion.values()) / n
    Mg = 100 * sum(v["marginal"] for v in per_motion.values()) / n
    F = 100 * sum(v["fail"] for v in per_motion.values()) / n
    print(f"\n=== OVERALL ({args.condition}, n={n}) ===")
    print(f"  Perfect  {P:5.1f}%")
    print(f"  Marginal {Mg:5.1f}%")
    print(f"  Failure  {F:5.1f}%   ({time.time()-t0:.0f}s)")

    by = defaultdict(list)
    for v in per_motion.values():
        by[v["class"]].append(v)
    if len(by) > 1:
        print("\n  per-class Perfect%:")
        for c in sorted(by):
            g = by[c]
            print(f"    {c:5s} n={len(g):>2}  P={100*sum(x['perfect'] for x in g)/len(g):5.1f}%")

    if args.out:
        json.dump({"condition": args.condition, "K": K, "overall": {"perfect": P, "marginal": Mg, "fail": F},
                   "per_motion": per_motion}, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
