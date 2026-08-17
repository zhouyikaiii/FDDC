<a href="https://www.pku.edu.cn/"><img src="assets/pku_logo.png" alt="Peking University" width="90" align="right"/></a>

# A Change of Frame Makes Balance Observable: Distillation-Free Humanoid Single-Leg Stance

**DDC (Deployable Dynamic-CoM)** — a unified policy and a method-agnostic sim2sim benchmark for humanoid single-leg balance.

**Yikai Zhou, Xingyun Wang, Jieming Cui, Bozhou Chen, Yikai Fan, Yixin Zhu\*, Wenxin Li\***<br>
[Peking University](https://www.pku.edu.cn/) &nbsp;·&nbsp; <sup>\*</sup>Corresponding authors (`yixin.zhu@pku.edu.cn`, `lwx@pku.edu.cn`)

📄 Paper: [arXiv:2608.00500](https://arxiv.org/abs/2608.00500) &nbsp;·&nbsp; 🌐 Project page: [estoil.github.io/DDC](https://estoil.github.io/DDC/) &nbsp;·&nbsp; 📺 Video: [YouTube](https://www.youtube.com/watch?v=ue3DhT5B3mU)

![DDC holds clean single-leg balance across 90 stratified poses and transfers to a real Unitree G1](assets/fig_teaser.png)

[![DDC on a real Unitree G1 — single-leg balance across distinct poses (click to watch)](assets/video_thumb.jpg)](https://www.youtube.com/watch?v=ue3DhT5B3mU)

▶️ Watch the full video on [YouTube](https://www.youtube.com/watch?v=ue3DhT5B3mU).

Unified humanoid policies track dances, runs, and backflips, yet stumble on a simple demand: staying
balanced on one leg. DDC is the first single-leg-balance policy to deploy **directly on a real Unitree
G1 without teacher–student distillation**. Its key idea is a *deployable* dynamic-CoM observation: the
capture point (xCoM) — the center of mass extrapolated by its velocity — expressed **relative to the
support foot**, so the base linear velocity (which no on-board humanoid sensor can measure) cancels
exactly and the balance signal becomes reconstructible from encoders + IMU alone. It is paired with a
reward library translated term-by-term from human postural-control science (margin of stability,
time-to-boundary, ankle→knee→hip hierarchy, jerk), and trained by asymmetric FastSAC with a privileged
critic.

![DDC method overview: an asymmetric actor–critic where the deployable actor sees a support-relative dynamic-CoM state and a human-science reward library shapes balance](assets/fig1_overview.png)

The same trained actor runs **directly on a real Unitree G1** (no distillation):

![DDC deployed on a real Unitree G1, holding single-leg balance across distinct poses](assets/fig_realrobot_montage.png)

This repository lets anyone **reproduce the benchmark for the deployed DDC checkpoint** (`model_0262000.pt`)
and score it under the exact conditions from the paper — from the `.pt` alone, no ONNX, no IsaacSim,
no training/deploy runtime.

---

## What's here

| | |
|---|---|
| **Policy** | `policy/model_0262000.pt` — the deployed DDC checkpoint (the benchmark-selected one); the inference actor + observation normalizer, which is all `run_eval.py` needs and reproduces every number here. |
| **Benchmark** | `eval/` — a self-contained MuJoCo sim2sim harness (`numpy` + `mujoco` + `torch`). |
| **Motions** | `data/data_stratified_900/` — the 900-clip stratified single-leg set (720 train / 90 val / 90 test). |
| **Robot** | `robot/g1_29dof/` — the exact Unitree G1 29-DoF MuJoCo plant the policy was validated on. |
| **Training** | `training/` — the DDC-specific implementation (deployable dynamic-CoM obs + reward library) and how to retrain; see `training/TRAINING.md`. |
| **Deployment** | `DEPLOYMENT.md` — the real on-robot recipe: the Holosoma `run_sim` / `run_policy` two-process stack, the sim2sim + sim2real commands, and the safety caveats. |

Together these are the paper's released **"full stack — training code, benchmark, data, checkpoint, and deployment recipe"**:
data = `data/`, code = `training/` (method) + `eval/` (harness), policy = `policy/`, benchmark = `eval/`.

The policy is run **directly from the `.pt`** (`eval/fast_policy.py` reproduces the actor in numpy,
bit-exact with the IsaacSim ONNX export). The robot constants (kp/kd/default pose/joint limits) live in
`policy/robot_config.json`.

## The outcome tiers

Every trial is graded into one of three mutually exclusive tiers, judged from the robot's **true
physical state** (not the reference):

- **Perfect** — a clean single-leg stance across the single-support window: the support foot never
  hops, the swing foot never touches down, the robot does not fall, and the body tracks the reference
  within HuB's average 12-keypoint 0.5 m gate.
- **Marginal** — does not fall, but stays upright only by breaking the single-leg constraint (hopping
  the support foot or touching the swing foot down) — a recovery, not a genuine single-leg hold.
- **Failure** — a fall.

Two conditions (paper §5.1): **clean** (deterministic, no observation noise, `K=1`) and **noisy**
(deployment-relevant observation noise — per-step dof-velocity noise + 1-step delay + a temporally
correlated HuB IMU-orientation drift — averaged over `K=10` fixed seeds).

## Quickstart

```bash
# 0. fetch the motion clips from HuggingFace (not in git; ~645 MB) -> data/data_stratified_900/
pip install huggingface_hub socksio   # socksio only needed if you download via a SOCKS proxy (e.g. ALL_PROXY=socks5://...)
python download_data.py

# 1. environment (CPU is fine; a CPU-only torch build works)
pip install -r requirements.txt

# 2. clean benchmark — reproduces the headline (expect Perfect 98.9 / Marginal 0.0 / Failure 1.1)
cd eval
python run_eval.py --condition clean            # all 90 held-out test motions, K=1   (~75 s)

# quick smoke on the first 8 motions
python run_eval.py --condition clean --limit 8

# 3. noisy benchmark — deployment-relevant obs noise, K=10 (expect Perfect 61.8)
python run_eval.py --condition noisy            # ~12 min

# score a different split (e.g. validation) or your own motions:
python run_eval.py --condition clean --motion-dir ../data/data_stratified_900/val
```

Thread caps for parallel/background runs: `WBT_ORT_THREADS=1 OMP_NUM_THREADS=1 python run_eval.py ...`
(or pass `--threads 1`).

## Expected results (held-out test set, n = 90)

The deployed checkpoint reproduces the paper's Table 1 (clean) / Table 2 (noisy) **exactly** from the `.pt`:

| condition | Perfect | Marginal | Failure |
|-----------|:-------:|:--------:|:-------:|
| clean (K=1)  | **98.9 %** (89/90) | 0.0 %  | 1.1 % (1 fall) |
| noisy (K=10) | **61.8 %**         | 37.1 % | 1.1 % |

*(Both rows reproduced bit-exact from `model_0262000.pt` with this harness.)*

Per-class clean Perfect-rate over the 3×3 pose grid (pelvis height × swing-foot height): every class
is 100 % except `P1S2` (mid pelvis × **high swing foot**, 90 %) — the single hardest corner, and the
one class holding the lone clean fall, as in the paper.

## Package layout

```
opensource_release/
├─ README.md            LICENSE (Apache-2.0)   CITATION.cff   requirements.txt
├─ eval/
│  ├─ run_eval.py       benchmark entry point (Perfect/Marginal/Failure, per-class)
│  ├─ fast_policy.py    DDC actor from the .pt in numpy (bit-exact vs the IsaacSim ONNX)
│  ├─ wbt_rollout.py    self-contained MuJoCo rollout (deploy-faithful obs, PD, noise switches)
│  └─ metrics.py        postural-control metric suite (tiers + MoS / TTB / slippage / jerk / tracking)
├─ policy/
│  ├─ model_0262000.pt  deployed DDC checkpoint
│  └─ robot_config.json G1 constants (dof_names, kp, kd, action_scale, default pose, joint limits)
├─ robot/g1_29dof/      g1_29dof.xml + meshes/ + NOTICE   (Unitree-derived plant)
├─ training/            DDC method on Holosoma: fddc_src/ (obs + reward + config) + train_full.sh + TRAINING.md + NOTICE
└─ data/
   ├─ data_stratified_900/{train,val,test}/  sample_*_mj.npz  (+ dataset_info.json, manifest.csv)
   ├─ LICENSE           GPL-3.0 (motions; derived from AMS)
   └─ NOTICE            AMS attribution + what we changed
```

## Licensing & attribution

This project **mixes licenses** — please read before redistributing:

- **Code + policy** (`eval/`, `run_eval.py`, `policy/`) — **Apache-2.0** (`LICENSE`).
- **Motions** (`data/data_stratified_900/`) — **GPL-3.0** (`data/LICENSE`), because they are a
  derivative of the **AMS** synthetic balance dataset (Pan et al., 2025, GPL-3.0). Attribution and
  what-we-changed are in `data/NOTICE`. AMS motions are *synthetic* (no motion-capture terms).
- **Robot model** (`robot/g1_29dof/`) — **Unitree G1, BSD-3-Clause** (Copyright Unitree Robotics; the
  full license is in `robot/g1_29dof/LICENSE`, attribution + stated modifications in
  `robot/g1_29dof/NOTICE`). Redistributed under BSD-3-Clause; the `<actuator>` / visual-geom edits are
  Holosoma's (Apache-2.0).
- **Training code** (`training/fddc_src/`, `training/train_full.sh`) — **Apache-2.0**, modified from the
  public **Holosoma** framework (Amazon FAR, Apache-2.0); the changes we made are stated in
  `training/NOTICE`. It is a small readable subset, not a redistribution of the whole framework.

The GPL-3.0 data and the Apache-2.0 code sit side-by-side as an *aggregate* (GPL-3.0 §5): bundling the
GPL-3.0 motions does not relicense the independent evaluation code or the policy weights.
*(This is an engineering-practice summary, not legal advice.)*

Built on the public **Holosoma** framework (Amazon FAR, 2025).

## Citation

Paper: **[arXiv:2608.00500](https://arxiv.org/abs/2608.00500)**

```bibtex
@article{zhou2026ddc,
  title   = {A Change of Frame Makes Balance Observable: Distillation-Free Humanoid Single-Leg Stance},
  author  = {Zhou, Yikai and Wang, Xingyun and Cui, Jieming and Chen, Bozhou and Fan, Yikai and Zhu, Yixin and Li, Wenxin},
  journal = {arXiv preprint arXiv:2608.00500},
  year    = {2026}
}
```

See also `CITATION.cff`. If you use the motion set, please **also** cite AMS (see `data/NOTICE`).
