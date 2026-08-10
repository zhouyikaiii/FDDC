# Training DDC

This directory documents **how DDC is trained**, so the method is transparent and reproducible in
principle. DDC is implemented on top of the public **Holosoma** framework (Amazon FAR, Apache-2.0);
what is *specific to DDC* is a small set of files — the deployable dynamic-CoM observation and the
human-science reward library — collected here under `fddc_src/` with the exact training command in
`train_full.sh`. We do **not** vendor the whole framework: install upstream Holosoma to run this.

> The released policy (`../policy/model_0262000.pt`) and the benchmark (`../eval/`) are self-contained
> and need none of this. This folder is for readers who want to see or reproduce the training itself.

## What is DDC-specific (method → file)

Everything else in a Holosoma checkout is upstream; DDC lives in these files (all under
`fddc_src/`, mirroring the Holosoma package layout `src/holosoma/holosoma/…`):

| File | What it implements (paper) |
|------|----------------------------|
| `managers/observation/terms/wbt.py` | **The deployable support-relative dynamic-CoM observation** — `whole_body_com_rel_support_center` (the actor term; §3.1). The base linear velocity cancels in the CoM-minus-support-foot difference, so it is reconstructible from encoders + IMU alone. Also the privileged-critic `whole_body_xcom_rel_support_center`. |
| `managers/reward/terms/wbt.py` | **The human-science balance reward library** (§3.2): capture-point / support-polygon **Margin-of-Stability** (`_support_polygon_margin`, `_convex_hull_halfspace_margin`), the xCoM computation (`_xcom_xy`), and the support determination the balance rewards use. |
| `config_values/wbt/g1/reward.py` | **Reward weights = paper Table 6** (e.g. `support_xcom_polygon_margin` −20 (MoS), `xcom_ttb` −15 (time-to-boundary), stance ankle/knee action-rate, action jerk). |
| `config_values/wbt/g1/experiment.py` | **FastSAC training hyperparameters = paper Table 6.** |
| `config_values/wbt/g1/observation.py` | Wires the actor / critic observation groups (deployable actor terms vs privileged critic terms). |
| `train_full.sh` | The multi-GPU FastSAC launch command for the `g1-29dof-wbt-fast-sac` experiment. |

The `fddc_src/` files are **modified Holosoma files** (Apache-2.0; see `NOTICE`). To reproduce, drop
them into the matching paths of a Holosoma checkout (`src/holosoma/holosoma/…`), replacing the upstream
versions.

## Reproducing the training

**1. Install the framework** (not vendored here):

- **Holosoma** (Apache-2.0): https://github.com/amazon-far/holosoma — `pip install -e src/holosoma`
  (the DDC extras `[unitree,booster]` are for real-robot deploy only and are *not* needed for training).
- **IsaacSim 5.1.0** — NVIDIA, installed via pip wheels under the NVIDIA license/EULA.
- **IsaacLab v2.3.0**: https://github.com/isaac-sim/IsaacLab — `./isaaclab.sh --install`.

**2. Environment** (the configuration the released run used; a from-scratch recipe is common with any
recent CUDA GPU box):

| | version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7.0 + cu128 |
| IsaacSim | 5.1.0.0 |
| IsaacLab | v2.3.0 |
| rsl-rl-lib | 3.0.1 |
| numpy | 1.26.x (do **not** upgrade to 2.x — IsaacSim/numba compatibility) |
| wandb | 0.22.0 |

**3. Apply the DDC files** — run `bash apply_ddc_to_holosoma.sh <holosoma-checkout>`: it verifies the
checkout is at the pinned commit `5b61d5768bc8e44710e2983db6263e174193981c` and that the DDC files match
`fddc_src/SHA256SUMS`, then overlays the five files into `src/holosoma/holosoma/…`. (Or copy `fddc_src/*`
over the same paths manually.)

**4. Train** (multi-GPU — the paper used 2x RTX 3090; edit `CUDA_VISIBLE_DEVICES` / `NUM_ENVS` for your box):

```bash
# reproduce the deployed policy — 2x RTX 3090 (paper Table 6): 2 GPUs, 8192 total envs (4096/GPU), 400k iters.
# The deployed checkpoint model_0262000.pt is step 262k, selected on the val split (not the final iter).
CUDA_VISIBLE_DEVICES=0,1 NUM_ENVS=8192 ITERS=400000 \
    HOLOSOMA_ROOT=/path/to/holosoma-checkout DATA=/path/to/data_stratified_900/train \
    bash train_full.sh
```

`HOLOSOMA_ROOT` must point at the Holosoma checkout you overlaid the DDC files onto in step 3 — the script
`cd`s there (the framework `train_agent.py` is not vendored in this release) and runs (from `$HOLOSOMA_ROOT`,
see `train_full.sh` for the overridable knobs):

```bash
torchrun --standalone --nproc_per_node=2 src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac  logger:wandb  --logger.video.enabled=False \
    --training.num-envs=8192 \
    --algo.config.buffer_size=384 \
    --algo.config.num_learning_iterations=400000 \
    "--command.setup_terms.motion_command.params.motion_config.motion_dir=$DATA"
```

## Training data

Train on the released stratified set: **`../data/data_stratified_900/train`** (720 clips). The deployed
DDC checkpoint (`model_0262000.pt`, step 262k) was trained on this set; checkpoints are selected on
`../data/data_stratified_900/val` and all numbers reported on `../data/data_stratified_900/test`
(paper §4.1 / Appendix E). The motions are GPL-3.0 (derived from AMS; see `../data/NOTICE`).

## Caveats (the honest version)

- **Heavy stack.** Training needs IsaacSim (NVIDIA EULA) + multiple GPUs (the released run used 2× RTX
  3090). Most users will only ever run the *benchmark* (`../eval/`), which needs none
  of this.
- **Version drift.** The `fddc_src/` files are DDC's modifications to Holosoma at upstream commit
  `5b61d5768bc8e44710e2983db6263e174193981c` (`amazon-far/holosoma`, 2026-07-24 — the version behind
  Table 6). To reconstruct the training stack, check out Holosoma at that commit and overlay these files;
  on a newer upstream, minor adaptation may be needed — the known trade-off of shipping the DDC layer
  rather than a frozen full-framework fork.
- **Exact reward weights and hyperparameters** are in `config_values/wbt/g1/{reward,experiment}.py` and,
  in paper form, in Table 6.
