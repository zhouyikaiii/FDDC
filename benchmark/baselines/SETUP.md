# Reproducing the eight baselines (paper Table 1)

Each adapter here is the **exact script used to score that baseline in the paper**. An adapter runs the
baseline's *own* policy — and, for **GMT / TWIST / OmniXtreme / Humanoid-GPT**, the baseline's *own*
observation and MuJoCo sim (its native deployment harness); for **ProtoMotions / MOSAIC / SONIC /
HoloMotion**, the shared DDC G1 plant (`../../eval/wbt_rollout.py`). Every method is scored on the
**same motion set** by the **same outcome metric** (`../../eval/metrics.py`), judged from the robot's
true physical state — but the physics plant is not byte-identical across all eight (see the plant note
in [`../README.md`](../README.md)).

We do **not** redistribute third-party code or weights: you obtain each baseline's **repository and
released weights from its authors** (links below), then point the adapter at them via environment
variables. The Unitree-G1 constants come from the bundled `robot_meta.onnx`.

Expected outcome: **all eight baselines score 0/90 Perfect** (they never hold a clean single-leg stance;
they stay up by hopping / stepping, or fall). Per-baseline Marginal/Failure are in the table below and
match the paper's Table 1.

---

## 1. Common setup (how every run is invoked)

```bash
cd benchmark/baselines
export PYTHONPATH="$(pwd)/../../eval"          # the shared kernel: wbt_rollout.py + metrics.py + fast_policy.py
export WBT_ORT_THREADS=1 OMP_NUM_THREADS=1     # cap threads (recommended)
DATA="$(pwd)/../../data/data_stratified_900/test"   # the 90 held-out test clips (run download_data.py first)
```

- **Python environment.** The lightweight baselines (ProtoMotions, MOSAIC, SONIC) need only
  `mujoco`, `onnx`, `onnxruntime`, `numpy` (and `torch` for TWIST/GMT weights). The repo-based baselines
  (GMT, TWIST, OmniXtreme, Humanoid-GPT, HoloMotion) additionally need **that baseline's own repo
  installed with its requirements** (each was run in its own conda env to avoid dependency clashes) —
  plus `mujoco_viewer` (GMT, OmniXtreme) or `warp-lang` (HoloMotion).
- **Device.** The eval runs on **CPU** (the shared MuJoCo kernel is CPU). If a repo-based baseline's own
  deploy code auto-selects CUDA and ends up mixing CPU/GPU tensors, run it CPU-only by exporting
  `CUDA_VISIBLE_DEVICES=""` (OmniXtreme sets this itself on startup — see its note below).
- **Sanity first.** Every adapter has a `sanity` sub-command that reports whether the wiring is correct —
  it tracks a normal motion (or, for ProtoMotions, which has no bundled normal clip, the opening
  double-support phase of a few test clips) without an immediate fall. Run `python <baseline>_eval.py sanity`
  and confirm it passes before trusting the single-leg number (paper §4.2 — a method that fails sanity is
  a wiring bug, not a real failure).
- **The metrics run** writes a per-motion JSON (`{tag}__sh{shard}.json` etc.) with `success` / `fell`
  per clip. Aggregate them (below) into Perfect / Marginal / Failure.

## 2. Per-baseline: repo, weights, env vars, and command

Run the full test set with `shard=0 nshards=1`. Use the **metrics sub-command** — the one that calls the
shared `metrics.py` and writes the unified `success` field. **Do not use a diagnostic branch** such as
mosaic's `run` (it emits `strict_success`, not `success`, so the aggregator below would silently read a
missing field as 0 and report a spurious 0% Perfect). The metrics sub-command differs per adapter (an
artifact of the original per-method scripts): ProtoMotions is positional; GMT / TWIST / OmniXtreme /
Humanoid-GPT / MOSAIC / SONIC use `runs`; HoloMotion uses `runm`. Confirm the output JSON has a
`success` field; see each adapter's docstring.

| Baseline | Repo / weights (from the authors) | Env vars | Command (from `benchmark/baselines/`) |
|---|---|---|---|
| **ProtoMotions** (weights only) | [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) → `unified_pipeline.onnx` | `PROTO_ONNX` | `PROTO_NOISE=0 PROTO_DELAY=0 python proto_eval.py "$DATA" proto 0 1 1 ./out_proto` |
| **MOSAIC** (weights only) | [BAAI-Humanoid/MOSAIC](https://github.com/BAAI-Humanoid/MOSAIC), HF [BAAI-Humanoid/MOSAIC_Model](https://huggingface.co/BAAI-Humanoid/MOSAIC_Model) → `gmt.onnx` | `MOSAIC_ONNX` | `python mosaic_eval.py runs "$DATA" mosaic 0 1 ./out_mosaic` |
| **SONIC** (weights only) | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), HF [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) → the **root** `model_encoder.onnx` (obs **1762**) + `model_decoder.onnx` (obs 994). **NOT** the `low_latency/` variant (1247-dim) — this adapter matches the root model. | `SONIC_DIR` (dir with both root ONNX) | `python sonic_eval.py runs "$DATA" sonic 0 1 ./out_sonic` |
| **GMT** (repo) | [zixuan417/humanoid-general-motion-tracking](https://github.com/zixuan417/humanoid-general-motion-tracking); weights `assets/pretrained_checkpoints/pretrained.pt` in-repo. Needs `mujoco_viewer`. | `GMT_REPO`, `GMT_WEIGHTS` | `python gmt_eval.py runs "$DATA" gmt 0 1 ./out_gmt` |
| **TWIST** (repo) | [YanjieZe/TWIST](https://github.com/YanjieZe/TWIST) → `twist_general_motion_tracker.pt` (TorchScript); adapter reads `$TWIST_REPO/assets/g1/g1_sim2sim_with_wrist_roll.xml`. | `TWIST_REPO`, `TWIST_WEIGHTS` | `python twist_eval.py runs "$DATA" twist 0 1 ./out_twist` |
| **OmniXtreme** (repo) | [Perkins729/OmniXtreme](https://github.com/Perkins729/OmniXtreme) → `policy/{base_policy_trt,residual_policy,fk_trt}.onnx`; runs its `deploy_mujoco.DeployNode`. Needs `mujoco`. **Runs CPU-only automatically** (see note). | `OMNI_REPO`, `OMNI_DIR` (=`$OMNI_REPO/policy`) | `python omni_eval.py runs "$DATA" omni 0 1 ./out_omni` |
| **Humanoid-GPT** (repo) | [GalaxyGeneralRobotics/Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) → `pns_wo_priv216.onnx`; runs its `tracking` module (`G1TrackMjSim`). **The adapter `chdir`s into `$HUMANOID_GPT_REPO` itself and is upstream-version-tolerant — see note.** | `HUMANOID_GPT_REPO`, `HGPT_ONNX` | `python hgpt_eval.py runs "$DATA" hgpt 0 1 ./out_hgpt` |
| **HoloMotion** (repo) | [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion), HF [HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models) → `model_14000.onnx`; uses its warp obs kernel. Needs `warp-lang`. | `HOLO_REPO`, `HOLO_ONNX` | `python holo_eval.py runm "$DATA" holo 0 1 ./out_holo` |

> **SONIC gotcha:** `nvidia/GEAR-SONIC` ships two variants. Use the **root** model (`model_encoder.onnx`
> is 1762-dim), whose SHA256 matches the file used in the paper; the `low_latency/` subfolder is a
> different 1247-dim model that this adapter does **not** match. Point `SONIC_DIR` at the root files.

> **OmniXtreme:** it runs **CPU-only** and **deterministically**. (i) Its deploy auto-selects CUDA and
> would place some tensors on the GPU while the shared kernel stays on CPU — a mismatch that errors on all
> 90 clips — so the adapter sets `CUDA_VISIBLE_DEVICES=""` itself on startup (before any torch/deploy
> import; export it yourself to override). (ii) Its policy is a flow model that samples an `initial_noise`
> (`torch.randn`) every run, and its observations carry upstream noise (`noise_scales`, e.g.
> `base_ang_vel=0.1`); the shipped adapter reseeds Python/NumPy/PyTorch per clip, runs single-threaded, and
> defaults `OMNI_CLEAN=1` (zeros the obs noise + action delay for the paper's clean condition). A single
> `OMNI_SEED` is bit-reproducible, but OmniXtreme is a flow model whose `initial_noise` sample decides a
> handful of borderline clips: **Perfect is a deterministic 0 / 90** (it never holds single-leg), while the
> Marginal / Failure split shifts a few clips across seeds/runs (with the consistent-metrics scoring,
> observed **0.0 / 2.2–10.0 / 90.0–97.8** over `OMNI_SEED=0,1,2`; and, because the upstream flow draws are
> not fully seeded, it also drifts run-to-run at a fixed seed — the continuous metrics, e.g. MoS, drift with
> it since they follow OmniXtreme's own trajectory). The paper reports the default-seed (`OMNI_SEED` unset)
> sample **0.0 / 6.7 / 93.3**; treat it as one sample, not an exact target. Set `OMNI_SEED=<n>` for a
> different fixed draw, or `OMNI_CLEAN=0` for the upstream noisy config.

> **Humanoid-GPT** (targets the Humanoid-GPT version used in the paper). The adapter now **self-handles**
> the two upstream quirks below, so no manual `cd` or source edit is needed — you only set
> `HUMANOID_GPT_REPO` and `HGPT_ONNX` and run it from anywhere:
> 1. **Working directory** — its `tracking` module reads `storage/...` *relative to the repo*, so the
>    adapter `os.chdir`s into `$HUMANOID_GPT_REPO` on startup (resolving the motion dir to an absolute
>    path first, so a relative `$DATA` still works).
> 2. **Upstream API drift** — a later upstream renamed the policy-loading argument `load_path` →
>    `onnx_track`. The adapter tries `load_path` first and falls back to `onnx_track`, so it runs on
>    either revision. If your checkout diverged further than this one argument, pin the upstream commit
>    the adapter targets.

> **ProtoMotions — clean vs noisy.** ProtoMotions is the **only** adapter that defaults to the *noisy*
> condition (`PROTO_NOISE=0.20 PROTO_DELAY=1`); every other adapter defaults to clean. Table 1 is the
> **clean** condition, so run ProtoMotions with `PROTO_NOISE=0 PROTO_DELAY=0`. This changes no tier
> (ProtoMotions is 0/90 Perfect either way) but does shift the continuous margins (§4).

Example (ProtoMotions, end to end):

```bash
PROTO_ONNX=/path/to/unified_pipeline.onnx  PROTO_NOISE=0 PROTO_DELAY=0 \
    python proto_eval.py "$DATA" proto 0 1 1 ./out_proto
```

Example (GMT, needs its repo cloned + `mujoco_viewer` installed in the active env):

```bash
GMT_REPO=/path/to/humanoid-general-motion-tracking \
GMT_WEIGHTS=$GMT_REPO/assets/pretrained_checkpoints/pretrained.pt \
    python gmt_eval.py runs "$DATA" gmt 0 1 ./out_gmt
```

## 3. Aggregate a run into Perfect / Marginal / Failure

```bash
python - <<'PY'
import json, glob
pm = {}
for f in glob.glob("./out_proto/*.json"):       # <- the run's output dir
    pm.update(json.load(open(f)).get("per_motion", {}))
assert pm and "success" in next(iter(pm.values())), \
    "no 'success' field -> wrong sub-command; use the metrics one (runs / positional / runm)"
s  = [v["success"] for v in pm.values()]         # Perfect (unified 'success' from metrics.py)
fl = [v["fell"]    for v in pm.values()]         # Failure
n = len(s)
P, F = 100*sum(s)/n, 100*sum(fl)/n
print(f"n={n}  Perfect={P:.1f}%  Marginal={max(0,100-P-F):.1f}%  Failure={F:.1f}%")
PY
```

## 4. Continuous metrics (Table 1 balance columns + Appendix G)

Section 3 gives the Perfect / Marginal / Failure tiers. The **continuous** columns — Margin-of-Stability
and xCoM-out (Table 1), plus the full Appendix G suite (support-foot slippage, keypoint tracking error,
jerk, time-to-fall) — come from each adapter's **continuous** sub-command, which writes the full
per-motion metric fields; average each field over the 90 clips.

| Adapter | Continuous sub-command |
|---|---|
| ProtoMotions | its normal positional run (already writes the full fields) |
| MOSAIC, SONIC | `fullruns` (in place of `runs`) |
| GMT, TWIST | `fullruns` (in place of `runs`) |
| OmniXtreme, Humanoid-GPT | `fullruns` (in place of `runs`) — via `fullmetrics_post.py` |
| HoloMotion | `runm` (already the full-metric sub-command) |

e.g. `CUDA_VISIBLE_DEVICES="" python omni_eval.py fullruns "$DATA" omni 0 1 ./out_omni`. Every adapter's
continuous output carries the same `metrics.py` fields (all averaged over the 90 clips):

| Paper quantity | `metrics.py` field |
|---|---|
| MoS (mediolateral / fore–aft) | `xcom_margin_ml_min` / `xcom_margin_ap_min` |
| xCoM-out (time outside support) | `xcom_margin_viol_dur` |
| support-foot slippage (mm/s) | `slippage_mm_s` |
| keypoint tracking error | `track_Epos` (+ `track_Evel`, `track_Eacc`) |
| jerk | `action_jerk_rms`, `dof_vel_jerk_rms` |
| time-to-fall | `time_to_fall` |

```bash
python - <<'PY'
import json, glob, numpy as np
pm = {}
for f in glob.glob("./out_omni/*.json"):          # <- the continuous run's output dir
    pm.update(json.load(open(f)).get("per_motion", {}))
rows = [v for v in pm.values() if "xcom_margin_ap_min" in v]
mean = lambda k: float(np.mean([r[k] for r in rows]))
print(f"n={len(rows)}  MoS_ml/ap={mean('xcom_margin_ml_min'):.3f}/{mean('xcom_margin_ap_min'):.3f}  "
      f"xCoM-out={mean('xcom_margin_viol_dur'):.2f}s  slip={mean('slippage_mm_s'):.0f}mm/s  "
      f"jerk={mean('action_jerk_rms'):.2f}")
PY
```

> **Reading these numbers.** Table 1 / Appendix G are the **clean** condition (K=1, no observation noise);
> run ProtoMotions with `PROTO_NOISE=0 PROTO_DELAY=0` (see its note above — the one adapter that defaults to
> noisy). For OmniXtreme / Humanoid-GPT the action channel is recovered from the recorded joint positions
> (their harness runs its own loop), so `action_jerk_rms` is a position-jerk proxy; `dof_vel_jerk_rms` uses
> the true joint velocities and is directly comparable. These continuous diagnostics reproduce up to
> condition-consistency, not necessarily cross-machine bit-exact — a small numerical spread on the
> diagnostic margins is expected; the Perfect / Marginal / Failure tiers are robust.

## 5. Expected results (paper Table 1, clean, n=90)

| Baseline | Perfect | Marginal | Failure |
|----------|:-------:|:--------:|:-------:|
| ProtoMotions | 0.0 | 51.1 | 48.9 |
| OmniXtreme   | 0.0 | 6.7† | 93.3 |
| GMT          | 0.0 | 18.9 | 81.1 |
| MOSAIC       | 0.0 | 66.7 | 33.3 |
| TWIST        | 0.0 | 47.8 | 52.2 |
| Humanoid-GPT | 0.0 | 75.6 | 24.4 |
| HoloMotion   | 0.0 | 75.6 | 24.4 |
| SONIC        | 0.0 | 81.1 | 18.9 |

> † OmniXtreme's Marginal/Failure split is one fixed-seed sample: Perfect is a robust 0/90, but the split
> shifts a few borderline clips across seeds/runs (flow-model `initial_noise` — see the OmniXtreme note in §2).

The weights-only adapters (ProtoMotions, MOSAIC, SONIC) were spot-checked to reproduce **0/90 Perfect**
directly from this released package (ProtoMotions full-90: Perfect 0.0 / Failure 48.9, matching the
table). The repo-based baselines reproduce the same Perfect = 0 once their repo + weights are supplied.

## 6. Robot metadata

`robot_meta.onnx` is a tiny (~3 KB) metadata-only ONNX carrying the Unitree-G1 constants the adapters
read for the shared PD (`dof_names`, `kp`, `kd`, `action_scale`, default pose). Override with
`DDC_ROBOT_META_ONNX`. These constants equal `../../policy/robot_config.json`.
