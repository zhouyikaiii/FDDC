# The DDC single-leg-balance benchmark — method-agnostic scoring

DDC's benchmark is **method-agnostic**: every policy is scored on the **same motion set** by the
**same outcome metric**, judged from the robot's **true physical state** — not from the policy's own
reference or reward. Where a method's observation and simulator can be cleanly reused, it runs in the
shared DDC G1 MuJoCo plant; where they are tightly coupled to the method's own deployment stack, it runs
in that method's **native MuJoCo deployment harness** (more faithful than force-porting it). This
directory documents both, and how the paper's eight baselines were scored (Table 1).

## How it works

**Shared across all methods:**

1. **Motion set** — the same 90 held-out single-leg clips (`../data/data_stratified_900/test`).
2. **Outcome metric** — the Perfect / Marginal / Failure tiers, computed from the robot's **true physical
   state** (`eval/metrics.py`; the native-harness methods reach the same tiers via `success_metric.py`).
3. **Protocol** — initialized from the reference's first frame, judged from the true state, sanity-first.

**Physics plant — shared for some, native for others:**

- **Shared DDC G1 plant** (`eval/wbt_rollout.py` `G1Sim`, 50 Hz PD): **ProtoMotions, MOSAIC, SONIC,
  HoloMotion** — their observation/sim reuse the shared kernel.
- **The method's own upstream MuJoCo** (its model / PD / timestep / sim wrapper): **GMT, TWIST,
  OmniXtreme, Humanoid-GPT** — their obs/sim are tightly coupled, so we run their native deployment
  harness rather than force-port them (arguably fairer: each is scored in the sim it was validated on).

So the benchmark guarantees the **same motions + same outcome metric + same true-state judging** for
every method — it is **not** a byte-identical physics plant for all eight. Each method's **adapter**
assembles that method's observation, runs its network, and decodes to joint targets; `robot_meta.onnx`
carries the Unitree-G1 constants the shared-plant methods use for PD.

> **Metric-parity note.** `eval/metrics.py` includes HuB's 0.5 m 12-keypoint tracking gate; the
> `success_metric.py` used by the native-harness baselines does not compute it. This changes no baseline
> number — the gate is an anti-gaming check that only binds once a method *sustains* single-leg, and every
> baseline is 0/90 sustained — but the two success functions are not byte-identical in code.

## Reproducing the eight baselines (paper Table 1)

`baselines/` holds the **exact adapters used in the paper** (one per baseline), plus the shared helpers
(`success_metric.py`, `fulllog*.py`). They are provided so the 0/90 baseline result is reproducible and
transparent.

> **These are not turnkey.** Each baseline runs its *own* policy, and several run their *own* sim/obs
> code, so an adapter needs that baseline's **repository and released weights, obtained from the original
> authors** (we do not redistribute third-party code or weights). Per-baseline setup — repo, weights,
> extra dependencies, and the exact command — is in [`baselines/SETUP.md`](baselines/SETUP.md).

The adapters import the shared kernel from `../eval/`, so run them with that on the path, e.g.:

```bash
cd benchmark/baselines
PYTHONPATH=../../eval  GMT_REPO=/path/to/GMT  GMT_WEIGHTS=$GMT_REPO/assets/pretrained_checkpoints/pretrained.pt \
    python gmt_eval.py runs  ../../data/data_stratified_900/test  gmt  0 1  ./out    # metrics sub-command = "runs"
```

## Scoring your own policy

The simplest worked example is the DDC path itself: `../eval/run_eval.py` + `../eval/fast_policy.py`
score a policy that already speaks the WBT observation. To score a policy with a **different**
observation / action space, write an adapter following the pattern in `baselines/`:

- read the shared sim state and the per-frame motion reference from `wbt_rollout` (`G1Sim`,
  `load_motion_npz`, the reference terms),
- assemble your policy's observation, run your network,
- return joint-position targets for the shared PD (`kp`/`kd`/default pose from `robot_meta.onnx`),
- let `metrics.compute_metrics` judge the outcome from the true state.

`mosaic_eval.py` and `proto_eval.py` are the lightest examples (they drive the shared `wbt_rollout`
plant directly and need only their weights, no external repo).
