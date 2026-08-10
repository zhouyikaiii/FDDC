"""Humanoid-GPT faithful harness evaluation (its own sim/obs/action full pipeline, zero rewiring).
Usage:
  python hgpt_eval.py sanity                         # end-to-end sanity with its bundled walking clip
  python hgpt_eval.py run <motion_dir> <tag> <shard> <nsh> <K> <outdir>   # batch over my single-foot clips
"""
import os, sys, json, glob, re, time
os.environ.setdefault("MUJOCO_GL", "egl")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
_HGPT_REPO = os.environ["HUMANOID_GPT_REPO"]
sys.path.insert(0, _HGPT_REPO)
if len(sys.argv) > 2 and sys.argv[1] in ("run", "runs", "fullruns"):
    sys.argv[2] = os.path.abspath(sys.argv[2])   # resolve motion_dir to absolute before we chdir into the repo
os.chdir(_HGPT_REPO)                              # its tracking module reads storage/... relative to the repo root, so run from there
import numpy as np, mujoco
_HS=_ROOT
os.environ.setdefault("WBT_EVAL_ROBOT_XML", os.path.join(_HS, "robot", "g1_29dof", "g1_29dof.xml"))
if len(sys.argv)>2 and sys.argv[1] in ("runs","fullruns"): os.environ["WBT_EVAL_MOTION_DIR"]=sys.argv[2]
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HS, "eval"))
from success_metric import foot_fz as FZ, success_of
from tracking import constants as consts
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.infer_utils import (G1TrackMjSim, G1TrackInferFn, g1_infer_env_config, apply_ema_qpos)
from tracking.policy import Args as PolicyArgs, get_policy_onnx
try:  # force onnxruntime single-thread (avoid saturating cores)
    import onnxruntime as _ort
    _OS = _ort.InferenceSession
    def _S1(*a, **k):
        so = k.pop("sess_options", None) or _ort.SessionOptions()
        so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        return _OS(*a, sess_options=so, **k)
    _ort.InferenceSession = _S1
except Exception: pass

ONNX = os.environ["HGPT_ONNX"]
FREQ = 50; CTRL_DT = 1.0 / FREQ

def build_harness():
    env_cfg = g1_infer_env_config(ctrl_dt=CTRL_DT)
    try:
        _pa = PolicyArgs(load_path=ONNX, policy_type="mlp")       # the upstream revision used in the paper
    except TypeError:
        _pa = PolicyArgs(onnx_track=ONNX, policy_type="mlp")      # later upstream renamed the arg load_path -> onnx_track
    policy = get_policy_onnx(_pa)
    mj_sim = G1TrackMjSim(init_qpos=consts.DEFAULT_QPOS.copy(), headless=True, ctrl_dt=CTRL_DT)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, policy, privileged=False)
    convert_model = mujoco.MjModel.from_xml_path(str(consts.TRACK_XML))
    return env_cfg, mj_sim, infer_fn, convert_model

def make_ref(convert_model, qpos, freq_src):
    qpos = apply_ema_qpos(np.float32(qpos))
    return qpos2kpt(convert_model, qpos, freq_src=float(freq_src), freq_tgt=FREQ,
                    interp_sec=0.5, end_default_sec=0.5, debug=False,
                    foot_contact_est=False, height_clip_mode=None, video_path=None)

def idx(ref, t):
    return {k: (v[t][None] if isinstance(v, np.ndarray) else v) for k, v in ref.items()}

_FKD = {}
def foot_z(model, qpos):
    """FK: return (left foot z, right foot z), world height of ankle_roll_link."""
    if "d" not in _FKD:
        _FKD["d"] = mujoco.MjData(model)
        _FKD["lf"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        _FKD["rf"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    d = _FKD["d"]; d.qpos[:len(qpos)] = qpos; mujoco.mj_kinematics(model, d)
    return float(d.xpos[_FKD["lf"], 2]), float(d.xpos[_FKD["rf"], 2])

_QV = []
def rollout(mj_sim, infer_fn, ref_traj):
    """Run one reference, return per-frame qpos log + fall info. sim starts from DEFAULT_QPOS (its deployment protocol)."""
    state = mj_sim.init_state(); state = mj_sim.reset(state)
    infer_fn.info["last_action"][:] = 0.0; infer_fn.info["step"] = 0
    T = len(ref_traj["qpos"]); hs = []; fzs = []; _QV.clear()
    gm = mj_sim.mj_model
    _lf = mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    _rf = mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    for t in range(T):
        ref_curr = idx(ref_traj, t); ref_next = idx(ref_traj, min(t + 1, T - 1))
        mt = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})
        state = mj_sim.step(state, mt)
        hs.append(state.mj_data.qpos.copy())
        _QV.append(state.mj_data.qvel.copy())
        fzs.append([FZ(gm, state.mj_data, _lf), FZ(gm, state.mj_data, _rf)])
    return np.array(hs), ref_traj, np.array(fzs)

def tilt_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    gz = -(1.0 - 2.0 * (x * x + y * y))  # -R.T[...,2].z, gz=-1 when upright
    return float(np.arccos(np.clip(-gz, -1, 1)))

if sys.argv[1] == "sanity":
    env_cfg, mj_sim, infer_fn, convert_model = build_harness()
    print("DEFAULT_QPOS z=", consts.DEFAULT_QPOS[2])
    d = dict(np.load(f"{os.environ['HUMANOID_GPT_REPO']}/storage/test/human_walking_50Hz_29dof.npz", allow_pickle=True))
    qpos = d["qpos"]; print("walking clip qpos", qpos.shape)
    ref = make_ref(convert_model, qpos, 50)
    print("ref keys:", [k for k in ref.keys()])
    print("ref len:", len(ref["qpos"]), " (includes head/tail ease-in/out)")
    hs, _, _ = rollout(mj_sim, infer_fn, ref)
    heights = hs[:, 2]; tilts = np.array([np.degrees(tilt_of(q[3:7])) for q in hs])
    fell = (heights < 0.4).any() or (tilts > 70).any()
    ffi = int(np.argmax((heights < 0.4) | (tilts > 70))) if fell else len(hs)
    print(f"walking sanity: {'fell' if fell else 'stood/completed'} @step{ffi}/{len(hs)} final_h={heights[-1]:.3f} max_tilt={tilts.max():.1f}deg final_tilt={tilts[-1]:.1f}deg")
    print("-> if completed without falling = full harness wiring correct")
    sys.exit(0)

if sys.argv[1] == "run":
    motion_dir, tag, shard, nsh, K, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    of = f"{outdir}/{tag}__sh{shard}.json"
    if os.path.exists(of): print(f"[skip]{of}"); sys.exit(0)
    os.makedirs(outdir, exist_ok=True)
    env_cfg, mj_sim, infer_fn, convert_model = build_harness()
    NOM = float(consts.DEFAULT_QPOS[2])  # 0.78
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    my = mots[shard::nsh]; res = {}; t0 = time.time()
    for mid in my:
        d = dict(np.load(f"{motion_dir}/sample_{mid}_mj.npz", allow_pickle=True))
        qpos = d["joint_pos"]  # (T,36) model order wxyz
        fps = float(np.array(d["fps"]).reshape(-1)[0])
        try:
            ref = make_ref(convert_model, qpos, fps)
            hs, ref_traj, _ = rollout(mj_sim, infer_fn, ref)
        except Exception as e:
            res[mid] = {"err": str(e)[:120]}; print(f"[err]{mid}:{str(e)[:80]}"); continue
        heights = hs[:, 2]; tilts = np.array([tilt_of(q[3:7]) for q in hs])  # rad
        bad = (heights < 0.5 * NOM) | (tilts > 1.0)
        fell = bool(bad.any()); ffi = int(np.argmax(bad)) if fell else len(hs)
        T = len(hs); Tr = len(ref_traj["qpos"])
        # joint tracking error (robot vs reference, model-order qpos[7:]), only up to fall
        upto = ffi if fell else T
        refq = ref_traj["qpos"][:T, 7:]; robq = hs[:, 7:]
        jerr = float(np.abs(robq[:upto] - refq[:upto]).mean()) if upto > 5 else float("nan")
        # foot-height FK: swing-foot peak lift (robot vs reference), to judge whether truly single-foot
        gm = mj_sim.mj_model
        rob_fz = np.array([foot_z(gm, hs[t]) for t in range(min(upto, T))])   # (upto,2)
        ref_fz = np.array([foot_z(gm, ref_traj["qpos"][t]) for t in range(T)])  # (T,2)
        ground = 0.03  # approx foot height when standing
        # swing foot = the one lifted higher in the reference; take its peak lift
        ref_lift = float(ref_fz.max(axis=0).max() - ground)
        swing_side = int(np.argmax(ref_fz.max(axis=0)))  # 0 left 1 right
        rob_lift = float(rob_fz[:, swing_side].max() - ground) if len(rob_fz) else 0.0
        lift_ratio = rob_lift / ref_lift if ref_lift > 0.02 else float("nan")
        # strict success: not fallen + swing-foot lift >= 50% of reference + max tilt < 40deg
        strict = bool((not fell) and (lift_ratio >= 0.5) and (np.degrees(tilts.max()) < 40.0))
        res[mid] = {"fell": fell, "fall_step": ffi, "T": T, "completion": ffi / T,
                    "final_h": float(heights[-1]), "max_tilt_deg": float(np.degrees(tilts.max())),
                    "joint_track_err": jerr, "ref_lift": ref_lift, "rob_lift": rob_lift,
                    "lift_ratio": lift_ratio, "strict_success": strict}
    json.dump({"tag": tag, "K": K, "harness": "humanoid_gpt_native", "per_motion": res}, open(of, "w"))
    n = len([v for v in res.values() if "fell" in v])
    fr = np.mean([v["fell"] for v in res.values() if "fell" in v]) if n else 0
    print(f"[done]{tag} sh{shard} {len(my)}mot fall_rate={fr:.2%} {time.time()-t0:.0f}s")
    sys.exit(0)

if sys.argv[1] == "runs":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(outdir, exist_ok=True)
    env_cfg, mj_sim, infer_fn, convert_model = build_harness()
    gm = mj_sim.mj_model; mass = float(np.sum(gm.body_mass)); NOM = float(consts.DEFAULT_QPOS[2])
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        d = dict(np.load(f"{motion_dir}/sample_{mid}_mj.npz", allow_pickle=True))
        try:
            ref = make_ref(convert_model, d["joint_pos"], float(np.array(d["fps"]).reshape(-1)[0]))
            hs, ref_traj, fzs = rollout(mj_sim, infer_fn, ref)
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}; continue
        # qpos2kpt pads 0.5s at head and tail (25 frames @50Hz), align back to original clip segment
        Torig = len(d["joint_pos"]); off = round(0.5 * FREQ)
        sl = slice(off, min(off + Torig, len(hs)))
        base_z = hs[sl, 2]; tilt = np.array([tilt_of(q[3:7]) for q in hs[sl]])
        res[mid] = success_of(fzs[sl, 0], fzs[sl, 1], base_z, tilt, NOM, 0.02, mass, mid)
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot metrics.py_success={np.mean(sv):.1%}")
    sys.exit(0)

if sys.argv[1] == "fullruns":
    import fullmetrics_post as FP
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(outdir, exist_ok=True)
    env_cfg, mj_sim, infer_fn, convert_model = build_harness()
    gm = mj_sim.mj_model; NOM = float(consts.DEFAULT_QPOS[2])
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        d = dict(np.load(f"{motion_dir}/sample_{mid}_mj.npz", allow_pickle=True))
        try:
            ref = make_ref(convert_model, d["joint_pos"], float(np.array(d["fps"]).reshape(-1)[0]))
            hs, ref_traj, fzs = rollout(mj_sim, infer_fn, ref)
            qv = np.array(_QV)
            Torig = len(d["joint_pos"]); off = round(0.5 * FREQ)
            sl = slice(off, min(off + Torig, len(hs)))
            res[mid] = FP.compute(hs[sl], qv[sl], fzs[sl], gm, mid, NOM, 50.0)
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot success={np.mean(sv) if sv else float('nan'):.1%}")
    sys.exit(0)
