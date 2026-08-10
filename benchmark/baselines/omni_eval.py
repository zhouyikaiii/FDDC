"""OmniXtreme faithful evaluation: directly run its deploy_mujoco.DeployNode.main_loop (its own obs/base+residual+fk/PD/sim),
only stub out viewer/visual, wrap mj_step to record qpos. Feed my single-foot clips (converted to its npz format).
Usage: python omni_eval.py sanity | run <motion_dir> <tag> <shard> <nsh> <outdir>
"""
import os, sys, glob, re, json, time, pickle
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # run CPU-only: its deploy auto-selects CUDA and would mix GPU tensors with the shared CPU kernel (errors on all 90); export CUDA_VISIBLE_DEVICES yourself to override. Must precede any torch/deploy import.
os.environ.setdefault("OMNI_CLEAN", "1")            # clean condition by default: zero the upstream obs noise (noise_scales) + action delay (paper "clean, no observation noise"). Set OMNI_CLEAN=0 to run the upstream noisy config.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
REPO = os.environ["OMNI_REPO"]; sys.path.insert(0, REPO); os.chdir(REPO)
import mujoco, torch, random
torch.set_num_threads(1)   # single-thread -> deterministic + don't saturate cores

_SEED = int(os.environ.get("OMNI_SEED", "0"))
def _seed_all(s=_SEED):
    """Fix Python/NumPy/PyTorch RNG so the upstream policy's initial_noise (torch.randn, deploy_mujoco.py)
    and any residual sampling are reproducible. OmniXtreme is a flow policy that REQUIRES an initial-noise
    input, so 'deterministic' means a FIXED (seeded) noise sample, not a zeroed one; we reseed identically
    per clip so each clip's result is independent of run order."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

PERM = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28])
# clip 51-body -> OmniXtreme 30-body (BeyondMimic native order) body remap
OMNI_IDX = [1,3,15,27,4,16,28,5,17,29,6,18,31,41,8,20,32,42,9,21,33,43,34,44,35,45,36,46,37,47]

# ---- stub viewer / visual to make it headless ----
import mujoco.viewer as _mjv
class _Dummy:
    def __init__(s): s.pos = np.zeros(3)
class _GeomList:
    def __init__(s): s._d = _Dummy()
    def __getitem__(s, i): return s._d
class _Scn:
    def __init__(s): s.geoms = _GeomList(); s.ngeom = 0
class StubViewer:
    def __init__(s, *a, **k): s.user_scn = _Scn()
    def sync(s): pass
    def close(s): pass
    def is_running(s): return True
_mjv.launch_passive = lambda *a, **k: StubViewer()

# ---- success_metric (unified metrics.py metric) ----
_HS = _ROOT
os.environ.setdefault("WBT_EVAL_ROBOT_XML", os.path.join(_HS, "robot", "g1_29dof", "g1_29dof.xml"))
if len(sys.argv) > 2 and sys.argv[1] in ("runs", "fullruns"):
    os.environ["WBT_EVAL_MOTION_DIR"] = sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HS, "eval"))
from success_metric import foot_fz as FZ, success_of

# ---- wrap mj_step to record qpos + fz ----
_REC = {"on": False, "qpos": [], "fz": [], "qvel": []}
_orig_step = mujoco.mj_step
def _rec_step(m, d, *a, **k):
    r = _orig_step(m, d, *a, **k)
    if _REC["on"]:
        _REC["qpos"].append(d.qpos.copy())
        _REC["qvel"].append(d.qvel.copy())
        if "_lf" not in _REC:
            _REC["_lf"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
            _REC["_rf"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        _REC["fz"].append([FZ(m, d, _REC["_lf"]), FZ(m, d, _REC["_rf"])])
    return r
mujoco.mj_step = _rec_step

try:  # force onnxruntime single-thread (avoid saturating cores; deploy's onnx session sets no SessionOptions)
    import onnxruntime as _ort
    _OS = _ort.InferenceSession
    def _S1(*a, **k):
        so = k.pop("sess_options", None) or _ort.SessionOptions()
        so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        return _OS(*a, sess_options=so, **k)
    _ort.InferenceSession = _S1
except Exception: pass
import deploy_mujoco as D
D.VISUAL = False
D.add_visual_capsule = lambda *a, **k: None
# force cpu (source line365 hardcodes 'cuda')
_orig_getmotion = D.get_npz_motion
D.get_npz_motion = lambda fp, dev="cpu": _orig_getmotion(fp, "cpu")

def clip_to_omni_npz(npz_in, npz_out, static=False):
    d = dict(np.load(npz_in, allow_pickle=True))
    jp = d["joint_pos"]; jv = d["joint_vel"]  # (T,36),(T,35)
    dof_urdf = jp[:, 7:].astype(np.float32)      # (T,29) URDF
    dofv_urdf = jv[:, 6:].astype(np.float32)
    T = len(dof_urdf)
    # store in motionlib order: out[:,PERM]=urdf -> its get_npz_motion's [:,PERM] restores URDF
    jpos_out = np.zeros_like(dof_urdf); jpos_out[:, PERM] = dof_urdf
    jvel_out = np.zeros_like(dofv_urdf); jvel_out[:, PERM] = dofv_urdf
    bpos = d["body_pos_w"][:, OMNI_IDX, :].astype(np.float32)   # (T,30,3)
    bquat = d["body_quat_w"][:, OMNI_IDX, :].astype(np.float32)  # (T,30,4) wxyz
    if static:  # standing still: all frames = frame0
        jpos_out = np.repeat(jpos_out[:1], T, 0); jvel_out = np.zeros_like(jvel_out)
        bpos = np.repeat(bpos[:1], T, 0); bquat = np.repeat(bquat[:1], T, 0)
    np.savez(npz_out, joint_pos=jpos_out, joint_vel=jvel_out, body_pos_w=bpos, body_quat_w=bquat)
    return T

def run_one(motion_npz):
    """Write policy/motion.npz, instantiate DeployNode, run main_loop, return recorded qpos (T,nq)."""
    import shutil
    shutil.copy(motion_npz, f"{REPO}/policy/motion.npz")
    _seed_all()                                      # reseed per clip -> initial_noise + sampling reproducible
    _REC["on"] = False; _REC["qpos"] = []; _REC["fz"] = []; _REC["qvel"] = []
    node = D.DeployNode()
    if os.environ.get("OMNI_CLEAN", "1") != "0":     # clean (default): zero obs noise + action delay; OMNI_CLEAN=0 -> upstream noisy config
        try:
            for _k in list(node.config.noise_scales.keys()): node.config.noise_scales[_k] = 0.0
        except Exception: pass
        for _key in ("action_depaly_decimation", "action_delay_decimation"):
            try: node.config[_key] = [0, 0]
            except Exception: pass
    _REC["on"] = True
    try:
        node.main_loop()
    except SystemExit:
        pass
    _REC["on"] = False
    return np.array(_REC["qpos"]), node.env.mj_model, np.array(_REC["fz"])

def foot_z_fk(model, qpos):
    if not hasattr(foot_z_fk, "d"):
        foot_z_fk.d = mujoco.MjData(model)
        foot_z_fk.lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        foot_z_fk.rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    dd = foot_z_fk.d; dd.qpos[:len(qpos)] = qpos; mujoco.mj_kinematics(model, dd)
    return float(dd.xpos[foot_z_fk.lf, 2]), float(dd.xpos[foot_z_fk.rf, 2])

def tilt_rad(q_wxyz):
    w, x, y, z = q_wxyz
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1, 1)))

if sys.argv[1] == "sanity":
    tmp = "/tmp/omni_sanity.npz"
    easy = sorted(glob.glob(os.path.join(_ROOT, "data", "data_stratified_900", "test", "sample_*_mj.npz")))[0]
    print("=== A. standing-still sanity (robot should stay standing) ===")
    clip_to_omni_npz(easy, tmp, static=True)
    hs, m, _ = run_one(tmp)
    h = hs[:, 2]; tl = np.array([np.degrees(tilt_rad(q[3:7])) for q in hs])
    print(f"  standing: {'fell' if (h<0.4).any() or (tl>50).any() else 'stood'}  final_h={h[-1]:.3f} max_tilt={tl.max():.1f}deg n={len(hs)}")
    print("=== B. real single-foot clip (check whether it follows reasonably) ===")
    clip_to_omni_npz(easy, tmp, static=False)
    hs, m, _ = run_one(tmp)
    h = hs[:, 2]; tl = np.array([np.degrees(tilt_rad(q[3:7])) for q in hs])
    print(f"  {os.path.basename(easy)}: {'fell' if (h<0.4).any() or (tl>60).any() else 'completed'}  final_h={h[-1]:.3f} max_tilt={tl.max():.1f}deg n={len(hs)}")
    sys.exit(0)

if sys.argv[1] == "run":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    of = f"{outdir}/{tag}__sh{shard}.json"
    if os.path.exists(of): print(f"[skip]{of}"); sys.exit(0)
    os.makedirs(outdir, exist_ok=True)
    tmp = f"/tmp/omni_run_sh{shard}.npz"
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    my = mots[shard::nsh]; res = {}; t0 = time.time(); NOM = 0.78
    import contextlib as _c, io as _io
    for mid in my:
        try:
            clip_to_omni_npz(f"{motion_dir}/sample_{mid}_mj.npz", tmp, static=False)
            with _c.redirect_stdout(_io.StringIO()):
                hs, m, _ = run_one(tmp)
        except Exception as e:
            res[mid] = {"err": str(e)[:120]}; print(f"[err]{mid}:{str(e)[:70]}"); continue
        heights = hs[:, 2]; tilts = np.array([tilt_rad(q[3:7]) for q in hs])
        bad = (heights < 0.5 * NOM) | (tilts > 1.0)
        fell = bool(bad.any()); ffi = int(np.argmax(bad)) if fell else len(hs)
        T = len(hs); upto = ffi if fell else T
        rob_fz = np.array([foot_z_fk(m, hs[t]) for t in range(min(upto, T))])
        d = dict(np.load(f"{motion_dir}/sample_{mid}_mj.npz", allow_pickle=True))
        bz = d["body_pos_w"][:, :, 2]; bn = [str(x) for x in d["body_names"]]
        li = bn.index("left_ankle_roll_link"); ri = bn.index("right_ankle_roll_link")
        ref_fz = np.stack([bz[:, li], bz[:, ri]], 1)[:T]; ground = 0.03
        ref_lift = float(ref_fz.max(0).max() - ground); swing = int(np.argmax(ref_fz.max(0)))
        rob_lift = float(rob_fz[:, swing].max() - ground) if len(rob_fz) else 0.0
        lr = rob_lift / ref_lift if ref_lift > 0.02 else float("nan")
        strict = bool((not fell) and (lr >= 0.5) and (np.degrees(tilts.max()) < 40.0))
        res[mid] = {"fell": fell, "fall_step": ffi, "T": T, "completion": ffi / T,
                    "final_h": float(heights[-1]), "max_tilt_deg": float(np.degrees(tilts.max())),
                    "ref_lift": ref_lift, "rob_lift": rob_lift, "lift_ratio": lr, "strict_success": strict}
    json.dump({"tag": tag, "harness": "omnixtreme_native", "per_motion": res}, open(of, "w"))
    n = len([v for v in res.values() if "fell" in v])
    fr = np.mean([v["fell"] for v in res.values() if "fell" in v]) if n else 0
    print(f"[done]{tag} sh{shard} {len(my)}mot fall_rate={fr:.1%} {time.time()-t0:.0f}s")
    sys.exit(0)

if sys.argv[1] == "runs":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(outdir, exist_ok=True); tmp = f"/tmp/omni_runs_sh{shard}.npz"
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        try:
            clip_to_omni_npz(f"{motion_dir}/sample_{mid}_mj.npz", tmp, static=False)
            import contextlib as _c, io as _io
            with _c.redirect_stdout(_io.StringIO()):
                hs, m, fz = run_one(tmp)
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}; continue
        # 5 mj_steps per control step -> subsample to control steps (take end of each step), align to clip frames
        qs = hs[4::5]; fzs = fz[4::5]
        Torig = len(np.load(f"{motion_dir}/sample_{mid}_mj.npz", allow_pickle=True)["joint_pos"])
        n = min(len(qs), Torig); qs = qs[:n]; fzs = fzs[:n]
        base_z = qs[:, 2]; tilt = np.array([tilt_rad(q[3:7]) for q in qs])
        mass = float(np.sum(m.body_mass))
        res[mid] = success_of(fzs[:, 0], fzs[:, 1], base_z, tilt, 0.78, 0.02, mass, mid)
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot metrics.py_success={np.mean(sv):.1%}")
    sys.exit(0)

if sys.argv[1] == "fullruns":
    import fullmetrics_post as FP
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(outdir, exist_ok=True); tmp = f"/tmp/omni_full_sh{shard}.npz"
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        try:
            clip_to_omni_npz(f"{motion_dir}/sample_{mid}_mj.npz", tmp, static=False)
            import contextlib as _c, io as _io
            with _c.redirect_stdout(_io.StringIO()):
                hs, m, fz = run_one(tmp)
            qv = np.array(_REC["qvel"])
            qs = hs[4::5]; fzs = fz[4::5]; qvs = qv[4::5]
            Torig = len(np.load(f"{motion_dir}/sample_{mid}_mj.npz", allow_pickle=True)["joint_pos"])
            n = min(len(qs), Torig); qs = qs[:n]; fzs = fzs[:n]; qvs = qvs[:n]
            res[mid] = FP.compute(qs, qvs, fzs, m, mid, 0.78, 50.0)
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot success={np.mean(sv) if sv else float('nan'):.1%}")
    sys.exit(0)
