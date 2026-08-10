"""GMT (General Motion Tracking, zixuan417) faithful harness evaluation: run its own sim2sim logic (its obs/policy/PD/sim), headless.
Usage:
  python gmt_eval.py sanity
  python gmt_eval.py run <motion_dir> <tag> <shard> <nsh> <outdir>
"""
import os, sys, glob, re, json, time, pickle
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
REPO = os.environ["GMT_REPO"]; sys.path.insert(0, REPO); os.chdir(REPO)

# ---- stub out the on-screen viewer to make it headless ----
import mujoco_viewer
class _StubCam:
    def __init__(s): s.distance = 5.0; s.lookat = np.zeros(3); s.azimuth = 0; s.elevation = 0; s.type = 0
class _StubViewer:
    def __init__(s, *a, **k): s.cam = _StubCam()
    def render(s): pass
    def read_pixels(s, *a, **k): return np.zeros((64, 64, 3), np.uint8)
    def close(s): pass
mujoco_viewer.MujocoViewer = _StubViewer

import torch, mujoco
torch.set_num_threads(1)
from sim2sim import HumanoidEnv
_HS = _ROOT
os.environ.setdefault("WBT_EVAL_ROBOT_XML", os.path.join(_HS, "robot", "g1_29dof", "g1_29dof.xml"))
if len(sys.argv) > 2 and sys.argv[1] in ("runs", "fullruns"):
    os.environ["WBT_EVAL_MOTION_DIR"] = sys.argv[2]  # must be set before import success_metric (-> wbt_rollout cache)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HS, "eval"))
from success_metric import foot_fz as FZ, success_of
import fulllog_native as FN

PT = os.environ["GMT_WEIGHTS"]
# copy to its expected path
os.makedirs(f"{REPO}/assets/pretrained_checkpoints", exist_ok=True)
if not os.path.exists(f"{REPO}/assets/pretrained_checkpoints/pretrained.pt"):
    import shutil; shutil.copy(PT, f"{REPO}/assets/pretrained_checkpoints/pretrained.pt")

# GMT 23-dof joint names (strip _joint suffix to align with my 29)
GMT_DOFS = ["left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
            "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
            "waist_yaw","waist_roll","waist_pitch",
            "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
            "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow"]

def foot_z_fk(model, qpos):
    if not hasattr(foot_z_fk, "d"):
        foot_z_fk.d = mujoco.MjData(model)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        cand = [n for n in names if n and "ankle_roll" in n]
        foot_z_fk.lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand[0])
        foot_z_fk.rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand[1])
    d = foot_z_fk.d; d.qpos[:len(qpos)] = qpos; mujoco.mj_kinematics(model, d)
    return float(d.xpos[foot_z_fk.lf, 2]), float(d.xpos[foot_z_fk.rf, 2])

def tilt_rad(quat_wxyz):
    w, x, y, z = quat_wxyz
    gz = (1.0 - 2.0 * (x * x + y * y))  # z component of world-z in base frame (upright=1)
    return float(np.arccos(np.clip(gz, -1, 1)))

def run_motion(pkl_path, n_steps=None, flog=None):
    """Run one pkl, return per-frame (base qpos log) + fall. sim starts from keyframe0 (standing) = its deployment protocol."""
    env = HumanoidEnv(policy_path=f"{REPO}/assets/pretrained_checkpoints/pretrained.pt",
                      motion_path=pkl_path, robot_type="g1", device="cpu", record_video=False)
    m, d = env.model, env.data
    nom_h = float(d.qpos[2])
    if flog is not None: flog["_half_nom"] = 0.5 * 0.78; _pc = None
    _lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    _rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    total = n_steps if n_steps else int(env.sim_duration / env.sim_dt)
    hs = []; fzs = []
    for i in range(total):
        dof_pos, dof_vel, quat, ang_vel = env.extract_data()
        if i % env.sim_decimation == 0:
            ct = i // env.sim_decimation
            mimic_obs = env._get_mimic_obs(ct)
            from sim2sim import quatToEuler
            rpy = quatToEuler(quat)
            odv = dof_vel.copy(); odv[[4, 5, 10, 11]] = 0.0
            obs_prop = np.concatenate([ang_vel * env.ang_vel_scale, rpy[:2],
                                       (dof_pos - env.default_dof_pos) * env.dof_pos_scale,
                                       odv * env.dof_vel_scale, env.last_action])
            obs_hist = np.array(env.proprio_history_buf).flatten()
            obs_buf = np.concatenate([mimic_obs, obs_prop, obs_hist])
            with torch.no_grad():
                raw = env.policy_jit(torch.from_numpy(obs_buf).float().unsqueeze(0).to(env.device)).cpu().numpy().squeeze()
            env.last_action = raw.copy()
            pd_target = np.clip(raw, -10., 10.) * env.action_scale + env.default_dof_pos
            env.proprio_history_buf.append(obs_prop)
            hs.append(d.qpos.copy())
            fzl = FZ(m, d, _lf); fzr = FZ(m, d, _rf); fzs.append([fzl, fzr])
            if flog is not None:
                _pc = FN.log_step(flog, m, d, _lf, _rf, ct, raw, 50.0, _pc, fz_l=fzl, fz_r=fzr)
        torque = (pd_target - dof_pos) * env.stiffness - dof_vel * env.damping
        torque = np.clip(torque, -env.torque_limits, env.torque_limits)
        d.ctrl = torque; mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)  # sync xpos/contacts to the just-integrated qpos (mj_step leaves them one substep stale — same fix as G1Sim.pd_step); the next logged fz/foot_z then match qpos
    return np.array(hs), m, nom_h, np.array(fzs)

if sys.argv[1] == "sanity":
    hs, m, nom_h, _ = run_motion(f"{REPO}/assets/motions/walk_stand.pkl", n_steps=int(8.0/0.001))
    heights = hs[:, 2]; tilts = np.array([np.degrees(tilt_rad(q[3:7])) for q in hs])
    fell = (heights < 0.5 * nom_h).any() or (tilts > 60).any()
    print(f"nom_h={nom_h:.3f}")
    print(f"GMT walk_stand sanity: {'fell' if fell else 'stood/completed'} final_h={heights[-1]:.3f} max_tilt={tilts.max():.1f}deg final_tilt={tilts[-1]:.1f}deg")
    print("-> completed without falling = harness wiring correct")
    sys.exit(0)

def clip_to_pkl(npz_path, pkl_path):
    """My 29-dof _mj.npz -> GMT 23-dof .pkl (wxyz->xyzw, select 23 joints). Return (T, fps)."""
    d = dict(np.load(npz_path, allow_pickle=True))
    jp = d["joint_pos"]  # (T,36) root3+quat(wxyz)4+29dof
    fps = float(np.array(d["fps"]).reshape(-1)[0])
    my_names = [str(n).replace("_joint", "") for n in d["joint_names"]]
    sel = [my_names.index(g) for g in GMT_DOFS]  # 29->23 mapping
    root_pos = jp[:, 0:3].astype(np.float32)
    q_wxyz = jp[:, 3:7]; root_rot = q_wxyz[:, [1, 2, 3, 0]].astype(np.float32)  # →xyzw
    dof_pos = jp[:, 7:][:, sel].astype(np.float32)
    pickle.dump({"fps": fps, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos},
                open(pkl_path, "wb"))
    return len(jp), fps

def ref_foot_z(model, root_pos, root_rot_xyzw, dof23):
    """Reference-frame FK foot height: build mujoco qpos (root3+wxyz4+23dof)."""
    d = foot_z_fk.d if hasattr(foot_z_fk, "d") else None
    q = np.zeros(model.nq); q[0:3] = root_pos
    q[3:7] = root_rot_xyzw[[3, 0, 1, 2]]  # xyzw→wxyz
    q[7:7 + 23] = dof23
    return foot_z_fk(model, q)

if sys.argv[1] == "run":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    of = f"{outdir}/{tag}__sh{shard}.json"
    if os.path.exists(of): print(f"[skip]{of}"); sys.exit(0)
    os.makedirs(outdir, exist_ok=True)
    tmpd = f"/tmp/gmt_pkls_sh{shard}"; os.makedirs(tmpd, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    my = mots[shard::nsh]; res = {}; t0 = time.time()
    for mid in my:
        pkl = f"{tmpd}/{mid}.pkl"
        try:
            T, fps = clip_to_pkl(f"{motion_dir}/sample_{mid}_mj.npz", pkl)
            n_steps = int((T / fps) / 0.001)  # run for one clip duration
            hs, m, _, _ = run_motion(pkl, n_steps=n_steps)
        except Exception as e:
            res[mid] = {"err": str(e)[:120]}; print(f"[err]{mid}:{str(e)[:70]}"); continue
        heights = hs[:, 2]; tilts = np.array([tilt_rad(q[3:7]) for q in hs])  # rad
        NOM = 0.78
        bad = (heights < 0.5 * NOM) | (tilts > 1.0)
        fell = bool(bad.any()); ffi = int(np.argmax(bad)) if fell else len(hs)
        Th = len(hs); upto = ffi if fell else Th
        # foot height: robot (hs full qpos, but GMT qpos is only 7+23=30) vs reference
        rob_fz = np.array([foot_z_fk(m, hs[t]) for t in range(min(upto, Th))])
        pd = pickle.load(open(pkl, "rb"))
        Tr = min(len(pd["root_pos"]), Th)
        ref_fz = np.array([ref_foot_z(m, pd["root_pos"][t], pd["root_rot"][t], pd["dof_pos"][t]) for t in range(Tr)])
        ground = float(min(ref_fz.min(), rob_fz.min()) if len(rob_fz) else ref_fz.min())
        ref_lift = float(ref_fz.max(axis=0).max() - ground)
        swing = int(np.argmax(ref_fz.max(axis=0)))
        rob_lift = float(rob_fz[:, swing].max() - ground) if len(rob_fz) else 0.0
        lr = rob_lift / ref_lift if ref_lift > 0.02 else float("nan")
        strict = bool((not fell) and (lr >= 0.5) and (np.degrees(tilts.max()) < 40.0))
        res[mid] = {"fell": fell, "fall_step": ffi, "T": Th, "completion": ffi / Th,
                    "final_h": float(heights[-1]), "max_tilt_deg": float(np.degrees(tilts.max())),
                    "ref_lift": ref_lift, "rob_lift": rob_lift, "lift_ratio": lr, "strict_success": strict}
        os.remove(pkl)
    json.dump({"tag": tag, "harness": "gmt_native", "per_motion": res}, open(of, "w"))
    n = len([v for v in res.values() if "fell" in v])
    fr = np.mean([v["fell"] for v in res.values() if "fell" in v]) if n else 0
    print(f"[done]{tag} sh{shard} {len(my)}mot fall_rate={fr:.1%} {time.time()-t0:.0f}s")
    sys.exit(0)

if sys.argv[1] == "runs":  # unified metrics.py success metric
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.environ["WBT_EVAL_MOTION_DIR"] = motion_dir; os.makedirs(outdir, exist_ok=True)
    tmpd = f"/tmp/gmt_pkls_s{shard}"; os.makedirs(tmpd, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        pkl = f"{tmpd}/{mid}.pkl"
        try:
            T, fps = clip_to_pkl(f"{motion_dir}/sample_{mid}_mj.npz", pkl)
            hs, m, _, fzs = run_motion(pkl, n_steps=int((T / fps) / 0.001))
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}; continue
        base_z = hs[:, 2]; tilt = np.array([tilt_rad(q[3:7]) for q in hs])
        mass = float(m.body_subtreemass[0]) if hasattr(m, "body_subtreemass") else float(np.sum(m.body_mass))
        res[mid] = success_of(fzs[:, 0], fzs[:, 1], base_z, tilt, 0.78, 0.02, mass, mid)
        os.remove(pkl)
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot metrics.py_success={np.mean(sv):.1%}")
    sys.exit(0)

if sys.argv[1] == "fullruns":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.environ["WBT_EVAL_MOTION_DIR"] = motion_dir; os.makedirs(outdir, exist_ok=True)
    tmpd = f"/tmp/gmt_pkls_f{shard}"; os.makedirs(tmpd, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        pkl = f"{tmpd}/{mid}.pkl"
        try:
            T, fps = clip_to_pkl(f"{motion_dir}/sample_{mid}_mj.npz", pkl)
            fl_ = FN.new_log(); hs, m, _, fzs = run_motion(pkl, n_steps=int((T / fps) / 0.001), flog=fl_)
            mass = float(np.sum(m.body_mass))
            res[mid] = FN.finalize(fl_, mid, 0.78, mass, 50.0)
            os.remove(pkl)
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot success={np.mean(sv):.1%}")
    sys.exit(0)
