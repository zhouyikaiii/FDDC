"""HoloMotion (model_14000.onnx, 604 obs + KV-cache + MoE) evaluation: use its own warp obs kernel (zero re-implementation) + threaded KV.
In hsmujoco env (warp+torch+onnxruntime+mujoco+wbt_rollout). Usage: sanity | run <motion_dir> <tag> <shard> <nsh> <outdir>"""
import os, sys, json, glob, re, time
import numpy as np, torch, onnx, onnxruntime as ort
os.environ.setdefault("WBT_ORT_THREADS", "1")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
sys.path.insert(0, os.environ["HOLO_REPO"])
import warp as wp
wp.init()
from holomotion.src.motion_tracking.actor_observation import launch_motion_actor_observation_warp, motion_actor_observation_dim
from holomotion.src.motion_tracking.reference_observation import derive_reference_kinematics_torch
HS = _ROOT
os.environ["WBT_EVAL_ROBOT_XML"] = os.path.join(HS, "robot", "g1_29dof", "g1_29dof.xml")
os.environ["WBT_EVAL_MOTION_DIR"] = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[1] in ("run", "runm")) else os.path.join(_ROOT, "data", "data_stratified_900", "test")
sys.path.insert(0, os.path.join(HS, "eval"))
import wbt_rollout as W
try:
    import onnxruntime as _ort
    _OS=_ort.InferenceSession
    def _S1(*a,**k):
        so=k.pop("sess_options",None) or _ort.SessionOptions()
        so.intra_op_num_threads=1; so.inter_op_num_threads=1
        return _OS(*a,sess_options=so,**k)
    _ort.InferenceSession=_S1
except Exception: pass


HM = os.environ["HOLO_ONNX"]
_md = {p.key: p.value for p in onnx.load(HM, load_external_data=False).metadata_props}
O = _md["joint_names"].split(",")                                     # ONNX interleaved order
ASCALE = np.array([float(x) for x in _md["action_scale"].split(",")], np.float64)   # ONNX order
DEFAULT_O = np.array([float(x) for x in _md["default_joint_pos"].split(",")], np.float64)
KP_O = np.array([float(x) for x in _md["joint_stiffness"].split(",")], np.float64)
KD_O = np.array([float(x) for x in _md["joint_damping"].split(",")], np.float64)
# my 29-dof order (R, sequential = g1.xml)
R = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
     "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
     "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
     "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
     "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
IDX = np.array([R.index(n) for n in O])         # ONNX index -> my index (= reference_dof_indices)
KP_R = np.empty(29); KP_R[IDX] = KP_O; KD_R = np.empty(29); KD_R[IDX] = KD_O  # PD gains in my order
DEFAULT_R = np.empty(29); DEFAULT_R[IDX] = DEFAULT_O  # default pose in my order
NFUT = 10; OBSDIM = motion_actor_observation_dim(NFUT)  # 604
sess = ort.InferenceSession(HM, providers=["CPUExecutionProvider"])
DEV = wp.get_device("cpu")

# pre-build constant warp arrays
_ref_idx_wp = wp.from_numpy(IDX.astype(np.int32), dtype=wp.int32, device=DEV)
_default_wp = wp.from_numpy(DEFAULT_O.reshape(1, 29).astype(np.float32), dtype=wp.float32, device=DEV)
_yaw_id_wp = wp.from_numpy(np.array([[1, 0, 0, 0]], np.float32), dtype=wp.float32, device=DEV)

def build_obs604(qpos_win, rquat_wxyz, rangvel, rdof_pos_o, rdof_vel_o, last_act_o):
    # qpos_win: (13,36) np; convert to torch derive kinematics
    qt = torch.from_numpy(qpos_win.reshape(1, 13, 36).astype(np.float32))
    kin = derive_reference_kinematics_torch(qt, fps=50.0)
    f2w = lambda t: wp.from_torch(t.contiguous(), dtype=wp.float32)
    qpos_wp = wp.from_numpy(qpos_win.reshape(1, 13, 36).astype(np.float32), dtype=wp.float32, device=DEV)
    out = wp.zeros((1, OBSDIM), dtype=wp.float32, device=DEV)
    launch_motion_actor_observation_warp(
        qpos_wp=qpos_wp,
        kinematics=(f2w(kin.dof_vel), f2w(kin.root_linvel_local), f2w(kin.root_angvel_local), f2w(kin.projected_gravity)),
        robot_root_quat_wp=wp.from_numpy(rquat_wxyz.reshape(1, 4).astype(np.float32), dtype=wp.float32, device=DEV),
        robot_root_angvel_wp=wp.from_numpy(rangvel.reshape(1, 3).astype(np.float32), dtype=wp.float32, device=DEV),
        robot_dof_pos_wp=wp.from_numpy(rdof_pos_o.reshape(1, 29).astype(np.float32), dtype=wp.float32, device=DEV),
        robot_dof_vel_wp=wp.from_numpy(rdof_vel_o.reshape(1, 29).astype(np.float32), dtype=wp.float32, device=DEV),
        last_action_wp=wp.from_numpy(last_act_o.reshape(1, 29).astype(np.float32), dtype=wp.float32, device=DEV),
        default_dof_pos_wp=_default_wp, reference_dof_indices_wp=_ref_idx_wp, reference_yaw_alignment_wp=_yaw_id_wp,
        use_yaw_alignment=True, current_index=1, num_future_frames=NFUT, output_wp=out, device=DEV)
    wp.synchronize_device(DEV)
    return out.numpy().reshape(-1)

def wxyz_of(sim):
    q = sim.base_quat_xyzw(); return np.array([q[3], q[0], q[1], q[2]])

def rollout(mid):
    ref = W.load_motion_npz(mid); T = ref["T"]
    qpos_all = ref["npz"]["joint_pos"] if "joint_pos" in ref["npz"] else None
    d = dict(np.load(f"{os.environ['WBT_EVAL_MOTION_DIR']}/sample_{mid}_mj.npz", allow_pickle=True))
    qpos_all = d["joint_pos"].astype(np.float64)  # (T,36) root3+quat_wxyz4+dof29 (R order)
    sim = W.G1Sim([n.replace("_joint", "") for n in R]) if False else W.G1Sim(R)
    sim.reset_to(ref["base_pos0"], ref["base_quat0"], ref["joint_pos0"]); sim.settle_freeze(n_steps=50)
    nom_h = float(ref["base_pos0"][2]); last_o = np.zeros(29)
    pkv = np.zeros((1, 2, 1, 32, 4, 64), np.float32); hs = []
    c = lambda i: min(max(i, 0), T - 1)
    for t in range(T):
        win = np.stack([qpos_all[c(t - 1 + k)] for k in range(13)])  # (13,36)
        rdof_o = sim.dof_pos()[IDX]; rdofv_o = sim.dof_vel()[IDX]
        obs = build_obs604(win, wxyz_of(sim), sim.base_ang_vel_base(), rdof_o, rdofv_o, last_o)
        out = sess.run(["actions", "present_key_values"], {"obs": obs.reshape(1, -1).astype(np.float32), "past_key_values": pkv, "step_idx": np.array([t], np.int64)})
        act_o = out[0].reshape(-1)[:29]; pkv = out[1]
        last_o = act_o.copy()
        qt_o = act_o.astype(np.float64) * ASCALE + DEFAULT_O
        q_target = np.empty(29); q_target[IDX] = qt_o  # ONNX -> my order
        sim.pd_step(q_target, KP_R, KD_R)
        lz = sim.foot_state(sim.lfoot)[2]; rz = sim.foot_state(sim.rfoot)[2]
        hs.append(np.concatenate([sim.data.qpos[:7], [lz, rz]]))
    return np.array(hs), nom_h, d

def tilt_of(qw):
    w, x, y, z = qw; return float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))

# ---- re-evaluate with metrics.py full success metric ----
import metrics as M
class _Dummy:
    num_dofs = 29
_dummy = _Dummy(); _dummy.dof_names = [n.replace("_joint", "") for n in R]; _dummy.default_dof = np.zeros(29)

def rollout_metrics(mid):
    ref = W.load_motion_npz(mid); mref = W.NpzMotionReference(ref["npz"]); T = ref["T"]
    d = dict(np.load(f"{os.environ['WBT_EVAL_MOTION_DIR']}/sample_{mid}_mj.npz", allow_pickle=True))
    qpos_all = d["joint_pos"].astype(np.float64)
    sim = W.G1Sim(R); sim.reset_to(ref["base_pos0"], ref["base_quat0"], ref["joint_pos0"]); sim.settle_freeze(n_steps=50)
    nominal_h = float(ref["base_pos0"][2]); last_o = np.zeros(29)
    pkv = np.zeros((1, 2, 1, 32, 4, 64), np.float32); fell = False
    c = lambda i: min(max(i, 0), T - 1)
    keys = ["t", "base_pos", "base_quat", "base_angvel", "base_linvel", "qpos", "qvel", "fz_l", "fz_r",
            "foot_l_pos", "foot_r_pos", "foot_l_z", "foot_r_z", "foot_l_quat", "foot_r_quat",
            "com", "com_vel", "com_rel_obs", "action", "motor_cmd", "tilt", "fell", "kp12_robot"]
    log = {k: [] for k in keys}
    for t in range(T):
        motion = mref.infer(np.zeros(463, np.float32), t)
        win = np.stack([qpos_all[c(t - 1 + k)] for k in range(13)])
        rdof_o = sim.dof_pos()[IDX]; rdofv_o = sim.dof_vel()[IDX]
        obs = build_obs604(win, wxyz_of(sim), sim.base_ang_vel_base(), rdof_o, rdofv_o, last_o)
        out = sess.run(["actions", "present_key_values"], {"obs": obs.reshape(1, -1).astype(np.float32), "past_key_values": pkv, "step_idx": np.array([t], np.int64)})
        act_o = out[0].reshape(-1)[:29]; pkv = out[1]; last_o = act_o.copy()
        qt_o = act_o.astype(np.float64) * ASCALE + DEFAULT_O
        q_target = np.empty(29); q_target[IDX] = qt_o
        action = q_target - DEFAULT_R  # approximate action (only for jerk metric, magnitude suffices)
        com_rel = W.build_obs(sim, _dummy, motion, np.zeros(29, np.float32))[-4:].copy()
        tilt = float(np.arccos(np.clip(-sim.projected_gravity_base()[2], -1, 1)))
        lp, lv, lz = sim.foot_state(sim.lfoot); rp, rv, rz = sim.foot_state(sim.rfoot)
        com = sim.com_true(); comv = (com - log["com"][-1]) * W.CONTROL_HZ if log["com"] else np.zeros(3); cur_h = float(sim.data.qpos[2])  # true xipos CoM + finite-diff (unified); obs com_rel above stays proxy
        fell = fell or (cur_h < 0.5 * nominal_h) or (tilt > 1.0)
        log["t"].append(t / W.CONTROL_HZ); log["base_pos"].append(sim.data.qpos[0:3].copy())
        log["base_quat"].append(sim.base_quat_xyzw()); log["base_angvel"].append(sim.base_ang_vel_base())
        log["base_linvel"].append(sim.data.qvel[0:3].copy()); log["qpos"].append(sim.dof_pos()); log["qvel"].append(sim.dof_vel())
        log["fz_l"].append(sim.foot_contact_fz(sim.lfoot)); log["fz_r"].append(sim.foot_contact_fz(sim.rfoot))
        log["foot_l_pos"].append(lp); log["foot_r_pos"].append(rp); log["foot_l_z"].append(lz); log["foot_r_z"].append(rz)
        log["foot_l_quat"].append(sim.foot_quat_xyzw(sim.lfoot)); log["foot_r_quat"].append(sim.foot_quat_xyzw(sim.rfoot))
        log["com"].append(com); log["com_vel"].append(comv); log["com_rel_obs"].append(com_rel)
        log["action"].append(action.copy()); log["motor_cmd"].append(q_target.copy()); log["tilt"].append(tilt); log["fell"].append(fell)
        log["kp12_robot"].append(sim.data.xpos[sim.kp12_ids].copy())   # HuB 12 keypoints (world)
        sim.pd_step(q_target, KP_R, KD_R)
    ol = {k: np.array(v) for k, v in log.items()}
    ol["ref_support_state"] = ref["ref_support_state"][:T]; ol["nominal_h"] = nominal_h
    ol["motion_id"] = mid; ol["seed"] = 0; ol["dt"] = 1.0 / W.CONTROL_HZ; ol["body_mass_total"] = sim.total_mass
    return M.compute_metrics(ol)

if sys.argv[1] == "runm":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    of = f"{outdir}/{tag}__sh{shard}.json"
    os.makedirs(outdir, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    my = mots[shard::nsh]; res = {}; t0 = time.time()
    _MK = ["success", "success_sustained", "track_fail", "fell", "hop_count", "swing_touched", "time_to_fall", "xcom_margin_ap_min",
           "xcom_margin_ml_min", "xcom_margin_ap_mean", "xcom_margin_ml_mean", "com_margin_ap_mean", "com_margin_ml_mean", "com_margin_ap_min", "com_margin_ml_min", "xcom_margin_viol_dur", "xcom_min_ttb", "xcom_low_ttb_events", "slippage_mm_s", "action_jerk_rms", "dof_vel_jerk_rms",
           "track_Epos", "track_Evel", "track_Eacc"]
    for mid in my:
        try:
            m = rollout_metrics(mid); res[mid] = {k: float(m.get(k, 0)) for k in _MK}
        except Exception as e:
            res[mid] = {"err": str(e)[:120]}; print(f"[err]{mid}:{str(e)[:70]}")
    json.dump({"tag": tag, "per_motion": res}, open(of, "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot metrics.py_success_rate={np.mean(sv):.1%} {time.time()-t0:.0f}s")
    sys.exit(0)

if sys.argv[1] == "sanity":
    MD = os.environ["WBT_EVAL_MOTION_DIR"]
    allm = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(MD + "/sample_*_mj.npz"))
    mids = allm[::80][:6]; ok = 0
    print("=== HoloMotion sanity (normal motions) ===")
    for mid in mids:
        hs, nom_h, d = rollout(mid)
        h = hs[:, 2]; tl = np.array([np.degrees(tilt_of(q[3:7])) for q in hs])
        fell = (h < 0.5 * nom_h).any() or (tl > 60).any(); ff = int(np.argmax((h < 0.5 * nom_h) | (tl > 60))) if fell else len(h)
        print(f"  {mid}: {'fell@'+str(ff) if fell else 'completed'}/{len(h)} final_h={h[-1]:.3f} max_tilt={tl.max():.0f}deg"); ok += (not fell)
    print(f"total: completed {ok}/{len(mids)} -> {'wiring likely correct' if ok >= len(mids)//2 else 'wiring possibly wrong'}")
    sys.exit(0)

if sys.argv[1] == "run":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    of = f"{outdir}/{tag}__sh{shard}.json"
    if os.path.exists(of): print(f"[skip]{of}"); sys.exit(0)
    os.makedirs(outdir, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    my = mots[shard::nsh]; res = {}; t0 = time.time()
    for mid in my:
        try:
            hs, nom_h, d = rollout(mid)
        except Exception as e:
            res[mid] = {"err": str(e)[:120]}; print(f"[err]{mid}:{str(e)[:70]}"); continue
        h = hs[:, 2]; tl = np.array([tilt_of(q[3:7]) for q in hs])
        bad = (h < 0.5 * nom_h) | (tl > 1.0); fell = bool(bad.any()); ffi = int(np.argmax(bad)) if fell else len(hs)
        T = len(hs); upto = ffi if fell else T; rob_fz = hs[:upto, 7:9]
        bz = d["body_pos_w"][:, :, 2]; bn = [str(x) for x in d["body_names"]]
        li = bn.index("left_ankle_roll_link"); ri = bn.index("right_ankle_roll_link")
        ref_fz = np.stack([bz[:, li], bz[:, ri]], 1)[:T]; ground = 0.03
        ref_lift = float(ref_fz.max(0).max() - ground); swing = int(np.argmax(ref_fz.max(0)))
        rob_lift = float(rob_fz[:, swing].max() - ground) if len(rob_fz) else 0.0
        lr = rob_lift / ref_lift if ref_lift > 0.02 else float("nan")
        strict = bool((not fell) and (lr >= 0.5) and (np.degrees(tl.max()) < 40.0))
        res[mid] = {"fell": fell, "fall_step": ffi, "T": T, "completion": ffi / T, "final_h": float(h[-1]),
                    "max_tilt_deg": float(np.degrees(tl.max())), "ref_lift": ref_lift, "rob_lift": rob_lift, "lift_ratio": lr, "strict_success": strict}
    json.dump({"tag": tag, "harness": "holomotion_native_kernel", "per_motion": res}, open(of, "w"))
    n = len([v for v in res.values() if "fell" in v]); fr = np.mean([v["fell"] for v in res.values() if "fell" in v]) if n else 0
    print(f"[done]{tag} sh{shard} {len(my)}mot fall_rate={fr:.1%} {time.time()-t0:.0f}s")
    sys.exit(0)
