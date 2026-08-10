"""Score protomotions on the single-foot balance benchmark: replicate run_rollout logging, actions via protomotions adapter, same 0.20 dof_vel noise condition.
Usage:
  python proto_eval.py <motion_dir> <tag> <shard> <nsh> <K> <outdir>
  python proto_eval.py sanity                        # wiring check on a few released test clips (still needs PROTO_ONNX)"""
import os, sys, json, glob, re, time, numpy as np, onnxruntime as ort, onnx
from onnx import numpy_helper
os.environ.setdefault("WBT_ORT_THREADS", "1")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
HS = _ROOT
os.environ["WBT_EVAL_ROBOT_XML"] = os.path.join(HS, "robot", "g1_29dof", "g1_29dof.xml")
_SANITY = len(sys.argv) > 1 and sys.argv[1] == "sanity"
if _SANITY:
    motion_dir = os.path.join(HS, "data", "data_stratified_900", "test")   # sanity runs on the released test clips
else:
    motion_dir, tag, shard, nsh, K, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
os.environ["WBT_EVAL_MOTION_DIR"] = motion_dir
sys.path.insert(0, os.path.join(HS, "eval"))
import wbt_rollout as W, metrics as M
try:
    import onnxruntime as _ort
    _OS=_ort.InferenceSession
    def _S1(*a,**k):
        so=k.pop("sess_options",None) or _ort.SessionOptions()
        so.intra_op_num_threads=1; so.inter_op_num_threads=1
        return _OS(*a,sess_options=so,**k)
    _ort.InferenceSession=_S1
except Exception: pass


PROTO = os.environ["PROTO_ONNX"]
sess = ort.InferenceSession(PROTO, providers=["CPUExecutionProvider"])
outn = [o.name for o in sess.get_outputs()]
# default pose + dof_names (dummy policy for computing com_rel)
md = {p.key: p.value for p in onnx.load(os.environ.get("DDC_ROBOT_META_ONNX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_meta.onnx")), load_external_data=False).metadata_props}
dof_names = json.loads(md["dof_names"])
dja = json.loads(md["experiment_config"])["robot"]["init_state"]["default_joint_angles"]
default_dof = np.array([dja[n] for n in dof_names], dtype=np.float64)

class DummyPol:  # only used by build_obs to compute com_rel
    num_dofs = 29; dof_names = None; default_dof = None
dummy = DummyPol(); dummy.dof_names = dof_names; dummy.default_dof = default_dof

MK = ["success", "success_sustained", "track_fail", "fell", "swing_touched", "hop_count", "hop_height_max", "flight_events", "time_to_fall", "foot_activity_area",
      "foot_yaw_drift", "com_dev_pos_mean", "com_dev_pos_max", "com_dev_vel_mean", "xcom_margin_ap_min",
      "xcom_margin_ml_min", "xcom_margin_ap_mean", "xcom_margin_ml_mean", "com_margin_ap_mean", "com_margin_ml_mean", "com_margin_ap_min", "com_margin_ml_min", "xcom_margin_viol_dur", "xcom_min_ttb", "xcom_low_ttb_events", "slippage_mm_s", "action_jerk_rms", "dof_vel_jerk_rms", "track_Epos", "track_Evel", "track_Eacc"]

def rollout(motion_id, seed, dof_vel_noise=float(os.environ.get("PROTO_NOISE","0.20")), dof_vel_delay=int(os.environ.get("PROTO_DELAY","1"))):
    ref = W.load_motion_npz(motion_id); mref = W.NpzMotionReference(ref["npz"]); T = ref["T"]
    sim = W.G1Sim(dof_names); rng = np.random.RandomState(seed)
    sim.reset_to(ref["base_pos0"], ref["base_quat0"], ref["joint_pos0"]); sim.settle_freeze(n_steps=50)
    nominal_h = float(ref["base_pos0"][2]); last_act = np.zeros(29, np.float32); hist = np.zeros((1, 1, 29), np.float32); fell = False
    c = lambda i: min(max(int(i), 0), T - 1)
    keys = ["t", "base_pos", "base_quat", "base_angvel", "base_linvel", "qpos", "qvel", "fz_l", "fz_r",
            "foot_l_pos", "foot_r_pos", "foot_l_z", "foot_r_z", "foot_l_quat", "foot_r_quat",
            "com", "com_vel", "com_rel_obs", "action", "motor_cmd", "tilt", "fell", "kp12_robot"]
    log = {k: [] for k in keys}
    for t in range(T):
        motion = mref.infer(np.zeros(463, np.float32), t)
        clean_dv = sim.dof_vel(); sub = sim._vel_hist_sub
        delayed = sub[-1 - dof_vel_delay] if dof_vel_delay > 0 and len(sub) > dof_vel_delay else clean_dv
        dv_obs = delayed + rng.uniform(-1.0, 1.0, clean_dv.shape) * dof_vel_noise
        obs = W.build_obs(sim, dummy, motion, last_act, dof_vel_obs=dv_obs)  # only take com_rel
        com_rel = obs[-4:].copy()
        # protomotions 8 inputs
        anchor = sim.torso_quat_xyzw()  # xyzw
        fidx = [c(t + k) for k in range(1, 5)]
        feed = {
            "current_anchor_rot": anchor.reshape(1, 4).astype(np.float32),
            "current_dof_pos": sim.dof_pos().reshape(1, 29).astype(np.float32),
            "current_dof_vel": dv_obs.reshape(1, 29).astype(np.float32),
            "current_root_local_ang_vel": sim.base_ang_vel_base().reshape(1, 3).astype(np.float32),
            "historical_processed_actions": hist,
            "mimic_future_anchor_rot": np.stack([mref.tq[i] for i in fidx]).reshape(1, 4, 4).astype(np.float32),
            "mimic_future_dof_pos": np.stack([mref.jp[i] for i in fidx]).reshape(1, 4, 29).astype(np.float32),
            "mimic_future_dof_vel": np.stack([mref.jv[i] for i in fidx]).reshape(1, 4, 29).astype(np.float32),
        }
        od = dict(zip(outn, sess.run(outn, feed)))
        q_target = np.asarray(od["joint_pos_targets"]).reshape(-1)[:29].astype(np.float64)
        kp = np.asarray(od["stiffness_targets"]).reshape(-1)[:29].astype(np.float64)
        kd = np.asarray(od["damping_targets"]).reshape(-1)[:29].astype(np.float64)
        action = np.asarray(od["actions"]).reshape(-1)[:29].astype(np.float64)
        hist = action.reshape(1, 1, 29).astype(np.float32); last_act = action.astype(np.float32)
        tilt = float(np.arccos(np.clip(-sim.projected_gravity_base()[2], -1, 1)))
        lp, lv, lz = sim.foot_state(sim.lfoot); rp, rv, rz = sim.foot_state(sim.rfoot)
        com = sim.com_true(); comv = (com - log["com"][-1]) * W.CONTROL_HZ if log["com"] else np.zeros(3); cur_h = float(sim.data.qpos[2])  # true xipos CoM + finite-diff (unified); obs com_rel above stays proxy
        fell = fell or (cur_h < 0.5 * nominal_h) or (tilt > 1.0)
        log["t"].append(t / W.CONTROL_HZ); log["base_pos"].append(sim.data.qpos[0:3].copy())
        log["base_quat"].append(sim.base_quat_xyzw()); log["base_angvel"].append(sim.base_ang_vel_base())
        log["base_linvel"].append(sim.data.qvel[0:3].copy()); log["qpos"].append(sim.dof_pos()); log["qvel"].append(clean_dv)
        log["fz_l"].append(sim.foot_contact_fz(sim.lfoot)); log["fz_r"].append(sim.foot_contact_fz(sim.rfoot))
        log["foot_l_pos"].append(lp); log["foot_r_pos"].append(rp); log["foot_l_z"].append(lz); log["foot_r_z"].append(rz)
        log["foot_l_quat"].append(sim.foot_quat_xyzw(sim.lfoot)); log["foot_r_quat"].append(sim.foot_quat_xyzw(sim.rfoot))
        log["com"].append(com); log["com_vel"].append(comv); log["com_rel_obs"].append(com_rel)
        log["action"].append(action.copy()); log["motor_cmd"].append(q_target.copy()); log["tilt"].append(tilt); log["fell"].append(fell)
        log["kp12_robot"].append(sim.data.xpos[sim.kp12_ids].copy())   # HuB 12 keypoints (world)
        sim.pd_step(q_target, kp, kd)
    ol = {k: np.array(v) for k, v in log.items()}
    ol["ref_support_state"] = ref["ref_support_state"][:T]; ol["nominal_h"] = nominal_h
    ol["motion_id"] = motion_id; ol["seed"] = seed; ol["dt"] = 1.0 / W.CONTROL_HZ; ol["body_mass_total"] = sim.total_mass
    return ol

if _SANITY:
    # Wiring check: with observation noise off, a correctly wired policy tracks the opening
    # (double-support) phase of a clip without an immediate fall. A generalist is still not
    # expected to hold the single-leg phase itself -- that is the benchmark, not this check.
    _mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    _mids = _mots[::max(1, len(_mots) // 4)][:4]
    print("=== ProtoMotions sanity (wiring: no immediate fall on the opening phase) ===")
    _ok = 0
    for _mid in _mids:
        _fell = np.asarray(rollout(_mid, 0, dof_vel_noise=0.0, dof_vel_delay=0)["fell"])
        _ff = int(np.argmax(_fell)) if bool(_fell.any()) else len(_fell)
        _early = _ff < 25   # fell within ~0.5 s -> wiring bug
        print(f"  {_mid}: {'fell@' + str(_ff) if bool(_fell.any()) else 'stood full'}/{len(_fell)}  "
              f"{'<- immediate fall (wiring?)' if _early else 'opening tracked'}")
        _ok += (not _early)
    print(f"-> {_ok}/{len(_mids)} tracked the opening without an immediate fall = wiring OK "
          f"(a clean single-leg hold is the benchmark, not this check)")
    sys.exit(0)

of = f"{outdir}/{tag}__sh{shard}.json"
if os.path.exists(of): print(f"[skip]{of}"); sys.exit(0)
os.makedirs(outdir, exist_ok=True)
mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
my = mots[shard::nsh]; res = {}; t0 = time.time()
for mid in my:
    acc = {k: 0.0 for k in MK}
    for s in range(K):
        m = M.compute_metrics(rollout(mid, s))
        for k in MK: acc[k] += float(m.get(k, 0))
    res[mid] = {k: acc[k] / K for k in MK}
json.dump({"tag": tag, "K": K, "per_motion": res}, open(of, "w"))
print(f"[done]{tag} sh{shard} {len(my)}mot {time.time()-t0:.0f}s")
