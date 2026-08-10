"""Post-process a recorded (qpos + qvel + fz) trajectory (at control rate) into the full metric
suite via M.compute_metrics. Used by the omni / hgpt adapters, which run the baseline's own
main_loop and cannot expose the policy action directly, so foot/com are recovered by forward
kinematics. The action channel uses the dof positions (qpos[7:]), so action_jerk is a
position-jerk proxy (noted in the paper); dof_vel_jerk uses the true qvel and is directly comparable."""
import numpy as np, mujoco, metrics as M
MK = ["success", "success_sustained", "track_fail", "fell", "hop_count", "swing_touched", "time_to_fall",
      "xcom_margin_ap_min", "xcom_margin_ml_min", "xcom_margin_ap_mean", "xcom_margin_ml_mean", "com_margin_ap_mean", "com_margin_ml_mean", "com_margin_ap_min", "com_margin_ml_min", "xcom_margin_viol_dur", "xcom_min_ttb", "xcom_low_ttb_events",
      "slippage_mm_s", "action_jerk_rms", "dof_vel_jerk_rms", "track_Epos", "track_Evel", "track_Eacc"]

def _xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])

def compute(qpos_traj, qvel_traj, fz_traj, model, mid, nom_h, HZ):
    T = len(qpos_traj)
    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    kp12_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in M.KP12_NAMES]
    mass = model.body_mass.reshape(-1, 1); Mtot = float(model.body_mass.sum())
    d = mujoco.MjData(model)
    K = ["t", "base_pos", "base_quat", "base_angvel", "base_linvel", "qvel", "fz_l", "fz_r",
         "foot_l_pos", "foot_r_pos", "foot_l_z", "foot_r_z", "foot_l_quat", "foot_r_quat",
         "com", "com_vel", "com_rel_obs", "action", "tilt", "fell", "kp12_robot"]
    log = {k: [] for k in K}
    prev_com = None; fell = False
    for t in range(T):
        q = qpos_traj[t]
        d.qpos[:len(q)] = q; mujoco.mj_kinematics(model, d); mujoco.mj_comPos(model, d)
        bp = q[0:3].copy(); bq = _xyzw(q[3:7])
        tilt = float(np.arccos(np.clip(1 - 2 * (bq[0] ** 2 + bq[1] ** 2), -1, 1)))
        fell = fell or (bp[2] < 0.5 * nom_h) or (tilt > 1.0)
        com = (d.xipos * mass).sum(0) / Mtot
        cv = (com - prev_com) * HZ if prev_com is not None else np.zeros(3)
        qv = qvel_traj[t][6:] if qvel_traj is not None else np.zeros(len(q) - 7)
        log["t"].append(t / HZ); log["base_pos"].append(bp); log["base_quat"].append(bq)
        log["base_angvel"].append(qvel_traj[t][3:6] if qvel_traj is not None else np.zeros(3))
        log["base_linvel"].append(qvel_traj[t][0:3] if qvel_traj is not None else np.zeros(3))
        log["qvel"].append(qv.copy())
        log["fz_l"].append(float(fz_traj[t][0])); log["fz_r"].append(float(fz_traj[t][1]))
        log["foot_l_pos"].append(d.xpos[lf][:2].copy()); log["foot_r_pos"].append(d.xpos[rf][:2].copy())
        log["foot_l_z"].append(float(d.xpos[lf][2])); log["foot_r_z"].append(float(d.xpos[rf][2]))
        log["foot_l_quat"].append(_xyzw(d.xquat[lf])); log["foot_r_quat"].append(_xyzw(d.xquat[rf]))
        log["com"].append(com); log["com_vel"].append(cv); log["com_rel_obs"].append(np.zeros(4))
        log["action"].append(q[7:].copy()); log["tilt"].append(tilt); log["fell"].append(fell)
        log["kp12_robot"].append(d.xpos[kp12_ids].copy())   # HuB 12 keypoints (world)
        prev_com = com
    ol = {k: np.array(v) for k, v in log.items()}
    ol["motion_id"] = mid; ol["nominal_h"] = float(nom_h); ol["dt"] = 1.0 / HZ; ol["body_mass_total"] = Mtot
    m = M.compute_metrics(ol)
    return {k: float(m.get(k, 0)) for k in MK}
