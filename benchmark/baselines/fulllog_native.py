"""Unified: native harness (each with its own mujoco m/d) records full per-frame state -> M.compute_metrics.
com = mass-weighted xipos; com_vel = finite-diff (needs threaded prev_com); com_rel_obs zeroed (only take metrics not depending on it).
Usage: log=new_log(); pc=None; per step pc=log_step(log,m,d,lf,rf,t,action,HZ,pc); at end finalize(log,mid,nom_h,mass,HZ)."""
import os, numpy as np, metrics as M
MK = ["success", "success_sustained", "track_fail", "fell", "hop_count", "swing_touched", "time_to_fall",
      "xcom_margin_ap_min", "xcom_margin_ml_min", "xcom_margin_ap_mean", "xcom_margin_ml_mean", "com_margin_ap_mean", "com_margin_ml_mean", "com_margin_ap_min", "com_margin_ml_min", "xcom_margin_viol_dur", "xcom_min_ttb", "xcom_low_ttb_events",
      "slippage_mm_s", "action_jerk_rms", "dof_vel_jerk_rms", "track_Epos", "track_Evel", "track_Eacc"]
_KEYS = ["t", "base_pos", "base_quat", "base_angvel", "base_linvel", "qvel", "fz_l", "fz_r",
         "foot_l_pos", "foot_r_pos", "foot_l_z", "foot_r_z", "foot_l_quat", "foot_r_quat",
         "com", "com_vel", "com_rel_obs", "action", "tilt", "fell", "kp12_robot", "qpos", "jnt_name", "jnt_qposadr"]

def new_log():
    return {k: [] for k in _KEYS}

def _xyzw(q_wxyz):  # mujoco xquat/qpos quat = wxyz -> xyzw
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

def log_step(log, m, d, lf, rf, t, action, HZ, prev_com, fz_l=None, fz_r=None):
    from success_metric import foot_fz as FZ
    bp = d.qpos[0:3].copy(); bq = _xyzw(d.qpos[3:7])
    tilt = float(np.arccos(np.clip(1 - 2 * (bq[0] ** 2 + bq[1] ** 2), -1, 1)))
    fll = log.get("_fell", False) or (bp[2] < log["_half_nom"]) or (tilt > 1.0)
    log["_fell"] = fll
    com = (d.xipos * m.body_mass[:, None]).sum(0) / m.body_mass.sum()
    cv = (com - prev_com) * HZ if prev_com is not None else np.zeros(3)
    zl = float(fz_l) if fz_l is not None else float(FZ(m, d, lf))
    zr = float(fz_r) if fz_r is not None else float(FZ(m, d, rf))
    log["t"].append(t / HZ); log["base_pos"].append(bp); log["base_quat"].append(bq)
    log["base_angvel"].append(d.qvel[3:6].copy()); log["base_linvel"].append(d.qvel[0:3].copy())
    log["qvel"].append(d.qvel[6:].copy())
    log["fz_l"].append(zl); log["fz_r"].append(zr)
    log["foot_l_pos"].append(d.xpos[lf][:2].copy()); log["foot_r_pos"].append(d.xpos[rf][:2].copy())
    log["foot_l_z"].append(float(d.xpos[lf][2])); log["foot_r_z"].append(float(d.xpos[rf][2]))
    log["foot_l_quat"].append(_xyzw(d.xquat[lf])); log["foot_r_quat"].append(_xyzw(d.xquat[rf]))
    log["com"].append(com); log["com_vel"].append(cv); log["com_rel_obs"].append(np.zeros(4))
    log["action"].append(np.asarray(action, np.float64).copy()); log["tilt"].append(tilt); log["fell"].append(fll)
    log["kp12_robot"].append(M.kp12_world(m, d))   # HuB 12 keypoints (world)
    log["qpos"].append(d.qpos.copy())              # each model's full qpos (for rendering)
    if not log["jnt_name"]:                          # joint names + qpos addresses (once): for name-based remapping to shared scene at render time
        log["jnt_name"] = [m.joint(i).name for i in range(m.njnt)]
        log["jnt_qposadr"] = [int(m.jnt_qposadr[i]) for i in range(m.njnt)]
    return com

def finalize(log, mid, nom_h, mass, HZ):
    for k in ("_fell", "_half_nom"): log.pop(k, None)
    ol = {k: np.array(v) for k, v in log.items()}
    ol["motion_id"] = mid; ol["nominal_h"] = float(nom_h); ol["dt"] = 1.0 / HZ; ol["body_mass_total"] = float(mass)
    _dp = os.environ.get("WBT_DUMP_FLOG")
    if _dp:
        import pickle; pickle.dump(ol, open(_dp, "wb"))
    m = M.compute_metrics(ol)
    return {k: float(m.get(k, 0)) for k in MK}
