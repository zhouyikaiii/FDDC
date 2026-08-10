"""Shared: for my-harness baseline (using W.G1Sim), record full per-frame state -> M.compute_metrics full metrics.
com_rel_obs zeroed (baseline does not compute it) -> only take metrics not depending on it (xcom/jerk/track/hop/fall/ttb)."""
import os, numpy as np, metrics as M
MK = ["success", "success_sustained", "track_fail", "fell", "hop_count", "swing_touched", "time_to_fall",
      "xcom_margin_ap_min", "xcom_margin_ml_min", "xcom_margin_ap_mean", "xcom_margin_ml_mean", "com_margin_ap_mean", "com_margin_ml_mean", "com_margin_ap_min", "com_margin_ml_min", "xcom_margin_viol_dur", "xcom_min_ttb", "xcom_low_ttb_events",
      "slippage_mm_s", "action_jerk_rms", "dof_vel_jerk_rms", "track_Epos", "track_Evel", "track_Eacc"]
_KEYS = ["t", "base_pos", "base_quat", "base_angvel", "base_linvel", "qvel", "fz_l", "fz_r",
         "foot_l_pos", "foot_r_pos", "foot_l_z", "foot_r_z", "foot_l_quat", "foot_r_quat",
         "com", "com_vel", "com_rel_obs", "action", "tilt", "fell", "kp12_robot", "qpos"]

def new_log():
    return {k: [] for k in _KEYS}

def log_step(log, sim, t, action, tilt, fell, HZ):
    lp, lv, lz = sim.foot_state(sim.lfoot); rp, rv, rz = sim.foot_state(sim.rfoot)
    com, comv = sim.com_state()
    log["t"].append(t / HZ); log["base_pos"].append(sim.data.qpos[0:3].copy())
    log["base_quat"].append(sim.base_quat_xyzw()); log["base_angvel"].append(sim.base_ang_vel_base())
    log["base_linvel"].append(sim.data.qvel[0:3].copy()); log["qvel"].append(sim.dof_vel())
    log["fz_l"].append(sim.foot_contact_fz(sim.lfoot)); log["fz_r"].append(sim.foot_contact_fz(sim.rfoot))
    log["foot_l_pos"].append(lp); log["foot_r_pos"].append(rp); log["foot_l_z"].append(lz); log["foot_r_z"].append(rz)
    log["foot_l_quat"].append(sim.foot_quat_xyzw(sim.lfoot)); log["foot_r_quat"].append(sim.foot_quat_xyzw(sim.rfoot))
    log["com"].append(com); log["com_vel"].append(comv); log["com_rel_obs"].append(np.zeros(4))
    log["action"].append(np.asarray(action, np.float64).copy()); log["tilt"].append(float(tilt)); log["fell"].append(bool(fell))
    log["kp12_robot"].append(sim.data.xpos[sim.kp12_ids].copy())   # HuB 12 keypoints (world)
    log["qpos"].append(sim.data.qpos.copy())                        # full qpos(36) for pose reconstruction in rendering

def finalize(log, mid, nom_h, mass, HZ):
    ol = {k: np.array(v) for k, v in log.items()}
    ol["motion_id"] = mid; ol["nominal_h"] = float(nom_h); ol["dt"] = 1.0 / HZ; ol["body_mass_total"] = float(mass)
    _dp = os.environ.get("WBT_DUMP_FLOG")            # for video: dump raw full-frame log (incl qpos/com/com_vel/foot)
    if _dp:
        import pickle; pickle.dump(ol, open(_dp, "wb"))
    m = M.compute_metrics(ol)
    return {k: float(m.get(k, 0)) for k in MK}
