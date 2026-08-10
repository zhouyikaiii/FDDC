"""Sustained single-leg success (== metrics.py `success_sustained`): hop_count==0 AND not fell AND not
swing_touched. This is the keypoint-free tier check; the HuB 0.5 m 12-keypoint tracking gate that
metrics.py folds into its FINAL `success` needs the keypoint trajectories and is applied there
(compute_metrics), NOT here — for the gated, reported number use compute_metrics / an adapter's full-metric
sub-command. (No method scored 0/90 Perfect is affected: a fall/hop already fails the tier before the gate.)
Requires per-step fz_l, fz_r (foot contact normal force), base_z, tilt (rad) plus nominal_h, dt, mass, motion_id."""
import os
import sys
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
HS = _ROOT
sys.path.insert(0, os.path.join(HS, "eval"))
import metrics as M

def success_of(fz_l, fz_r, base_z, tilt, nominal_h, dt, mass, motion_id):
    fz_l = np.asarray(fz_l, float); fz_r = np.asarray(fz_r, float)
    base_z = np.asarray(base_z, float); tilt = np.asarray(tilt, float)
    T = len(base_z)
    ref = M.reference_signals(str(motion_id), T)
    win = ref["window"]; support_L = ref["support_foot"] == "L"
    fz_support = fz_l if support_L else fz_r
    fz_swing = fz_r if support_L else fz_l
    mg = float(mass) * M.G; f_thresh = 0.05 * mg
    hop = M.hop_metrics(fz_support, base_z, fz_l, fz_r, dt, f_thresh, mg)
    fall = M.fall_metrics(base_z, tilt, dt, float(nominal_h))
    swing = M.swing_touchdown_metrics(fz_swing, win, dt, f_thresh)
    return {"success": bool(hop["hop_count"] == 0 and not fall["fell"] and not swing["swing_touched"]),
            "hop_count": int(hop["hop_count"]), "fell": bool(fall["fell"]), "swing_touched": bool(swing["swing_touched"])}

def foot_fz(model, data, foot_bid):
    """mujoco: sum of foot-body normal contact forces (approx vertical GRF)."""
    import mujoco
    tot = 0.0; f = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.geom_bodyid[con.geom1]; b2 = model.geom_bodyid[con.geom2]
        if foot_bid == b1 or foot_bid == b2:
            mujoco.mj_contactForce(model, data, i, f)
            tot += abs(float(f[0]))
    return tot
