"""Metric library (design-spec §4) for the single-leg balance benchmark.

Pure functions: a rollout log (true world state, from wbt_rollout.run_rollout) + the dataset reference
-> a flat dict of metrics. Nothing here touches the policy observation (principle P2: method-agnostic),
so a policy that never used foot-contact phase is scored by the exact same ruler. Stage-aware: the
sim2sim caller gets the full set; a 2real caller would drop the CoM-based §4.5 terms.

Windows (design-spec §3): hop / fall are measured over the WHOLE motion (double->single->double, so a
jump during the lift/land transition still counts); foot-activity / CoM-deviation / xCoM-TTB / tracking
are measured inside the reference single-support window [t0, t1] given by the dataset.
"""
from __future__ import annotations

import numpy as np

from wbt_rollout import (
    load_motion_npz,
    model_body_masses_by_name,
    quat_rotate_inverse_xyzw,
    quat_to_matrix_xyzw,
    wxyz_to_xyzw,
)

G = 9.81
# G1 single-foot support rectangle in the ankle_roll_link body frame (x=fore/aft AP, y=lateral ML).
# From the training reward's _G1_FOOT_SUPPORT_POLYGON; ML (y) is the tight axis for single-leg stance.
FOOT_AP = (-0.05, 0.12)      # x_min, x_max  (m)
FOOT_ML = (-0.0275, 0.0275)  # y_min, y_max  (m)

# HuB-aligned 12 tracking keypoints [arXiv:2505.07294 §B.1]: hips, knees, ankles, shoulders, elbows,
# wrists (left/right). Cartesian keypoint tracking error over these is reported in mm; success requires
# the average keypoint error to stay <= 0.5 m (500 mm), same as HuB.
KP12_NAMES = [
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
TRACK_FAIL_THRESH_MM = 500.0   # HuB: average keypoint tracking error > 0.5 m => trial fails

_KP12_ID_CACHE: dict = {}


def kp12_world(model, data) -> np.ndarray:
    """(12, 3) world positions of the HuB tracking keypoints from a live mujoco model+data.
    Every rollout log-builder appends this per frame as log['kp12_robot']."""
    import mujoco
    ids = _KP12_ID_CACHE.get(id(model))
    if ids is None:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in KP12_NAMES]
        if any(i < 0 for i in ids):
            miss = [n for n, i in zip(KP12_NAMES, ids) if i < 0]
            raise ValueError(f"kp12 bodies missing from model: {miss}")
        _KP12_ID_CACHE[id(model)] = ids
    return data.xpos[ids].copy()


def ref_kp12(body_names, body_pos_w) -> np.ndarray:
    """(T, 12, 3) reference keypoint world positions from a motion npz (body_pos_w + body_names)."""
    idx = [list(body_names).index(n) for n in KP12_NAMES]
    return np.asarray(body_pos_w)[:, idx]


# ===================================================================================================
# reference signals (motion-only; policy-independent) -- computed once per motion
# ===================================================================================================
def reference_signals(motion_id: str, T: int) -> dict:
    """Per-frame reference quantities the metrics compare against:
      window     (t0, t1)   single-support span from reference_support_state (!=0)
      support_foot 'L'/'R'  the reference support foot in that window
      com_rel    (T, 4)     reference CoM-rel-support-center [pos_xy, vel_xy] in the REFERENCE base frame
                            (same construction as the deployable actor obs whole_body_com_rel_support_center)
      kp         (T, 3, 3)  reference keypoint world positions [pelvis, left_foot, right_foot]
    """
    ref = load_motion_npz(motion_id)
    d = ref["npz"]
    bn = ref["body_names"]
    pelvis, lf, rf = bn.index("pelvis"), bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")
    pos = d["body_pos_w"][:T].astype(np.float64)       # (T, nbody, 3) link origins
    vel = d["body_lin_vel_w"][:T].astype(np.float64)   # (T, nbody, 3)
    base_quat = d["body_quat_w"][:T, pelvis].astype(np.float64)  # wxyz

    # mass-weighted link-origin CoM, masses aligned to the npz body order (same links as the robot CoM)
    masses = model_body_masses_by_name()
    mvec = np.array([masses.get(n, 0.0) for n in bn], dtype=np.float64)
    M = mvec.sum()
    com = (pos * mvec[None, :, None]).sum(1) / M       # (T, 3)
    comv = (vel * mvec[None, :, None]).sum(1) / M

    lz, rz = pos[:, lf, 2], pos[:, rf, 2]
    fp = np.stack([pos[:, lf, :2], pos[:, rf, :2]], 1)  # (T, 2, 2)
    fv = np.stack([vel[:, lf, :2], vel[:, rf, :2]], 1)
    com_rel = np.zeros((T, 4))
    for t in range(T):
        mask = _height_support_mask(lz[t], rz[t])
        w = mask.astype(np.float64)[:, None]
        c_pos = (fp[t] * w).sum(0) / w.sum() if mask.any() else fp[t].mean(0)
        c_vel = (fv[t] * w).sum(0) / w.sum() if mask.any() else fv[t].mean(0)
        rel_p = np.array([com[t, 0] - c_pos[0], com[t, 1] - c_pos[1], 0.0])
        rel_v = np.array([comv[t, 0] - c_vel[0], comv[t, 1] - c_vel[1], 0.0])
        bq = wxyz_to_xyzw(base_quat[t])
        com_rel[t, :2] = quat_rotate_inverse_xyzw(bq, rel_p)[:2]
        com_rel[t, 2:] = quat_rotate_inverse_xyzw(bq, rel_v)[:2]

    ss = ref["ref_support_state"][:T]
    single = ss != 0
    if single.any():
        t0 = int(np.argmax(single))
        t1 = int(len(single) - np.argmax(single[::-1]))
    else:  # degenerate: no single-support phase in the reference
        t0, t1 = 0, T
    # support foot: the planted (lower) foot over the single window
    support_foot = "L" if pos[t0:t1, lf, 2].mean() <= pos[t0:t1, rf, 2].mean() else "R"

    kp = np.stack([pos[:, pelvis], pos[:, lf], pos[:, rf]], 1)  # (T, 3, 3) legacy 3-keypoint
    kp12 = pos[:, [bn.index(n) for n in KP12_NAMES]]             # (T, 12, 3) HuB tracking keypoints
    return {"window": (t0, t1), "support_foot": support_foot, "com_rel": com_rel, "kp": kp, "kp12": kp12}


def _height_support_mask(lz: float, rz: float, diff: float = 0.03) -> np.ndarray:
    if abs(lz - rz) <= diff:
        return np.array([True, True])
    return np.array([lz < rz, rz <= lz])


# ===================================================================================================
# §4.0 support state
# ===================================================================================================
def support_state_series(fz_l: np.ndarray, fz_r: np.ndarray, f_thresh: float) -> np.ndarray:
    """Per-step support label: 0 double, 1 single_L, 2 single_R, 3 flight (both feet off = jump)."""
    L, R = fz_l > f_thresh, fz_r > f_thresh
    s = np.full(len(fz_l), 3, dtype=int)          # flight
    s[L & R] = 0
    s[L & ~R] = 1
    s[R & ~L] = 2
    return s


# ===================================================================================================
# §4.2 hops (head-line metric)
# ===================================================================================================
def hop_metrics(fz_support: np.ndarray, base_z: np.ndarray, fz_l: np.ndarray, fz_r: np.ndarray,
                dt: float, f_thresh: float, mg: float) -> dict:
    """A hop = the support foot's normal force drops below threshold (loss of contact), counted over the
    WHOLE motion. height = base-z rise during each airborne interval; impulse = landing force above body
    weight * dt (how hard it slams back). Also counts full 'flight' events (both feet off)."""
    below = fz_support < f_thresh
    starts, ends = _rising_falling(below)
    count = len(starts)
    heights, impulses = [], []
    for s, e in zip(starts, ends):
        heights.append(float(base_z[s:e + 1].max() - base_z[s])) if e > s else heights.append(0.0)
        # landing impulse: sum of (Fz - mg)_+ * dt over a short window right after re-contact
        land = min(e + 1, len(fz_support) - 1)
        w = fz_support[land:land + 5]
        impulses.append(float(np.clip(w - mg, 0, None).sum() * dt))
    flight = (fz_l < f_thresh) & (fz_r < f_thresh)
    flight_events = len(_rising_falling(flight)[0])
    return {
        "hop_count": count,
        "hop_height_max": max(heights) if heights else 0.0,
        "hop_height_mean": float(np.mean(heights)) if heights else 0.0,
        "hop_impulse_max": max(impulses) if impulses else 0.0,
        "flight_events": flight_events,
    }


def _rising_falling(b: np.ndarray):
    """Start/end indices of True runs in a boolean series."""
    b = b.astype(int)
    d = np.diff(np.concatenate([[0], b, [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0] - 1


# §4.2b swing-foot touchdown (extra failure mode)
SWING_TOUCHDOWN_MARGIN_S = 0.2  # seconds trimmed from EACH end of the single-support window


def swing_touchdown_metrics(fz_swing: np.ndarray, window: tuple, dt: float, f_thresh: float,
                            margin_s: float = SWING_TOUCHDOWN_MARGIN_S) -> dict:
    """The NON-support (swing) foot bearing load during single support = a failed single-leg hold. Checked
    only inside [t0+margin, t1-margin]: a `margin_s`-second band is trimmed at each end so lifting the foot
    slightly early or setting it down slightly late (timing jitter vs the reference) is NOT counted."""
    t0, t1 = window
    mg = int(round(margin_s / dt))
    a, b = t0 + mg, t1 - mg
    if b <= a:  # window too short after trimming -> nothing to judge
        return {"swing_touchdown_frames": 0, "swing_touchdown_events": 0, "swing_touched": False}
    touch = fz_swing[a:b] > f_thresh
    return {"swing_touchdown_frames": int(touch.sum()),
            "swing_touchdown_events": len(_rising_falling(touch)[0]),
            "swing_touched": bool(touch.any())}


# ===================================================================================================
# §4.3 fall / time-to-fall
# ===================================================================================================
def fall_metrics(base_z: np.ndarray, tilt: np.ndarray, dt: float, nominal_h: float,
                 h_frac: float = 0.5, ang_thresh: float = 1.0) -> dict:
    """Fell if the base drops below h_frac*nominal OR tilts past ang_thresh (rad). time_to_fall = first
    such time, else the full episode length (right-censored)."""
    bad = (base_z < h_frac * nominal_h) | (tilt > ang_thresh)
    fell = bool(bad.any())
    ttf = float(np.argmax(bad) * dt) if fell else float(len(base_z) * dt)
    return {"fell": fell, "time_to_fall": ttf}


# ===================================================================================================
# §4.4 support-foot activity area
# ===================================================================================================
def foot_activity(foot_pos_xy: np.ndarray, foot_quat_xyzw: np.ndarray, window: tuple) -> dict:
    """A good policy pins the support foot; a bad one smears/pivots it. area = convex-hull area of the
    support-foot origin xy track in the window; yaw_drift = max |yaw - yaw(t0)|."""
    t0, t1 = window
    pts = foot_pos_xy[t0:t1]
    area = _convex_hull_area(pts)
    yaws = np.array([_yaw_of(q) for q in foot_quat_xyzw[t0:t1]])
    yaw_drift = float(np.max(np.abs(_unwrap(yaws) - yaws[0]))) if len(yaws) else 0.0
    return {"foot_activity_area": area, "foot_yaw_drift": yaw_drift}


def _yaw_of(q_xyzw: np.ndarray) -> float:
    R = quat_to_matrix_xyzw(q_xyzw)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _unwrap(a):
    return np.unwrap(a) if len(a) else a


def _convex_hull_area(pts: np.ndarray) -> float:
    p = np.unique(np.round(pts, 6), axis=0)
    if len(p) < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo = []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    hi = []
    for q in p[::-1]:
        while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
            hi.pop()
        hi.append(q)
    hull = np.array(lo[:-1] + hi[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ===================================================================================================
# §4.5(a) dynamic-CoM deviation relative to the reference  (MAIN balance metric)
# ===================================================================================================
def dyn_com_dev(robot_com_rel: np.ndarray, ref_com_rel: np.ndarray, window: tuple) -> dict:
    """Mean ||robot_com_rel - ref_com_rel|| over the single-support window, split into position and
    velocity. Both sides use the deployable CoM-rel-support-center quantity, so this measures 'did the
    balance state follow the reference INTENT' -- and does NOT punish a dynamic motion for its CoM
    correctly leaving the support polygon (design-spec P5)."""
    t0, t1 = window
    r, q = robot_com_rel[t0:t1], ref_com_rel[t0:t1]
    dpos = np.linalg.norm(r[:, :2] - q[:, :2], axis=1)
    dvel = np.linalg.norm(r[:, 2:] - q[:, 2:], axis=1)
    return {
        "com_dev_pos_mean": float(dpos.mean()) if len(dpos) else 0.0,
        "com_dev_pos_max": float(dpos.max()) if len(dpos) else 0.0,
        "com_dev_vel_mean": float(dvel.mean()) if len(dvel) else 0.0,
    }


# ===================================================================================================
# §4.5(b) absolute xCoM TTB / margin  (DIAGNOSTIC only -- why it jumped, not a headline)
# ===================================================================================================
def xcom_ttb_margin(com: np.ndarray, com_vel: np.ndarray, support_foot_xy: np.ndarray,
                    support_foot_quat: np.ndarray, window: tuple, dt: float, t_warn: float = 0.3) -> dict:
    """Extrapolated-CoM (capture point) margin & time-to-boundary against the single-foot support
    rectangle, in the support-foot frame so AP (fore/aft) and ML (lateral) separate. Reported for the
    single-support window. ML is usually the binding axis for one-leg stance."""
    t0, t1 = window
    # CoM acceleration (world frame, finite difference of com_vel) -- for the capture-point xCoM velocity d(xCoM)/dt = c_dot + c_ddot/omega
    com_acc = np.gradient(com_vel, dt, axis=0)
    margins_ap, margins_ml = [], []
    cmargins_ap, cmargins_ml = [], []   # actual-CoM (non-extrapolated) margin m_c: positive = inward
    xttbs, cttbs = [], []               # xCoM-TTB (primary, xCoM side) / CoM-TTB (honest, CoM side)
    viol = 0
    for t in range(t0, t1):
        h = max(com[t, 2], 0.25)
        omega = np.sqrt(G / h)
        yaw = _yaw_of(support_foot_quat[t])
        c, s = np.cos(yaw), np.sin(yaw)
        Rt = np.array([[c, s], [-s, c]])                 # world-xy -> foot-frame
        rel = Rt @ (com[t, :2] - support_foot_xy[t])
        vel = Rt @ com_vel[t, :2]
        xcom = rel + vel / omega
        # xCoM's own velocity (foot frame): d(xCoM)/dt = c_dot + c_ddot/omega, ignoring omega_dot (height changes very slowly)
        xcom_vel = vel + (Rt @ com_acc[t, :2]) / omega
        m_ap = min(xcom[0] - FOOT_AP[0], FOOT_AP[1] - xcom[0])
        m_ml = min(xcom[1] - FOOT_ML[0], FOOT_ML[1] - xcom[1])
        margins_ap.append(m_ap)
        margins_ml.append(m_ml)
        # actual-CoM rel margin to the support-rectangle boundary (positive = inward)
        cmargins_ap.append(min(rel[0] - FOOT_AP[0], FOOT_AP[1] - rel[0]))
        cmargins_ml.append(min(rel[1] - FOOT_ML[0], FOOT_ML[1] - rel[1]))
        if min(m_ap, m_ml) < 0:
            viol += 1
        # xCoM-TTB (primary): time for the capture point xcom, at its own velocity xcom_vel, to reach the boundary it heads toward; directional (out-of-bounds + diverging -> 0)
        xttbs.append(min(_ttb_axis(xcom[0], xcom_vel[0], *FOOT_AP), _ttb_axis(xcom[1], xcom_vel[1], *FOOT_ML)))
        # CoM-TTB (honest: actual-CoM rel + CoM velocity vel): kept for internal comparison, no longer primary in the report
        cttbs.append(min(_ttb_axis(rel[0], vel[0], *FOOT_AP), _ttb_axis(rel[1], vel[1], *FOOT_ML)))
    xttbs = np.array(xttbs); cttbs = np.array(cttbs)
    return {
        "xcom_margin_ap_min": float(np.min(margins_ap)) if margins_ap else 0.0,   # MoS AP min (Hof margin of stability)
        "xcom_margin_ml_min": float(np.min(margins_ml)) if margins_ml else 0.0,   # MoS ML min (binding axis for 1-leg)
        "xcom_margin_ap_mean": float(np.mean(margins_ap)) if margins_ap else 0.0,  # MoS AP mean
        "xcom_margin_ml_mean": float(np.mean(margins_ml)) if margins_ml else 0.0,  # MoS ML mean
        "com_margin_ap_min": float(np.min(cmargins_ap)) if cmargins_ap else 0.0,
        "com_margin_ml_min": float(np.min(cmargins_ml)) if cmargins_ml else 0.0,
        "com_margin_ap_mean": float(np.mean(cmargins_ap)) if cmargins_ap else 0.0,
        "com_margin_ml_mean": float(np.mean(cmargins_ml)) if cmargins_ml else 0.0,
        "xcom_min_ttb": float(np.min(xttbs)) if len(xttbs) else 0.0,          # xCoM side (primary)
        "xcom_low_ttb_events": int(np.sum(xttbs < t_warn)),                   # xCoM-side low-TTB frames
        "com_min_ttb": float(np.min(cttbs)) if len(cttbs) else 0.0,           # CoM side (honest comparison)
        "com_low_ttb_events": int(np.sum(cttbs < t_warn)),
        "xcom_margin_viol_dur": float(viol * dt),
    }


def _ttb_axis(p: float, v: float, lo: float, hi: float, big: float = 9.9) -> float:
    """Time for a point at p moving at v to reach the [lo, hi] bound it is heading toward."""
    if v > 1e-6:
        return max(0.0, (hi - p) / v)
    if v < -1e-6:
        return max(0.0, (p - lo) / (-v))
    return big


# ===================================================================================================
# §4.6 smoothness (jerk) -- catch policies that hold balance by high-frequency chatter
# ===================================================================================================
def jerk_metrics(action: np.ndarray, qvel: np.ndarray, dt: float) -> dict:
    """RMS of the 2nd difference of the action command and of joint velocity (design-spec P3: this is a
    SMOOTHNESS probe, not a 'less sway = better' reward)."""
    def rms2(x):
        if len(x) < 3:
            return 0.0
        return float(np.sqrt((np.diff(x, 2, axis=0) ** 2).mean()))

    return {"action_jerk_rms": rms2(action), "dof_vel_jerk_rms": rms2(qvel)}


# ===================================================================================================
# §4.7 tracking error (secondary / context; NOT a headline -- P5 lets balance deviate from reference)
# ===================================================================================================
def tracking_error(kp_robot: np.ndarray, kp_ref: np.ndarray, window: tuple, dt: float) -> dict:
    """HuB-aligned global keypoint tracking error over the single-support window [arXiv:2505.07294].
    kp_robot / kp_ref: (T, 12, 3) Cartesian world keypoints (the HuB 12 keypoints). Reported in HuB
    units: Epos (mm), Evel (mm/frame), Eacc (mm/frame^2) -- per-frame differences, NOT per-second."""
    t0, t1 = window
    P, Pr = kp_robot[t0:t1], kp_ref[t0:t1]
    epos = float(np.linalg.norm(P - Pr, axis=-1).mean() * 1000.0)
    if t1 - t0 >= 3:
        evel = float(np.linalg.norm(np.diff(P, axis=0) - np.diff(Pr, axis=0), axis=-1).mean() * 1000.0)
        eacc = float(np.linalg.norm(np.diff(P, 2, axis=0) - np.diff(Pr, 2, axis=0), axis=-1).mean() * 1000.0)
    else:
        evel = eacc = 0.0
    return {"track_Epos": epos, "track_Evel": evel, "track_Eacc": eacc}


def slippage_metric(sup_xy: np.ndarray, fz_support: np.ndarray, window: tuple, dt: float,
                    f_thresh: float) -> dict:
    """HuB Slippage (mm/s): the support foot's ground-relative horizontal speed while in contact,
    averaged over the single-support window. Higher = less stable foot contact (foot sliding)."""
    t0, t1 = window
    xy = np.asarray(sup_xy[t0:t1], dtype=np.float64)
    if len(xy) < 2:
        return {"slippage_mm_s": 0.0}
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / dt        # m/s per inter-frame step
    contact = np.asarray(fz_support[t0:t1])[1:] > f_thresh           # foot planted on the later frame
    v = speed[contact] if contact.any() else speed
    return {"slippage_mm_s": float(v.mean() * 1000.0)}


# ===================================================================================================
# top-level: log -> flat metric dict
# ===================================================================================================
def compute_metrics(log: dict, f_thresh_frac: float = 0.05) -> dict:
    """Run all §4 metrics on one rollout log. Returns a flat {name: value} dict."""
    dt = float(log["dt"])
    mg = float(log["body_mass_total"]) * G
    f_thresh = f_thresh_frac * mg
    T = len(log["t"])
    ref = reference_signals(str(log["motion_id"]), T)
    win = ref["window"]
    support_L = ref["support_foot"] == "L"
    fz_l, fz_r = log["fz_l"], log["fz_r"]
    fz_support = fz_l if support_L else fz_r
    fz_swing = fz_r if support_L else fz_l            # non-support foot: should stay lifted in the window
    base_z = log["base_pos"][:, 2]

    # robot keypoints: HuB 12 keypoints logged per frame by every harness's log-builder (kp12_world)
    kp_robot = np.asarray(log["kp12_robot"])            # (T, 12, 3)

    sup_xy = log["foot_l_pos"] if support_L else log["foot_r_pos"]
    sup_quat = log["foot_l_quat"] if support_L else log["foot_r_quat"]

    m = {"motion_id": str(log["motion_id"]), "seed": int(log.get("seed", 0)),
         "support_foot": ref["support_foot"], "T": T, "single_window": list(win)}
    m.update(hop_metrics(fz_support, base_z, fz_l, fz_r, dt, f_thresh, mg))
    m.update(fall_metrics(base_z, log["tilt"], dt, float(log["nominal_h"])))
    m.update(foot_activity(sup_xy, sup_quat, win))
    m.update(dyn_com_dev(log["com_rel_obs"], ref["com_rel"], win))
    m.update(xcom_ttb_margin(log["com"], log["com_vel"], sup_xy, sup_quat, win, dt))
    m.update(jerk_metrics(log["action"], log["qvel"], dt))
    m.update(tracking_error(kp_robot, ref["kp12"], win, dt))
    m.update(slippage_metric(sup_xy, fz_support, win, dt, f_thresh))
    m.update(swing_touchdown_metrics(fz_swing, win, dt, f_thresh))
    # sustained single-leg: 0 support-foot hops AND did not fall AND swing foot never touched down in-window
    # (early-lift / late-set-down at the window ends are excluded by the swing-touchdown margin)
    m["success_sustained"] = bool(m["hop_count"] == 0 and not m["fell"] and not m["swing_touched"])
    # HuB tracking gate: average 12-keypoint error must stay <= 0.5 m (500 mm) -- anti "stand still, ignore ref"
    m["track_fail"] = bool(m["track_Epos"] > TRACK_FAIL_THRESH_MM)
    # HuB-aligned success = sustained single-leg AND tracking within 0.5 m
    m["success"] = bool(m["success_sustained"] and not m["track_fail"])
    return m
