#!/usr/bin/env python
"""Deliverable (2): proxy (link-origin) vs TRUE (xipos) whole-body CoM error, over DDC test-90 frames.

Runs the deployed DDC policy (model_0262000.pt) clean over the test motions, then for EACH logged frame
reconstructs the full qpos and, by forward kinematics on the deploy model, computes BOTH CoM definitions:
  proxy = Sum_{i>=1} m_i * xpos[i]  / Sum_{i>=1} m_i   (link ORIGIN -- the old com_state / policy-obs proxy)
  true  = Sum_i     m_i * xipos[i] / Sum_i     m_i    (per-body CoM -- com_true / the baselines / new metric)
Reports 3D CoM position error, horizontal xCoM error, and AP/ML signed systematic bias (support-foot frame).
Also asserts the rollout's logged log['com'] now equals the offline TRUE com (validates the metric change).
No physics re-run for the comparison itself -- both CoMs come from the SAME logged configuration.
"""
import argparse, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--pt", default=os.path.join(PKG, "policy", "model_0262000.pt"))
ap.add_argument("--robot-config", default=os.path.join(PKG, "policy", "robot_config.json"))
ap.add_argument("--robot-xml", default=os.path.join(PKG, "robot", "g1_29dof", "g1_29dof.xml"))
ap.add_argument("--motion-dir", default=os.path.join(PKG, "data", "data_stratified_900", "test"))
ap.add_argument("--limit", type=int, default=0, help="first N motions (0 = all 90)")
ap.add_argument("--threads", default="1")
args = ap.parse_args()

os.environ["WBT_ORT_THREADS"] = args.threads
os.environ.setdefault("OMP_NUM_THREADS", args.threads)
os.environ["WBT_EVAL_ROBOT_XML"] = args.robot_xml
os.environ["WBT_EVAL_MOTION_DIR"] = args.motion_dir
sys.path.insert(0, HERE)
import numpy as np, mujoco
import wbt_rollout as W, metrics as MET
from fast_policy import FastPolicy

pol = FastPolicy(args.pt, args.robot_config)
sim = W.G1Sim(pol.dof_names)               # gives model + qpos_adr; we FK on a scratch MjData
m = sim.model
d = mujoco.MjData(m)
mass = m.body_mass.copy()
total_no_world = mass[1:].sum()            # com_state denominator (excludes world body 0)
total_all = mass.sum()                     # com_true denominator (world mass = 0, so same value)
qpos_adr = sim.qpos_adr

def full_qpos(base_pos, base_quat_xyzw, dof29):
    q = np.zeros(m.nq)
    q[0:3] = base_pos
    x, y, z, w = base_quat_xyzw
    q[3:7] = [w, x, y, z]                   # mujoco free-joint quat is wxyz
    q[qpos_adr] = dof29
    return q

def both_com(q):
    d.qpos[:] = 0.0; d.qpos[:m.nq] = q
    mujoco.mj_forward(m, d)
    proxy = (mass[1:, None] * d.xpos[1:]).sum(0) / total_no_world
    true = (mass[:, None] * d.xipos).sum(0) / total_all
    return proxy, true

mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p))
              for p in glob.glob(os.path.join(args.motion_dir, "sample_*_mj.npz")))
if args.limit:
    mots = mots[:args.limit]
print(f"DDC proxy-vs-true CoM over {len(mots)} test motions (clean)\n")

pos_err, xcom_err, ap_bias, ml_bias, ml_sign_disagree = [], [], [], [], []
G, DT = 9.81, 1.0 / W.CONTROL_HZ
max_logcom_mismatch = 0.0
for i, mid in enumerate(mots):
    log = W.run_rollout(pol, mid, seed=0, use_npz_ref=True,
                        dof_vel_noise=0.0, dof_vel_delay=0, imu_noise=False)
    ref = MET.reference_signals(mid, len(log["t"]))
    t0, t1 = ref["window"]
    supL = ref["support_foot"] == "L"
    sup_xy = log["foot_l_pos"] if supL else log["foot_r_pos"]
    sup_quat = log["foot_l_quat"] if supL else log["foot_r_quat"]
    P, Tr = [], []
    for t in range(len(log["t"])):
        q = full_qpos(log["base_pos"][t], log["base_quat"][t], log["qpos"][t])
        p, tr = both_com(q)
        P.append(p); Tr.append(tr)
    P, Tr = np.array(P), np.array(Tr)
    # sanity: the rollout's logged com (now the metric-side CoM) must equal the offline TRUE com
    max_logcom_mismatch = max(max_logcom_mismatch, float(np.abs(np.array(log["com"]) - Tr).max()))
    # finite-diff velocities (same estimator both sides) for the xCoM comparison
    Pv = np.vstack([np.zeros(3), np.diff(P, axis=0)]) / DT
    Trv = np.vstack([np.zeros(3), np.diff(Tr, axis=0)]) / DT
    for t in range(t0, t1):                # AP/ML bias + xCoM error over the single-support window
        pos_err.append(np.linalg.norm(Tr[t] - P[t]))
        h = max(Tr[t, 2], 0.25); omega = np.sqrt(G / h)
        xcom_p = P[t, :2] + Pv[t, :2] / omega
        xcom_t = Tr[t, :2] + Trv[t, :2] / omega
        xcom_err.append(np.linalg.norm(xcom_t - xcom_p))
        yaw = MET._yaw_of(sup_quat[t]); c, s = np.cos(yaw), np.sin(yaw)
        Rt = np.array([[c, s], [-s, c]])   # world-xy -> foot frame (AP=x, ML=y)
        db = Rt @ (Tr[t, :2] - P[t, :2])
        ap_bias.append(db[0]); ml_bias.append(db[1])
        # actual-CoM ML margin (support-foot frame): do proxy vs true DISAGREE on in/out of the support rect?
        pml = (Rt @ (P[t, :2] - sup_xy[t]))[1]; tml = (Rt @ (Tr[t, :2] - sup_xy[t]))[1]
        pm = min(pml - MET.FOOT_ML[0], MET.FOOT_ML[1] - pml)
        tm = min(tml - MET.FOOT_ML[0], MET.FOOT_ML[1] - tml)
        ml_sign_disagree.append((pm > 0) != (tm > 0))
    if (i + 1) % 10 == 0 or i + 1 == len(mots):
        print(f"  [{i+1:>3}/{len(mots)}] {mid}")

pos_err = np.array(pos_err) * 1000.0       # mm
xcom_err = np.array(xcom_err) * 1000.0     # mm
ap_bias = np.array(ap_bias) * 1000.0       # mm (signed)
ml_bias = np.array(ml_bias) * 1000.0
def stats(a): return f"mean={a.mean():6.2f}  P95={np.percentile(a,95):6.2f}  max={a.max():6.2f}"
print(f"\n=== proxy(link-origin) vs TRUE(xipos) CoM — DDC, {len(pos_err)} single-support frames over {len(mots)} motions ===")
print(f"  3D CoM position error (mm):   {stats(pos_err)}")
print(f"  horizontal xCoM error (mm):   {stats(xcom_err)}")
print(f"  AP (fore/aft) signed bias (mm, true-proxy): mean={ap_bias.mean():+6.2f}  |mean|P95={np.percentile(np.abs(ap_bias),95):6.2f}")
print(f"  ML (lateral) signed bias  (mm, true-proxy): mean={ml_bias.mean():+6.2f}  |mean|P95={np.percentile(np.abs(ml_bias),95):6.2f}")
print(f"  ML |error|                (mm): mean={np.abs(ml_bias).mean():6.2f}  P95={np.percentile(np.abs(ml_bias),95):6.2f}")
print(f"  support-domain ML sign-inconsistency (proxy vs true in/out of the support rect): {100*np.mean(ml_sign_disagree):.2f}% of frames")
print(f"  [note] xCoM uses the TRUE-CoM height for omega on BOTH proxy and true (fixed height isolates the horizontal CoM position/velocity difference)")
print(f"\n  [sanity] max |log['com'] - offline_true_com| = {max_logcom_mismatch:.2e} m (should be ~0 -> metric change wired correctly)")
print(f"  [context] single-foot ML half-width = {MET.FOOT_ML[1]*1000:.2f} mm; AP span = [{MET.FOOT_AP[0]*1000:.0f},{MET.FOOT_AP[1]*1000:.0f}] mm")
