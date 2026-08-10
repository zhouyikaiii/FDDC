"""TWIST (twist_general_motion_tracker.pt, TorchScript) single-leg-balance evaluation.
obs=1155=obs_full(105=mimic31+proprio74)+history10x105. 23-action/25-dof model. Usage as before:
  python twist_eval.py sanity | run <motion_dir> <tag> <shard> <nsh> <outdir>"""
import os, sys, json, glob, re, time
import numpy as np, torch, mujoco
torch.set_num_threads(1)
os.environ.setdefault("MUJOCO_GL", "egl")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
REPO = os.environ["TWIST_REPO"]
XML = f"{REPO}/assets/g1/g1_sim2sim_with_wrist_roll.xml"
PT = os.environ["TWIST_WEIGHTS"]
HS = _ROOT
META = os.environ.get("DDC_ROBOT_META_ONNX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_meta.onnx"))
os.environ.setdefault("WBT_EVAL_ROBOT_XML", os.path.join(HS, "robot", "g1_29dof", "g1_29dof.xml"))
if len(sys.argv) > 2 and sys.argv[1] in ("runs", "fullruns"):
    os.environ["WBT_EVAL_MOTION_DIR"] = sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HS, "eval"))
from success_metric import foot_fz as FZ, success_of
import fulllog_native as FN

# 25-dof model order; 23-body = exclude wrist_roll [19,24]
TW25 = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
        "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
        "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
        "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint",
        "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint"]
BODY_IDS = [i for i in range(25) if i not in (19, 24)]  # 23
TW23_NAMES = [TW25[i] for i in BODY_IDS]
DEFAULT_DOF = np.array([-0.2,0,0,0.4,-0.2,0, -0.2,0,0,0.4,-0.2,0, 0,0,0, 0,0.4,0,1.2, 0,-0.4,0,1.2], np.float64)  # 23
KP = np.array([100,100,100,150,40,40, 100,100,100,150,40,40, 150,150,150, 40,40,40,40,20, 40,40,40,40,20], np.float64)
KD = np.array([2,2,2,4,2,2, 2,2,2,4,2,2, 4,4,4, 5,5,5,5,1, 5,5,5,5,1], np.float64)
TLIM = np.array([88,139,88,139,50,50, 88,139,88,139,50,50, 88,50,50, 25,25,25,25,25, 25,25,25,25,25], np.float64)
ANKLE_IDX = [4, 5, 10, 11]  # 23-frame
ACTION_SCALE = 0.5

# the 29-dof order (= g1.xml/URDF order)
_MINE29 = ["left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
           "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
           "waist_yaw","waist_roll","waist_pitch",
           "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow","left_wrist_roll","left_wrist_pitch","left_wrist_yaw",
           "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow","right_wrist_roll","right_wrist_pitch","right_wrist_yaw"]
SEL23 = [_MINE29.index(n.replace("_joint", "")) for n in TW23_NAMES]  # indices selecting 23 out of my 29

policy = torch.jit.load(PT, map_location="cpu"); policy.eval()

def euler_xyzw(q):  # q=[x,y,z,w]
    x, y, z, w = q
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p = 2*(w*y-z*x); pitch = np.arcsin(np.clip(p, -1, 1))
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw

def quatToEuler_wxyz(q):  # q=[w,x,y,z]
    w, x, y, z = q
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p = 2*(w*y-z*x); pitch = np.copysign(np.pi/2, p) if abs(p) >= 1 else np.arcsin(p)
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.array([roll, pitch, yaw])

def qrot_inv_xyzw(q, v):  # rotate v into body frame, q=xyzw
    x, y, z, w = q; qv = np.array([x, y, z])
    a = v*(2*w*w-1); b = np.cross(qv, v)*2*w; c = qv*(qv@v)*2
    return a - b + c

def smooth(x, k=19):
    if len(x) < 3: return x
    ker = np.ones(k)/k; out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], ker, mode="same")
    return out

def load_ref(npz):
    d = dict(np.load(npz, allow_pickle=True))
    jp = d["joint_pos"]; fps = float(np.array(d["fps"]).reshape(-1)[0]); T = len(jp)
    root_pos = jp[:, 0:3].astype(np.float64)
    q_wxyz = jp[:, 3:7]; root_rot = q_wxyz[:, [1, 2, 3, 0]].astype(np.float64)  # xyzw
    dof23 = jp[:, 7:][:, SEL23].astype(np.float64)  # (T,23)
    rv = np.zeros_like(root_pos); rv[:-1] = fps*(root_pos[1:]-root_pos[:-1]); rv[-1] = rv[-2]; rv = smooth(rv)
    # root_ang_vel via quat diff -> exp map (approx: rotation vector of adjacent quaternions)
    rav = np.zeros_like(root_pos)
    for t in range(T-1):
        q0, q1 = root_rot[t], root_rot[t+1]
        dq = quat_mul(quat_conj(q0), q1)
        rav[t] = quat_to_expmap(dq)*fps
    rav[-1] = rav[-2]; rav = smooth(rav)
    return dict(root_pos=root_pos, root_rot=root_rot, dof23=dof23, rv=rv, rav=rav, T=T, d=d)

def quat_conj(q): return np.array([-q[0], -q[1], -q[2], q[3]])
def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz])
def quat_to_expmap(q):  # xyzw -> rotation vector
    q = q if q[3] >= 0 else -q
    w = np.clip(q[3], -1, 1); ang = 2*np.arccos(w); s = np.sqrt(1-w*w)
    if s < 1e-6: return np.zeros(3)
    return (ang/s)*q[:3]

def make_qpos_full(ref, t):
    q = np.zeros(32)
    q[0:3] = ref["root_pos"][t]; rr = ref["root_rot"][t]; q[3:7] = [rr[3], rr[0], rr[1], rr[2]]  # wxyz
    dof25 = np.zeros(25);
    for i, bi in enumerate(BODY_IDS): dof25[bi] = ref["dof23"][t, i]
    q[7:7+25] = dof25
    return q

def rollout(npz, flog=None):
    ref = load_ref(npz); T = ref["T"]
    m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = 0.001; d = mujoco.MjData(m)
    d.qpos[:] = make_qpos_full(ref, 0); mujoco.mj_forward(m, d)
    if flog is not None: flog["_half_nom"] = 0.5 * float(ref["root_pos"][0, 2]); _pc = None
    sid_o = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
    sid_a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    adr_o = m.sensor_adr[sid_o]; adr_a = m.sensor_adr[sid_a]
    lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    from collections import deque
    hist = deque([np.zeros(105) for _ in range(10)], maxlen=10)
    last_action = np.zeros(23); hs = []
    c = lambda i: min(max(i, 0), T-1)
    for ts in range(T):
        tf = c(ts+1)
        rr = ref["root_rot"][tf]
        roll, pitch, yaw = euler_xyzw(rr)
        rv_l = qrot_inv_xyzw(rr, ref["rv"][tf]); av_l = qrot_inv_xyzw(rr, ref["rav"][tf])
        mimic = np.concatenate([[ref["root_pos"][tf, 2]], [roll, pitch, yaw], rv_l, [av_l[2]], ref["dof23"][tf]])  # 31
        quat = d.sensordata[adr_o:adr_o+4].copy()  # wxyz
        ang_vel = d.sensordata[adr_a:adr_a+3].copy()
        rpy = quatToEuler_wxyz(quat)
        bdp = d.qpos[7:][BODY_IDS]; bdv = d.qvel[6:][BODY_IDS].copy()
        bdv_obs = bdv.copy(); bdv_obs[ANKLE_IDX] = 0.0
        proprio = np.concatenate([ang_vel*0.25, rpy[:2], bdp-DEFAULT_DOF, bdv_obs*0.05, last_action])  # 74
        obs_full = np.concatenate([mimic, proprio])  # 105
        obs_buf = np.concatenate([obs_full, np.array(hist).flatten()]).astype(np.float32)  # 1155
        with torch.no_grad():
            raw = policy(torch.from_numpy(obs_buf).unsqueeze(0)).cpu().numpy().reshape(-1)[:23]
        last_action = raw.copy()
        a = np.clip(raw, -10, 10)*ACTION_SCALE
        pdt_body = a.astype(np.float64) + DEFAULT_DOF
        pdt25 = np.zeros(25)
        for i, bi in enumerate(BODY_IDS): pdt25[bi] = pdt_body[i]
        hist.append(obs_full)
        for _ in range(20):
            tau = (pdt25 - d.qpos[7:])*KP - d.qvel[6:]*KD
            tau = np.clip(tau, -TLIM, TLIM); d.ctrl[:] = tau; mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)  # sync xpos/contacts to the just-integrated qpos (mj_step leaves derived qtys one substep stale — same fix as G1Sim.pd_step)
        fzl = FZ(m, d, lf); fzr = FZ(m, d, rf)
        hs.append(np.concatenate([d.qpos[:7], [float(d.xpos[lf, 2]), float(d.xpos[rf, 2]), fzl, fzr]]))
        if flog is not None:
            _pc = FN.log_step(flog, m, d, lf, rf, ts, raw, 50.0, _pc, fz_l=fzl, fz_r=fzr)
    return np.array(hs), ref, m

def tilt_of(qw):
    w, x, y, z = qw; return float(np.arccos(np.clip(1-2*(x*x+y*y), -1, 1)))

if sys.argv[1] == "sanity":
    MD = os.path.join(_ROOT, "data", "data_stratified_900", "test")
    allm = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(MD+"/sample_*_mj.npz"))
    mids = allm[::80][:6]; ok = 0
    print("=== TWIST sanity (normal motions) ===")
    for mid in mids:
        hs, ref, m = rollout(f"{MD}/sample_{mid}_mj.npz")
        h = hs[:, 2]; tl = np.array([np.degrees(tilt_of(q[3:7])) for q in hs])
        nom = float(ref["root_pos"][0, 2]); fell = (h < 0.5*nom).any() or (tl > 60).any()
        ff = int(np.argmax((h < 0.5*nom) | (tl > 60))) if fell else len(h)
        print(f"  {mid}: {'fell@'+str(ff) if fell else 'completed'}/{len(h)} final_h={h[-1]:.3f} max_tilt={tl.max():.0f}deg")
        ok += (not fell)
    print(f"total: completed {ok}/{len(mids)} -> {'wiring likely correct' if ok>=len(mids)//2 else 'wiring possibly wrong'}")
    sys.exit(0)

if sys.argv[1] == "run":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    of = f"{outdir}/{tag}__sh{shard}.json"
    if os.path.exists(of): print(f"[skip]{of}"); sys.exit(0)
    os.makedirs(outdir, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir+"/sample_*_mj.npz"))
    my = mots[shard::nsh]; res = {}; t0 = time.time()
    for mid in my:
        try:
            hs, ref, m = rollout(f"{motion_dir}/sample_{mid}_mj.npz")
        except Exception as e:
            res[mid] = {"err": str(e)[:120]}; print(f"[err]{mid}:{str(e)[:70]}"); continue
        h = hs[:, 2]; tl = np.array([tilt_of(q[3:7]) for q in hs]); nom = float(ref["root_pos"][0, 2])
        bad = (h < 0.5*nom) | (tl > 1.0); fell = bool(bad.any()); ffi = int(np.argmax(bad)) if fell else len(hs)
        T = len(hs); upto = ffi if fell else T; rob_fz = hs[:upto, 7:9]
        dd = ref["d"]; bz = dd["body_pos_w"][:, :, 2]; bn = [str(x) for x in dd["body_names"]]
        li = bn.index("left_ankle_roll_link"); ri = bn.index("right_ankle_roll_link")
        ref_fz = np.stack([bz[:, li], bz[:, ri]], 1)[:T]; ground = 0.03
        ref_lift = float(ref_fz.max(0).max()-ground); swing = int(np.argmax(ref_fz.max(0)))
        rob_lift = float(rob_fz[:, swing].max()-ground) if len(rob_fz) else 0.0
        lr = rob_lift/ref_lift if ref_lift > 0.02 else float("nan")
        strict = bool((not fell) and (lr >= 0.5) and (np.degrees(tl.max()) < 40.0))
        res[mid] = {"fell": fell, "fall_step": ffi, "T": T, "completion": ffi/T, "final_h": float(h[-1]),
                    "max_tilt_deg": float(np.degrees(tl.max())), "ref_lift": ref_lift, "rob_lift": rob_lift,
                    "lift_ratio": lr, "strict_success": strict}
    json.dump({"tag": tag, "harness": "twist_native", "per_motion": res}, open(of, "w"))
    n = len([v for v in res.values() if "fell" in v]); fr = np.mean([v["fell"] for v in res.values() if "fell" in v]) if n else 0
    print(f"[done]{tag} sh{shard} {len(my)}mot fall_rate={fr:.1%} {time.time()-t0:.0f}s")
    sys.exit(0)

if sys.argv[1] in ("runs", "fullruns"):
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(outdir, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        try:
            if sys.argv[1] == "fullruns":
                fl_ = FN.new_log(); hs, ref, m = rollout(f"{motion_dir}/sample_{mid}_mj.npz", flog=fl_)
                res[mid] = FN.finalize(fl_, mid, float(ref["root_pos"][0, 2]), float(np.sum(m.body_mass)), 50.0); continue
            hs, ref, m = rollout(f"{motion_dir}/sample_{mid}_mj.npz")
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}; continue
        base_z = hs[:, 2]; tilt = np.array([tilt_of(q[3:7]) for q in hs]); nom = float(ref["root_pos"][0, 2])
        mass = float(np.sum(m.body_mass))
        res[mid] = success_of(hs[:, 9], hs[:, 10], base_z, tilt, nom, 0.02, mass, mid)
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot metrics.py_success={np.mean(sv):.1%}")
    sys.exit(0)
