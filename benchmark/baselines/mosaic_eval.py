"""MOSAIC (gmt.onnx 770->29) evaluation: reuse the validated wbt_rollout (G1Sim / reference / term computation), assemble the 770 student obs.
770 = 5-frame history x 6 terms [command58, motion_anchor_ori_b6, base_ang_vel3, joint_pos29, joint_vel29, actions29]
Layout: [command x5, anchor x5, angvel x5, jpos x5, jvel x5, act x5] (blocked per term, old->new within block).
Usage: python mosaic_eval.py sanity [histflip] | run <motion_dir> <tag> <shard> <nsh> <outdir> [histflip]"""
import os, sys, json, glob, re, time, numpy as np, onnxruntime as ort, onnx
os.environ.setdefault("WBT_ORT_THREADS", "1")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
HS = _ROOT
os.environ["WBT_EVAL_ROBOT_XML"] = os.path.join(HS, "robot", "g1_29dof", "g1_29dof.xml")
sys.path.insert(0, os.path.join(HS, "eval"))
MOTION_DIR_ENV = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[1] in ("run", "runs", "fullruns")) else os.path.join(_ROOT, "data", "data_stratified_900", "test")
os.environ["WBT_EVAL_MOTION_DIR"] = MOTION_DIR_ENV
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fulllog as FL

MOSAIC = os.environ["MOSAIC_ONNX"]
META = os.environ.get("DDC_ROBOT_META_ONNX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_meta.onnx"))
md = {p.key: p.value for p in onnx.load(META, load_external_data=False).metadata_props}
dof_names = json.loads(md["dof_names"])
action_scale = np.asarray(json.loads(md["action_scale"]), dtype=np.float64).reshape(-1)
dja = json.loads(md["experiment_config"])["robot"]["init_state"]["default_joint_angles"]
default_dof = np.array([dja[n] for n in dof_names], dtype=np.float64)
KP = np.asarray(json.loads(md["kp"]), dtype=np.float64)
KD = np.asarray(json.loads(md["kd"]), dtype=np.float64)
sess = ort.InferenceSession(MOSAIC, providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name
# MOSAIC's own joint order (gmt.onnx metadata) + action_scale/default (MOSAIC order)
gmd = {p.key: p.value for p in onnx.load(MOSAIC, load_external_data=False).metadata_props}
_mj = gmd["joint_names"].split(",")
_minen = [n.replace("_joint", "") for n in dof_names]
IDX = np.array([_minen.index(n.replace("_joint", "")) for n in _mj])  # mosaic index -> my index
M_ASCALE = np.array([float(x) for x in gmd["action_scale"].split(",")], dtype=np.float64)  # MOSAIC order
M_DEFAULT = np.array([float(x) for x in gmd["default_joint_pos"].split(",")], dtype=np.float64)  # MOSAIC order

def obs6(sim, motion, last_action):
    # all joint-related terms in MOSAIC order ([IDX]); anchor/angvel are order-independent
    command = np.concatenate([motion["joint_pos"].reshape(-1)[IDX], motion["joint_vel"].reshape(-1)[IDX]])  # 58
    ref_quat = motion["ref_quat_xyzw"].reshape(-1)
    rel = W.subtract_frame_ori_xyzw(sim.torso_quat_xyzw(), ref_quat)
    anchor = W.quat_to_matrix_xyzw(rel)[:, :2].reshape(-1)  # 6
    jpos = (sim.dof_pos() - default_dof)[IDX]
    jvel = sim.dof_vel()[IDX]
    return [command, anchor, sim.base_ang_vel_base(), jpos, jvel, last_action.astype(np.float64)]  # last_action already in MOSAIC order

def assemble(hist, flip):
    # hist: deque of 5 frames, each = list of 6 arrays. layout: concatenate 5 frames per term
    frames = list(hist)
    if flip: frames = frames[::-1]
    blocks = []
    for term in range(6):
        blocks.append(np.concatenate([frames[f][term] for f in range(5)]))
    return np.concatenate(blocks).astype(np.float32)

def tilt_of(q_wxyz):
    w, x, y, z = q_wxyz
    return float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))

def rollout(mid, flip, dof_vel_noise=0.0, flog=None):
    from collections import deque
    ref = W.load_motion_npz(mid); mref = W.NpzMotionReference(ref["npz"]); T = ref["T"]
    sim = W.G1Sim(dof_names)
    sim.reset_to(ref["base_pos0"], ref["base_quat0"], ref["joint_pos0"]); sim.settle_freeze(n_steps=50)
    nom_h = float(ref["base_pos0"][2]); last_act = np.zeros(29, np.float64)
    hist = deque(maxlen=5)
    hs = []
    for t in range(T):
        motion = mref.infer(np.zeros(463, np.float32), t)
        f = obs6(sim, motion, last_act)
        if len(hist) == 0:
            for _ in range(5): hist.append(f)
        else:
            hist.append(f)
        obs = assemble(hist, flip)
        action = np.asarray(sess.run(None, {INAME: obs.reshape(1, -1)})[0]).reshape(-1)[:29].astype(np.float64)  # MOSAIC order
        last_act = action.copy()  # store in MOSAIC order (next-step actions term)
        qt_mosaic = action * M_ASCALE + M_DEFAULT       # target in MOSAIC order
        q_target = np.empty(29); q_target[IDX] = qt_mosaic  # map back to my order
        if flog is not None:
            import math
            qw = sim.base_quat_xyzw(); tl = float(np.arccos(np.clip(1 - 2 * (qw[0] ** 2 + qw[1] ** 2), -1, 1)))
            fll = flog.get("_fell", False) or (sim.data.qpos[2] < 0.5 * nom_h) or (tl > 1.0)
            flog["_fell"] = fll
            FL.log_step(flog, sim, t, action, tl, fll, W.CONTROL_HZ)
        sim.pd_step(q_target, KP, KD)
        lz = sim.foot_state(sim.lfoot)[2]; rz = sim.foot_state(sim.rfoot)[2]
        fzl = sim.foot_contact_fz(sim.lfoot); fzr = sim.foot_contact_fz(sim.rfoot)
        hs.append(np.concatenate([sim.data.qpos[:7], [lz, rz, fzl, fzr]]))
    return np.array(hs), nom_h, ref, sim

if sys.argv[1] == "sanity":
    flip = len(sys.argv) > 2 and sys.argv[2] == "flip"
    allm = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(MOTION_DIR_ENV + "/sample_*_mj.npz"))
    mids = allm[::80][:6]
    print(f"=== MOSAIC sanity (normal motions, hist{'reversed' if flip else 'old->new'}) ===")
    ok = 0
    for mid in mids:
        hs, nom_h, ref, sim = rollout(mid, flip)
        h = hs[:, 2]; tl = np.array([np.degrees(tilt_of(q[3:7])) for q in hs])
        fell = (h < 0.5 * nom_h).any() or (tl > 60).any()
        ff = int(np.argmax((h < 0.5 * nom_h) | (tl > 60))) if fell else len(h)
        print(f"  {mid}: {'fell@'+str(ff) if fell else 'completed'}/{len(h)} final_h={h[-1]:.3f} max_tilt={tl.max():.0f}deg")
        ok += (not fell)
    print(f"total: completed {ok}/{len(mids)} -> {'wiring likely correct' if ok>=len(mids)//2 else 'wiring/history-order/action possibly wrong'}")
    sys.exit(0)

if sys.argv[1] == "run":
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    of = f"{outdir}/{tag}__sh{shard}.json"
    if os.path.exists(of):
        print(f"[skip]{of}"); sys.exit(0)
    os.makedirs(outdir, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    my = mots[shard::nsh]; res = {}; t0 = time.time()
    for mid in my:
        try:
            hs, nom_h, ref, sim = rollout(mid, False)
        except Exception as e:
            res[mid] = {"err": str(e)[:120]}; print(f"[err]{mid}:{str(e)[:70]}"); continue
        heights = hs[:, 2]; tilts = np.array([tilt_of(q[3:7]) for q in hs])  # rad
        bad = (heights < 0.5 * nom_h) | (tilts > 1.0)
        fell = bool(bad.any()); ffi = int(np.argmax(bad)) if fell else len(hs)
        T = len(hs); upto = ffi if fell else T
        rob_fz = hs[:upto, 7:9]  # (upto,2) left/right foot z
        d = dict(np.load(f"{motion_dir}/sample_{mid}_mj.npz", allow_pickle=True))
        bz = d["body_pos_w"][:, :, 2]; bn = [str(x) for x in d["body_names"]]
        li = bn.index("left_ankle_roll_link"); ri = bn.index("right_ankle_roll_link")
        ref_fz = np.stack([bz[:, li], bz[:, ri]], 1)[:T]
        ground = 0.03
        ref_lift = float(ref_fz.max(0).max() - ground); swing = int(np.argmax(ref_fz.max(0)))
        rob_lift = float(rob_fz[:, swing].max() - ground) if len(rob_fz) else 0.0
        lr = rob_lift / ref_lift if ref_lift > 0.02 else float("nan")
        strict = bool((not fell) and (lr >= 0.5) and (np.degrees(tilts.max()) < 40.0))
        res[mid] = {"fell": fell, "fall_step": ffi, "T": T, "completion": ffi / T,
                    "final_h": float(heights[-1]), "max_tilt_deg": float(np.degrees(tilts.max())),
                    "ref_lift": ref_lift, "rob_lift": rob_lift, "lift_ratio": lr, "strict_success": strict}
    json.dump({"tag": tag, "harness": "mosaic_wbt", "per_motion": res}, open(of, "w"))
    n = len([v for v in res.values() if "fell" in v])
    fr = np.mean([v["fell"] for v in res.values() if "fell" in v]) if n else 0
    print(f"[done]{tag} sh{shard} {len(my)}mot fall_rate={fr:.1%} {time.time()-t0:.0f}s")
    sys.exit(0)

if sys.argv[1] in ("runs", "fullruns"):  # unified metrics.py success metric
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(outdir, exist_ok=True); of = f"{outdir}/{tag}__sh{shard}.json"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from success_metric import success_of
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        try:
            if sys.argv[1] == "fullruns":
                fl_ = FL.new_log(); hs, nom_h, ref, sim = rollout(mid, False, flog=fl_)
                res[mid] = FL.finalize(fl_, mid, nom_h, sim.total_mass, W.CONTROL_HZ); continue
            hs, nom_h, ref, sim = rollout(mid, False)
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}; continue
        base_z = hs[:, 2]; tilt = np.array([tilt_of(q[3:7]) for q in hs])
        res[mid] = success_of(hs[:, 9], hs[:, 10], base_z, tilt, nom_h, 1.0 / W.CONTROL_HZ, sim.total_mass, mid)
    json.dump({"tag": tag, "per_motion": res}, open(of, "w"))
    sv = [v["success"] for v in res.values() if "success" in v]
    print(f"[done]{tag} {len(sv)}mot metrics.py_success={np.mean(sv):.1%}")
    sys.exit(0)
