"""GEAR-SONIC (encoder 1762 + decoder 994) evaluation. Based on GR00T-WholeBodyControl's obs reconstruction recipe.
G1 mode: encoder fills only 4 terms (mode_4=0, motion_jpos/jvel 10 frames step5, motion_anchor_ori 10 frames 6D), rest all zero.
decoder = token_state(64) + his[ang_vel, jpos-def, jvel, lastact, grav] each 10 frames (old->new). Joint order = IsaacLab interleaved.
Usage: sanity | run <motion_dir> <tag> <shard> <nsh> <outdir>"""
import os, sys, json, glob, re, numpy as np, onnxruntime as ort, onnx
os.environ.setdefault("WBT_ORT_THREADS", "1")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # DDC release repo root
HS = _ROOT
os.environ["WBT_EVAL_ROBOT_XML"] = os.path.join(HS, "robot", "g1_29dof", "g1_29dof.xml")
if len(sys.argv) > 2 and sys.argv[1] in ("run", "runs", "fullruns"):
    os.environ["WBT_EVAL_MOTION_DIR"] = sys.argv[2]
else:
    os.environ["WBT_EVAL_MOTION_DIR"] = os.path.join(_ROOT, "data", "data_stratified_900", "test")
sys.path.insert(0, os.path.join(HS, "eval")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wbt_rollout as W
from success_metric import success_of
import fulllog as FL

SD = os.environ["SONIC_DIR"]
_so = ort.SessionOptions(); _so.intra_op_num_threads = 1; _so.inter_op_num_threads = 1
enc = ort.InferenceSession(f"{SD}/model_encoder.onnx", sess_options=_so, providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(f"{SD}/model_decoder.onnx", sess_options=_so, providers=["CPUExecutionProvider"])
ENC_IN = enc.get_inputs()[0].name; DEC_IN = dec.get_inputs()[0].name
# joint order (hpp): isaaclab_to_mujoco=[0,6,12,...] (obs: gather with it for mujoco->isaaclab); mujoco_to_isaaclab=[0,3,6,...] (action: gather with it for isaaclab->mujoco)
IL2MJ = np.array([0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28])  # for to_isaac
MJ2IL = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28])  # for action->mujoco
I2M = IL2MJ; M2I = MJ2IL
DEFAULT = np.array([-0.312,0,0,0.669,-0.363,0, -0.312,0,0,0.669,-0.363,0, 0,0,0, 0.2,0.2,0,0.6,0,0,0, 0.2,-0.2,0,0.6,0,0,0], np.float64)  # mujoco order
META = os.environ.get("DDC_ROBOT_META_ONNX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_meta.onnx"))
_md = {p.key: p.value for p in onnx.load(META, load_external_data=False).metadata_props}
ASCALE = np.asarray(json.loads(_md["action_scale"]), np.float64).reshape(-1)  # mujoco order (same as G1)
KP = np.asarray(json.loads(_md["kp"]), np.float64); KD = np.asarray(json.loads(_md["kd"]), np.float64)
dof_names = json.loads(_md["dof_names"])

def to_isaac(v): return v[I2M]   # mujoco order -> IsaacLab order
def tilt_of(qw):
    w, x, y, z = qw; return float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))

def enc_obs(ref, t):
    T = ref["T"]; c = lambda i: min(max(i, 0), T - 1)
    o = np.zeros(1762, np.float32)
    fidx = [c(t + 5 * k) for k in range(10)]  # 10 frames step5
    jpos = np.concatenate([to_isaac(ref["dof"][i]) for i in fidx])       # 290
    jvel = np.concatenate([to_isaac(ref["dofv"][i]) for i in fidx])      # 290
    an = []
    rq = W  # anchor: reference pelvis quaternion relative to robot pelvis (6D)
    for i in fidx:
        rel = W.subtract_frame_ori_xyzw(ref["cur_base_xyzw"], ref["pquat_xyzw"][i])
        an.append(W.quat_to_matrix_xyzw(rel)[:, :2].reshape(-1))
    anchor = np.concatenate(an)  # 60
    o[0:4] = 0.0; o[4:294] = jpos; o[294:584] = jvel; o[601:661] = anchor
    return o.reshape(1, -1)

def rollout(mid, flog=None):
    d = dict(np.load(f"{os.environ['WBT_EVAL_MOTION_DIR']}/sample_{mid}_mj.npz", allow_pickle=True))
    jp = d["joint_pos"]; jv = d["joint_vel"]; T = len(jp)
    bn = [str(x) for x in d["body_names"]]; pj = bn.index("pelvis")
    ref = {"T": T, "dof": jp[:, 7:].astype(np.float64), "dofv": jv[:, 6:].astype(np.float64),
           "pquat_xyzw": d["body_quat_w"][:, pj][:, [1, 2, 3, 0]].astype(np.float64)}  # wxyz->xyzw
    r0 = W.load_motion_npz(mid); sim = W.G1Sim(dof_names)
    sim.reset_to(r0["base_pos0"], r0["base_quat0"], r0["joint_pos0"]); sim.settle_freeze(n_steps=50)
    nom_h = float(r0["base_pos0"][2])
    from collections import deque
    hist = deque(maxlen=10); last_act = np.zeros(29)  # IsaacLab order
    hs = []; lf = sim.lfoot; rf = sim.rfoot; c = lambda i: min(max(i, 0), T - 1)
    for t in range(T):
        ref["cur_base_xyzw"] = sim.base_quat_xyzw()
        # encoder
        tok = np.asarray(enc.run(None, {ENC_IN: enc_obs(ref, t)})[0]).reshape(-1)[:64]
        # proprioception (IsaacLab order)
        ang = sim.base_ang_vel_base(); grav = sim.projected_gravity_base()
        jpos_r = to_isaac(sim.dof_pos() - DEFAULT); jvel_r = to_isaac(sim.dof_vel())
        frame = np.concatenate([ang, jpos_r, jvel_r, last_act, grav])  # 3+29+29+29+3=93
        if len(hist) == 0:
            for _ in range(10): hist.append(frame)
        else: hist.append(frame)
        H = list(hist)  # old->new
        his_ang = np.concatenate([H[k][0:3] for k in range(10)])
        his_jp = np.concatenate([H[k][3:32] for k in range(10)])
        his_jv = np.concatenate([H[k][32:61] for k in range(10)])
        his_la = np.concatenate([H[k][61:90] for k in range(10)])
        his_gr = np.concatenate([H[k][90:93] for k in range(10)])
        dobs = np.concatenate([tok, his_ang, his_jp, his_jv, his_la, his_gr]).astype(np.float32)  # 994
        act_i = np.asarray(dec.run(None, {DEC_IN: dobs.reshape(1, -1)})[0]).reshape(-1)[:29]  # IsaacLab order
        last_act = act_i.copy()
        act_m = act_i[M2I]  # -> mujoco order
        q_target = act_m.astype(np.float64) * ASCALE + DEFAULT
        if flog is not None:
            tl = tilt_of(sim.base_quat_xyzw()[[3, 0, 1, 2]])
            fll = flog.get("_fell", False) or (sim.data.qpos[2] < 0.5 * nom_h) or (tl > 1.0)
            flog["_fell"] = fll
            FL.log_step(flog, sim, t, act_m.astype(np.float64), tl, fll, W.CONTROL_HZ)
        sim.pd_step(q_target, KP, KD)
        hs.append(np.concatenate([sim.data.qpos[:7], [sim.foot_contact_fz(lf), sim.foot_contact_fz(rf)]]))
    return np.array(hs), nom_h, sim

if sys.argv[1] == "sanity":
    MD = os.environ["WBT_EVAL_MOTION_DIR"]
    allm = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(MD + "/sample_*_mj.npz"))
    mids = allm[::80][:6]; ok = 0
    print("=== SONIC sanity (normal motions) ===")
    for mid in mids:
        hs, nom_h, sim = rollout(mid)
        h = hs[:, 2]; tl = np.array([np.degrees(tilt_of(q[3:7])) for q in hs])
        fell = (h < 0.5 * nom_h).any() or (tl > 60).any(); ff = int(np.argmax((h < 0.5 * nom_h) | (tl > 60))) if fell else len(h)
        print(f"  {mid}: {'fell@'+str(ff) if fell else 'completed'}/{len(h)} final_h={h[-1]:.3f} max_tilt={tl.max():.0f}deg"); ok += (not fell)
    print(f"total: completed {ok}/{len(mids)} -> {'wiring likely correct' if ok>=len(mids)//2 else 'wiring wrong (check concat order/joint order/anchor)'}")
    sys.exit(0)

if sys.argv[1] in ("runs", "fullruns"):
    motion_dir, tag, shard, nsh, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(outdir, exist_ok=True)
    mots = sorted(re.sub(r"^sample_|_mj\.npz$", "", os.path.basename(p)) for p in glob.glob(motion_dir + "/sample_*_mj.npz"))
    myl = mots[shard::nsh]; res = {}
    for mid in myl:
        try:
            if sys.argv[1] == "fullruns":
                fl_ = FL.new_log(); hs, nom_h, sim = rollout(mid, flog=fl_)
                res[mid] = FL.finalize(fl_, mid, nom_h, sim.total_mass, W.CONTROL_HZ); continue
            hs, nom_h, sim = rollout(mid)
        except Exception as e:
            res[mid] = {"err": str(e)[:100]}; continue
        base_z = hs[:, 2]; tilt = np.array([tilt_of(q[3:7]) for q in hs])
        res[mid] = success_of(hs[:, 7], hs[:, 8], base_z, tilt, nom_h, 1.0 / W.CONTROL_HZ, sim.total_mass, mid)
    json.dump({"tag": tag, "per_motion": res}, open(f"{outdir}/{tag}__sh{shard}.json", "w"))
    sv = [v["success"] for v in res.values() if "success" in v]; fl = [v["fell"] for v in res.values() if "fell" in v]
    print(f"[done]{tag} n={len(sv)} full_success={np.mean(sv):.1%} fell={np.mean(fl):.1%}")
    sys.exit(0)
