"""Self-contained MuJoCo rollout of a WBT ONNX policy for the single-leg balance benchmark.

Runs one exported ONNX (policy + baked motion) through its motion in MuJoCo, reconstructing the
exact 463-dim actor observation, applying PD control, and logging the standard signals the metric
library (metrics.py) consumes. No dependency on the training/deploy runtime (obs re-implemented to
match `managers/observation/terms/wbt.py`; motion parts read from the ONNX's own reference outputs,
which are a pure time_step lookup independent of the obs input).

Only imports numpy / mujoco / onnxruntime. Does not modify anything outside evaluation/.
"""
from __future__ import annotations

import json
import os
import re

import mujoco
import numpy as np
# onnxruntime is imported lazily inside WBTPolicy (the legacy per-motion-ONNX path). The benchmark
# entry point run_eval.py scores model_0262000.pt via FastPolicy and needs only numpy / mujoco / torch.

# The deploy sim2sim MuJoCo model. 7 capsule collision geoms per foot paired ONLY with the floor
# (solref="0.01 1"), proper armature + per-joint effort limits, and default contype/conaffinity=0 (no
# self-collision -> only the explicit foot<->floor pairs generate contact). This is the exact plant the
# policy was validated against; the retargeting `g1_29dof_spherehand.xml` is a DIFFERENT model (sphere
# feet, no armature) on which even a good policy falls. The XML references a geom named "floor" that it
# does NOT define (the deploy sim scene injects it), so we patch in a floor plane + an absolute meshdir.
# ROBOT_XML / MOTION_DIR are the only machine-specific paths. Override via the environment so this
# folder is portable without editing code. Defaults point at the bundled robot model + test motions
# (this file lives in <pkg>/eval/); override via the environment to score other motion sets:
#   export WBT_EVAL_ROBOT_XML=/path/to/robot/g1_29dof/g1_29dof.xml   # its dir must contain meshes/
#   export WBT_EVAL_MOTION_DIR=/path/to/data_stratified_900/test     # holds sample_<motion>_mj.npz
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # package root (parent of eval/)
ROBOT_XML = os.environ.get(
    "WBT_EVAL_ROBOT_XML", os.path.join(_PKG, "robot", "g1_29dof", "g1_29dof.xml"))
MOTION_DIR = os.environ.get(
    "WBT_EVAL_MOTION_DIR", os.path.join(_PKG, "data", "data_stratified_900", "test"))
CONTROL_HZ = 50.0
SIM_DT = 0.0005  # 2000 Hz physics (Euler integrator), matching the deploy runtime; the 1-physics-step
                 # dof_vel delay is therefore 0.5 ms.
SUBSTEPS = 40    # 40 physics steps / control step -> policy at 50 Hz (2000/40); PD recomputed each physics
                 # step like the deploy (compute_torques runs every physics step). dof_vel_delay=1 = 1 physics
                 # step = 0.5 ms (deploy SIM2SIM_DOF_VEL_DELAY_STEPS=1 @ 2000 Hz).

# Deploy observation/command details, replicated for 1:1 consistency with the deployed controller:
WBT_WAIST_LPF_ALPHA = 0.5   # EMA on the 3 waist joints before the torso-orientation FK
OBS_CLIP = 100.0            # raw-obs clip before the ONNX (deploy clip_observations=100)
ACTION_CLIP = 100.0        # policy-action clip before scaling (deploy)
WBT_ENABLE_TARGET_CLAMP = False   # script-layer q_target joint-limit clamp; OFF to match the deploy
                                  # (which does not script-clamp; physics range + torque saturation bound it)

# ---------------------------------------------------------------------------------------------------
# quaternion helpers (numpy, w_last=xyzw unless noted). Mirror holosoma.utils.rotations semantics.
# ---------------------------------------------------------------------------------------------------
def quat_rotate_inverse_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by inverse of q (q in xyzw). q:(4,) v:(3,) -> (3,)."""
    q = q_xyzw
    qvec = q[:3]
    w = q[3]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qvec, v) * 2.0 * w
    c = qvec * (qvec @ v) * 2.0
    return a - b + c


def wxyz_to_xyzw(q):
    q = np.asarray(q).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_mul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conj_xyzw(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_to_matrix_xyzw(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def subtract_frame_ori_xyzw(q_a_xyzw, q_b_xyzw):
    """orientation of b expressed in a's frame = a^-1 * b (all xyzw)."""
    return quat_mul_xyzw(quat_conj_xyzw(q_a_xyzw), q_b_xyzw)


# ---------------------------------------------------------------------------------------------------
class WBTPolicy:
    """Wraps an exported WBT ONNX: runs inference and exposes the baked motion reference by time_step."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort  # lazy: only this legacy per-motion-ONNX path needs onnxruntime
        _so = ort.SessionOptions()
        _nt = int(os.environ.get("WBT_ORT_THREADS", "0"))  # cap ORT threads for parallel sweeps (OMP vars
        if _nt > 0:                                        # do NOT control onnxruntime's own thread pool)
            _so.intra_op_num_threads = _nt
            _so.inter_op_num_threads = _nt
        self.sess = ort.InferenceSession(onnx_path, sess_options=_so, providers=["CPUExecutionProvider"])
        md = {p.key: p.value for p in _load_meta(onnx_path)}
        self.dof_names = json.loads(md["dof_names"])
        self.num_dofs = len(self.dof_names)
        self.kp = np.asarray(json.loads(md["kp"]), dtype=np.float64)
        self.kd = np.asarray(json.loads(md["kd"]), dtype=np.float64)
        self.action_scale = _parse_scale(md["action_scale"], self.num_dofs)
        ec = json.loads(md["experiment_config"])
        dja = ec["robot"]["init_state"]["default_joint_angles"]
        self.default_dof = np.array([dja[n] for n in self.dof_names], dtype=np.float64)
        self.out_names = [o.name for o in self.sess.get_outputs()]
        # deploy joint-position safety clamp limits (present on ONNX exported after the clamp was added;
        # None on older ONNX -> clamp skipped, exactly like the deploy). See base.py q_target clamp.
        self.dof_pos_lower = _parse_limit(md.get("dof_pos_lower"))
        self.dof_pos_upper = _parse_limit(md.get("dof_pos_upper"))

    def infer(self, obs_463: np.ndarray, time_step: int):
        feed = {
            "obs": obs_463.reshape(1, -1).astype(np.float32),
            "time_step": np.array([[time_step]], dtype=np.float32),
        }
        outs = self.sess.run(None, feed)
        d = dict(zip(self.out_names, outs))
        return d  # keys: actions, joint_pos, joint_vel, ref_pos_xyz, ref_quat_xyzw, reference_support_phase, future_support_phase, future_cmd


def _load_meta(onnx_path):
    import onnx

    return onnx.load(onnx_path, load_external_data=False).metadata_props


def _parse_scale(raw, n):
    v = np.asarray(json.loads(raw), dtype=np.float64).reshape(-1)
    return np.full(n, v.item()) if v.size == 1 else v


def _parse_limit(raw):
    """dof_pos_lower/upper metadata (JSON list) -> (num_dofs,) array, else None (mirrors deploy wbt._parse_limit_metadata)."""
    if raw is None:
        return None
    try:
        v = np.asarray(json.loads(raw), dtype=np.float64).reshape(-1)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return v if v.size else None


def _load_deploy_model() -> "mujoco.MjModel":
    """Load the deploy robot XML and rebuild the ground contact EXACTLY like the deploy run_sim scene
    (holosoma scene_manager.py). In the deploy, the robot's own floor is stripped and the foot<->floor
    <pair> definitions break under the MjSpec attach-prefix, so the foot-floor contact falls back to
    geom<->geom MIXING between the foot capsules and a ground plane carrying the scene_manager params
    (solref=[0.001,1], friction=[0.7,0.005,0.001], solimp=[0.99,0.99,0.01,0.5,2]). MuJoCo averages the
    plane's solref (0.001) with the foot geoms' default (0.02) -> effective foot-floor solref = 0.0105,
    friction = max(1.0, 0.7) = 1.0 (matching the deploy's compiled contact). So we:
      (a) DROP the XML's explicit foot<->floor <pair> block (14 pairs, no <exclude>s) -- these carry
          Holosoma's solref="0.01 1" friction="0.8 0.8" and break under the deploy MjSpec attach-prefix
          (above), so keeping them would not reproduce the deploy's geom-mix (0.0105 / 1.0); and
      (b) inject the ground plane with the scene_manager params (not the old friction="1 1 1"/default 0.02).
    Patch as text -> from_xml_string so nothing outside evaluation/ is written and meshes resolve by CWD."""
    meshes = os.path.dirname(ROBOT_XML) + "/meshes/"
    txt = open(ROBOT_XML).read().replace('meshdir="./meshes/"', f'meshdir="{meshes}"')
    txt = re.sub(r"<contact>.*?</contact>", "", txt, flags=re.DOTALL)  # drop foot<->floor pairs (deploy geom-mixes)
    floor = ('<geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" contype="1" conaffinity="1" '
             'friction="0.7 0.005 0.001" solref="0.001 1" solimp="0.99 0.99 0.01 0.5 2"/>\n</worldbody>')
    txt = txt.replace("</worldbody>", floor, 1)
    return mujoco.MjModel.from_xml_string(txt)


_MODEL_MASS_BY_NAME = None


def model_body_masses_by_name() -> dict:
    """{body_name: mass} for the deploy model (cached). Lets the reference-motion CoM (indexed by the
    npz body_names order) use the SAME per-link masses as the robot CoM, so the §4.5(a) deviation
    compares like with like."""
    global _MODEL_MASS_BY_NAME
    if _MODEL_MASS_BY_NAME is None:
        m = _load_deploy_model()
        _MODEL_MASS_BY_NAME = {
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"): float(m.body_mass[i])
            for i in range(m.nbody)
        }
    return _MODEL_MASS_BY_NAME


# ---------------------------------------------------------------------------------------------------
class G1Sim:
    """MuJoCo G1 sim (deploy plant) with helpers for base/foot/CoM state and torque control."""

    def __init__(self, dof_names):
        self.model = _load_deploy_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        m.opt.timestep = SIM_DT  # 2000 Hz (matches deploy run_sim fps=2000); integrator = XML/MuJoCo default (Euler)
        # Foot ground-reaction / alignment geoms: the 7 capsule collision geoms per foot (names contain
        # "foot" + "collision"). Contact itself is defined by the XML's explicit foot<->floor <pair>s;
        # every other geom defaults to contype/conaffinity=0 so there is no self-collision to disable.
        self.foot_gids = [i for i in range(m.ngeom)
                          if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                          and "collision" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
        # joint qpos/qvel indices for the 29 policy dofs (skip the free joint = first 7 qpos / 6 qvel)
        self.jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in dof_names]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.qvel_adr = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.pelvis = self.bid("pelvis")
        self.torso = self.bid("torso_link")
        self.lfoot = self.bid("left_ankle_roll_link")
        self.rfoot = self.bid("right_ankle_roll_link")
        # HuB 12 tracking keypoints (hips, knees, ankles, shoulders, elbows, wrists L/R) for track error
        self.kp12_ids = [self.bid(n) for n in (
            "left_hip_pitch_link", "right_hip_pitch_link", "left_knee_link", "right_knee_link",
            "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_pitch_link",
            "right_shoulder_pitch_link", "left_elbow_link", "right_elbow_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link")]
        # per-joint effort limits from the XML actuatorfrcrange (88/139/50/25/5 N·m; fallback large)
        self.effort = np.array([
            m.jnt_actfrcrange[j][1] if m.jnt_actfrclimited[j] else 200.0 for j in self.jid
        ], dtype=np.float64)
        # body masses (for mass-weighted CoM). exclude world (id 0)
        self.body_mass = m.body_mass.copy()
        self.total_mass = self.body_mass[1:].sum()
        self.dt = m.opt.timestep
        self.substeps = SUBSTEPS
        # PHYSICS-step dof_vel history (filled by pd_step). The deploy bridge publishes the LowState every
        # PHYSICS step, so its DOF_VEL_DELAY_STEPS counts physics steps (1 step = 1/fps s = 0.5 ms @ 2000 Hz),
        # NOT control steps (20 ms). The dof_vel-obs delay must be applied at this substep rate to match.
        self._vel_hist_sub: list = []
        self._v6 = np.zeros(6)  # scratch for mj_objectVelocity
        # for the deploy waist-LPF on the torso-orientation obs: recompute torso quat from LPF'd waist
        # angles via FK on a scratch MjData (main sim untouched).
        self._scratch = mujoco.MjData(m)
        _waist = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        self.waist_qpos_adr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _waist])
        self.waist_dof_idx = [dof_names.index(n) for n in _waist]

    def reset_to(self, base_pos, base_quat_wxyz, joint_pos):
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[0:3] = base_pos
        d.qpos[3:7] = base_quat_wxyz  # mujoco free joint quat is wxyz
        d.qpos[self.qpos_adr] = joint_pos
        mujoco.mj_forward(self.model, d)
        # Shift base_z so the lowest foot-sphere rests 1 mm above the ground: the motion's ground
        # reference differs from the mujoco floor (feet start ~3 cm off), so a naive init drops/bounces.
        lowest = min(d.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.foot_gids)
        d.qpos[2] -= lowest - 0.001
        mujoco.mj_forward(self.model, d)

    def pd_step(self, q_target, kp, kd):
        """Hold q_target for one control period; recompute the PD torque EACH substep (stiff-PD stability).
        Records the post-substep dof_vel into `_vel_hist_sub` (physics-step rate) so the dof_vel-obs delay
        matches the deploy bridge, which publishes -> caches dof_vel every physics step, not every control step."""
        for _ in range(self.substeps):
            q = self.dof_pos()
            qd = self.dof_vel()
            tau = np.clip(kp * (q_target - q) - kd * qd, -self.effort, self.effort)
            self.apply_torque(tau)
            self.step()
            self._vel_hist_sub.append(self.dof_vel())
        if len(self._vel_hist_sub) > self.substeps + 2:
            self._vel_hist_sub = self._vel_hist_sub[-(self.substeps + 2):]
        # mj_step computes derived quantities (xpos/xipos/xquat/contacts) at the pre-integration qpos and
        # THEN integrates, so on return those lag qpos/qvel by one substep. Re-sync them so the next
        # build_obs + log read ONE consistent physical state (mirrors the settle_freeze mj_forward). This
        # only recomputes derived qtys — it does not integrate, so the physics trajectory is unchanged.
        mujoco.mj_forward(self.model, self.data)

    def settle_freeze(self, n_steps=50):
        """Deploy-style freeze-frame0. The robot is held RIGID at the exact frame-0 configuration (base
        xy + orientation + all 29 joints), with ONLY base-z free, so it drops straight down onto both feet
        and every velocity damps to ~0. Mirrors the deploy bridge, which hard-resets qpos to frame-0 each
        physics step until the policy engages -> a clean, symmetric, zero-velocity double-stance handoff.
        (A soft PD hold instead injects contact jitter and hands off with residual velocity -> tips over.)"""
        full0 = self.data.qpos.copy()
        self.data.qfrc_applied[:] = 0.0
        for _ in range(n_steps):
            for _ in range(self.substeps):
                self.step()
                self.data.qpos[0:2] = full0[0:2]   # base xy
                self.data.qpos[3:] = full0[3:]      # base orientation + all 29 joints = frame0
                self.data.qvel[0:2] = 0.0
                self.data.qvel[3:] = 0.0            # only base-z (qpos[2]/qvel[2]) is free
        # refresh xpos/xipos/derived from the final frozen qpos: the loop overwrites qpos AFTER the last
        # mj_step, so without this the first logged frame's CoM/keypoints lag one config (<1 mm). One
        # mj_forward removes that lag; it does not integrate, so the physics/handoff are unchanged.
        mujoco.mj_forward(self.model, self.data)

    # --- state getters ---
    def base_quat_xyzw(self):
        return wxyz_to_xyzw(self.data.qpos[3:7])

    def base_ang_vel_base(self):
        # mujoco free-joint qvel[3:6] = angular velocity in the LOCAL (base) frame
        return self.data.qvel[3:6].copy()

    def dof_pos(self):
        return self.data.qpos[self.qpos_adr].copy()

    def dof_vel(self):
        return self.data.qvel[self.qvel_adr].copy()

    def projected_gravity_base(self):
        return quat_rotate_inverse_xyzw(self.base_quat_xyzw(), np.array([0.0, 0.0, -1.0]))

    def torso_quat_xyzw(self):
        return wxyz_to_xyzw(self.data.xquat[self.torso])

    def torso_quat_from_waist(self, waist_qpos3):
        """Torso world quat recomputed with the 3 waist joints overridden by (LPF'd) angles, via FK on a
        scratch MjData so the main sim is untouched. Mirrors the deploy, which LPFs the waist joints
        before the torso-orientation FK. waist_qpos3 == the raw waist angles -> equals torso_quat_xyzw()."""
        self._scratch.qpos[:] = self.data.qpos
        self._scratch.qpos[self.waist_qpos_adr] = waist_qpos3
        mujoco.mj_kinematics(self.model, self._scratch)
        return wxyz_to_xyzw(self._scratch.xquat[self.torso])

    def com_state(self):
        """Whole-body mass-weighted CoM position + linear velocity (world), summed over per-body LINK
        ORIGINS -- matches the training `_whole_body_com_state`, which explicitly ignores each link's
        local CoM offset. NOT subtree_com (which includes the local CoM lever -> a different quantity the
        policy never observed)."""
        m, d = self.model, self.data
        com = np.zeros(3)
        comv = np.zeros(3)
        for i in range(1, m.nbody):
            mi = m.body_mass[i]
            if mi <= 0.0:
                continue
            com += mi * d.xpos[i]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, i, self._v6, 0)  # world-frame
            comv += mi * self._v6[3:6]
        return com / self.total_mass, comv / self.total_mass

    def com_true(self):
        """TRUE whole-body CoM (world): mass-weighted per-body CoM (d.xipos) = Sum_i m_i*xipos_i / Sum_i m_i.
        This is the METRIC-side CoM, byte-identical to fulllog_native.py and every baseline, so all 9
        methods are scored against the SAME CoM. Distinct from com_state() (link-ORIGIN proxy, ~cm off with
        a systematic fore/aft bias) which the POLICY OBS (build_obs whole_body_com_rel_support_center) keeps
        using UNCHANGED. com_vel is finite-differenced by the caller at CONTROL_HZ (matches the baselines)."""
        m, d = self.model, self.data
        return (d.xipos * m.body_mass[:, None]).sum(0) / m.body_mass.sum()

    def com_rel_jac_world(self, support_mask: np.ndarray) -> np.ndarray:
        """World-frame Jacobian of (link-origin mass-weighted CoM − mask-weighted support-foot center)
        w.r.t. the 29 joint velocities. (3, 29). Used to propagate the dof_vel OBS noise+delay into the
        com_rel velocity exactly like the deploy (`vel += (J_c − J_s) @ qdot`, wbt_utils.py): the harness's
        true CoM/foot velocity uses the CLEAN qdot, so without this the com_rel obs would be noise-free
        while the deploy's is not. MuJoCo FK == the deploy's Pinocchio FK for this robot (validated 0.00 cm)
        -> this Jacobian matches the deploy's."""
        m, d = self.model, self.data
        jp = np.zeros((3, m.nv))
        jc = np.zeros((3, m.nv))
        for i in range(1, m.nbody):                       # link-origin mass-weighted CoM Jacobian
            mi = m.body_mass[i]
            if mi <= 0.0:
                continue
            mujoco.mj_jac(m, d, jp, None, d.xpos[i], i)
            jc += mi * jp
        jc /= self.total_mass
        js = np.zeros((3, m.nv))                          # mask-weighted support-foot-center Jacobian
        w = support_mask.astype(np.float64)
        wsum = w.sum() if w.any() else 2.0
        for k, b in enumerate([self.lfoot, self.rfoot]):
            ww = (w[k] if w.any() else 1.0) / wsum
            if ww == 0.0:
                continue
            mujoco.mj_jac(m, d, jp, None, d.xpos[b], b)
            js += ww * jp
        return (jc - js)[:, self.qvel_adr]                # (3, 29) joint columns in dof_names order

    def foot_state(self, bid):
        pos = self.data.xpos[bid][:2].copy()
        vel = _body_linvel_world(self.model, self.data, bid)[:2]
        z = self.data.xpos[bid][2]
        return pos, vel, z

    def foot_quat_xyzw(self, bid):
        return wxyz_to_xyzw(self.data.xquat[bid])

    def foot_contact_fz(self, bid):
        """sum of normal (z) ground reaction force on the given foot body's geoms."""
        fz = 0.0
        d, m = self.data, self.model
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            # foot geoms belong to bid or its child sphere links -> check body chain root
            if _is_foot(m, b1, bid) or _is_foot(m, b2, bid):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                # contact frame: force[0] is normal along contact normal (world z ~ up for flat ground)
                fz += abs(f[0])
        return fz

    def apply_torque(self, tau):
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.qvel_adr] = tau

    def step(self):
        mujoco.mj_step(self.model, self.data)


def _is_foot(m, body_id, foot_root):
    """True if body_id is foot_root or a descendant of it (sphere contact links)."""
    b = body_id
    for _ in range(8):
        if b == foot_root:
            return True
        if b == 0:
            return False
        b = m.body_parentid[b]
    return False


def _body_linvel_world(model, data, bid):
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)  # 0 = world frame
    return v[3:6]  # linear part (mj_objectVelocity returns [ang(3), lin(3)])


# ---------------------------------------------------------------------------------------------------
def build_obs(sim: G1Sim, policy: WBTPolicy, motion: dict, last_action: np.ndarray,
              height_diff: float = 0.03, dof_vel_obs: np.ndarray | None = None,
              torso_quat: np.ndarray | None = None, ang_vel_obs: np.ndarray | None = None,
              proj_g_obs: np.ndarray | None = None, dof_pos_obs: np.ndarray | None = None) -> np.ndarray:
    """Assemble the 463-dim actor obs (alphabetical term order) from sim state + ONNX motion outputs.
    `dof_vel_obs` overrides the raw joint velocity with the deploy bridge's delayed+noised value; when
    None the clean sim velocity is used (deterministic/clean condition). `torso_quat` (xyzw) overrides the
    robot torso orientation for motion_ref_ori_b with the deploy's waist-LPF'd FK value; None -> true quat.
    `ang_vel_obs`/`proj_g_obs`/`dof_pos_obs` override the raw IMU angular velocity, projected gravity and
    joint positions with the realism-switch (imu_noise/full_real) noised values; each None -> clean (the
    default -> byte-identical to the deploy sim2sim). These affect the OBS ONLY; the true state used for
    fall detection / metrics stays clean."""
    nd = policy.num_dofs
    # --- robot-state terms ---
    base_ang_vel = sim.base_ang_vel_base() if ang_vel_obs is None else ang_vel_obs   # (3)
    dof_pos = (sim.dof_pos() if dof_pos_obs is None else dof_pos_obs) - policy.default_dof  # (29)
    dof_vel = sim.dof_vel() if dof_vel_obs is None else dof_vel_obs  # (29)
    proj_g = sim.projected_gravity_base() if proj_g_obs is None else proj_g_obs      # (3)

    # --- motion terms (from ONNX reference outputs, obs-independent lookup) ---
    joint_pos = motion["joint_pos"].reshape(-1)
    joint_vel = motion["joint_vel"].reshape(-1)
    motion_command = np.concatenate([joint_pos, joint_vel])     # (58)
    ref_support = motion["reference_support_phase"].reshape(-1)  # (2)
    future_support = motion["future_support_phase"].reshape(-1)  # (10)
    future_cmd = motion["future_cmd"].reshape(-1)                # (290)

    # --- motion_ref_ori_b (6): relative orientation robot-torso vs ref-torso, matrix first 2 cols ---
    ref_quat = motion["ref_quat_xyzw"].reshape(-1)              # xyzw
    robot_ori = sim.torso_quat_xyzw() if torso_quat is None else torso_quat
    rel_ori = subtract_frame_ori_xyzw(robot_ori, ref_quat)
    mat = quat_to_matrix_xyzw(rel_ori)
    motion_ref_ori_b = mat[:, :2].reshape(-1)                   # (6)

    # --- whole_body_com_rel_support_center (4): CoM & vel rel support center, base frame ---
    com, comv = sim.com_state()
    lp, lv, lz = sim.foot_state(sim.lfoot)
    rp, rv, rz = sim.foot_state(sim.rfoot)
    # height-based support (matches deploy): lower foot supports; both within height_diff -> double
    mask = _support_height_mask(lz, rz, height_diff)
    fp = np.stack([lp, rp]); fv = np.stack([lv, rv])
    if mask.any():
        w = mask.astype(np.float64)[:, None]
        center_pos = (fp * w).sum(0) / w.sum()
        center_vel = (fv * w).sum(0) / w.sum()
    else:
        center_pos = fp.mean(0); center_vel = fv.mean(0)
    rel_pos_w = np.array([com[0] - center_pos[0], com[1] - center_pos[1], 0.0])
    rel_vel_w = np.array([comv[0] - center_vel[0], comv[1] - center_vel[1], 0.0])
    # Deploy: the com_rel VELOCITY is `(J_c - J_s) @ qdot` with the (noised+delayed) encoder qdot, so the
    # dof_vel obs noise+delay propagates into the balance-core com_rel obs (the amplification that matters).
    # The harness's true CoM/foot velocity above used the CLEAN qdot -> add back the noise/delay difference
    # projected through the CoM-relative Jacobian, exactly like the deploy.
    if dof_vel_obs is not None:
        dqd = dof_vel_obs - sim.dof_vel()                          # (noised+delayed) - clean
        dvw = sim.com_rel_jac_world(mask) @ dqd                    # world-frame velocity contribution
        rel_vel_w[0] += dvw[0]
        rel_vel_w[1] += dvw[1]
    bq = sim.base_quat_xyzw()
    pos_b = quat_rotate_inverse_xyzw(bq, rel_pos_w)[:2]
    vel_b = quat_rotate_inverse_xyzw(bq, rel_vel_w)[:2]
    com_rel = np.concatenate([pos_b, vel_b])                    # (4)

    # alphabetical concat: actions, base_ang_vel, dof_pos, dof_vel, future_cmd, future_support_phase,
    #                      motion_command, motion_ref_ori_b, projected_gravity, reference_support_phase,
    #                      whole_body_com_rel_support_center
    obs = np.concatenate([
        last_action.reshape(-1), base_ang_vel, dof_pos, dof_vel, future_cmd, future_support,
        motion_command, motion_ref_ori_b, proj_g, ref_support, com_rel,
    ]).astype(np.float32)
    return obs


def _support_height_mask(lz, rz, height_diff):
    if abs(lz - rz) <= height_diff:
        return np.array([True, True])
    return np.array([lz < rz, rz <= lz])


# ---------------------------------------------------------------------------------------------------
def load_motion_npz(motion_id: str):
    """Return the reference npz + derived per-frame reference support state & CoM (world)."""
    path = os.path.join(MOTION_DIR, f"sample_{motion_id}_mj.npz")
    d = np.load(path)
    body_names = [str(x) for x in d["body_names"]]
    pelvis_i = body_names.index("pelvis")
    # npz joint_pos is the FULL qpos (T,36) = [base_pos(3), base_quat_wxyz(4), joints(29)].
    qpos0 = d["joint_pos"][0].astype(np.float64)
    base_pos0 = qpos0[:3]
    base_quat0 = qpos0[3:7]  # wxyz
    joint_pos0 = qpos0[7:36]
    return {
        "npz": d,
        "T": int(d["joint_pos"].shape[0]),
        "base_pos0": base_pos0,
        "base_quat0": base_quat0,
        "joint_pos0": joint_pos0,
        "ref_support_state": d["reference_support_state"].astype(int),  # per-frame: 0=double,1=left,2=right (convention TBD)
        "body_names": body_names,
        "pelvis_i": pelvis_i,
    }


class NpzMotionReference:
    """Serves the motion-reference obs terms straight from the motion npz, byte-for-byte reproducing the
    baked-ONNX reference branch: joint_pos = npz joints[t];
    joint_vel = npz joint_vel[t][6:35]; ref_quat_xyzw = torso_link quat (wxyz->xyzw); reference_support_phase
    = npz reference_support_phase[t]; future (5 frames t+1..t+5, clamped to the last frame at the end) drives
    future_cmd (joint_pos+joint_vel per frame, 5*58=290) and future_support_phase (5*2=10). Lets ONE policy
    ONNX be scored against ANY motion without re-baking a per-motion ONNX."""

    def __init__(self, npz):
        self.jp = np.asarray(npz["joint_pos"])[:, 7:36].astype(np.float32)      # (T,29) reference joint pos
        self.jv = np.asarray(npz["joint_vel"])[:, 6:35].astype(np.float32)      # (T,29) reference joint vel
        self.sp = np.asarray(npz["reference_support_phase"], dtype=np.float32)  # (T,2)
        bn = [str(x) for x in npz["body_names"]]
        tq = np.asarray(npz["body_quat_w"])[:, bn.index("torso_link")]          # (T,4) wxyz
        self.tq = tq[:, [1, 2, 3, 0]].astype(np.float32)                        # -> xyzw
        self.T = self.jp.shape[0]

    def infer(self, obs, time_step):
        T = self.T
        c = lambda i: min(max(int(i), 0), T - 1)  # noqa: E731  clamp index (hold last frame past the end)
        t = c(time_step)
        fut = [c(time_step + k) for k in range(1, 6)]
        future_cmd = np.concatenate([np.concatenate([self.jp[f], self.jv[f]]) for f in fut])  # 5*58
        future_sp = np.concatenate([self.sp[f] for f in fut])                                 # 5*2
        return {"joint_pos": self.jp[t], "joint_vel": self.jv[t], "ref_quat_xyzw": self.tq[t],
                "reference_support_phase": self.sp[t], "future_support_phase": future_sp,
                "future_cmd": future_cmd}


def run_rollout(onnx_path: str, motion_id: str, max_frames: int | None = None, seed: int = 0,
                dof_vel_noise: float = 0.20, dof_vel_delay: int = 1, push: dict | None = None,
                fall_h_frac: float = 0.5, fall_tilt: float = 1.0, verbose: bool = False,
                use_npz_ref: bool = False, imu_noise: bool = False, full_real: bool = False):
    """Run one exported ONNX through its motion in the deploy MuJoCo plant; return a per-step log of the
    TRUE world state (metrics.py consumes it). Faithful to the deploy sim2sim bridge:
      * the dof_vel OBSERVATION gets a `dof_vel_delay`-step delay THEN fresh uniform +-`dof_vel_noise`
        rad/s noise (seeded -> reproducible; this is the intended K-trial variation source). The PD
        controller keeps using the clean fresh velocity, and the log stores the clean velocity too.
      * `push` = dict(interval_s, vel) injects a seeded root linear-velocity kick every interval_s (the
        section 5.3 test-time perturbation condition). Set dof_vel_noise=0, push=None for the clean run.

    Realism switches (BOTH default False -> byte-identical to the clean deploy sim2sim; they only ADD
    obs-side noise, never change the true physics/fall/metrics or the existing dof_vel/push RNG stream):
      * `imu_noise`: inject the training IMU model (HuB arXiv:2505.07294) into the actor obs -- a shared
        slowly-drifting orientation error delta (OU/AR(1): theta=0.1, sigma=0.016 per 50 Hz control step,
        steady std ~0.036 rad) applied COUPLED to both base_ang_vel and projected_gravity via
        `v + cross(delta, v)`, PLUS the per-term uniform white noise (ang_vel +-0.2, gravity +-0.03) the
        actor was always trained with. delta starts at 0 and steps once per control tick.
      * `full_real`: imu_noise PLUS 1-control-step actuator latency (q_target delayed one 50 Hz step; NOTE
        action-delay DR was OFF in the 20260718 training -> this is an OOD stress test) PLUS joint-position
        encoder noise (dof_pos +-0.01 per step + a per-rollout fixed bias +-0.01).
    The IMU noise draws from a dedicated `rng_imu` and the dof_pos/actuator noise from `rng_real`, both
    seeded off `seed`, so: switches OFF -> those streams are never touched (identical); imu_noise and
    full_real share the SAME delta + dof_vel/push sequences -> conditions are strictly paired per seed.
    """
    policy = onnx_path if hasattr(onnx_path, 'infer') else WBTPolicy(onnx_path)
    sim = G1Sim(policy.dof_names)
    ref = load_motion_npz(motion_id)
    # DECOUPLED mode (use_npz_ref): take the motion-reference obs terms from the npz instead of the ONNX's
    # own baked reference branch, so ONE policy ONNX scores against ANY motion (500-motion sweep). The policy
    # ONNX is still used for the ACTIONS (its actor is motion-independent). Off = original baked-ONNX path.
    motion_ref = NpzMotionReference(ref["npz"]) if use_npz_ref else policy
    T = ref["T"] if max_frames is None else min(ref["T"], max_frames)
    rng = np.random.RandomState(seed)

    # --- realism switches (default OFF -> none of this runs -> byte-identical to the clean sim2sim) ---
    apply_imu = imu_noise or full_real          # HuB IMU OU drift + per-term white noise on the actor obs
    apply_actdelay = full_real                  # 1 control-step (20 ms) actuator latency on q_target
    apply_dofpos = full_real                    # joint-position encoder white noise + per-rollout bias
    rng_imu = np.random.RandomState(seed + 100003)   # dedicated streams: OFF -> untouched (identical); and
    rng_real = np.random.RandomState(seed + 200003)  # imu_noise & full_real share the SAME IMU delta seq
    IMU_OU_THETA, IMU_OU_SIGMA = 0.1, 0.016          # server training constants (HuB arXiv:2505.07294)
    IMU_ANGVEL_WHITE, IMU_GRAV_WHITE = 0.2, 0.03     # actor per-term uniform IMU noise (baked config)
    DOFPOS_WHITE, DOFPOS_BIAS = 0.01, 0.01           # dof_pos.noise + dof_pos_bias_range (baked config)
    imu_ou = np.zeros(3)                              # shared IMU orientation-error delta, starts at zero
    dofpos_bias = rng_real.uniform(-DOFPOS_BIAS, DOFPOS_BIAS, policy.num_dofs) if apply_dofpos else None
    act_delay = 1 if apply_actdelay else 0
    qtarget_hist: list = []

    sim.reset_to(ref["base_pos0"], ref["base_quat0"], ref["joint_pos0"])
    # Deploy-style freeze-frame0: hold the robot rigid at frame-0, drop straight onto both feet and damp
    # velocities -> a clean in-contact, zero-velocity double stance before the policy takes over.
    sim.settle_freeze(n_steps=50)
    nominal_h = float(ref["base_pos0"][2])
    last_action = np.zeros(policy.num_dofs, dtype=np.float32)
    waist_lpf = None  # deploy waist-joint EMA state for the torso-orientation obs
    push_every = int(round(push["interval_s"] * CONTROL_HZ)) if push else 0
    push_vel = np.asarray(push["vel"], dtype=np.float64) if push else None

    keys = ["t", "base_pos", "base_quat", "base_angvel", "base_linvel", "qpos", "qvel", "fz_l", "fz_r",
            "foot_l_pos", "foot_r_pos", "foot_l_z", "foot_r_z", "foot_l_quat", "foot_r_quat",
            "com", "com_vel", "com_rel_obs", "action", "motor_cmd", "tilt", "fell", "kp12_robot"]
    log = {k: [] for k in keys}
    fell = False
    for t in range(T):
        # optional test-time root velocity perturbation (section 5.3)
        if push_every and t > 0 and t % push_every == 0:
            sim.data.qvel[0:3] += rng.uniform(-1.0, 1.0, 3) * push_vel

        motion = motion_ref.infer(np.zeros(463, dtype=np.float32), t)  # obs-independent reference lookup

        # dof_vel observation: `dof_vel_delay` PHYSICS-step delay THEN fresh uniform noise (deploy bridge
        # order + rate). `_vel_hist_sub` is filled by pd_step at the physics-step rate; its newest entry is
        # the current velocity, so a delay of `dof_vel_delay` physics steps reads `[-1 - dof_vel_delay]`
        # (deploy DOF_VEL_DELAY_STEPS=1 = 0.5 ms @ 2000 Hz, NOT the 20 ms control step -- the earlier control-step
        # delay over-lagged com_rel 4x and made the policy fall where the real deploy only hops occasionally).
        clean_dv = sim.dof_vel()
        sub = sim._vel_hist_sub
        delayed = sub[-1 - dof_vel_delay] if dof_vel_delay > 0 and len(sub) > dof_vel_delay else clean_dv
        dv_obs = delayed + rng.uniform(-1.0, 1.0, clean_dv.shape) * dof_vel_noise if dof_vel_noise > 0 else delayed

        # deploy waist-LPF: EMA the 3 waist joints, recompute the torso quat used by motion_ref_ori_b
        if WBT_WAIST_LPF_ALPHA < 1.0:
            waist_raw = sim.dof_pos()[sim.waist_dof_idx]
            waist_lpf = waist_raw.copy() if waist_lpf is None else (
                WBT_WAIST_LPF_ALPHA * waist_raw + (1.0 - WBT_WAIST_LPF_ALPHA) * waist_lpf)
            torso_quat = sim.torso_quat_from_waist(waist_lpf)
        else:
            torso_quat = None
        # --- realism switches: OBS-ONLY noise (true state used for fall/metrics below stays clean) ---
        ang_vel_obs = proj_g_obs = dof_pos_obs = None
        if apply_imu:
            # step the shared IMU orientation-error delta once per control tick (AR(1)/OU), then apply it
            # COUPLED to both IMU vectors (v + delta x v) + the per-term uniform white noise
            imu_ou = imu_ou * (1.0 - IMU_OU_THETA) + IMU_OU_SIGMA * rng_imu.standard_normal(3)
            av = sim.base_ang_vel_base()
            ang_vel_obs = av + np.cross(imu_ou, av) + rng_imu.uniform(-1.0, 1.0, 3) * IMU_ANGVEL_WHITE
            pg = sim.projected_gravity_base()
            proj_g_obs = pg + np.cross(imu_ou, pg) + rng_imu.uniform(-1.0, 1.0, 3) * IMU_GRAV_WHITE
        if apply_dofpos:
            dof_pos_obs = sim.dof_pos() + dofpos_bias + rng_real.uniform(-1.0, 1.0, policy.num_dofs) * DOFPOS_WHITE
        obs = build_obs(sim, policy, motion, last_action, dof_vel_obs=dv_obs, torso_quat=torso_quat,
                        ang_vel_obs=ang_vel_obs, proj_g_obs=proj_g_obs, dof_pos_obs=dof_pos_obs)
        obs = np.clip(obs, -OBS_CLIP, OBS_CLIP)                        # deploy clip_observations=100
        action = policy.infer(obs, t)["actions"].reshape(-1)
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)           # deploy policy-action clip
        last_action = action.astype(np.float32)                       # deploy stores the CLIPPED action
        q_target = action * policy.action_scale + policy.default_dof
        if WBT_ENABLE_TARGET_CLAMP and policy.dof_pos_lower is not None:  # deploy joint clamp (default OFF)
            q_target = np.clip(q_target, policy.dof_pos_lower, policy.dof_pos_upper)

        # log TRUE world state (pre-step: the state the policy acted on); qvel is the CLEAN velocity
        tilt = float(np.arccos(np.clip(-sim.projected_gravity_base()[2], -1, 1)))
        lp, lv, lz = sim.foot_state(sim.lfoot); rp, rv, rz = sim.foot_state(sim.rfoot)
        # metric-side CoM = TRUE whole-body CoM (mass-weighted per-body xipos), unified across all 9 methods
        # (matches fulllog_native / the baselines); com_vel = finite-diff at CONTROL_HZ. The POLICY OBS
        # com_rel (build_obs, above) keeps the link-origin proxy com_state() -- this only changes the metrics.
        com = sim.com_true()
        comv = (com - log["com"][-1]) * CONTROL_HZ if log["com"] else np.zeros(3)
        cur_h = float(sim.data.qpos[2])
        fell = fell or (cur_h < fall_h_frac * nominal_h) or (tilt > fall_tilt)
        log["t"].append(t / CONTROL_HZ)
        log["base_pos"].append(sim.data.qpos[0:3].copy())
        log["base_quat"].append(sim.base_quat_xyzw())
        log["base_angvel"].append(sim.base_ang_vel_base())
        log["base_linvel"].append(sim.data.qvel[0:3].copy())
        log["qpos"].append(sim.dof_pos()); log["qvel"].append(clean_dv)
        log["fz_l"].append(sim.foot_contact_fz(sim.lfoot)); log["fz_r"].append(sim.foot_contact_fz(sim.rfoot))
        log["foot_l_pos"].append(lp); log["foot_r_pos"].append(rp)
        log["foot_l_z"].append(lz); log["foot_r_z"].append(rz)
        log["foot_l_quat"].append(sim.foot_quat_xyzw(sim.lfoot))
        log["foot_r_quat"].append(sim.foot_quat_xyzw(sim.rfoot))
        log["com"].append(com); log["com_vel"].append(comv)
        log["com_rel_obs"].append(obs[-4:].copy())
        log["action"].append(action.copy()); log["motor_cmd"].append(q_target.copy())
        log["tilt"].append(tilt); log["fell"].append(fell)
        log["kp12_robot"].append(sim.data.xpos[sim.kp12_ids].copy())   # HuB 12 keypoints (world)

        # actuator latency (full_real): apply the q_target from `act_delay` control steps ago (act_delay=0
        # -> q_apply is q_target -> identical). The policy's last_action memory is unaffected (only the
        # motor command is delayed, matching the deploy action-delay buffer).
        qtarget_hist.append(q_target)
        q_apply = qtarget_hist[-1 - act_delay] if act_delay and len(qtarget_hist) > act_delay else q_target
        sim.pd_step(q_apply, policy.kp, policy.kd)

    out_log = {k: np.array(v) for k, v in log.items()}
    out_log["ref_support_state"] = ref["ref_support_state"][:T]
    out_log["nominal_h"] = nominal_h
    out_log["motion_id"] = motion_id
    out_log["seed"] = seed
    out_log["dt"] = 1.0 / CONTROL_HZ
    out_log["body_mass_total"] = sim.total_mass
    if verbose:
        print(f"  [{motion_id} seed{seed}] T={T} fell={bool(out_log['fell'][-1])} "
              f"base_h={out_log['base_pos'][-1,2]:.3f} tilt={np.degrees(out_log['tilt'][-1]):.1f}deg")
    return out_log


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python wbt_rollout.py <exported_model_<step>_<motion>.onnx>\n"
              "  (legacy per-motion-ONNX smoke test; for the benchmark use run_eval.py, which scores\n"
              "   policy/model_0262000.pt against the motion set from the .pt alone.)")
        sys.exit(0)
    onnx = sys.argv[1]
    mid = re.sub(r"^model_\d+_", "", os.path.basename(onnx))[:-5]
    log = run_rollout(onnx, mid, verbose=True)
    print("keys:", list(log.keys()))
    print("base height trajectory (every 30 steps):", np.round(log["base_pos"][::30, 2], 3))
    print("tilt deg (every 30):", np.round(np.degrees(log["tilt"][::30]), 1))
