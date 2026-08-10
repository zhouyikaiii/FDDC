"""Run the DDC actor directly from a .pt (numpy) — no IsaacSim, no ONNX. Bit-exact with the IsaacSim
ONNX export. The robot config (dof_names / kp / kd / PD action_scale / default pose /
joint limits) is read from a static robot_config.json — these are Unitree-G1 constants, identical across
every training run — so the ONLY per-policy artifact needed is the .pt."""
import json, numpy as np, torch


def _silu(x): return x / (1.0 + np.exp(-x))


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _obs_slice(obs, dim):
    if dim == 463: return obs                                 # full (dyn-CoM: pos2 + vel2)
    if dim == 461: return obs[..., :461]                      # static-CoM: drop com_rel velocity (last 2)
    if dim == 459: return obs[..., :459]                      # drop com_rel (last 4)
    if dim == 163: return np.concatenate([obs[..., :90], obs[..., 390:]], axis=-1)  # drop future (90:390)
    raise ValueError(f"unknown obs_dim={dim}")


class FastPolicy:
    """DDC deployable actor from a .pt + robot_config.json. Drop-in for wbt_rollout (exposes .infer,
    .dof_names, .kp, .kd, .action_scale, .default_dof, .dof_pos_lower/upper, .num_dofs, .obs_dim)."""

    def __init__(self, pt_path, robot_config):
        d = torch.load(pt_path, map_location="cpu", weights_only=True)
        sd = d["actor_state_dict"]; norm = d["obs_normalizer_state"]
        f = lambda x: x.detach().cpu().numpy().astype(np.float64)
        # normalization divisor = std + 0.01 (RunningMeanStd eps; the raw std is 0 on the constant dims,
        # which would otherwise divide-by-zero -> NaN).
        self._mean = f(norm["_mean"]); self._std = f(norm["_std"]) + 0.01
        self._W = [f(sd[f"net.{i}.weight"]) for i in (0, 3, 6)]
        self._b = [f(sd[f"net.{i}.bias"]) for i in (0, 3, 6)]
        self._lw = [f(sd[f"net.{i}.weight"]) for i in (1, 4, 7)]
        self._lb = [f(sd[f"net.{i}.bias"]) for i in (1, 4, 7)]
        self._muW = f(sd["fc_mu.0.weight"]); self._mub = f(sd["fc_mu.0.bias"])
        self._as = f(sd["action_scale"]); self._ab = f(sd["action_bias"])  # network tanh-squash bounds
        self.obs_dim = self._W[0].shape[1]
        # robot constants from the static json
        cfg = robot_config if isinstance(robot_config, dict) else json.load(open(robot_config))
        self.dof_names = list(cfg["dof_names"]); self.num_dofs = len(self.dof_names)
        self.kp = np.asarray(cfg["kp"], dtype=np.float64)
        self.kd = np.asarray(cfg["kd"], dtype=np.float64)
        sc = np.asarray(cfg["action_scale"], dtype=np.float64).reshape(-1)   # PD residual scale
        self.action_scale = np.full(self.num_dofs, sc.item()) if sc.size == 1 else sc
        self.default_dof = np.asarray(cfg["default_dof_pos"], dtype=np.float64)
        self.dof_pos_lower = np.asarray(cfg["dof_pos_lower"], dtype=np.float64) if cfg.get("dof_pos_lower") else None
        self.dof_pos_upper = np.asarray(cfg["dof_pos_upper"], dtype=np.float64) if cfg.get("dof_pos_upper") else None

    def infer(self, obs_463, time_step):
        x = _obs_slice(np.asarray(obs_463, dtype=np.float64).reshape(1, -1), self.obs_dim)
        x = (x - self._mean) / self._std
        for i in range(3):
            x = x @ self._W[i].T + self._b[i]; x = _ln(x, self._lw[i], self._lb[i]); x = _silu(x)
        mu = x @ self._muW.T + self._mub
        act = np.tanh(mu) * self._as + self._ab
        return {"actions": act.astype(np.float32)}
