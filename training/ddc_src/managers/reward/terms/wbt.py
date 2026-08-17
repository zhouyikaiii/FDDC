# ---------------------------------------------------------------------------------------------------
# Modified from the Holosoma framework (Amazon FAR): https://github.com/amazon-far/holosoma
# Copyright Amazon.com, Inc. or its affiliates. Licensed under the Apache License, Version 2.0.
# This file was CHANGED for DDC: it carries the deployable support-relative dynamic-CoM observation
# and/or the human-science balance reward library and its config (see training/TRAINING.md).
# The Apache-2.0 license text is in the repository LICENSE; attribution is in training/NOTICE.
# ---------------------------------------------------------------------------------------------------

"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate_batched

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


# ================================================================================================
# Reference Support Contact Mismatch (anti foot-hopping)
# ================================================================================================


class ReferenceSupportContactMismatchPenalty(RewardTermBase):
    """Penalize mismatch between the reference support phase and the robot's actual foot contact.

    Faithful port of whole_body_tracking's ``reference_support_contact_mismatch_penalty``: the logic
    is identical; only ``weight`` is rescaled in the config to holosoma's reward magnitude (WBT used
    -10; here -2). Per foot, the actual vertical contact force is squashed through a sigmoid and
    compared to the soft reference support phase in [0, 1] (precomputed offline into each motion npz
    and exposed via ``MotionCommand.reference_support_phase``). A two-sided product loss is weighted
    by a transition-confidence factor ``|2*ref - 1|`` and summed over both feet:

      * stance-miss  (``stance_miss_weight``):         reference says contact but the foot is airborne
                                                       -- this is the single-foot-hopping case.
      * swing-extra  (``swing_extra_contact_weight``): reference says swing but the foot is touching.

    Returns a non-negative penalty magnitude [num_envs]; the negative sign comes from the config weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        # Foot bodies in [left, right] order -- MUST match the reference_support_phase columns.
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_body_indexes = self._get_index_of_a_in_b(
            foot_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.force_threshold = cfg.params.get("force_threshold", 5.0)
        self.force_tau = cfg.params.get("force_tau", 0.5)
        self.stance_miss_weight = cfg.params.get("stance_miss_weight", 1.0)
        self.swing_extra_contact_weight = cfg.params.get("swing_extra_contact_weight", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)

        # Actual contact (soft, [0, 1]): vertical (z) net force, relu'd, max over the history window.
        # contact_forces_history: (num_envs, history_length, num_bodies, 3) -> (num_envs, 2)
        net_contact_forces = self.env.simulator.contact_forces_history
        normal_force = torch.relu(net_contact_forces[:, :, self.foot_body_indexes, 2]).max(dim=1)[0]
        actual_contact = torch.sigmoid((normal_force - self.force_threshold) / self.force_tau)

        # Reference contact (soft, [0, 1]) for the current frame, columns [left, right]. (num_envs, 2)
        reference_contact = motion_command.reference_support_phase.to(actual_contact.dtype)
        certainty = torch.abs(2.0 * reference_contact - 1.0)

        loss = certainty * (
            self.stance_miss_weight * reference_contact * (1.0 - actual_contact)
            + self.swing_extra_contact_weight * (1.0 - reference_contact) * actual_contact
        )
        return torch.sum(loss, dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


# ================================================================================================
# Balance rewards (xCoM / support-polygon margin / time-to-boundary) + single-support no-slip
# Ported from whole_body_tracking (logic identical; param VALUES set in config). holosoma exposes
# no whole-body CoM, so it is computed from per-body world pos/vel weighted by per-body masses
# (masses fetched once via simulator.get_body_masses()).
# ================================================================================================

_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
# G1 foot support rectangle: corners in the ankle body frame (x, y, z); both feet share it.
_G1_FOOT_SUPPORT_POLYGON = (
    (-0.05, -0.025, -0.03),
    (0.12, -0.030, -0.03),
    (0.12, 0.030, -0.03),
    (-0.05, 0.025, -0.03),
)


def _resolve_foot_body_indexes(env) -> torch.Tensor:
    body_names = env.simulator.body_names  # type: ignore[attr-defined]
    return torch.tensor([body_names.index(n) for n in _FOOT_BODY_NAMES], dtype=torch.long, device=env.device)


def _whole_body_com_state(env, body_masses: torch.Tensor, total_mass: torch.Tensor):
    """Mass-weighted whole-body CoM position & linear velocity (world frame).

    Approximation: uses per-body link-origin pos/vel (ignores each link's local CoM offset);
    mass-averaged, this is accurate enough for the LIP/xCoM balance signal.
    """
    pos = env.simulator._rigid_body_pos  # (N, num_bodies, 3)
    vel = env.simulator._rigid_body_vel  # (N, num_bodies, 3)
    w = body_masses.view(1, -1, 1)
    com_pos = (pos * w).sum(dim=1) / total_mass
    com_vel = (vel * w).sum(dim=1) / total_mass
    return com_pos, com_vel


def _xcom_xy(
    com_pos: torch.Tensor,
    com_vel: torch.Tensor,
    gravity: float,
    com_height_floor: float,
    use_velocity: bool = True,
) -> torch.Tensor:
    """Extrapolated CoM (Hof / LIP): xcom_xy = com_xy + com_vel_xy / omega, omega = sqrt(g / h).

    use_velocity=False -> degenerates to the static geometric CoM (com_xy only).
    - polygon / xcom_ttb rewards: use this function's default use_velocity=True (always dynamic);
    - critic obs whole_body_xcom_rel_support_center: passes use_velocity=CRITIC_XCOM_USE_VELOCITY explicitly;
    - actor obs whole_body_com_rel_support_center: does not use this function; it reads ACTOR_XCOM_USE_VELOCITY in its own func.
    """
    if not use_velocity:
        return com_pos[:, :2]
    h = com_pos[:, 2].clamp_min(com_height_floor)
    omega = torch.sqrt(torch.full_like(h, float(gravity)) / h)
    return com_pos[:, :2] + com_vel[:, :2] / omega.unsqueeze(-1)


def _support_contact_mask(env, foot_ids: torch.Tensor, threshold: float) -> torch.Tensor:
    """Per-foot boolean contact: max-over-history of the 3D net-force norm > threshold. (N, num_feet)"""
    forces = env.simulator.contact_forces_history[:, :, foot_ids, :]  # (N, hist, num_feet, 3)
    return forces.norm(dim=-1).max(dim=1)[0] > threshold


def _support_height_mask(env, foot_ids: torch.Tensor, double_support_height_diff: float = 0.03) -> torch.Tensor:
    """Per-foot boolean support from gravity-aligned foot HEIGHT (DEPLOYABLE proxy). (N, 2).

    Matches the deploy-side support rule (sim2sim/sim2real: HoloFKEstimator /
    PinocchioRobot.compute_com_rel_support_center_b): the lower foot supports; if both feet are within
    ``double_support_height_diff`` -> double support. In sim, world-up = +z, so foot world-z IS the
    gravity-aligned height. Used by the DEPLOYED actor obs whole_body_com_rel_support_center so training
    and deployment select the support foot identically (no train->deploy gap at foot lift/land).
    foot_ids order is [left, right] (see _FOOT_BODY_NAMES).
    """
    foot_z = env.simulator._rigid_body_pos[:, foot_ids, 2]  # (N, 2) world height (== gravity-aligned in sim)
    z_left, z_right = foot_z[:, 0], foot_z[:, 1]
    double = (z_left - z_right).abs() < double_support_height_diff
    left_lower = z_left < z_right
    left_support = double | left_lower
    right_support = double | (~left_lower)
    return torch.stack([left_support, right_support], dim=1)  # (N, 2) bool


def _foot_support_vertices_xy_w(env, foot_ids: torch.Tensor, polygon_b: torch.Tensor) -> torch.Tensor:
    """World XY of each foot's support-polygon corners. polygon_b: (V,3) in ankle frame -> (N, num_feet, V, 2)."""
    foot_pos = env.simulator._rigid_body_pos[:, foot_ids, :]   # (N, F, 3)
    foot_quat = env.simulator._rigid_body_rot[:, foot_ids, :]  # (N, F, 4) xyzw
    n_env = foot_pos.shape[0]
    n_v = polygon_b.shape[0]
    verts = []
    for f in range(foot_ids.shape[0]):
        corners_b = polygon_b.unsqueeze(0).expand(n_env, n_v, 3)  # (N, V, 3)
        corners_w = quat_rotate_batched(foot_quat[:, f, :], corners_b) + foot_pos[:, f, :].unsqueeze(1)
        verts.append(corners_w[..., :2])  # (N, V, 2)
    return torch.stack(verts, dim=1)  # (N, F, V, 2)


def _convex_hull_halfspace_margin(point_xy: torch.Tensor, vertices_xy: torch.Tensor) -> torch.Tensor:
    """Signed margin of point to the convex hull of vertices via support halfspaces (SAT).

    Positive => inside all halfspaces, negative => outside. vertices_xy: (M, V, 2), point_xy: (M, 2).
    """
    num_vertices = vertices_xy.shape[1]
    idx0, idx1 = torch.triu_indices(num_vertices, num_vertices, offset=1, device=vertices_xy.device)
    edge = vertices_xy[:, idx1] - vertices_xy[:, idx0]
    axes = torch.stack([-edge[..., 1], edge[..., 0]], dim=-1)
    axes = axes / torch.norm(axes, dim=-1, keepdim=True).clamp_min(1e-8)
    axes = torch.cat([axes, -axes], dim=1)
    vertex_projection = torch.sum(vertices_xy[:, :, None, :] * axes[:, None, :, :], dim=-1)
    support_projection = torch.max(vertex_projection, dim=1)[0]
    point_projection = torch.sum(point_xy[:, None, :] * axes, dim=-1)
    return torch.min(support_projection - point_projection, dim=1)[0]


def _support_polygon_margin(env, foot_ids, point_xy, threshold, polygon_b):
    """Signed margin of point_xy (xCoM) to the support polygon. Returns (margin, single_mask, double_mask)."""
    foot_vertices_xy = _foot_support_vertices_xy_w(env, foot_ids, polygon_b)  # (N, 2, V, 2)
    support_mask = _support_contact_mask(env, foot_ids, threshold)  # (N, 2)
    left_support = support_mask[:, 0]
    right_support = support_mask[:, 1]
    margin = torch.zeros(point_xy.shape[0], device=point_xy.device, dtype=point_xy.dtype)
    left_only = left_support & ~right_support
    right_only = right_support & ~left_support
    single_support = left_only | right_only
    double_support = left_support & right_support
    if torch.any(left_only):
        margin[left_only] = _convex_hull_halfspace_margin(point_xy[left_only], foot_vertices_xy[left_only, 0])
    if torch.any(right_only):
        margin[right_only] = _convex_hull_halfspace_margin(point_xy[right_only], foot_vertices_xy[right_only, 1])
    if torch.any(double_support):
        double_vertices = torch.cat([foot_vertices_xy[double_support, 0], foot_vertices_xy[double_support, 1]], dim=1)
        margin[double_support] = _convex_hull_halfspace_margin(point_xy[double_support], double_vertices)
    return margin, single_support, double_support


class SupportXcomPolygonMarginPenalty(RewardTermBase):
    """Penalize the extrapolated CoM (xCoM) approaching/leaving the support polygon.

    penalty = relu(safety_margin - signed_margin)^2 per env (optionally capped by max_penalty).
    Faithful port of whole_body_tracking.support_xcom_polygon_margin_penalty; param values in config.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.foot_body_indexes = _resolve_foot_body_indexes(env)
        self.body_masses = env.simulator.get_body_masses()  # type: ignore[attr-defined]
        self.total_mass = self.body_masses.sum().clamp_min(1e-6)
        p = cfg.params
        self.threshold = p.get("threshold", 1.0)
        self.single_support_safety_margin = p.get("single_support_safety_margin", 0.005)
        self.double_support_safety_margin = p.get("double_support_safety_margin", 0.01)
        self.gravity = p.get("gravity", 9.81)
        self.com_height_floor = p.get("com_height_floor", 0.25)
        self.max_penalty = p.get("max_penalty", 0.04)
        self.penalty_power = p.get("penalty_power", 2.0)  # 1 = linear hinge, 2 = squared (original behavior)
        self.polygon_b = torch.tensor(
            p.get("foot_polygon", _G1_FOOT_SUPPORT_POLYGON), dtype=torch.float32, device=env.device
        )

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        com_pos, com_vel = _whole_body_com_state(env, self.body_masses, self.total_mass)
        xcom_xy = _xcom_xy(com_pos, com_vel, self.gravity, self.com_height_floor)
        margin, single_support, double_support = _support_polygon_margin(
            env, self.foot_body_indexes, xcom_xy, self.threshold, self.polygon_b
        )
        safety = torch.zeros_like(margin)
        safety[single_support] = float(self.single_support_safety_margin)
        safety[double_support] = float(self.double_support_safety_margin)
        active = single_support | double_support
        penalty = torch.zeros_like(margin)
        penalty[active] = torch.relu(safety[active] - margin[active]) ** self.penalty_power
        if self.max_penalty is not None and self.max_penalty > 0.0:
            penalty = penalty.clamp(max=float(self.max_penalty))
        return penalty

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class XcomTtbPenalty(RewardTermBase):
    """Penalize small Time-To-Boundary (TTB) of the xCoM w.r.t. the support polygon.

    ttb = margin / max(omega * ||xcom - support_center||, v_min);  penalty = relu(ttb_threshold - ttb)^2.
    Faithful port of whole_body_tracking.xcom_ttb_penalty; param values in config.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.foot_body_indexes = _resolve_foot_body_indexes(env)
        self.body_masses = env.simulator.get_body_masses()  # type: ignore[attr-defined]
        self.total_mass = self.body_masses.sum().clamp_min(1e-6)
        p = cfg.params
        self.threshold = p.get("threshold", 1.0)
        self.single_support_ttb_threshold = p.get("single_support_ttb_threshold", 0.30)
        self.double_support_ttb_threshold = p.get("double_support_ttb_threshold", 0.20)
        self.v_min = p.get("v_min", 0.01)
        self.gravity = p.get("gravity", 9.81)
        self.com_height_floor = p.get("com_height_floor", 0.25)
        self.max_penalty = p.get("max_penalty", 0.09)
        self.polygon_b = torch.tensor(
            p.get("foot_polygon", _G1_FOOT_SUPPORT_POLYGON), dtype=torch.float32, device=env.device
        )

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        com_pos, com_vel = _whole_body_com_state(env, self.body_masses, self.total_mass)
        xcom_xy = _xcom_xy(com_pos, com_vel, self.gravity, self.com_height_floor)
        margin, single_support, double_support = _support_polygon_margin(
            env, self.foot_body_indexes, xcom_xy, self.threshold, self.polygon_b
        )
        support_mask = _support_contact_mask(env, self.foot_body_indexes, self.threshold)
        foot_xy = env.simulator._rigid_body_pos[:, self.foot_body_indexes, :2]
        center = 0.5 * (foot_xy[:, 0] + foot_xy[:, 1])
        left_only = support_mask[:, 0] & ~support_mask[:, 1]
        right_only = support_mask[:, 1] & ~support_mask[:, 0]
        if torch.any(left_only):
            center[left_only] = foot_xy[left_only, 0]
        if torch.any(right_only):
            center[right_only] = foot_xy[right_only, 1]
        com_height = com_pos[:, 2].clamp_min(float(self.com_height_floor))
        omega = torch.sqrt(torch.full_like(com_height, float(self.gravity)) / com_height)
        xcom_dist = torch.norm(xcom_xy - center, dim=-1)
        v_xcom = (omega * xcom_dist).clamp_min(float(self.v_min))
        ttb = margin / v_xcom
        ttb_threshold = torch.zeros_like(margin)
        ttb_threshold[single_support] = float(self.single_support_ttb_threshold)
        ttb_threshold[double_support] = float(self.double_support_ttb_threshold)
        active = single_support | double_support
        penalty = torch.zeros_like(margin)
        penalty[active] = torch.relu(ttb_threshold[active] - ttb[active]) ** 2
        if self.max_penalty is not None and self.max_penalty > 0.0:
            penalty = penalty.clamp(max=float(self.max_penalty))
        return penalty

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class SingleSupportFootSlipPenalty(RewardTermBase):
    """During single-leg support, penalize the support foot sliding: xy speed^2 while in contact.

    The reference support STATE selects which foot should be the sole support (1=left, 2=right);
    the penalty is gated by actual contact. New term (no WBT equivalent); design per request.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.foot_body_indexes = _resolve_foot_body_indexes(env)
        self.force_threshold = cfg.params.get("force_threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        state = motion_command.reference_support_state  # (N,) 0=double 1=left 2=right 3=flight
        foot_vel_xy = env.simulator._rigid_body_vel[:, self.foot_body_indexes, :2]  # (N, 2, 2)
        speed_sq = foot_vel_xy.square().sum(dim=-1)  # (N, 2)
        normal_force = torch.relu(
            env.simulator.contact_forces_history[:, :, self.foot_body_indexes, 2]
        ).max(dim=1)[0]
        contact = (normal_force > self.force_threshold).to(speed_sq.dtype)  # (N, 2)
        # single-support gate: column 0 active iff state==left-only, column 1 iff state==right-only
        support_gate = torch.stack(
            [(state == 1).to(speed_sq.dtype), (state == 2).to(speed_sq.dtype)], dim=-1
        )  # (N, 2)
        return torch.sum(speed_sq * contact * support_gate, dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class StanceAnkleActionRatePenalty(RewardTermBase):
    """During single-leg support, penalize the action-rate of the STANCE foot's ankle joints.

    Faithful port of whole_body_tracking.stance_ankle_action_rate_penalty. Gates on ACTUAL
    single-support contact (left_only / right_only, exactly like WBT), then penalizes the sum of
    squared action deltas on that foot's ankle action dims. Ankle action indices are resolved by
    joint name against env.simulator.dof_names (action index i <-> dof_names[i]). Values in config.

    NOTE: WBT (and this port) gate on ACTUAL contact; the sibling SingleSupportFootSlipPenalty
    instead gates on the REFERENCE support state. Swap _support_contact_mask for the reference
    state if you want both to use the same gate.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.foot_body_indexes = _resolve_foot_body_indexes(env)
        dof_names = list(env.simulator.dof_names)  # type: ignore[attr-defined]  # action index i <-> dof_names[i]
        left_names = cfg.params["left_joint_names"]
        right_names = cfg.params["right_joint_names"]
        self.left_action_ids = torch.tensor(
            [dof_names.index(n) for n in left_names], dtype=torch.long, device=env.device
        )
        self.right_action_ids = torch.tensor(
            [dof_names.index(n) for n in right_names], dtype=torch.long, device=env.device
        )
        self.threshold = cfg.params.get("threshold", 1.0)
        joint_weights = cfg.params.get("joint_weights")
        if joint_weights is None:
            self.joint_weights = torch.ones(self.left_action_ids.numel(), dtype=torch.float32, device=env.device)
        else:
            self.joint_weights = torch.tensor(joint_weights, dtype=torch.float32, device=env.device)
        assert self.left_action_ids.numel() == self.right_action_ids.numel() == self.joint_weights.numel(), (
            "left/right ankle joint lists and joint_weights must have equal length."
        )

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        support_mask = _support_contact_mask(env, self.foot_body_indexes, self.threshold)  # (N, 2) actual contact
        left_only = support_mask[:, 0] & ~support_mask[:, 1]
        right_only = support_mask[:, 1] & ~support_mask[:, 0]
        action_rate = env.action_manager.action - env.action_manager.prev_action
        penalty = torch.zeros(env.num_envs, device=action_rate.device, dtype=action_rate.dtype)
        if torch.any(left_only):
            penalty[left_only] = torch.sum(
                torch.square(action_rate[left_only][:, self.left_action_ids]) * self.joint_weights, dim=-1
            )
        if torch.any(right_only):
            penalty[right_only] = torch.sum(
                torch.square(action_rate[right_only][:, self.right_action_ids]) * self.joint_weights, dim=-1
            )
        return penalty

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class StanceKneeActionRatePenalty(StanceAnkleActionRatePenalty):
    """During single-leg support, penalize the STANCE leg's KNEE action-rate.

    Structurally identical to StanceAnkleActionRatePenalty (single-support gating + per-joint action-rate
    on the stance foot), just applied to the knee joint(s) supplied via config. Separate subclass only for
    a clear per-term name in configs/logs; it inherits the ankle implementation verbatim. Ankle-over-knee
    weighting is intended, so keep this term's weight below the ankle term's (-0.3).
    """


class ActionJerkPenalty(RewardTermBase):
    """Penalize the second difference of actions (action "jerk"): (a_t - a_{t-1}) - (a_{t-1} - a_{t-2}).

    Complements action_rate. action_rate (first difference) penalizes ALL action change, so a heavier
    weight over-smooths and hurts tracking of dynamic motion. Jerk
    (second difference) is ~0 for smooth motion of ANY speed and only spikes on high-frequency reversals
    -- it targets the foot-chatter failure mode directly, without the over-regularization of a heavier
    action_rate. Global: all action dims, every step.

    Stateful: caches the previous step's action-rate, so it never needs a_{t-2} from the action manager.
    The cache is zeroed per-env on episode reset (the action manager likewise zeros prev_action on reset),
    so jerk does not spike across the reset boundary beyond the same one-step transient action_rate has.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        action_dim = env.action_manager.action.shape[1]
        self._prev_action_rate = torch.zeros((env.num_envs, action_dim), device=env.device)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        action_rate = env.action_manager.action - env.action_manager.prev_action
        jerk = action_rate - self._prev_action_rate
        self._prev_action_rate.copy_(action_rate.detach())
        return torch.sum(torch.square(jerk), dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_action_rate.zero_()
        else:
            self._prev_action_rate[env_ids] = 0.0
