# ---------------------------------------------------------------------------------------------------
# Modified from the Holosoma framework (Amazon FAR): https://github.com/amazon-far/holosoma
# Copyright Amazon.com, Inc. or its affiliates. Licensed under the Apache License, Version 2.0.
# This file was CHANGED for DDC: it carries the deployable support-relative dynamic-CoM observation
# and/or the human-science balance reward library and its config (see training/TRAINING.md).
# The Apache-2.0 license text is in the repository LICENSE; attribution is in training/NOTICE.
# ---------------------------------------------------------------------------------------------------

"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def _apply_imu_ou(env: WholeBodyTrackingManager, vec_base: torch.Tensor) -> torch.Tensor:
    """Apply the shared IMU orientation-error (OU noise ``env._imu_ou_euler``) to a base-frame vector.

    HuB (arXiv:2505.07294) idea: a single temporally-correlated root-orientation error should perturb
    ALL orientation-derived observations together (coupled), unlike independent per-term white noise.
    Small-angle rotation R(δ)·v ≈ v + δ × v, so projected_gravity and base_ang_vel get the SAME δ.
    ``_imu_ou_euler`` is advanced once per step in the env and zeroed during eval -> this is a no-op there.
    """
    ou = getattr(env, "_imu_ou_euler", None)
    if ou is None:
        return vec_base
    return vec_base + torch.cross(ou, vec_base, dim=-1)


def base_ang_vel_ou(env: WholeBodyTrackingManager) -> torch.Tensor:
    """base_ang_vel + coupled IMU OU orientation noise (actor-only; critic keeps the clean term)."""
    return _apply_imu_ou(env, get_base_ang_vel(env))


def projected_gravity_ou(env: WholeBodyTrackingManager) -> torch.Tensor:
    """projected_gravity + coupled IMU OU orientation noise (actor-only; critic keeps the clean term)."""
    return _apply_imu_ou(env, get_projected_gravity(env))


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def reference_support_phase(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Reference support phase (soft per-foot contact [0,1], columns [left, right]) at the current frame. (N, 2)"""
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.reference_support_phase


def future_support_phase(env: WholeBodyTrackingManager, num_future_frames: int = 10) -> torch.Tensor:
    """Reference support phase for t+1 ... t+K, concatenated. Shape (N, K*2)."""
    motion_command = _get_motion_command_and_assert_type(env)
    phases = [motion_command.future_support_phase(k) for k in range(1, num_future_frames + 1)]
    return torch.cat(phases, dim=-1)


def future_cmd(env: WholeBodyTrackingManager, num_future_frames: int = 10) -> torch.Tensor:
    """Reference [joint_pos, joint_vel] for t+1 ... t+K, concatenated. Shape (N, K*2*num_joints)."""
    motion_command = _get_motion_command_and_assert_type(env)
    frames = []
    for k in range(1, num_future_frames + 1):
        jp = motion_command.future_joint_pos(k)
        jv = motion_command.future_joint_vel(k)
        frames.append(torch.cat([jp, jv], dim=-1))
    return torch.cat(frames, dim=-1)


# Dynamic vs static CoM: the actor and critic independently control whether the xCoM uses the
# velocity-extrapolation term. True = dynamic (xCoM = CoM + velocity extrapolation, capture-point/LIP);
# False = static (pure geometric CoM, drops the velocity term). The two flags are independent, so the
# actor observation can be varied without changing the critic. (The polygon / xcom_ttb rewards are not
# controlled by these flags -- they always use the dynamic term.)
ACTOR_XCOM_USE_VELOCITY = True   # controls the actor whole_body_com_rel_support_center (deployable); False -> return position only (N,2)
CRITIC_XCOM_USE_VELOCITY = True  # controls the critic whole_body_xcom_rel_support_center (privileged / training-only)


def whole_body_xcom_rel_support_center(
    env: WholeBodyTrackingManager,
    threshold: float = 1.0,
    gravity: float = 9.81,
    com_height_floor: float = 0.25,
) -> torch.Tensor:
    """xCoM relative to the support-foot center, in base frame (xy). Shape (N, 2).

    PRIVILEGED / CRITIC-ONLY: built from the whole-body mass-weighted CoM + CoM velocity, which are
    NOT reliably observable in world frame on the real robot (base linear velocity has no direct
    sensor). Keep this OUT of the deployed actor obs to avoid a sim2real gap. Faithful port of
    whole_body_tracking.whole_body_xcom_rel_support_center_b: support center = contact-weighted mean
    of the in-contact feet (fallback = mean of both feet); vector rotated into the base frame.
    """
    from holosoma.managers.reward.terms.wbt import (
        _resolve_foot_body_indexes,
        _support_contact_mask,
        _whole_body_com_state,
        _xcom_xy,
    )

    foot_ids = _resolve_foot_body_indexes(env)
    masses = getattr(env, "_wb_body_masses_cache", None)
    if masses is None:
        masses = env.simulator.get_body_masses()
        env._wb_body_masses_cache = masses
        env._wb_total_mass_cache = masses.sum().clamp_min(1e-6)
    com_pos, com_vel = _whole_body_com_state(env, masses, env._wb_total_mass_cache)
    xcom_xy = _xcom_xy(com_pos, com_vel, gravity, com_height_floor, use_velocity=CRITIC_XCOM_USE_VELOCITY)

    foot_xy = env.simulator._rigid_body_pos[:, foot_ids, :2]  # (N, 2, 2)
    support_mask = _support_contact_mask(env, foot_ids, threshold)  # (N, 2) bool
    w = support_mask.to(foot_xy.dtype).unsqueeze(-1)
    weighted = (foot_xy * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    fallback = foot_xy.mean(dim=1)
    has_support = torch.any(support_mask, dim=1, keepdim=True)
    center_xy = torch.where(has_support, weighted, fallback)

    rel_w = torch.zeros((env.num_envs, 3), device=xcom_xy.device, dtype=xcom_xy.dtype)
    rel_w[:, :2] = xcom_xy - center_xy
    return quat_rotate_inverse(_base_quat(env), rel_w, w_last=True)[:, :2]


def whole_body_com_rel_support_center(
    env: WholeBodyTrackingManager, double_support_height_diff: float = 0.03
) -> torch.Tensor:
    """CoM-rel-support-center POSITION + relative VELOCITY (base frame, xy). (N, 4).

    DEPLOYABLE actor obs: both are RELATIVE (CoM minus support center), so base LINEAR velocity and
    absolute base position CANCEL in the difference -> computable from joint encoders + IMU (no
    base-velocity estimate / leg odometry needed). Position = pure FK; velocity = CoM_vel - support_center_vel
    (during stance the foot is ~stationary, so this ~= world CoM velocity = the capture-point/xCoM velocity
    term). Residual sim2real gap is only q-dot noise + CoM mass-model error (add DR noise before deploy).
    Support center = mean of in-contact feet; support is selected from gravity-aligned FOOT HEIGHT
    (lower foot supports; both within ``double_support_height_diff`` -> double support), matching the
    deploy-side rule EXACTLY (sim2sim/sim2real) so there is no train->deploy gap at foot lift/land.
    """
    from holosoma.managers.reward.terms.wbt import (
        _resolve_foot_body_indexes,
        _support_height_mask,
        _whole_body_com_state,
    )

    foot_ids = _resolve_foot_body_indexes(env)
    masses = getattr(env, "_wb_body_masses_cache", None)
    if masses is None:
        masses = env.simulator.get_body_masses()
        env._wb_body_masses_cache = masses
        env._wb_total_mass_cache = masses.sum().clamp_min(1e-6)
    com_pos, com_vel = _whole_body_com_state(env, masses, env._wb_total_mass_cache)

    foot_pos = env.simulator._rigid_body_pos[:, foot_ids, :2]  # (N, 2, 2)
    foot_vel = env.simulator._rigid_body_vel[:, foot_ids, :2]  # (N, 2, 2)
    # Height-based support (matches deploy) instead of contact force -> no train->deploy gap.
    support_mask = _support_height_mask(env, foot_ids, double_support_height_diff)  # (N, 2) bool
    w = support_mask.to(foot_pos.dtype).unsqueeze(-1)
    wsum = w.sum(dim=1).clamp_min(1.0)
    has_support = torch.any(support_mask, dim=1, keepdim=True)
    center_pos = torch.where(has_support, (foot_pos * w).sum(dim=1) / wsum, foot_pos.mean(dim=1))

    rel_pos_w = torch.zeros((env.num_envs, 3), device=com_pos.device, dtype=com_pos.dtype)
    rel_pos_w[:, :2] = com_pos[:, :2] - center_pos
    pos_b = quat_rotate_inverse(_base_quat(env), rel_pos_w, w_last=True)[:, :2]

    # ACTOR_XCOM_USE_VELOCITY=False -> return only the CoM-rel-support position (N,2), dropping the
    # dynamic relative-velocity term below.
    if not ACTOR_XCOM_USE_VELOCITY:
        return pos_b  # (N, 2): static CoM position (base frame)

    # Direct relative velocity (CoM - support center): base LINEAR velocity cancels in the difference,
    # so deployable from encoders+IMU (no base-vel estimate). No buffer -> immune to the obs-runs-twice
    # bug. During stance (foot ~stationary) it ~= world CoM velocity = the xCoM/capture-point velocity term.
    center_vel = torch.where(has_support, (foot_vel * w).sum(dim=1) / wsum, foot_vel.mean(dim=1))
    rel_vel_w = torch.zeros((env.num_envs, 3), device=com_vel.device, dtype=com_vel.dtype)
    rel_vel_w[:, :2] = com_vel[:, :2] - center_vel
    vel_b = quat_rotate_inverse(_base_quat(env), rel_vel_w, w_last=True)[:, :2]
    return torch.cat([pos_b, vel_b], dim=-1)  # (N, 4): [CoM-rel-support position, relative velocity], base frame


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)
