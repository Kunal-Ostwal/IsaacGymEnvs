import numpy as np
import os

# Set CUDA_VISIBLE_DEVICES

import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine
from .base.vec_task import VecTask

class KinovaBoxReacher(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.arm_dof_noise = self.cfg["env"]["armDofNoise"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        # [Kunal] - Curriculum & BO variables
        self.curr_joint_stiffness = 0.0
        self.curr_joint_damping = 0.0

        self.horizon_length = 16

        self.theta1 = self.cfg["env"]["theta1"]
        self.theta2 = self.cfg["env"]["theta2"]

        self.alpha1 = self.cfg["env"]["alpha1"]
        self.alpha2 = self.cfg["env"]["alpha2"]

        self.c1 = self.cfg["env"]["c1"]
        self.c2 = self.cfg["env"]["c2"]

        self.final_stiffness = self.cfg["env"]["finalStiffness"]
        self.final_damping = self.cfg["env"]["finalDamping"]

        self.epoch = 0.0

        self.reward_settings = {
            'r_dist_reward_scale': self.cfg["env"]["distRewardScale"],
            'r_rot_reward_scale': self.cfg["env"]["rotRewardScale"],
            'r_around_ball_reward_scale': self.cfg["env"]["aroundBallRewardScale"],
            'r_action_penalty_scale': self.cfg["env"]["actionPenaltyScale"],
            'r_bonus_dist_reward_scale': self.cfg["env"]["bonusDistRewardScale"],
            'r_penalty_dist_reward_scale': self.cfg["env"]["penaltyDistRewardScale"],
        }

        self.states = {}

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        self.cfg["env"]["numObservations"] = 27
        self.cfg["env"]["numActions"] = 12  # 6 joints
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.arm_default_dof_pos = to_torch([0.0, 3.0], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_dof_state = self.dof_state.view(self.num_envs, -1, 2)

        self.arm_dof_pos = self.arm_dof_state[..., 0]
        self.arm_dof_vel = self.arm_dof_state[..., 1]

        # Get box dofs & ball dofs
        # self.box_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, 2:2]  # No DOFs for box
        # self.ball_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, 2:2]  # No DOFs for ball
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.arm_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # Global indices: arm, box, ball
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        print("global_indices: ", self.global_indices.shape)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        arm_asset_file = "kinova_description/urdf/kinova_mod.urdf"

        # Load box and ball assets
        box_asset_file = "urdf/sektion_cabinet_model/urdf/box_scaled_90.urdf"
        ball_asset_file = "urdf/ball.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            arm_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", arm_asset_file)
            box_asset_file = self.cfg["env"]["asset"].get("assetFileNameBox", box_asset_file)
            ball_asset_file = self.cfg["env"]["asset"].get("assetFileNameBall", ball_asset_file)

        # load arm asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        self.arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_file, asset_options)

        # [Kunal] load box asset
        asset_options.thickness = 0.001
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005

        init_box_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file, asset_options)

        # ----------------------------
        # [Kunal] load ball asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, asset_options)
        # ----------------------------

        arm_dof_stiffness = to_torch([800, 800], dtype=torch.float, device=self.device)
        arm_dof_damping = to_torch([160, 160], dtype=torch.float, device=self.device)

        self.num_arm_bodies = self.gym.get_asset_rigid_body_count(self.arm_asset)
        self.num_arm_dofs = self.gym.get_asset_dof_count(self.arm_asset)

        # Get number of box & ball bodies and dofs
        self.num_box_bodies = self.gym.get_asset_rigid_body_count(init_box_asset)
        self.num_box_dofs = self.gym.get_asset_dof_count(init_box_asset)
        self.num_ball_bodies = self.gym.get_asset_rigid_body_count(ball_asset)
        self.num_ball_dofs = self.gym.get_asset_dof_count(ball_asset)

        print("num arm bodies: ", self.num_arm_bodies)
        print("num arm dofs: ", self.num_arm_dofs)
        print("num box bodies: ", self.num_box_bodies)
        print("num box dofs: ", self.num_box_dofs)
        print("num ball bodies: ", self.num_ball_bodies)
        print("num ball dofs: ", self.num_ball_dofs)

        # set arm dof properties
        arm_dof_props = self._config_arm_dofs_props()

        # Set box and ball dof properties (if any)
        box_dof_props = self.gym.get_asset_dof_properties(init_box_asset)
        
        for i in range(self.num_box_dofs):
            box_dof_props['damping'][i] = 10.0

        ball_dof_props = self.gym.get_asset_dof_properties(ball_asset)
        for i in range(self.num_ball_dofs):
            ball_dof_props['damping'][i] = 10.0
    
        arm_start_pose = gymapi.Transform()
        arm_start_pose.p = gymapi.Vec3(1.25, 0.0, 1.0)
        arm_start_pose.r = gymapi.Quat(0.0, 1.0, 0.0, 0.0)

        # Box and ball start poses
        box_start_pose = gymapi.Transform()
        box_start_pose.p = gymapi.Vec3(0.0, 0.0, 1000.0)
        box_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(0.5, 0.0, 1.0)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_arm_bodies = self.gym.get_asset_rigid_body_count(self.arm_asset)
        num_arm_shapes = self.gym.get_asset_rigid_shape_count(self.arm_asset)

        num_box_bodies = self.gym.get_asset_rigid_body_count(init_box_asset)
        num_box_shapes = self.gym.get_asset_rigid_shape_count(init_box_asset)
        num_ball_bodies = self.gym.get_asset_rigid_body_count(ball_asset)
        num_ball_shapes = self.gym.get_asset_rigid_shape_count(ball_asset)

        max_agg_bodies = num_arm_bodies + num_ball_bodies + num_box_bodies
        max_agg_shapes = num_arm_shapes + num_ball_shapes + num_box_shapes

        self.arms = []
        self.boxes = []
        self.balls = []
        self.ball_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            arm_actor = self.gym.create_actor(env_ptr, self.arm_asset, arm_start_pose, "arm", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, arm_actor, arm_dof_props)

            # Create box actor
            box_pose = box_start_pose
            box_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0
            dy = np.random.rand() - 0.5
            box_pose.p.y += self.start_position_noise * dy
            box_pose.p.z += self.start_position_noise * dz
            box_actor = self.gym.create_actor(env_ptr, init_box_asset, box_pose, "box", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, box_actor, box_dof_props)

            # Create ball actor
            ball_pose = ball_start_pose
            ball_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dy = np.random.rand() - 0.5
            ball_pose.p.y += self.start_position_noise * dy
            ball_actor = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, ball_actor, ball_dof_props)
            ball_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ball_actor, "ball")

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arms.append(arm_actor)
            self.boxes.append(box_actor)
            self.balls.append(ball_actor)
            self.ball_handles.append(ball_handle)

        self.end_effector_handle = self.gym.find_actor_rigid_body_handle(env_ptr, arm_actor, "kinova_end_effector")
        self.ball_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ball_actor, "ball")

        self.init_data()


    def init_data(self):
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        arm_local_grasp_pose = gymapi.Transform()
        arm_local_grasp_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        arm_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.arm_local_grasp_pos = to_torch([arm_local_grasp_pose.p.x, arm_local_grasp_pose.p.y,
                                             arm_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.arm_local_grasp_rot = to_torch([arm_local_grasp_pose.r.x, arm_local_grasp_pose.r.y,
                                             arm_local_grasp_pose.r.z, arm_local_grasp_pose.r.w],
                                            device=self.device).repeat((self.num_envs, 1))

        self.arm_grasp_pos = torch.zeros_like(self.arm_local_grasp_pos)
        self.arm_grasp_rot = torch.zeros_like(self.arm_local_grasp_rot)
        self.arm_grasp_rot[..., -1] = 1  #

        self.end_effector_pos = self.rigid_body_states[:, self.end_effector_handle][:, 0:3]
        self.end_effector_rot = self.rigid_body_states[:, self.end_effector_handle][:, 3:7]

        # Ball grasp pos
        self.ball_grasp_pos = self.rigid_body_states[:, self.ball_handle][:, 0:3]
        self.ball_grasp_rot = self.rigid_body_states[:, self.ball_handle][:, 3:7]

    def _config_arm_dofs_props(self):
        # get joint limits and ranges for Kinova
        arm_dof_props = self.gym.get_asset_dof_properties(self.arm_asset)
        self.n_dofs = len(arm_dof_props)
        self.arm_lower_limits_pos = to_torch(arm_dof_props['lower'], device=self.device, dtype=torch.float32)
        self.arm_upper_limits_pos = to_torch(arm_dof_props['upper'], device=self.device, dtype=torch.float32)
        self.arm_lower_limits_vel = to_torch(-arm_dof_props['velocity'], device=self.device, dtype=torch.float32)
        self.arm_upper_limits_vel = to_torch(arm_dof_props['velocity'], device=self.device, dtype=torch.float32)

        # set default joint_pos to mid_pos
        self.arm_mids_pos = 0.5 * (self.arm_upper_limits_pos + self.arm_lower_limits_pos)
        self.default_dof_pos = self.arm_mids_pos

        # set dof_props for joint_vel control
        arm_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS  # control with joint pos, but action is vel
        self.action_scale = self.arm_upper_limits_vel  # since actions are cliped between [-1, 1], action scales should be the upper limits
        self.arm_dof_speed_scales = torch.ones_like(self.arm_lower_limits_pos)
        # kinova_dof_props['stiffness'].fill(4000.0)
        # kinova_dof_props['damping'].fill(40)

        return arm_dof_props

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_arm_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.states, self.num_envs, self.reward_settings, self.distX_offset, self.max_episode_length,
        )


    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Refresh rigid body states
        self._update()

    def _update(self):
        self.states.update({
            'end_effector_pos': self.rigid_body_states[:, self.end_effector_handle][:, 0:3],
            'end_effector_rot': self.rigid_body_states[:, self.end_effector_handle][:, 3:7],
            'ball_grasp_pos': self.rigid_body_states[:, self.ball_handle][:, 0:3],
            'ball_grasp_rot': self.rigid_body_states[:, self.ball_handle][:, 3:7],
        })

    def compute_observations(self):

        self._refresh()

        dof_pos_scaled = (2.0 * (self.arm_dof_pos - self.arm_lower_limits_pos)
                          / (self.arm_upper_limits_pos - self.arm_lower_limits_pos) - 1.0)

        to_target = self.ball_grasp_pos - self.end_effector_pos

        self.obs_buf = torch.cat((dof_pos_scaled, self.arm_dof_vel * self.dof_vel_scale, to_target),
                                 dim=-1)
        return self.obs_buf

    def _update_current_epoch(self):
        self.epoch = self.total_train_env_frames / (self.num_envs * self.horizon_length)

    @property
    def _calc_curriculum_step(self):
        self.curr_joint_stiffness = self.alpha1 * (
            1 - (self.epoch/self.c1)
        ) * np.exp(-self.theta1 * self.epoch) + self.final_stiffness

        self.curr_joint_damping = self.alpha2 * (
            1 - (self.epoch/self.c2)
        ) * np.exp(-self.theta2 * self.epoch) + self.final_damping

        return self.curr_joint_stiffness, self.curr_joint_damping

    def _update_curriculum_step(self, env_ids):
        if self.epoch > self.c1:
            new_joint_stiffness = 50.0
            new_joint_damping = 10.0
        else:
            new_joint_stiffness, new_joint_damping = self._calc_curriculum_step

        for i in env_ids:
            # change the asset options on the arm
            env_ptr = self.envs[i]
            arm_actor = self.arms[i]

            dof_props = self.gym.get_actor_dof_properties(env_ptr, arm_actor)
            dof_props['stiffness'][:] = new_joint_stiffness
            dof_props['damping'][:] = new_joint_damping

            self.gym.set_actor_dof_properties(env_ptr, arm_actor, dof_props)

    def reset_idx(self, env_ids):

        # reset arm
        self._reset_arm(env_ids)

        self._update_current_epoch()
        self._update_curriculum_step(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_arm(self, env_ids):
        reset_noise = torch.rand((len(env_ids), self.n_dofs), device=self.device, dtype=torch.float32)
        pos = torch.clamp(self.default_dof_pos.unsqueeze(0) +
                          self.arm_dof_noise * 2.0 * (reset_noise - 0.5),
                          self.arm_lower_limits_pos.unsqueeze(0), self.arm_upper_limits_pos.unsqueeze(0))

        # refresh pos and vel
        self.arm_dof_pos[env_ids, :] = pos
        self.arm_dof_vel[env_ids, :] = torch.zeros_like(self.arm_dof_vel[env_ids])

        # reset kinova
        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        targets = self.arm_dof_targets[:,
                  :self.num_arm_dofs] + self.arm_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.arm_dof_targets[:, :self.num_arm_dofs] = tensor_clamp(
            targets, self.arm_lower_limits_pos, self.arm_upper_limits_pos)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.arm_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_arm_reward(
        reset_buf, progress_buf, actions,
        states, num_envs, reward_settings, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], int, Dict[str, float], float, float) -> Tuple[Tensor, Tensor]
    print(states.keys())
    # Compute distance reward
    d = torch.norm(states['end_effector_pos'] - states['ball_grasp_pos'], dim=-1)
    dist_reward = 1.0 / (1.0 + d)
    print(f"dist_reward: {dist_reward[0]}")
    print(f"d: {d[0]}")

    # If the arm is very close to the ball, give a bonus reward
    bonus_dist_reward = torch.where(d <= 0.1, 1.0, 0.0)

    # If the arm is too far, give a penalty
    penalty_dist_reward = torch.where(d >= 1.5, -1.0, 0.0)

    # Regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = ((dist_reward * reward_settings['r_dist_reward_scale'] -
               action_penalty * reward_settings['r_action_penalty_scale'] +
                bonus_dist_reward * reward_settings['r_bonus_dist_reward_scale']) +
                penalty_dist_reward * reward_settings['r_penalty_dist_reward_scale'])

    print(f"rewards: {rewards[0]}")

    # Reset when max episodes reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # Reset if half of the max episode length is reached and distance is greater than 1.5
    reset_buf = torch.where((progress_buf >= max_episode_length // 2) & (d >= 1.0), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
