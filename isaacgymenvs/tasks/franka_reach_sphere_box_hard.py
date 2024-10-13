import numpy as np
import os

# Set CUDA_VISIBLE_DEVICES

import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine
from .base.vec_task import VecTask



# [Kunal] This is the next step after adding box asset to the environment.
#         In this, the ball asset will be transformed into a sphere object.
class FrankaReachSphereBoxHard(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"] *2

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        self.reward_settings = {
            'r_dist_reward_scale': self.cfg["env"]["distRewardScale"],
            'r_rot_reward_scale': self.cfg["env"]["rotRewardScale"],
            'r_around_ball_reward_scale': self.cfg["env"]["aroundBallRewardScale"],
            'r_open_reward_scale': self.cfg["env"]["openRewardScale"],
            'r_finger_dist_reward_scale': self.cfg["env"]["fingerDistRewardScale"],
            'r_action_penalty_scale': self.cfg["env"]["actionPenaltyScale"],
        }

        self.states = {}

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        self.cfg["env"]["numObservations"] = 21
        self.cfg["env"]["numActions"] = 9

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
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035],
                                               device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        # [Kunal] - Get box dofs & ball dofs
        self.box_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.box_dof_pos = self.box_dof_state[..., 0]
        self.box_dof_vel = self.box_dof_state[..., 1]
        self.ball_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.ball_dof_pos = self.ball_dof_state[..., 0]
        self.ball_dof_vel = self.ball_dof_state[..., 1]
        # --------------------------------------------


        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        print("rigid_body_states shape: ", self.rigid_body_states.shape)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # [Kunal] - global_indices keeps track of amount of actors in the environment.
        #           Make sure this corresponds to the number of actors in the environment.
        self.global_indices = torch.arange(self.num_envs * (3), dtype=torch.int32,
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
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        # [Kunal] Added this line to load box & ball asset
        box_asset_file = "urdf/sektion_cabinet_model/urdf/box_scaled_40.urdf"
        ball_asset_file = "urdf/ball.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

            # [Kunal] Added this line to load box asset & ball asset
            box_asset_file = self.cfg["env"]["asset"].get("assetFileNameBox", box_asset_file)
            ball_asset_file = self.cfg["env"]["asset"].get("assetFileNameBall", ball_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # [Kunal] load box asset
        asset_options.thickness = 0.001
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        box_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file, asset_options)
        # ----------------------------
        # [Kunal] load ball asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, asset_options)
        # ----------------------------

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e2, 1.0e2], dtype=torch.float,
                                        device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        # [Kunal] - Get number of box bodies and dofs
        self.num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)
        self.num_box_dofs = self.gym.get_asset_dof_count(box_asset)
        # --------------------------------------------
        # [Kunal] - Get number of ball bodies and dofs
        self.num_ball_bodies = self.gym.get_asset_rigid_body_count(ball_asset)
        self.num_ball_dofs = self.gym.get_asset_dof_count(ball_asset)
        # --------------------------------------------

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # [Kunal] - Print number of box bodies and dofs
        print("num box bodies: ", self.num_box_bodies)
        print("num box dofs: ", self.num_box_dofs)
        # --------------------------------------------
        # [Kunal] - Print number of ball bodies and dofs
        print("num ball bodies: ", self.num_ball_bodies)
        print("num ball dofs: ", self.num_ball_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # [Kunal] - Set box dof properties
        box_dof_props = self.gym.get_asset_dof_properties(box_asset)
        for i in range(self.num_box_dofs):
            box_dof_props['damping'][i] = 10.0

        # [Kunal] - Set ball dof properties
        ball_dof_props = self.gym.get_asset_dof_properties(ball_asset)
        for i in range(self.num_ball_dofs):
            ball_dof_props['damping'][i] = 10.0
        # --------------------------------------------

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.85, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # [Kunal] - Box & Ball start pose
        box_start_pose = gymapi.Transform()
        box_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        box_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # --------------------------------------------

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)

        # [Kunal] - Get number of box & ball bodies and shapes
        num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)
        num_box_shapes = self.gym.get_asset_rigid_shape_count(box_asset)
        num_ball_bodies = self.gym.get_asset_rigid_body_count(ball_asset)
        num_ball_shapes = self.gym.get_asset_rigid_shape_count(ball_asset)
        # --------------------------------------------

        max_agg_bodies = num_franka_bodies + num_box_bodies + num_ball_bodies
        max_agg_shapes = num_franka_shapes + num_box_shapes + num_ball_shapes

        self.frankas = []
        self.boxes = []
        self.balls = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            # [Kunal] - Create box actor
            box_pose = box_start_pose
            box_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0
            dy = np.random.rand() - 0.5
            box_pose.p.y += self.start_position_noise * dy
            box_pose.p.z += self.start_position_noise * dz
            box_actor = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, box_actor, box_dof_props)

            # [Kunal] - Create ball actor
            ball_pose = ball_start_pose
            ball_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            # dz = 0.5 * np.random.rand()
            dy = 1
            ball_pose.p.y += self.start_position_noise * dy
            # ball_pose.p.z += self.start_position_noise * dz
            ball_pose.p = gymapi.Vec3(*get_axis_params(0.25, self.up_axis_idx))
            ball_actor = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, ball_actor, ball_dof_props)

            # # [Kunal] - Add marker
            # marker_position = gymapi.Vec3(0.0, 0.0, 0.25)
            # self.draw_debug_marker(marker_position)
            # --------------------------------------------

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

            # [Kunal] - Append box actor to boxes & balls list
            self.boxes.append(box_actor)
            self.balls.append(ball_actor)
            # --------------------------------------------

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.ball_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ball_actor, "ball")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.finger_grip_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_grip_site")

        self.init_data()

    def init_data(self):
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger_tip")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger_tip")
        finger_grip = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_grip_site")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)
        finger_grip_pose = self.gym.get_rigid_transform(self.envs[0], finger_grip)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        franka_local_grasp_pose = finger_pose * hand_pose_inv
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.001, grasp_pose_axis))
        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w],
                                               device=self.device).repeat((self.num_envs, 1))

        # [Kunal] - Ball grasp pose
        ball_local_grasp_pose = gymapi.Transform()
        ball_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.0, grasp_pose_axis, 0.0))
        ball_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_local_grasp_pos = to_torch([ball_local_grasp_pose.p.x, ball_local_grasp_pose.p.y,
                                                  ball_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.ball_local_grasp_rot = to_torch([ball_local_grasp_pose.r.x, ball_local_grasp_pose.r.y,
                                                  ball_local_grasp_pose.r.z, ball_local_grasp_pose.r.w],
                                                 device=self.device).repeat((self.num_envs, 1))

        # # [Kunal] - Gripper axes (assuming the forward direction is along the -X axis and up direction is along the Z axis)
        # self.gripper_forward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.gripper_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        #
        # # [Kunal] - Ball axes (assuming the ball is vertical, with inward direction along the -X axis and up direction along the Z axis)
        # self.ball_inward_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        # self.ball_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        #
        # self.states.update({
        #     'gripper_forward_axis': self.gripper_forward_axis,
        #     'gripper_up_axis': self.gripper_up_axis,
        #     'ball_inward_axis': self.ball_inward_axis,
        #     'ball_up_axis': self.ball_up_axis
        # })

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  #

        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        # [Kunal] - Finger grip pos
        self.finger_grip_pos = self.rigid_body_states[:, self.finger_grip_handle][:, 0:3]
        self.finger_grip_rot = self.rigid_body_states[:, self.finger_grip_handle][:, 3:7]

        # [Kunal] - Ball grasp pos
        self.ball_grasp_pos = self.rigid_body_states[:, self.ball_handle][:, 0:3]
        self.ball_grasp_rot = self.rigid_body_states[:, self.ball_handle][:, 3:7]



    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
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
            'franka_grasp_pos': self.rigid_body_states[:, self.hand_handle][:, 0:3],
            'ball_grasp_pos': self.rigid_body_states[:, self.ball_handle][:, 0:3],
            'finger_grip_pos': self.rigid_body_states[:, self.finger_grip_handle][:, 0:3],
            'franka_grasp_rot': self.rigid_body_states[:, self.hand_handle][:, 3:7],
            'ball_grasp_rot': self.rigid_body_states[:, self.ball_handle][:, 3:7],
            'finger_grip_rot': self.rigid_body_states[:, self.finger_grip_handle][:, 3:7],
            'franka_lfinger_pos': self.rigid_body_states[:, self.lfinger_handle][:, 0:3],
            'franka_rfinger_pos': self.rigid_body_states[:, self.rfinger_handle][:, 0:3],
        })


    def compute_observations(self):

        self._refresh()

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        to_target = self.ball_grasp_pos - self.finger_grip_pos

        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, to_target),
                                 dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (
                    torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:,
                  :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

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
def compute_franka_reward(
        reset_buf, progress_buf, actions,
        states, num_envs, reward_settings, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], int, Dict[str, float], float, float) -> Tuple[Tensor, Tensor]


    # [Kunal] - Compute distance reward
    d = torch.norm(states['finger_grip_pos'] - states['ball_grasp_pos'], dim=-1)
    d_lf = torch.norm(states['franka_lfinger_pos'] - states['ball_grasp_pos'], dim=-1)
    d_rf = torch.norm(states['franka_rfinger_pos'] - states['ball_grasp_pos'], dim=-1)

    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
    dist_reward = torch.where(d <= 0.05, dist_reward * 2, dist_reward)

    # [Kunal] - regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # [Kunal] - Reward for opening the gripper when it is close to the ball
    d_gripper = torch.norm(states['franka_lfinger_pos'] - states['franka_rfinger_pos'], dim=-1)
    open_gripper_reward = torch.where(d_gripper > 0.03, 1.0, 0.0)

    # print(f"Finger Distance: {d[0]}")

    rewards = (dist_reward * reward_settings['r_dist_reward_scale'] -
               action_penalty * reward_settings['r_action_penalty_scale'] +
               open_gripper_reward * reward_settings['r_open_reward_scale'])

#     print(f"Rewards: {rewards[0]}")

    # [Kunal] - Reset when max episodes reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
