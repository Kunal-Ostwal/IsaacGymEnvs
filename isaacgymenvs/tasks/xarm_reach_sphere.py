import numpy as np
import os

# Set CUDA_VISIBLE_DEVICES

import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine
from .base.vec_task import VecTask



# [Kunal] This is the next step after adding box asset to the environment.
#         In this, the cylinder asset will be transformed into a sphere object.
class XArmReachSphere(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"] *2

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_ball_reward_scale = self.cfg["env"]["aroundBallRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.
        # self.cylinder_height = 0.4

        num_obs = 25
        num_acts = 9

        self.cfg["env"]["numObservations"] = 25
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
        self.xarm_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035],
                                               device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.xarm_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.xarm_dofs]
        self.xarm_dof_pos = self.xarm_dof_state[..., 0]
        self.xarm_dof_vel = self.xarm_dof_state[..., 1]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.xarm_dofs:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        # [Kunal] - Get box dofs & ball dofs
        self.box_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.xarm_dofs:]
        self.box_dof_pos = self.box_dof_state[..., 0]
        self.box_dof_vel = self.box_dof_state[..., 1]
        self.ball_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.xarm_dofs:]
        self.ball_dof_pos = self.ball_dof_state[..., 0]
        self.ball_dof_vel = self.ball_dof_state[..., 1]
        # --------------------------------------------


        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        print("rigid_body_states shape: ", self.rigid_body_states.shape)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.xarm_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # [Kunal] - global_indices keeps track of amount of actors in the environment.
        #           Make sure this corresponds to the number of actors in the environment.
        self.global_indices = torch.arange(self.num_envs * (4), dtype=torch.int32,
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
        xarm_asset_file = "urdf/xarm/xarm6_with_gripper.urdf"
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        # [Kunal] Added this line to load box & ball asset
        box_asset_file = "urdf/sektion_cabinet_model/urdf/box_scaled_20.urdf"
        ball_asset_file = "urdf/ball.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            xarm_asset_file = self.cfg["env"]["asset"].get("assetFileNameXArm", xarm_asset_file)
            cabinet_asset_file = self.cfg["env"]["asset"].get("assetFileNameCabinet", cabinet_asset_file)

            # [Kunal] Added this line to load box asset & ball asset
            box_asset_file = self.cfg["env"]["asset"].get("assetFileNameBox", box_asset_file)
            ball_asset_file = self.cfg["env"]["asset"].get("assetFileNameBall", ball_asset_file)

        # load xarm asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        xarm_asset = self.gym.load_asset(self.sim, asset_root, xarm_asset_file, asset_options)

        # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

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

        xarm_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float,
                                        device=self.device)
        xarm_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_xarm_bodies = self.gym.get_asset_rigid_body_count(xarm_asset)
        self.num_xarm_dofs = self.gym.get_asset_dof_count(xarm_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        # [Kunal] - Get number of box bodies and dofs
        self.num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)
        self.num_box_dofs = self.gym.get_asset_dof_count(box_asset)
        # --------------------------------------------
        # [Kunal] - Get number of ball bodies and dofs
        self.num_ball_bodies = self.gym.get_asset_rigid_body_count(ball_asset)
        self.num_ball_dofs = self.gym.get_asset_dof_count(ball_asset)
        # --------------------------------------------

        print("num xarm bodies: ", self.num_xarm_bodies)
        print("num xarm dofs: ", self.num_xarm_dofs)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

        # [Kunal] - Print number of box bodies and dofs
        print("num box bodies: ", self.num_box_bodies)
        print("num box dofs: ", self.num_box_dofs)
        # --------------------------------------------
        # [Kunal] - Print number of ball bodies and dofs
        print("num ball bodies: ", self.num_ball_bodies)
        print("num ball dofs: ", self.num_ball_dofs)

        # set xarm dof properties
        xarm_dof_props = self.gym.get_asset_dof_properties(xarm_asset)
        self.xarm_dof_lower_limits = []
        self.xarm_dof_upper_limits = []
        for i in range(self.num_xarm_dofs):
            xarm_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                xarm_dof_props['stiffness'][i] = xarm_dof_stiffness[i]
                xarm_dof_props['damping'][i] = xarm_dof_damping[i]
            else:
                xarm_dof_props['stiffness'][i] = 7000.0
                xarm_dof_props['damping'][i] = 50.0

            self.xarm_dof_lower_limits.append(xarm_dof_props['lower'][i])
            self.xarm_dof_upper_limits.append(xarm_dof_props['upper'][i])

        self.xarm_dof_lower_limits = to_torch(self.xarm_dof_lower_limits, device=self.device)
        self.xarm_dof_upper_limits = to_torch(self.xarm_dof_upper_limits, device=self.device)
        self.xarm_dof_speed_scales = torch.ones_like(self.xarm_dof_lower_limits)
        self.xarm_dof_speed_scales[[7, 8]] = 0.1
        xarm_dof_props['effort'][7] = 200
        xarm_dof_props['effort'][8] = 200

        # # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        # [Kunal] - Set box dof properties
        box_dof_props = self.gym.get_asset_dof_properties(box_asset)
        for i in range(self.num_box_dofs):
            box_dof_props['damping'][i] = 10.0

        # [Kunal] - Set ball dof properties
        ball_dof_props = self.gym.get_asset_dof_properties(ball_asset)
        for i in range(self.num_ball_dofs):
            ball_dof_props['damping'][i] = 10.0
        # --------------------------------------------

        xarm_start_pose = gymapi.Transform()
        xarm_start_pose.p = gymapi.Vec3(0.85, 0.0, 0.0)
        xarm_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(500.4, self.up_axis_idx))

        # [Kunal] - Box & Ball start pose
        box_start_pose = gymapi.Transform()
        box_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        box_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 1.5, 0.0)

        # --------------------------------------------

        # compute aggregate size
        num_xarm_bodies = self.gym.get_asset_rigid_body_count(xarm_asset)
        num_xarm_shapes = self.gym.get_asset_rigid_shape_count(xarm_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)

        # [Kunal] - Get number of box & ball bodies and shapes
        num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)
        num_box_shapes = self.gym.get_asset_rigid_shape_count(box_asset)
        num_ball_bodies = self.gym.get_asset_rigid_body_count(ball_asset)
        num_ball_shapes = self.gym.get_asset_rigid_shape_count(ball_asset)
        # --------------------------------------------

        max_agg_bodies = num_xarm_bodies + num_cabinet_bodies + num_box_bodies + num_ball_bodies
        max_agg_shapes = num_xarm_shapes + num_cabinet_shapes + num_box_shapes + num_ball_shapes

        self.xarms = []
        self.cabinets = []
        self.boxes = []
        self.balls = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            xarm_actor = self.gym.create_actor(env_ptr, xarm_asset, xarm_start_pose, "xarm", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, xarm_actor, xarm_dof_props)


            cabinet_pose = cabinet_start_pose
            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)

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
            # cylinder_pose.p.z += self.start_position_noise * dz
            ball_pose.p = gymapi.Vec3(*get_axis_params(0.25, self.up_axis_idx))
            ball_actor = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, 4, 0)
            self.gym.set_actor_dof_properties(env_ptr, ball_actor, ball_dof_props)
            # --------------------------------------------

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.xarms.append(xarm_actor)
            self.cabinets.append(cabinet_actor)

            # [Kunal] - Append box actor to boxes & balls list
            self.boxes.append(box_actor)
            self.balls.append(ball_actor)
            # --------------------------------------------

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, xarm_actor, "xarm_gripper_base_link")
        self.ball_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ball_actor, "ball")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, xarm_actor, "left_finger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, xarm_actor, "right_finger")

        # self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(
        #     self.num_envs, self.num_props, 13)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.xarms[0], "xarm_gripper_base_link")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.xarms[0], "left_finger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.xarms[0], "right_finger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        xarm_local_grasp_pose = hand_pose_inv * finger_pose
        xarm_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.001, grasp_pose_axis))
        self.xarm_local_grasp_pos = to_torch([xarm_local_grasp_pose.p.x, xarm_local_grasp_pose.p.y,
                                                xarm_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.xarm_local_grasp_rot = to_torch([xarm_local_grasp_pose.r.x, xarm_local_grasp_pose.r.y,
                                                xarm_local_grasp_pose.r.z, xarm_local_grasp_pose.r.w],
                                               device=self.device).repeat((self.num_envs, 1))

        # [Kunal] - Ball grasp pose
        ball_local_grasp_pose = gymapi.Transform()
        ball_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        ball_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_local_grasp_pos = to_torch([ball_local_grasp_pose.p.x, ball_local_grasp_pose.p.y,
                                                  ball_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.ball_local_grasp_rot = to_torch([ball_local_grasp_pose.r.x, ball_local_grasp_pose.r.y,
                                                  ball_local_grasp_pose.r.z, ball_local_grasp_pose.r.w],
                                                 device=self.device).repeat((self.num_envs, 1))

        # [Kunal] - Gripper axes (assuming the forward direction is along the -X axis and up direction is along the Z axis)
        self.gripper_forward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        # [Kunal] - Ball axes (assuming the ball is vertical, with inward direction along the -X axis and up direction along the Z axis)
        self.ball_inward_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.ball_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.xarm_grasp_pos = torch.zeros_like(self.xarm_local_grasp_pos)
        self.xarm_grasp_rot = torch.zeros_like(self.xarm_local_grasp_rot)
        self.xarm_grasp_rot[..., -1] = 1  #
        # self.drawer_grasp_pos = torch.zeros_like(self.ball_local_grasp_pos)
        # self.drawer_grasp_rot = torch.zeros_like(self.ball_local_grasp_rot)
        # self.drawer_grasp_rot[..., -1] = 1
        self.xarm_lfinger_pos = torch.zeros_like(self.xarm_local_grasp_pos)
        self.xarm_rfinger_pos = torch.zeros_like(self.xarm_local_grasp_pos)
        self.xarm_lfinger_rot = torch.zeros_like(self.xarm_local_grasp_rot)
        self.xarm_rfinger_rot = torch.zeros_like(self.xarm_local_grasp_rot)

        # [Kunal] - Ball grasp pos
        self.ball_grasp_pos = torch.zeros_like(self.ball_local_grasp_pos)
        self.ball_grasp_rot = torch.zeros_like(self.ball_local_grasp_rot)
        self.ball_grasp_rot[..., -1] = 1


    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_xarm_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.xarm_grasp_pos, self.ball_grasp_pos, self.xarm_grasp_rot, self.ball_grasp_rot,
            self.xarm_lfinger_pos, self.xarm_rfinger_pos,
            self.gripper_forward_axis, self.ball_inward_axis, self.gripper_up_axis, self.ball_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_ball_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length,
            # self.cylinder_height
        )

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        ball_pos = self.rigid_body_states[:, self.ball_handle][:, 0:3]
        ball_rot = self.rigid_body_states[:, self.ball_handle][:, 3:7]

        # print("ball_pos of first env: ", ball_pos[0])
        # print("ball_rot of first env: ", ball_rot[0])
        #
        # print("self.ball_local_grasp_pos of first env: ", self.ball_local_grasp_pos[0])
        # print("self.ball_local_grasp_rot of first env: ", self.ball_local_grasp_rot[0])

        self.xarm_grasp_rot[:], self.xarm_grasp_pos[:], self.ball_grasp_rot[:], self.ball_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.xarm_local_grasp_rot, self.xarm_local_grasp_pos,
                                     ball_rot, ball_pos, self.ball_local_grasp_rot, self.ball_local_grasp_pos
                                     )
        # print("self.ball_grasp_pos of first env: ", self.ball_grasp_pos[0])
        # print("Left finger pos of first env: ", self.xarm_lfinger_pos[0])
        # print("Right finger pos of first env: ", self.xarm_rfinger_pos[0])
        # print("self.ball_grasp_rot of first env: ", self.ball_grasp_rot[0])


        self.xarm_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.xarm_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.xarm_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.xarm_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.xarm_dof_pos - self.xarm_dof_lower_limits)
                          / (self.xarm_dof_upper_limits - self.xarm_dof_lower_limits) - 1.0)
        to_target = self.ball_grasp_pos - self.xarm_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.xarm_dof_vel * self.dof_vel_scale, to_target,
                                  # self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1),
                                 # [Kunal] - Added box dofs & ball dofs
                                 self.box_dof_pos[:, 3].unsqueeze(-1), self.box_dof_vel[:, 3].unsqueeze(-1),
                                 self.ball_dof_pos[:, 3].unsqueeze(-1), self.ball_dof_vel[:, 3].unsqueeze(-1)),
                                 dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset xarm
        pos = tensor_clamp(
            self.xarm_default_dof_pos.unsqueeze(0) + 0.25 * (
                    torch.rand((len(env_ids), self.num_xarm_dofs), device=self.device) - 0.5),
            self.xarm_dof_lower_limits, self.xarm_dof_upper_limits)
        self.xarm_dof_pos[env_ids, :] = pos
        self.xarm_dof_vel[env_ids, :] = torch.zeros_like(self.xarm_dof_vel[env_ids])
        self.xarm_dof_targets[env_ids, :self.num_xarm_dofs] = pos

        # reset cabinet
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.xarm_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.xarm_dof_targets[:,
                  :self.num_xarm_dofs] + self.xarm_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.xarm_dof_targets[:, :self.num_xarm_dofs] = tensor_clamp(
            targets, self.xarm_dof_lower_limits, self.xarm_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.xarm_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.xarm_grasp_pos[i] + quat_apply(self.xarm_grasp_rot[i], to_torch([1, 0, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                py = (self.xarm_grasp_pos[i] + quat_apply(self.xarm_grasp_rot[i], to_torch([0, 1, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                pz = (self.xarm_grasp_pos[i] + quat_apply(self.xarm_grasp_rot[i], to_torch([0, 0, 1],
                                                                                               device=self.device) * 0.2)).cpu().numpy()

                p0 = self.xarm_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                   [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                   [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                   [0.1, 0.1, 0.85])

                # px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0],
                #                                                                                device=self.device) * 0.2)).cpu().numpy()
                # py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0],
                #                                                                                device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1],
                #                                                                                device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.drawer_grasp_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # [Kunal] - Add step for ball
                px = (self.ball_grasp_pos[i] + quat_apply(self.ball_grasp_rot[i], to_torch([1, 0, 0],
                                                                                               device=self.device) * 0.2)).cpu().numpy()
                py = (self.ball_grasp_pos[i] + quat_apply(self.ball_grasp_rot[i], to_torch([0, 1, 0],
                                                                                                  device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ball_grasp_pos[i] + quat_apply(self.ball_grasp_rot[i], to_torch([0, 0, 1],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ball_grasp_pos
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])
                # --------------------------------------------


                px = (self.xarm_lfinger_pos[i] + quat_apply(self.xarm_lfinger_rot[i], to_torch([1, 0, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                py = (self.xarm_lfinger_pos[i] + quat_apply(self.xarm_lfinger_rot[i], to_torch([0, 1, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                pz = (self.xarm_lfinger_pos[i] + quat_apply(self.xarm_lfinger_rot[i], to_torch([0, 0, 1],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()

                p0 = self.xarm_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.xarm_rfinger_pos[i] + quat_apply(self.xarm_rfinger_rot[i], to_torch([1, 0, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                py = (self.xarm_rfinger_pos[i] + quat_apply(self.xarm_rfinger_rot[i], to_torch([0, 1, 0],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()
                pz = (self.xarm_rfinger_pos[i] + quat_apply(self.xarm_rfinger_rot[i], to_torch([0, 0, 1],
                                                                                                   device=self.device) * 0.2)).cpu().numpy()

                p0 = self.xarm_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])




#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_alignment_reward(xarm_grasp_rot, ball_grasp_rot, gripper_forward_axis, ball_inward_axis, gripper_up_axis, ball_up_axis, num_envs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tensor

    axis1 = tf_vector(xarm_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(ball_grasp_rot, ball_inward_axis)
    axis3 = tf_vector(xarm_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(ball_grasp_rot, ball_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    alignment_reward = 0.5 * (torch.abs(dot1) + torch.sign(dot2) * dot2 ** 2)
    return alignment_reward

@torch.jit.script
def compute_xarm_reward(
        reset_buf, progress_buf, actions,
        xarm_grasp_pos, ball_grasp_pos, xarm_grasp_rot, ball_grasp_rot,
        xarm_lfinger_pos, xarm_rfinger_pos,
        gripper_forward_axis, ball_inward_axis, gripper_up_axis, ball_up_axis,
        num_envs, dist_reward_scale, rot_reward_scale, around_ball_reward_scale, open_reward_scale,
        finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length, # cylinder_height
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # Adjust the ball grasp position to the middle of the ball
    ball_middle_pos = ball_grasp_pos.clone()
    ball_middle_pos[..., 0] = ball_middle_pos[..., 0] + 0.3
    ball_middle_pos[..., 1] = ball_middle_pos[..., 1] + 0.01
    ball_middle_pos[..., 2] = ball_middle_pos[..., 2] + 0.05

    print(f"Ball Middle Pos: {ball_middle_pos[0]}")
    print(f"Left Finger Pos: {xarm_lfinger_pos[0]}")
    print(f"Right Finger Pos: {xarm_rfinger_pos[0]}")

    # # distance from hand to the middle of the ball
    # d = torch.norm(xarm_grasp_pos - ball_middle_pos, p=2, dim=-1)
    # d = d - distX_offset
    #
    # # Adjusted distance reward to account for minimum observed distance
    # dist_reward = torch.exp(-d * dist_reward_scale)

    # Distance of each finger from the ball
    lfinger_d = torch.norm(xarm_lfinger_pos - ball_middle_pos, p=2, dim=-1)
    rfinger_d = torch.norm(xarm_rfinger_pos - ball_middle_pos, p=2, dim=-1)
    avg_finger_d = (lfinger_d + rfinger_d) / 2

    # Minimize the distance of each finger from the ball
    finger_dist_reward = torch.exp(-(lfinger_d + rfinger_d))

    # Add bonus reward for reaching the target
    bonus_distance_reward = torch.zeros_like(finger_dist_reward)
    bonus_distance_reward = torch.where(avg_finger_d < 0.125, bonus_distance_reward + 2.0, bonus_distance_reward)

    # Add penalty for going away from target
    excess_distance_penalty = torch.zeros_like(finger_dist_reward)
    excess_distance_penalty = torch.where(avg_finger_d > 1.0, excess_distance_penalty + 1.0, excess_distance_penalty)

    # Add bonus reward for opening the gripper
    open_reward = torch.zeros_like(finger_dist_reward)
    open_reward = torch.where((actions[:, 7] > 0.8) & (avg_finger_d < 0.3), open_reward + 2.0, open_reward)

    rot_reward = compute_alignment_reward(xarm_grasp_rot, ball_grasp_rot, gripper_forward_axis,
                                          ball_inward_axis, gripper_up_axis, ball_up_axis, num_envs)

    # Increase the reward for gripper closing around the ball, with adjusted threshold
    grasp_reward = torch.zeros_like(finger_dist_reward)
    grasp_reward = torch.where((actions[:, 7] < -0.5) & (avg_finger_d < 0.3), grasp_reward + 2.0, grasp_reward)

    # Stability reward for maintaining the grasp
    stability_reward = torch.zeros_like(finger_dist_reward)
    stability_reward = torch.where((actions[:, 7] < -0.5) & (avg_finger_d < 0.3), stability_reward + 1.0, stability_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Scale rewards to ensure proper weights

    rewards = (finger_dist_reward * finger_dist_reward_scale +
               # dist_reward_scale * distance_reward_scale +
               bonus_distance_reward -
               excess_distance_penalty +
               open_reward_scale * open_reward +
               # around_ball_reward_scale * around_ball_reward +
               rot_reward_scale * rot_reward -
               # perpendicular_reward -
               action_penalty_scale * action_penalty +
               # finger_dist_reward_scale * finger_dist_reward +
               grasp_reward +
               stability_reward # +
               # retraction_penalty
    )

    # Adding condition to reset if arm is close to the ball
    reset_buf = torch.where(avg_finger_d < 0.1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    print("---Index 0---")
    print("Reward: ", rewards[0])
    print(f"Left Finger Distance: {lfinger_d[0]}\tRight Finger Distance: {rfinger_d[0]}")
    print(f"Finger Distance Reward: {finger_dist_reward[0] * dist_reward_scale}\t Bonus Distance Reward: {bonus_distance_reward[0]}\t"
          f"Excess Distance Penalty: {excess_distance_penalty[0]}\tOpen Reward: {open_reward_scale * open_reward[0]}\tactions[0, 7]: {actions[0, 7]}\t"
          f"Rot Reward: {rot_reward[0] * rot_reward_scale}\tAction Penalty: {action_penalty[0]}\t"
          f"Grasp Reward: {grasp_reward[0]}\tStability Reward: {stability_reward[0]}")

    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, xarm_local_grasp_rot, xarm_local_grasp_pos,
                             ball_rot, ball_pos, ball_local_grasp_rot, ball_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_xarm_rot, global_xarm_pos = tf_combine(
        hand_rot, hand_pos, xarm_local_grasp_rot, xarm_local_grasp_pos)
    global_ball_rot, global_ball_pos = tf_combine(
        ball_rot, ball_pos, ball_local_grasp_rot, ball_local_grasp_pos)

    return global_xarm_rot, global_xarm_pos, global_ball_rot, global_ball_pos