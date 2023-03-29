from typing import Literal
import mujoco
from os import path
import numpy as np
from mycobotgym.utils import *
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium_robotics.envs.franka_kitchen.ik_controller import IKController
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.utils import mujoco_utils
from gymnasium_robotics.utils import rotations
from gymnasium_robotics.utils.rotations import euler2quat


DEFAULT_CAMERA_CONFIG = {
    "distance": 2,
    "azimuth": 0.0,
    "elevation": -35.0,
    "lookat": np.array([0, 0, 0.8]),
}

MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5
MAX_JOINT_DISPLACEMENT = 0.5


def limit_obj_loc(pos):
    y_threshold = -0.15
    pos[1] = max(pos[1], y_threshold)


class MyCobotEnv(MujocoEnv):
    metadata = {"render_modes": [
        "human", "rgb_array", "depth_array"], "render_fps": 25}

    def __init__(self, model_path: str = "./assets/mycobot280.xml", has_object=True, block_gripper=False, control_steps=5, controller_type: Literal['mocap', 'IK', 'joint', 'delta_joint'] = 'IK', obj_range: float = 0.1, target_range: float = 0.1, target_offset: float = 0.0, target_in_the_air=True, distance_threshold=0.05, initial_qpos: dict = {}, fetch_env: bool = False, reward_type="sparse", frame_skip: int = 20, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs) -> None:

        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.control_steps = control_steps
        self.controller_type = controller_type
        self.initial_qpos = initial_qpos
        self.fetch_env = fetch_env
        self.target_range = target_range
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.goal = np.zeros(3)

        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        observation_space = (
            spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
        )

        super().__init__(
            xml_file_path,
            frame_skip,
            observation_space,
            default_camera_config=default_camera_config,
            ** kwargs,
        )
        self.model_names = MujocoModelNames(self.model)

        self._env_setup(self.initial_qpos)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.init_ctrl = self.data.ctrl.ravel().copy()

        self.controller = None
        if self.controller_type == 'IK':
            self.controller = IKController(self.model, self.data)
            if self.fetch_env:
                action_size = 4    # 3 translation + 1 gripper
            else:
                # 3 translation + 3 rotation (euler) + 1 gripper
                action_size = 7
        elif self.controller_type in ['joint', 'delta_joint']:
            action_size = 7  # 6 joint positions + 1 gripper
        elif self.controller_type == 'mocap':
            if self.fetch_env:
                action_size = 4  # 3 mocap cartesian position + 1 gripper
            else:
                # 3 mocap position + 4 mocap rotation (quat) + 1 gripper
                action_size = 8

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=(action_size,)
        )

        # Actuator ranges
        ctrlrange = self.model.actuator_ctrlrange
        self.actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        self.actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.controller_type == 'IK':
            assert self.controller is not None
            current_eef_pose = self.data.site_xpos[
                self.model_names.site_name2id["EEF"]
            ].copy()
            target_eef_pose = current_eef_pose + \
                action[:3] * MAX_CARTESIAN_DISPLACEMENT
            if self.fetch_env:
                quat_rot = euler2quat(np.zeros(3) * MAX_ROTATION_DISPLACEMENT)
            else:
                quat_rot = euler2quat(action[3:6] * MAX_ROTATION_DISPLACEMENT)
            current_eef_quat = np.empty(
                4
            )  # current orientation of the end effector site in quaternions
            target_orientation = np.empty(
                4
            )  # desired end effector orientation in quaternions
            mujoco.mju_mat2Quat(
                current_eef_quat,
                self.data.site_xmat[self.model_names.site_name2id["EEF"]].copy(
                ),
            )
            mujoco.mju_mulQuat(target_orientation,
                               quat_rot, current_eef_quat)

            ctrl_action = np.zeros(7)

            # Denormalize gripper action
            ctrl_action[-1] = (
                self.actuation_center[-1] +
                action[-1] * self.actuation_range[-1]
            )

            for _ in range(self.control_steps):
                delta_qpos = self.controller.compute_qpos_delta(
                    target_eef_pose, target_orientation
                )
                ctrl_action[:6] = self.data.ctrl.copy()[:6] + delta_qpos[:6]

                # Do not use `do_simulation`` method from MujocoEnv: value error due to discrepancy between
                # the action space and the simulation control input when using IK controller.
                self.data.ctrl[:] = ctrl_action
                mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        elif self.controller_type == "mocap":
            mocap_action = np.zeros(7)
            mocap_action[:3] = action[:3] * 0.05
            grip_tcp_quat = self.data.xquat[self.model_names.body_name2id["gripper_tcp"]]
            if self.fetch_env:
                mocap_action[3:7] = np.array(
                    [0.5, -0.5, -0.5, 0.5])
            else:
                mocap_action[3:7] = action[3:7]
            mocap_action[3:7] -= grip_tcp_quat
            mujoco_utils.mocap_set_action(self.model, self.data, mocap_action)
            self.data.ctrl[-1] = (
                self.actuation_center[-1] +
                action[-1] * self.actuation_range[-1]
            )
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip * 5)
        elif self.controller_type == 'joint':
            # Denormalize the input action from [-1, 1] range to the each actuators control range
            action = self.actuation_center + \
                action * self.actuation_range
            if self.fetch_env:
                action[4:6] = self.init_ctrl[4:6]
            self.do_simulation(action, self.frame_skip)
        elif self.controller_type == 'delta_joint':
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[-1] = (
                self.actuation_center[-1] +
                action[-1] * self.actuation_range[-1]
            )
            self.data.ctrl[:-1] += action[:-1] * MAX_JOINT_DISPLACEMENT
            if self.fetch_env:
                self.data.ctrl[3:6] = self.init_ctrl[3:6]
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        self._step_callback()

        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal)}
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(
            obs["achieved_goal"], obs["desired_goal"], info)
        truncated = self.compute_truncated(
            obs["achieved_goal"], obs["desired_goal"], info)

        if self.render_mode == "human":
            self.mujoco_renderer.viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "is_success", str(info["is_success"]))
            if self.has_object:
                self.mujoco_renderer.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "distance_object_target", "%.3f" %
                                                        goal_distance(obs["achieved_goal"], self.goal))
                self.mujoco_renderer.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "distance_gripper_object", "%.3f" %
                                                        goal_distance(obs["achieved_goal"], obs["observation"][:3]))
            else:
                self.mujoco_renderer.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "distance_gripper_target", "%.3f" %
                                                        goal_distance(obs["achieved_goal"], self.goal))
            self.render()
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.data.qpos[:] = np.copy(self.init_qpos)
        self.data.qvel[:] = np.copy(self.init_qvel)
        self.data.ctrl[:] = np.copy(self.init_ctrl)
        if self.model.na != 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

        if self.model.nmocap > 0 and self.fetch_env:
            self._reset_mocap()

        # Randomize object location
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = mujoco_utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            limit_obj_loc(object_xpos)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            mujoco_utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        mujoco.mj_forward(self.model, self.data)

        self.goal = self._sample_goal().copy()

        obs = self._get_obs()
        return obs

    def _sample_goal(self):
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3
        )
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.2)

        limit_obj_loc(goal)

        return goal.copy()

    def _get_obs(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == "reward_shaping":
            grip_pos = mujoco_utils.get_site_xpos(self.model, self.data, "EEF")
            reward = int(self._is_success(grip_pos, achieved_goal, goal))
            reward += max(self.stage_rewards())
            return reward

    def _step_callback(self):
        if self.block_gripper:
            mujoco_utils.set_joint_qpos(
                self.model, self.data, "right_finger_joint", 0.0
            )
            mujoco_utils.set_joint_qpos(
                self.model, self.data, "left_finger_joint", 0.0
            )
            mujoco.mj_forward(self.model, self.data)

    def _render_callback(self):
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        self._render_callback()
        return super().render()

    def generate_mujoco_observations(self):
        # positions
        grip_pos = mujoco_utils.get_site_xpos(
            self.model, self.data, "EEF")

        dt = self.frame_skip * self.model.opt.timestep
        grip_velp = (
            mujoco_utils.get_site_xvelp(
                self.model, self.data, "EEF") * dt
        )

        robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(
            self.model, self.data, self.model_names.joint_names
        )
        if self.has_object:
            object_pos = mujoco_utils.get_site_xpos(
                self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                mujoco_utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                mujoco_utils.get_site_xvelp(
                    self.model, self.data, "object0") * dt
            )
            object_velr = (
                mujoco_utils.get_site_xvelr(
                    self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
        if info["is_success"]:
            return True
        return False

    def compute_truncated(self, achievec_goal, desired_goal, info):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gymnasium TimeLimit wrapper."""
        return False

    def _check_contact(self, gripper_id, object_id):
        for contact in self.data.contact:
            if (gripper_id == contact.geom1 and object_id == contact.geom2) or (object_id == contact.geom1 and gripper_id == contact.geom2):
                return True
        return False

    def stage_rewards(self):
        """
        Returns staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.7

        grip_pos = mujoco_utils.get_site_xpos(
            self.model, self.data, "EEF")
        object_pos = mujoco_utils.get_site_xpos(
            self.model, self.data, "object0")
        target_pos = mujoco_utils.get_site_xpos(
            self.model, self.data, "target0")

        r_reach = 0.0
        r_reach = (
            1 - np.tanh(10 * goal_distance(grip_pos, object_pos))) * reach_mult

        right_finger_layer = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_layer")
        left_finger_layer = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_layer")
        object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object0")
        r_grasp = (int(self._check_contact(right_finger_layer, object_id)
                   or self._check_contact(left_finger_layer, object_id)) * grasp_mult)

        r_lift = 0.0
        if self._check_contact(right_finger_layer, object_id) and self._check_contact(left_finger_layer, object_id):
            r_lift = grasp_mult + \
                (1 - np.tanh(15.0 * goal_distance(target_pos, object_pos))) * \
                (lift_mult - grasp_mult)

        return r_reach, r_grasp, r_lift

    def _env_setup(self, initial_qpos):
        if self.fetch_env:
            if self.controller_type == 'mocap':
                self._reset_mocap()
            else:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                mujoco.mj_forward(self.model, self.data)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = mujoco_utils.get_site_xpos(
            self.model, self.data, "EEF").copy()

        # Hide object in Reach env
        if not self.has_object:
            object_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, 'object0')
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, 'object0')
            self.model.geom_size[object_id] = np.zeros(3)
            self.model.site_size[site_id] = np.zeros(3)

        self.height_offset = mujoco_utils.get_site_xpos(
            self.model, self.data, "object0")[2]

    def _reset_mocap(self):
        mujoco_utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", np.array([0.70809474, 0, -0.70611744, 0]))
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip * 8)
        mujoco_utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", np.array([0.50235287, -0.499, -0.5, 0.49764296]))
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip * 8)
        mujoco_utils.set_mocap_pos(
            self.model, self.data, "robot0:mocap", np.array([0.0138673, -0.13135342, 1.010216]))
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip * 8)
