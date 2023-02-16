from typing import Literal
import mujoco
from os import path
import numpy as np
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

MAX_CARTESIAN_DISPLACEMENT = 0.05
MAX_ROTATION_DISPLACEMENT = 0.5
OBJECT_RANGE = [0.12, 0.21]
TARGET_HEIGHT_RANGE = [0, 0.25]


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PickAndPlaceEnv(MujocoEnv):
    metadata = {"render_modes": [
        "human", "rgb_array", "depth_array"], "render_fps": 20}

    def __init__(self, model_path: str = "./assets/pick_and_place.xml", has_object=True, block_gripper=False, control_steps=5, controller_type: Literal['mocap', 'IK', 'joint'] = 'mocap', gripper_extra_height=0, target_in_the_air=True, distance_threshold=0.05, reward_type="sparse", frame_skip: int = 25, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs) -> None:

        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.control_steps = control_steps
        self.controller_type = controller_type
        self.goal = np.zeros(0)

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

        self.init_ctrl = np.array([0, 0, 0, 0, 0, 0, 0])

        if self.controller_type == "mocap":
            mujoco_utils.reset_mocap_welds(self.model, self.data)
            mujoco_utils.reset_mocap2body_xpos(self.model, self.data)
        else:
            # Hide mocap geoms.
            mocap_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
            mocap_geom_start_id = self.model.body_geomadr[mocap_id]
            mocap_geom_end_id = (
                mocap_geom_start_id + self.model.body_geomnum[mocap_id]
            )
            for geom_id in range(mocap_geom_start_id, mocap_geom_end_id):
                self.model.geom_rgba[geom_id, :] = 0.0
            # Disable the mocap weld constraint
            if self.model.nmocap > 0 and self.model.eq_data is not None:
                for i in range(self.model.eq_data.shape[0]):
                    if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                        self.model.eq_active[i] = 0
            mujoco.mj_forward(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        if self.has_object:
            self.height_offset = mujoco_utils.get_site_xpos(
                self.model, self.data, "object0")[2]

        if self.controller_type == 'IK':
            self.controller = IKController(self.model, self.data)
            action_size = 7  # 3 translation + 3 rotation (euler) + 1 gripper
        elif self.controller_type == 'joint':
            self.controller = None
            action_size = 7  # 6 joint positions + 1 gripper
        elif self.controller_type == 'mocap':
            self.controller = None
            # 3 end effector position + 4 end effector rotation (quat) + 1 gripper
            action_size = 8

        self.model_names = MujocoModelNames(self.model)

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
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.controller_type == 'IK':
            assert self.controller is not None
            current_eef_pose = self.data.site_xpos[
                self.model_names.site_name2id["EEF"]
            ].copy()
            target_eef_pose = current_eef_pose + \
                action[:3] * MAX_CARTESIAN_DISPLACEMENT
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
            mujoco.mju_mulQuat(target_orientation, quat_rot, current_eef_quat)

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
                # TODO: eliminate error check in MujocoEnv (action space can be different from simulaton control input).
                self.data.ctrl[:] = ctrl_action
                mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

                if self.render_mode == "human":
                    self.render()
        elif self.controller_type == "mocap":
            mocap_action = action[:-1].copy()
            mocap_action[:3] *= MAX_CARTESIAN_DISPLACEMENT
            mocap_action[3:7] *= MAX_ROTATION_DISPLACEMENT
            mujoco_utils.mocap_set_action(self.model, self.data, mocap_action)
            # Denormalize gripper action
            gripper_action = (
                self.actuation_center[-1] +
                action[-1] * self.actuation_range[-1]
            )
            self.data.ctrl[-1] = gripper_action
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
            if self.render_mode == "human":
                self.render()
        elif self.controller_type == 'joint':
            # Denormalize the input action from [-1, 1] range to the each actuators control range
            self.data.ctrl[:] = self.actuation_center + \
                action * self.actuation_range
            self.do_simulation(action, self.frame_skip)
            if self.render_mode == "human":
                self.render()

        self._step_callback()

        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(
            obs["achieved_goal"], obs["desired_goal"], info)
        truncated = self.compute_truncated(
            obs["achieved_goal"], obs["desired_goal"], info)
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.data.ctrl[:] = self.init_ctrl
        self.set_state(qpos, qvel)

        # Randomize start position of object
        if self.has_object:
            object_xpos = self._sample_object()
            object_qpos = mujoco_utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            mujoco_utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        if self.controller_type == "mocap":
            mujoco_utils.reset_mocap2body_xpos(self.model, self.data)
            mujoco_utils.reset_mocap_welds(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.goal = self._sample_goal().copy()
        while np.linalg.norm(np.array(self.goal) - np.array(object_qpos[:3])) < self.distance_threshold * 2:
            self.goal = self._sample_goal().copy()

        obs = self._get_obs()
        return obs

    def _sample_object(self):
        object_x = self.np_random.choice(
            [self.np_random.uniform(-OBJECT_RANGE[1], -OBJECT_RANGE[0]), self.np_random.uniform(OBJECT_RANGE[0], OBJECT_RANGE[1])])
        object_y = self.np_random.choice(
            [self.np_random.uniform(-OBJECT_RANGE[1], -OBJECT_RANGE[0]), self.np_random.uniform(OBJECT_RANGE[0], OBJECT_RANGE[1])])
        object_xpos = [object_x, object_y]
        return object_xpos.copy()

    def _sample_goal(self):
        goal_x = self.np_random.choice(
            [self.np_random.uniform(-OBJECT_RANGE[1], -OBJECT_RANGE[0]), self.np_random.uniform(OBJECT_RANGE[0], OBJECT_RANGE[1])])
        goal_y = self.np_random.choice(
            [self.np_random.uniform(-OBJECT_RANGE[1], -OBJECT_RANGE[0]), self.np_random.uniform(OBJECT_RANGE[0], OBJECT_RANGE[1])])
        goal_z = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal_z += self.np_random.uniform(*TARGET_HEIGHT_RANGE)
        goal = np.array([goal_x, goal_y, goal_z])
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
        else:
            return -d

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
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        self._render_callback()
        return super().render()

    def generate_mujoco_observations(self):
        # positions
        grip_pos = mujoco_utils.get_site_xpos(
            self.model, self.data, "EEF")

        dt = self.control_steps * self.model.opt.timestep
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
