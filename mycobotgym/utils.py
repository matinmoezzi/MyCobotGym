import random
import numpy as np
from typing import Optional
import mujoco
from gymnasium_robotics.utils import mujoco_utils
import glfw
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
from enum import Enum
from gymnasium_robotics.utils import rotations
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


def create_random_3d_coord(lower_bound, upper_bound):
    """
    Create a random 3D coordinate within the specified ranges.
    The function accepts two arguments, lower_bound and upper_bound, for each axis.
    """
    coord = []
    for _ in range(3):
        # Choose whether to use the lower or upper range for each axis
        if random.choice([True, False]):
            # Lower range
            coord.append(random.uniform(lower_bound[0], lower_bound[1]))
        else:
            # Upper range
            coord.append(random.uniform(upper_bound[0], upper_bound[1]))
    return np.array(coord)


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def print_state(model, data):
    print(f"qpos: {data.qpos}\n")
    print(f"qvel: {data.qvel}\n")
    print(f"mocap_pos: {data.mocap_pos[0]}\n")
    print(f"mocap_quat: {data.mocap_quat[0]}\n")
    print(f"eef pos: {mujoco_utils.get_site_xpos(model, data, 'EEF')}\n")


class Direction(Enum):
    POS: int = 1
    NEG: int = -1


class PrincipalAxis(Enum):
    ROLL = 0
    PITCH = 2
    YAW = 1


class MyCobotArmController:
    """
    Controller to convert human interpretable action into relative action control.

    This controller assumes relative action control vector in [-1, 1] of

    [mocap_x, mocap_y, mocap_z, wrist_joint_angle, gripper_joint_angle]
    """

    dof_dims = [PrincipalAxis.ROLL, PrincipalAxis.PITCH]
    dof_dims_axes = [axis.value for axis in dof_dims]
    alignment_axis: Optional[PrincipalAxis] = None

    # The max speed.
    MAX_SPEED = 1.0

    # The minimum speed.
    MIN_SPEED = 0.0

    SPEED_CHANGE_PERCENT = 0.2

    def __init__(self, model, data):
        self._speeds = np.array([0.01, 5, 0.1, 0.1])
        self.model = model
        self.data = data

    @property
    def arm_speed(self):
        """
        The speed that arm moves.
        """
        return self._speeds[0]

    @property
    def wrist_speed(self):
        """
        The speed that wrist rotates.
        """
        return self._speeds[1]

    @property
    def gripper_speed(self):
        """
        The speed that gripper opens/closes.
        """
        return self._speeds[2]

    @property
    def rot_speed(self):
        """
        The speed that wrist rotates.
        """
        return self._speeds[3]

    def zero_control(self):
        """
        Returns zero control, meaning gripper shouldn't move by applying this action.
        """
        zero_ctrl = np.zeros(self.model.nu, dtype=float)
        return zero_ctrl

    def speed_up(self):
        """
        Increase gripper moving speed.
        """
        self._speeds = np.minimum(
            self._speeds * (1 + self.SPEED_CHANGE_PERCENT), self.MAX_SPEED
        )

    def speed_down(self):
        """
        Decrease gripper moving speed.
        """
        self._speeds = np.maximum(
            self._speeds * (1 - self.SPEED_CHANGE_PERCENT), self.MIN_SPEED
        )

    def get_tcp_quat(self, ctrl: np.ndarray) -> np.ndarray:
        assert len(ctrl) == 2

        euler = np.zeros(3)
        euler[self.dof_dims_axes] = ctrl
        quat = rotations.euler2quat(euler)
        gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_tcp"
        )
        gripper_quat = self.data.xquat[gripper_id]

        return rotations.quat_mul(gripper_quat, quat) - gripper_quat

    def move_x(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along x axis.
        """
        return self._move(0, direction)

    def move_y(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along y axis.
        """
        return self._move(1, direction)

    def move_z(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along z axis.
        """
        return self._move(2, direction)

    def _move(self, axis: int, direction: Direction):
        """
        Move gripper along given axis and direction.
        """
        action = np.zeros(3)
        action[axis] = self.arm_speed * direction.value
        self.data.mocap_pos[0] += action
        mujoco.mj_step(self.model, self.data, nstep=50)

    def move_gripper(self, direction: Direction):
        """
        Open/close gripper.
        """
        gripper_ctrlrange = self.model.actuator_ctrlrange[0]
        ctrl = self.zero_control()
        ctrl[-1] = self.gripper_speed * direction.value
        self.data.ctrl += ctrl
        self.data.ctrl = np.clip(
            self.data.ctrl, gripper_ctrlrange[0], gripper_ctrlrange[1]
        )
        mujoco.mj_step(self.model, self.data, nstep=10)

    def rot_x(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along x axis.
        """
        return self._rot(0, direction)

    def rot_y(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along y axis.
        """
        return self._rot(1, direction)

    def rot_z(self, direction: Direction) -> np.ndarray:
        """
        Move gripper along z axis.
        """
        return self._rot(2, direction)

    def _rot(self, axis: int, direction: Direction):
        """
        Move gripper along given axis and direction.
        """
        e = rotations.quat2euler(self.data.mocap_quat[0])
        e[axis] += self.rot_speed * direction.value
        quat = rotations.euler2quat(e)
        self.data.mocap_quat[0] += quat
        mujoco.mj_step(self.model, self.data, nstep=50)

    def tilt_gripper(self, direction: Direction) -> np.ndarray:
        """
        Tilt the gripper
        """
        quat = self.get_tcp_quat(np.array([self.wrist_speed * direction.value, 0]))
        self.data.mocap_quat[0] += quat
        mujoco.mj_step(self.model, self.data)

    def rotate_wrist(self, direction: Direction) -> np.ndarray:
        """
        Rotate the wrist joint.
        """
        quat = self.get_tcp_quat(np.array([0, self.wrist_speed * direction.value]))
        self.data.mocap_quat[0] += quat
        mujoco.mj_step(self.model, self.data)


class RobotControlViewer(WindowViewer):
    """
    A viewer which support controlling the robot via keyboard control.

    The key bindings are as follows:

    Key binds in EnvViewer (unless override) +

    - UP/DOWN/LEFT/RIGHT: Go backward/forward/left/right.
    - Z/X: Go down/up.
    - C/V: Close/open gripper.
    - Q/W: Rotate wrist CW/CCW.
    - Y/U: Tilt the wrist.
    - -/=: Slow down/speed up gripper moving.
    """

    def __init__(self, model, data):
        self.controller = MyCobotArmController(model, data)
        self.model = model
        self.data = data

        super().__init__(model, data)

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            self._press_key_callback(window, key, scancode, mods)
        elif action == glfw.RELEASE:
            self._release_key_callback(window, key, scancode, mods)

    def _press_key_callback(self, window, key, scancode, mods):
        """
        Key callback on press action.
        """
        if key == glfw.KEY_X:
            self.controller.move_z(Direction.NEG)
        elif key == glfw.KEY_Z:
            self.controller.move_z(Direction.POS)
        elif key == glfw.KEY_V:
            self.controller.move_gripper(Direction.NEG)
        elif key == glfw.KEY_C:
            self.controller.move_gripper(Direction.POS)
        elif key == glfw.KEY_UP:
            self.controller.move_x(Direction.NEG)
        elif key == glfw.KEY_DOWN:
            self.controller.move_x(Direction.POS)
        elif key == glfw.KEY_LEFT:
            self.controller.move_y(Direction.NEG)
        elif key == glfw.KEY_RIGHT:
            self.controller.move_y(Direction.POS)
        elif key == glfw.KEY_N:
            self.controller.rot_x(Direction.POS)
        elif key == glfw.KEY_M:
            self.controller.rot_x(Direction.NEG)
        elif key == glfw.KEY_R:
            self.controller.rot_y(Direction.POS)
        elif key == glfw.KEY_T:
            self.controller.rot_y(Direction.NEG)
        elif key == glfw.KEY_Y:
            self.controller.rot_z(Direction.POS)
        elif key == glfw.KEY_U:
            self.controller.rot_z(Direction.NEG)
        # elif key == glfw.KEY_Q:
        #     self.controller.rotate_wrist(Direction.POS)
        # elif key == glfw.KEY_W:
        #     self.controller.rotate_wrist(Direction.NEG)
        # elif key == glfw.KEY_Y:
        #     self.controller.tilt_gripper(Direction.POS)
        # elif key == glfw.KEY_U:
        #     self.controller.tilt_gripper(Direction.NEG)
        else:
            super()._key_callback(window, key, scancode, glfw.PRESS, mods)

    def _release_key_callback(self, window, key, scancode, mods):
        if key == glfw.KEY_MINUS:
            self.controller.speed_down()
        elif key == glfw.KEY_EQUAL:
            self.controller.speed_up()
        elif key in [
            glfw.KEY_Z,
            glfw.KEY_X,
            glfw.KEY_C,
            glfw.KEY_V,
            glfw.KEY_UP,
            glfw.KEY_DOWN,
            glfw.KEY_LEFT,
            glfw.KEY_RIGHT,
            glfw.KEY_N,
            glfw.KEY_M,
            glfw.KEY_R,
            glfw.KEY_T,
            glfw.KEY_Y,
            glfw.KEY_U,
        ]:
            # Don't respond on release for sticky control keys.
            return
        else:
            super()._key_callback(window, key, scancode, glfw.RELEASE, mods)

    def _create_overlay(self):
        super()._create_overlay()

        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "Go backward/forward/left/right",
            "[up]/[down]/[left]/[right] arrow",
        )
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Go up/down", "[Z]/[X]")
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Open/Close gripper", "[V]/[C]"
        )
        # self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        #                  "Rotate wrist CW/CCW", "[Q]/[W]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Rotate z axis", "[Y]/[U]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Rotate y axis", "[R]/[T]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Rotate x axis", "[N]/[M]")
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Slow down/Speed up", "[-]/[=]"
        )
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "Controller Speed",
            "%s" % self.controller._speeds,
        )

        mujoco.mj_step(self.model, self.data, nstep=200)
        goal = mujoco_utils.get_site_xpos(self.model, self.data, "target0")
        obj = mujoco_utils.get_site_xpos(self.model, self.data, "object0")
        eef = mujoco_utils.get_site_xpos(self.model, self.data, "EEF")

        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "distance_object_target",
            "%.3f" % goal_distance(obj, goal),
        )
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "distance_gripper_object",
            "%.3f" % goal_distance(eef, obj),
        )
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "distance_gripper_target",
            "%.3f" % goal_distance(eef, goal),
        )

        model_names = MujocoModelNames(self.model)
        robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(
            self.model, self.data, model_names.joint_names
        )
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "qpos", "%s" % robot_qpos
        )
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "mocap_pos",
            "%s" % self.data.mocap_pos,
        )
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "mocap_quat",
            "%s" % self.data.mocap_quat,
        )
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "gripper_pos", "%s" % eef
        )
        gripper_mocap_rel_pos = self.data.mocap_pos[0].copy() - eef
        self.add_overlay(
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "gripper_mocap_rel_pos",
            "%s" % gripper_mocap_rel_pos,
        )


import mujoco
import numpy as np


class IKController:
    def __init__(self, model, data, regularization_strength: float = 0.3):
        """Inverse Kinematic solver.

         The input of the controller are a target cartesian position and quaternion orientation for
         the frame of the end-effector.
         The controller will output angular displacements in the joint space to achieve the desired
         end-effector consiguration.

         This controller uses Damped Least Squares (DLS) to iteratively approximate to the solution and avoid singularities.
         More information can be read in the following paper. [Introduction to Inverse Kinematics with Jacobian Transpose,
         Pseudoinverse and Damped Least Squares methods](https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf)

        The controller implementation has also been inspired by the following repositories:
        [dexterity](https://github.com/kevinzakka/dexterity/tree/main/dexterity/inverse_kinematics), and
        [dm_robotics](https://github.com/deepmind/dm_robotics/blob/main/py/moma/utils/ik_solver.py). Even though these
        implementations use an integration technique to compute joint velocities instead of joint position displacements.

         Args:
             model (MjModel): mujoco model structure.
             data (MjData): mujoco data structure.
             regularization_strength (float): regularization parameter for DLS
        """
        self.model = model
        self.data = data

        self.regularization_strength = regularization_strength
        # End effector frame to compute IK
        self.eef_id = self.model.site("EEF").id

    def compute_qpos_delta(self, target_pos, target_quat):
        """Return joint position displacement to achieve a target end-effector pose.

        Args:
            target_pos (np.ndarray): target end-effector position
            target_quat (np.ndarray): target end-effector quaternion orientation

        Returns:
            qpos_increase (np.ndarray): joint displacement order by its mujoco id
        """
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        err = np.empty(6)
        err_pos, err_rot = err[:3], err[3:]
        eef_xquat = np.empty(4)
        neg_eef_xquat = np.empty(4)
        err_rot_quat = np.empty(4)

        eef_xpos = self.data.site_xpos[self.eef_id]
        eef_xmat = self.data.site_xmat[self.eef_id]

        # Translation error
        err_pos[:] = target_pos - eef_xpos

        # Rotation error
        mujoco.mju_mat2Quat(eef_xquat, eef_xmat)
        mujoco.mju_negQuat(neg_eef_xquat, eef_xquat)
        mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_eef_xquat)
        mujoco.mju_quat2Vel(err_rot, err_rot_quat, 50)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.eef_id)
        jac = np.concatenate((jac_pos, jac_rot), axis=0)
        qpos_increase = self.solve_DLS(jac, err)

        return qpos_increase

    def solve_DLS(self, jac_joints, error):
        """Computes the Least Mean Squares algorithm over the following equation.

        Where `e` is the end-effector error, J is the joint space Jacobian,
        `Δθ` is the joint displacement to solve, and `τ` is the regularization factor.

        .. math::

            J^{T}JΔθ=J^{T}e + Iτ

        Args:
            jac_joints (np.ndarray): system joint space Jacobian
            error (np.ndarray): end-effector target error

        Returns:
            Δqpos (np.ndarray): joint position displacement
        """
        hess_approx = jac_joints.T.dot(jac_joints)
        hess_approx += np.eye(hess_approx.shape[0]) * self.regularization_strength
        joint_delta = jac_joints.T.dot(error)
        # Least-squares solution
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]


def combine_images(image1, image2, image3, image4):
    height, width, channels = image1.shape
    assert all(
        image.shape == (height, width, channels) for image in [image2, image3, image4]
    )

    # Create a new array with twice the width and height
    combined_image = np.zeros((height * 2, width * 2, channels), dtype=image1.dtype)

    # Place each image in the combined array
    combined_image[0:height, 0:width] = image1  # Top left
    combined_image[0:height, width : width * 2] = image2  # Top right
    combined_image[height : height * 2, 0:width] = image3  # Bottom left
    combined_image[height : height * 2, width : width * 2] = image4  # Bottom right

    return combined_image
