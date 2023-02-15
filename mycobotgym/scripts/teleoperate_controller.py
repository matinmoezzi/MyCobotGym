from os import path
from typing import Optional
import mujoco
from gymnasium_robotics.utils import mujoco_utils
import glfw
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
from enum import Enum
from gymnasium_robotics.utils import rotations


import numpy as np


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
        self._speeds = np.array([0.01, 5, 0.5, 0.1])
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
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_tcp')
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
        mujoco.mj_step(self.model, self.data)

    def move_gripper(self, direction: Direction):
        """
        Open/close gripper.
        """
        ctrl = self.zero_control()
        ctrl[-1] = self.gripper_speed * direction.value
        self.data.ctrl = ctrl
        mujoco.mj_step(self.model, self.data)

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
        mujoco.mj_step(self.model, self.data)

    def tilt_gripper(self, direction: Direction) -> np.ndarray:
        """
        Tilt the gripper
        """
        quat = self.get_tcp_quat(
            np.array([self.wrist_speed * direction.value, 0]))
        self.data.mocap_quat[0] += quat
        mujoco.mj_step(self.model, self.data)

    def rotate_wrist(self, direction: Direction) -> np.ndarray:
        """
        Rotate the wrist joint.
        """
        quat = self.get_tcp_quat(
            np.array([0, self.wrist_speed * direction.value]))
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
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                         "Go up/down", "[Z]/[X]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                         "Open/Close gripper", "[V]/[C]")
        # self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        #                  "Rotate wrist CW/CCW", "[Q]/[W]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                         "Rotate z axis", "[Y]/[U]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                         "Rotate y axis", "[R]/[T]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                         "Rotate x axis", "[N]/[M]")
        self.add_overlay(mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                         "Slow down/Speed up", "[-]/[=]")


def main():
    # load model
    model_path = "../envs/assets/pick_and_place.xml"
    xml_file_path = path.join(
        path.dirname(path.realpath(__file__)),
        model_path,
    )
    model = mujoco.MjModel.from_xml_path(xml_file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # viewer set up
    viewer = RobotControlViewer(model, data)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_base')
    lookat = data.xpos[body_id]
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 2
    viewer.cam.azimuth = -90
    viewer.cam.elevation = -35

    while True:
        mujoco.mj_step(model, data, nstep=10)
        viewer.render()


if __name__ == '__main__':
    main()
