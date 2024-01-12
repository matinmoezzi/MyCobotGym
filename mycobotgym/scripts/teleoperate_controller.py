from mycobotgym.utils import *
from os import path


def main():
    # load model
    model_path = "../envs/assets/mycobot280_mocap.xml"
    xml_file_path = path.join(
        path.dirname(path.realpath(__file__)),
        model_path,
    )
    model = mujoco.MjModel.from_xml_path(xml_file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # viewer set up
    viewer = RobotControlViewer(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    lookat = data.xpos[body_id]
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 2
    viewer.cam.azimuth = -90
    viewer.cam.elevation = -35

    mujoco.mj_step(model, data, nstep=10)
    # mujoco.mj_resetDataKeyframe(model, data, 0)
    print_state(model, data)
    while True:
        mujoco.mj_step(model, data, nstep=10)
        viewer.render()


if __name__ == "__main__":
    main()
