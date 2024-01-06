from setuptools import find_packages, setup

setup(
    name="MyCobotGym",
    version="1.0",
    author="Matin Moezzi",
    author_email="matin.moezzi@mail.utoronto.ca",
    description="Reinforcement Learning Framework for Robotic Arm Tasks using MyCobot",
    packages=find_packages(),
    install_requires=["gymnasium_robotics", "mujoco"],
    python_requires=">=3.9",
    package_data={"mycobotgym": ["envs/assets/*", "envs/assets/meshes/*"]},
)
