from setuptools import find_packages, setup

setup(
    name="MyCobotGym",
    version="1.0",
    author="Matin Moezzi",
    author_email="matin.moezzi@mail.utoronto.ca",
    description="Reinforcement Learning Framework for Robotic Arm Tasks using MyCobot",
    packages=find_packages(),
    install_requires=["gymnasium_robotics==1.2.0", "mujoco==2.3.2"],
    python_requires=">=3.9",
)
