from setuptools import find_packages, setup

setup(
    name="a2c-ppo-acktr",
    packages=find_packages(),
    version="1.0.0",
    install_requires=["gym", "matplotlib", "pybullet", "moviepy", "imageio"],
)
