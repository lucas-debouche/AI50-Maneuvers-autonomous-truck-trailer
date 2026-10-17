from setuptools import find_packages, setup

setup(
    name="gym_truck_backerupper",
    version="0.1.0",
    description="Gymnasium tractor-trailer environment",
    packages=find_packages(),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "numpy>=1.23,<1.27",
        "scipy>=1.10,<1.14",
        "matplotlib>=3.7,<3.10",
        "gymnasium>=0.29,<1.2",
        "pygame>=2.5,<3",
    ],
)
