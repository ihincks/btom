from setuptools import setup, find_packages
import os

install_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(install_dir, "requirements.txt"), "r") as f:
    requirements = f.readlines()

setup(
    name='btom',
    version='0.1.0',
    description='Tools for Bayesian quantum tomography',
    author='Ian Hincks',
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    python_requires='>=3.6',
    install_requires=requirements
)
