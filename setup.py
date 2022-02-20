from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    install_requires=required,
    version='0.1.0',
    description='This is the effort to build a pipeline for voxel-wise brain encoding models, for Algonauts Challenge',
    author='huze',
    license='MIT',
)
