#!/usr/bin/env python
# -*- encoding: UTF-8 -*-

from setuptools import find_packages
from setuptools import setup

setup(
    # metadata
    name="liteasr",
    version="0.1.0",
    author="Cao Juncheng",
    author_email="soukunsei@gmail.com",
    description="Lite ASR Framework",
    # packages & static resource files
    packages=find_packages(),
    # dependency
    install_requires=[
        "hydra-core==1.1.0",
        "soundfile>=0.10.2",
    ],
    # target python versions
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "liteasr-train = liteasr.train:cli_main",
            "liteasr-infer = liteasr.infer:cli_main",
        ],
    },
)
