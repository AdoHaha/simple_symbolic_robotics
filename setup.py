import setuptools
from setuptools import find_packages

# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple_symbolic_robotics",
    version="0.1.0",  # Make sure this is updated for new versions
    author="Igor Zubrycki",
    author_email="igorzubrycki@gmail.com",
    description="Helper functions for fundamental robotics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdoHaha/simple_symbolic_robotics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
    ],
)
