import os

from setuptools import find_packages, setup

NAME = "greedy-mpi"

def get_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


setup(
    name=NAME,
    version="0.1",
    author="Rory Smith and Avi Vajpeyi",
    author_email="email",
    url="https://git.ligo.org/rory-smith/greedy-mpi/-/project_members",
    license="MIT",
    description="parallelize greedy reduced basis code",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "black",
        "isort",
        "scipy",
        "pytest",
    ],
    keywords=["reduced order modeling"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
