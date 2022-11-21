
from setuptools import find_packages
from setuptools import setup


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


setup(
    name="fabmodel",
    version="0.0.1",
    url="https://github.com/Dusandinho/PreFab",
    license="MIT",
    author="Dusan",
    author_email="dusan@gmail.com",
    description="(Prediction of Fabrication) is used for modelling fabrication process",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    install_requires=get_install_requires(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
