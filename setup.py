from setuptools import setup, find_packages

setup(
    name="ekarus",
    version="0.1.0",
    description="ekarus modeling and simulation package for adaptive optics and pyramid wavefront sensors",
    author="ekarus team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "arte",
        "matplotlib",
    ],
    python_requires=">=3.7",
)