import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="privpgd",
    version="0.0.1",
    description="Python implementation of the methods introduced in the paper: Privacy-preserving data release leveraging optimal transport and particle gradient descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anonymous",
    author_email="Anonymous",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, privacy, differential privacy, optimal transport, particle gradient descent",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy==1.26.2",
        "scipy==1.11.4",
        "pandas==2.1.4",
        "scikit-learn==1.2.2",
        "torch==2.1.2",
        "cvxpy==1.4.1",
        "disjoint-set==0.7.4",
        "networkx==3.1",
        "autodp==0.2.3.1",
        "openml==0.14.1",
        "POT==0.9.1",
        "folktables==0.0.12",
        "seaborn==0.13.0",
        "click==8.1.7",
    ],
    python_requires="==3.11.5",
)
