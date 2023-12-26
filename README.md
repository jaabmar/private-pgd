# PrivPGD: Particle Gradient Descent and Optimal Transport for Private Tabular Data Synthesis

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2312.03871)
[![Python 3.11.5](https://img.shields.io/badge/python-3.11.5-blue.svg)](https://python.org/downloads/release/python-3115/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pytorch 2.1.2](https://img.shields.io/badge/pytorch-2.1.2-green.svg)](https://pytorch.org/)


This repository contains the Python implementation of **PrivPGD**, a generation method for marginal-based private data synthesis introduced in the paper [Privacy-preserving data release leveraging optimal transport and particle gradient descent](https://arxiv.org/abs/2312.03871).

* [Overview](#overview)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Example Scripts and Tutorial](#example-scripts-and-tutorial)
* [Contributing](#contributing)
* [Contact](#contact)
* [Citation](#citation)

## Overview

## Getting Started

### Dependencies

- Python 3.11.5
- Numpy 1.26.2
- Scipy 1.11.4
- Scikit-learn 1.2.2
- Pandas 2.1.4
- Torch 2.1.2
- CVXPY 1.4.1
- Disjoint Set 0.7.4
- Networkx 3.1
- Autodp 0.2.3.1
- POT 0.9.1
- Folktables 0.0.12
- Openml 0.14.1
- Seaborn 0.13.0

### Installation

To set up your environment and install the package, follow these steps:

#### Create and Activate a Conda Environment

Start by creating a Conda environment with Python 3.11.5. This step ensures your package runs in an environment with the correct Python version. 
```bash
conda create -n myenv python=3.11.5
conda activate myenv
```
#### Install the Package

There are two ways to install the package:

1. **Local Installation:**
   If you have the package locally, upgrade `pip` to its latest version. Then, use the local setup files to install your package. This method is ideal for development or when you have the source code.
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```
2. **Direct Installation from GitHub:**
   You can also install the package directly from GitHub. This method is straightforward and ensures you have the latest version.
   ```bash
   pip install git+https://github.com/jaabmar/private-pgd.git
   ```
## Usage

## Example Scripts and Tutorial

## Contributing

We welcome contributions to improve this project. Here's how you can contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact

For any inquiries, please reach out:

- Javier Abad Martinez - [javier.abadmartinez@ai.ethz.ch](mailto:javier.abadmartinez@ai.ethz.ch)
- Konstantin Donhauser - [konstantin.donhauser@ai.ethz.ch](mailto:konstantin.donhauser@ai.ethz.ch)
- Neha Hulkund - [nhulkund@mit.edu](mailto:nhulkund@mit.edu)

## Citation

If you find this code useful, please consider citing our paper:
 ```
@article{donhauser2023leveraging,
  title={Privacy-preserving data release leveraging optimal transport and particle gradient descent},
  author={Konstantin Donhauser and Javier Abad and Neha Hulkund and Fanny Yang},
  year={2023},
  journal={arXiv preprint arXiv:2312.03871},
  eprint={2312.03871},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```
