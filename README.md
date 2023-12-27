# PrivPGD: Particle Gradient Descent and Optimal Transport for Private Tabular Data Synthesis

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2312.03871)
[![Python 3.11.5](https://img.shields.io/badge/python-3.11.5-blue.svg)](https://python.org/downloads/release/python-3115/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pytorch 2.1.2](https://img.shields.io/badge/pytorch-2.1.2-green.svg)](https://pytorch.org/)


This repository contains the Python implementation of **PrivPGD**, a generation method for marginal-based private data synthesis introduced in the paper [Privacy-preserving data release leveraging optimal transport and particle gradient descent](https://arxiv.org/abs/2312.03871).

* [Overview](#overview)
* [Contents](#contents)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Examples and Tutorial](#examples-and-tutorial)
* [Contributing](#contributing)
* [Contact](#contact)
* [Citation](#citation)

## Overview

The distribution of sensitive datasets plays a key role in data-driven decision-making across many fields, including healthcare and government. Nevertheless, the release of such datasets often leads to significant privacy concerns. Differential Privacy (DP) has emerged as an effective solution to address these concerns, ensuring privacy preservation in our increasingly data-centric world.

PrivPGD is a novel approach for differentially private tabular data synthesis. It creates high-quality, privacy-preserving copies of protected tabular datasets from noisy measurements of their marginals. PrivPGD leverages particle gradient descent coupled with an optimal transport-based divergence, which facilitates the efficient integration of marginal information during the dataset generation process.

Key advantages of PrivPGD include:

- State-of-the-Art Performance: Demonstrates superior performance in benchmarks and downstream tasks, especially with large datasets.
- Scalability: Features an optimized gradient computation suitable for parallelization on modern GPUs, making it suitable for handling large datasets and many marginals.
- Geometry Preservation: Retains the geometry of dataset features, suck as rankings, aligning more naturally with the nuances of real-world data.
- Domain-Specific Constraints Incorporation: Enables the inclusion of additional constraints in the synthetic data generation process.

## Contents

The `src` folder contains the core code of the package, organized into several subfolders, each catering to specific functionalities:

### 1. Mechanisms (`src/mechanisms`):
   - Handles marginal selection and privatization.
   - Key files and their corresponding mechanisms:
     - `kway.py`: Implements the K-Way mechanism.
     - `mwem.py`: Implements the MWEM.
     - `mst.py`: Implements the MST mechanism.
   - Additional utility files supporting these mechanisms are also located in this folder.

### 2. Data Handling (`src/data`):
   - Dedicated to downloading and processing data.
   - Contains scripts and modules for data manipulation, preparation, and loading.

### 3. Inference Methods (`src/inference`):
   - Contains the code for generation methods.
   - Subfolders and their specific methods:
     - `pgm`: Contains the implementation of the PGM method.
     - `privpgd`: Houses the PrivPGD method, our novel approach for differentially private data generation.

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

## Examples and Tutorial

## Examples and Tutorial

In the `examples` folder, you'll find practical examples showcasing how to use the package effectively. These examples are designed to help you understand the application of different mechanisms and methods included in the package.

### Key Experiment Scripts

1. **`experiment.py`**: This is a general file for running experiments. It's a versatile script that can be used for various experiment configurations.

2. **`mst+pgm.py`**: Use this script to run experiments with the PGM generation method, utilizing MST for marginal selection.

3. **`privpgd.py`**: This script is dedicated to running experiments with PrivPGD, the novel approach for differentially private data synthesis introduced in our paper.

### Running Experiments

To run experiments, you will interact with the scripts via the command line, and command handling is facilitated by Click (version 8.1.7). For example, to run an experiment with PrivPGD using the default hyperparameters and the setup described in our paper on the ACS Income California 2018 dataset, follow these steps:

1. Change directory (cd) to the `experiments` folder.
2. Run the command:

    ```bash
    python privpgd.py
    ```

This command will initiate the experiment with PrivPGD using the specified dataset and default settings.

### Step-by-Step Tutorial

For a detailed, step-by-step understanding of how PrivPGD works, refer to the `Tutorial.ipynb` notebook in the `examples` folder. This Jupyter notebook includes comprehensive explanations and visualizations, walking you through the entire process of using PrivPGD for differentially private data synthesis. 

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
