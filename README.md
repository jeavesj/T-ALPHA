![Project Logo](assets/logo.png)

-----------------------------------

[![](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gregory-kyro/CardioGenAI/blob/main/LICENSE)


# Abstract

There is significant interest in targeting disease-causing proteins with small molecule inhibitors to restore healthy cellular states. The ability to accurately predict the binding affinity of small molecules to a protein target in silico enables the rapid identification of candidate inhibitors and facilitates the optimization of on-target potency. In this work, we present T-ALPHA, a novel deep learning model that enhances protein-ligand binding affinity prediction by integrating multimodal feature representations within a hierarchical transformer framework to capture information critical to accurately predicting binding affinity. T-ALPHA outperforms all existing models reported in the literature on multiple benchmarks designed to evaluate protein-ligand binding affinity scoring functions. Remarkably, T-ALPHA maintains state-of-the-art performance when utilizing predicted structures rather than crystal structures, a powerful capability in real-world drug discovery applications where experimentally determined structures are often unavailable or incomplete. Additionally, we present an uncertainty-aware self-learning method for protein-specific alignment that does not require additional experimental data, and demonstrate that it improves T-ALPHA’s ability to rank compounds by binding affinity to biologically significant targets such as the SARS-CoV-2 main protease and the epidermal growth factor receptor. To facilitate implementation of T-ALPHA and reproducibility of all results presented in this paper, we have made all of our software available at this repository.


# Table of Contents
1. Installation
2. Accessing Data Files
3. Running the Scripts
    - Training the Model
    - Performing Inference
    - Monte Carlo Dropout and Semi-Supervised Fine-Tuning
4. On-the-Fly Inference Notebook
5. Contact


# Installation and Setup

### 1. Clone the Repository

```
git clone https://github.com/gregory-kyro/T-ALPHA.git
cd T-ALPHA
```

### 2. Set Up a Virtual Environment (Recommended)

```
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```


# Accessing Data Files

The pre-trained model parameters and test datasets are hosted on Zenodo for easy access.

### 1. Download the model parameters and test files using the following links:

- Model Parameters: [Download Here](https://zenodo.org/records/14510963/files/model_parameters.tar.gz?download=1)
- Test Files: [Download Here](https://zenodo.org/records/14510963/files/test_files.tar.gz?download=1)

### 2. Place the downloaded files in the data/ directory of the repository:

```
T-ALPHA/
├── data/
│   ├── model_parameters.tar.gz
│   ├── test_files.tar.gz
```

### 3. Extract the files. For Linux/macOS, you can use:

```
tar -xvzf test_files.tar.gz
tar -xvzf model_parameters.tar.gz
```


# Running the Scripts

This repository provides three primary scripts to train the model, perform inference, and run Monte Carlo Dropout for uncertainty estimation.

### 1. Training the Model

To train the model from scratch, run:

```
python scripts/train.py --train_set <path_to_train_hdf> \
                        --val_set <path_to_val_hdf> \
                        --checkpoint_dir <path_to_save_checkpoints> \
                        --save_dir <path_to_logs> \
                        --save_name <experiment_name>
```

### 2. Performing Inference

To evaluate the model on test data and compute key metrics, run:

```
python scripts/inference.py --ckpt_path <path_to_checkpoint> \
                            --test_set_path <path_to_test_hdf>
```

This will produce:
- A CSV file with predictions
- A scatter plot of predictions vs. true values

### 3. Estimating Uncertainty via Monte Carlo Dropout

To perform uncertainty estimation with Monte Carlo Dropout, run:

```
python scripts/mc_dropout.py --checkpoint <path_to_checkpoint.ckpt> \
                             --test_set <path_to_test_dataset> \
                             --output_results <path_to_save_results.csv> \
                             --output_weighted <path_to_save_weighted_results.csv>
                                
```


# On-the-Fly Inference Notebook

We are actively developing a Jupyter Notebook for facile inference. This notebook will allow users to swiftly predict protein-ligand binding affinity using either sequence-based inputs (protein sequence and ligand SMILES) or structural inputs (PDB files, SDF files).

Stay tuned for updates!


# Contact

For questions, issues, or collaborations, feel free to reach out:
- Name: Gregory W. Kyro
- Email: gregory.kyro@yale.edu
