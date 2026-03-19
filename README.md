# Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling — Official Code Release

*Formerly titled:* **Strengthening Generative Robot Policies through Predictive World Modeling**

Official implementation of:

> **Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling**  
> *(formerly titled: Strengthening Generative Robot Policies through Predictive World Modeling)*  
> Han Qi, Haocheng Yin, Aris Zhu, Yilun Du, Heng Yang  
> Accepted to IEEE Robotics and Automation Letters (RA-L)  
> arXiv 2025  
> Paper: https://arxiv.org/abs/2502.00622  

---

## Overview

This repository provides the official implementation of the framework proposed in the paper:

> We strengthen diffusion-based generative robot policies by integrating a predictive world model that enables long-horizon reasoning and improved robustness.

The framework consists of two main components:

1. **Diffusion-based Action Policy**  
   Generates action sequences using a generative diffusion model.

2. **Predictive World Model**  
   Learns environment dynamics to evaluate and refine candidate action trajectories.

At inference time, the world model enhances policy performance through trajectory prediction and ranking/optimization.

We use push-T experiment as an example in code.

---

## Repository Structure

```
.
├── all_checkpoint/                   # Pretrained checkpoints (policy + world model)
├── diffusion_policy_data/            # Training data for diffusion action policy
├── diffusion_policy_training/        # Training code for diffusion-based action policy
├── gpc_opt_evaluation/               # Evaluation with GPC-OPT (trajectory optimization)
├── gpc_rank_evaluation/              # Evaluation with GPC-RANK (trajectory ranking)
├── world_model_data/                 # Training data for predictive world model
├── world_model_train_phase_one/      # Phase I: single-step world model warmup training
└── world_model_train_phase_two/      # Phase II: multi-step world model training
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/han20192019/gpc_code.git
cd gpc_code
```

### 2. Install dependencies

We recommend using a clean conda environment:

```bash
conda env create -f environment.yml
conda activate gpc
```

---

## Checkpoints & Datasets

- **Pretrained checkpoints:**  
  `https://huggingface.co/han2019/gpc_checkpoints/tree/main`

Please download the checkpoints under a folder named 'all_checkpoint' in the root folder.

- **Diffusion policy training dataset:**  
  `https://huggingface.co/datasets/han2019/gpc_pushT_data/tree/main/diffusion_policy_data`

- **World model training dataset:**  
  `https://huggingface.co/datasets/han2019/gpc_pushT_data/tree/main/world_model_data`

Please download the datasets and place them in the root folder.

---

# Training

There are **two independent modules** to train:

---

## 1️⃣ Train the Diffusion Action Policy

Directory:

```
diffusion_policy_training/
```

Run:

```bash
python train_model.py
```

This trains the diffusion-based generative policy that produces candidate action sequences.

---

## 2️⃣ Train the Predictive World Model

World model training is performed in **two stages**, as described in the paper.

---

### (a) Phase One — Single-Step Warmup Training

Directory:

```
world_model_train_phase_one/
```

Run:

```bash
python train.py
```

This stage trains the world model for **single-step prediction**, which stabilizes early training and improves multi-step rollout performance.

---

### (b) Phase Two — Multi-Step Training

Directory:

```
world_model_train_phase_two/
```

Run:

```bash
python train.py
```

This stage trains the model for **multi-step rollouts**, enabling long-horizon trajectory evaluation.

---

# Evaluation

After training both the policy and world model, you can evaluate the integrated system.

---

## GPC-RANK (Trajectory Ranking)

Directory:

```
gpc_rank_evaluation/
```

Run:

```bash
python gpc_rank_evaluation.py
```

This mode:
- Samples candidate action sequences from the diffusion policy
- Uses the predictive world model to simulate future states
- Ranks trajectories
- Executes the highest-scoring candidate

---

## GPC-OPT (Trajectory Optimization)

Directory:

```
gpc_opt_evaluation/
```

Run:

```bash
python gpc_opt_evaluation.py
```

This mode:
- Uses the world model to iteratively optimize action sequences
- Improves performance through predictive refinement

---

## Community Implementations

A Rust reimplementation of this work is available:

- **gpc_rs (Rust workspace)**  
  https://github.com/AbdelStark/gpc_rs  

This project provides a Rust-based implementation of the GPC framework,
including inference-time trajectory ranking and optimization.

We thank the author for their contribution to making this work more accessible
to the Rust and systems programming community.

---



# Citation

If you find this work useful, please cite:

```bibtex
@article{qi25ral-gpc,
    title={Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling},
    note={Previously titled: Strengthening Generative Robot Policies through Predictive World Modeling},
    author={Qi, Han and Yin, Haocheng and Zhu, Aris and Du, Yilun and Yang, Heng},
    journal={IEEE Robotics and Automation Letters (RAL)},
    year={2025}
  }
```



