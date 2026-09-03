# Data-Driven Full-Body Pose Tracking Control of a Cable-Actuated Soft Manipulator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Table of Contents

- [Framework Overview](#-framework-overview)
- [Repository Structure](#-repository-structure)
- [Installation and Environment Configuration](#-installation-and-environment-configuration)
- [Data Preparation](#-data-preparation)
- [Bayesian Hyperparameter Optimization](#-bayesian-hyperparameter-optimization)
- [Training Pipeline](#-training-pipeline)
- [Adaptation to New Manipulator Morphologies](#-adaptation-to-new-manipulator-morphologies)
- [Inference and Evaluation](#-inference-and-evaluation)
- [Re-learning Mechanism](#-re-learning-mechanism)
- [Frequently Asked Questions](#-frequently-asked-questions)
- [Citation](#-citation)

---

## 🎯 Framework Overview

This framework comprises three core modules, implementing the complete control pipeline from data acquisition to physical deployment:

| Module | Function | Core Algorithm |
|:----:|:---------|:---------|
| **FBPoseNet** | Full-body pose analysis and feature extraction | HMDS + RDAM + M-DCN |
| **MMoE + PBM** | Forward dynamics modeling | Multi-gate MoE + PBM + Encoder |
| **TD3 Agent** | Optimal control policy learning | Twin Delayed DDPG |

---

## 📁 Repository Structure

```bash
Data-Driven Full-Body Pose Tracking Control of a Cable-Actuated Soft Manipulator/
├── fbposenet/                      # FBPoseNet module
│   ├── hmds.py                     # Hybrid multidimensional scaling
│   ├── rdam.py                     # Regional differential attention mechanism
│   └── mdcn.py                     # Multi-channel deep convolutional network
│
├── mmoe/                           # MMoE + PBM module
│   ├── encoder.py                  # Control input encoding + random masking
│   ├── experts.py                  # Multi-expert networks + gating
│   └── pbm.py                      # Parameter balancing mechanism
│
├── td3/                            # TD3 reinforcement learning module
│   ├── actor.py                    # Actor network
│   ├── critic.py                   # Twin Critic networks
│   └── trainer.py                  # TD3 trainer
│
├── utils/
│   ├── bilut.py                    # Bijective lookup table
│   └── data_loader.py              # Data loading and preprocessing
│
├── configs/
│   └── default.yaml                # Configuration file template
│
├── scripts/
│   ├── train_fbposenet.py
│   ├── train_mmoe.py
│   └── train_td3.py
│
├── optimization/                   # Bayesian hyperparameter optimization
│   ├── optimizer.py
│   └── search_space.yaml
│
├── data/
│   └── CASM.csv                   # Data format documentation
│
├── environment.yml
└── README.md
```

---

## ⚙️ Installation and Environment Configuration

```bash
# Clone the repository
git clone https://github.com/your-org/FBPoseNet.git
cd FBPoseNet

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate fbposenet
```

**Primary Dependencies:**

| Dependency | Version |
|:-----|:----:|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.12 |
| NumPy | ≥ 1.21 |
| SciPy | ≥ 1.7 |
| scikit-learn | ≥ 1.0 |
| PyYAML | ≥ 6.0 |

---

## 📊 Data Preparation

### Data Format Requirements

Raw data CSV file format:

| Columns 1–90 | Columns 91–94 |
|:--------|:---------|
| Coordinates of 30 markers (x, y, z) | 4 servo control signals |

### Data Preprocessing

```bash
# Convert to training format
python utils/preprocess.py \
    --input data/raw/dataset.csv \
    --output data/processed/train.npz

# Split into training/validation/test sets (80%/10%/10%)
python utils/split_data.py \
    --input data/processed/train.npz \
    --output_dir data/processed/
```

---

## 🔍 Bayesian Hyperparameter Optimization

```bash
python optimization/optimizer.py \
    --config configs/default.yaml \
    --search_space optimization/search_space.yaml \
    --n_trials 200 \
    --output_dir optimization/results/
```

**Search Space Definition** (`optimization/search_space.yaml`):

```yaml
fbposenet:
  mds_dim: [8, 16, 32, 64]
  num_heads: [2, 4, 8]
  mdcn_channels: [16, 32, 64, 128]
  hidden_layers: [2, 3, 4]
  hidden_dim: [64, 128, 256]
  learning_rate: [1e-5, 1e-3]

mmoe:
  num_experts: [4, 6, 8, 12]
  expert_hidden_dim: [64, 128, 256]
  num_hidden_layers: [2, 3, 4]
  learning_rate: [1e-5, 1e-3]

td3:
  actor_hidden_dim: [128, 256, 512]
  critic_hidden_dim: [128, 256, 512]
  num_hidden_layers: [2, 3, 4]
  actor_lr: [1e-5, 1e-3]
  critic_lr: [1e-5, 1e-3]
  gamma: [0.95, 0.999]
  tau: [0.001, 0.01]
  exploration_noise: [0.05, 0.2]
```

---

## 🚀 Training Pipeline

### Step 1: Train FBPoseNet

```bash
python scripts/train_fbposenet.py \
    --config configs/default.yaml \
    --data_dir data/processed/ \
    --epochs 100 \
    --batch_size 64 \
    --output_dir checkpoints/fbposenet/
```

| Configuration | Value |
|:-------|:--:|
| Input dimension | 90 (30 markers × 3) |
| Loss function | Reconstruction error + perturbation robustness regularization |
| Converged loss | Training 0.23 / Test 1.33 |

### Step 2: Train MMoE

```bash
python scripts/train_mmoe.py \
    --config configs/default.yaml \
    --fbposenet_path checkpoints/fbposenet/fbposenet_best.pth \
    --data_dir data/processed/ \
    --epochs 400 \
    --batch_size 128 \
    --output_dir checkpoints/mmoe/
```

| Configuration | Value |
|:-------|:--:|
| Input | Servo signals encoded by the Encoder |
| Loss function | RMSE + PBM gradient correction |
| Converged loss | Training 0.38 mm / Test 1.50 mm |

### Step 3: Train TD3 Agent

```bash
python scripts/train_td3.py \
    --config configs/default.yaml \
    --mmoe_path checkpoints/mmoe/mmoe_best.pth \
    --sim_env isaac_sim \
    --episodes 5000 \
    --output_dir checkpoints/td3/
```

| Configuration | Value |
|:-------|:--:|
| State space | Pose + target + error + obstacle distance |
| Action space | Increments of 4 servo control signals |
| Reward function | Target-approaching reward + collision penalty (−15) + out-of-bounds penalty (−20) |
| Converged error | 1.83 mm (MED) |

### Accelerated Inference Deployment (TensorRT)

```bash
# Export to ONNX
python scripts/export_onnx.py \
    --checkpoint checkpoints/td3/td3_best.pth \
    --output models/policy.onnx

# Convert to TensorRT
python scripts/convert_trt.py \
    --onnx models/policy.onnx \
    --output models/policy.plan \
    --fp16
```

> **Inference performance:** Single inference ≤ 30 ms, satisfying real-time control requirements.

---

## 🔄 Adaptation to New Manipulator Morphologies

### Summary of Module Transfer Requirements

When the robot morphology changes (e.g., variations in the number of segments, marker layout, or actuation mechanism), the transfer requirements for each module are as follows:

| Module | Network Architecture Reused? | Items Requiring Redesign | Hyperparameters to be Determined via Bayesian Optimization | Retraining Required? |
|:-------|:---------------------------:|:-------------------------|:-----------------------------------------------------------|:--------------------:|
| **FBPoseNet** | Partially reusable: architecture transferable, but non-metric MDS geometric constraints are morphology-specific and require redesign. | The geometric constraint formulation in HMDS and the input layer size must be re-derived and re-calculated for new manipulator morphologies. | MDS embedding dimension, number of attention heads, number of depthwise convolution channels, number of MLP layers and hidden units, learning rate | ✅ Yes |
| **Encoder** | Partially reusable: linear projection and random masking transferable, but encoding logic is servo-specific and requires redesign. | The feature encoding logic must be redesigned for different actuation mechanisms (e.g., pneumatic, hydraulic, or other transmission forms). | Encoded feature dimension, mask ratio, learning rate | ✅ Yes |
| **MMoE** | Fully reusable: expert networks, gating networks, and PBM mechanism transferable. | None | Number of experts, expert hidden layer size and depth, gating network structure, task-specific loss weights, learning rate | ✅ Yes |
| **TD3 Agent** | Fully reusable: Actor/Critic MLP structure transferable. | The state and action space dimensions must be re-calculated based on the actuation mechanism and workspace of the new manipulator morphology. | Number of Actor/Critic layers and hidden units, Actor/Critic learning rates, discount factor, target network update rate, exploration noise standard deviation | ✅ Yes |
| **Re-learning MLP** | Fully reusable: linear and dropout alternating structure transferable. | None | Number of MLP layers and hidden units, dropout ratio, learning rate | ✅ Yes |

### Adaptation Steps for a New Morphology

#### Step 1: Modify the Configuration File

```yaml
# configs/default_new.yaml
data:
  marker_count: <new total marker count>        # e.g., 30 for the original CASM, 24 for the new morphology
  actuator_count: <new number of actuation units>    # e.g., 4 for the original CASM, 6 for the new morphology

fbposenet:
  input_dim: <marker_count × 3>      # 24 × 3 = 72
  output_dim: <target feature dimension>          # set according to the complexity of the new morphology; 16–64 recommended

mmoe:
  encoder_input_dim: <actuator_count>
  num_outputs: <fbposenet.output_dim>

td3:
  state_dim: <fbposenet.output_dim + actuator_count + target pose + obstacles>
  action_dim: <actuator_count>
```

#### Step 2: Run Bayesian Optimization

```bash
python optimization/optimizer.py \
    --config configs/default_new.yaml \
    --search_space optimization/search_space_new.yaml \
    --n_trials 200
```

#### Steps 3–5: Train Sequentially

```bash
python scripts/train_fbposenet.py --config configs/default_new.yaml
python scripts/train_mmoe.py --config configs/default_new.yaml
python scripts/train_td3.py --config configs/default_new.yaml
```

> **Key point:** When transferring to a new morphology, the network architectures of MMoE, the TD3 Agent, and the Re-learning MLP can be directly reused. The HMDS geometric constraints of FBPoseNet and the feature encoding logic of the Encoder must be redesigned according to the new morphology. All hyperparameters are automatically determined via Bayesian optimization, and all modules are retrained on the new dataset. **No manual hyperparameter tuning is required.**

---

## 🧪 Inference and Evaluation

```bash
# Single inference
python scripts/inference.py \
    --checkpoint checkpoints/td3/td3_best.pth \
    --target data/targets/target_pose.npy

# Batch evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/td3/td3_best.pth \
    --test_data data/processed/test.npz \
    --output_dir results/
```

**Evaluation metric:** Mean Euclidean Distance (MED)

---

## 🧬 Re-learning Mechanism

When the end-effector payload changes, the base network is frozen and only the MLP adaptation module is trained:

```bash
python scripts/relearn.py \
    --base_checkpoint checkpoints/td3/td3_best.pth \
    --load_data data/load_50g/train.npz \
    --epochs 50 \
    --output_dir checkpoints/td3_load50g/
```

---

## ❓ Frequently Asked Questions

<details>
<summary><b>Why is the data acquisition rate 1 step/s?</b></summary>

Low-speed acquisition ensures that each sampled configuration reaches quasi-static equilibrium, thereby yielding high-quality, low-noise training samples that accurately characterize the full-body pose space of the manipulator, rather than defining the operational speed boundary of the system.
</details>

<details>
<summary><b>How to handle a change in the number of markers?</b></summary>

Simply modify the configuration file by setting `input_dim = new marker count × 3`. No changes to the network architecture are required.
</details>

<details>
<summary><b>How to handle a change in the number of actuators?</b></summary>

Simply modify the configuration file by setting `encoder_input_dim = new actuator count` and `action_dim = new actuator count`. No changes to the network architecture are required.
</details>

<details>
<summary><b>Can a trained policy be directly applied to a different morphology?</b></summary>

No, it cannot be directly applied. The network architecture can be reused, but the network parameters (weights) must be retrained on a new dataset.
</details>

<details>
<summary><b>Is manual hyperparameter tuning required?</b></summary>

No. All hyperparameters are automatically optimized via Bayesian optimization.
</details>

<details>
<summary><b>What is the simulation-to-reality transfer gap?</b></summary>

The error distributions of the physical system and simulation are highly consistent, with an average error increase of only 0.04–0.09 mm and a standard deviation increase of approximately 0.02–0.03 mm.
</details>

---

## 📄 Citation

```bibtex
@article{fbposenet2026,
  title={Data-Driven-Full-Body-Pose-Tracking-Control-of-a-Cable-Actuated-Soft-Manipulator},
  author={...},
  journal={...},
  year={2026}
}
```

---

## 📧 Contact

For questions, please contact us via [GitHub Issues](https://github.com/your-org/FBPoseNet/issues).
