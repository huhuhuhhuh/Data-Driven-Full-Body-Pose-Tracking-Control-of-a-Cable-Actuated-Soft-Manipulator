# Data-Driven-Full-Body-Pose-Tracking-Control-of-a-Cable-Actuated-Soft-Manipulator
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 目录

- [框架概述](#-框架概述)
- [仓库结构](#-仓库结构)
- [安装与环境配置](#-安装与环境配置)
- [数据准备](#-数据准备)
- [贝叶斯超参数优化](#-贝叶斯超参数优化)
- [训练流程](#-训练流程)
- [应用于新机械臂构型](#-应用于新机械臂构型)
- [推理与评估](#-推理与评估)
- [重新学习机制](#-重新学习机制)
- [常见问题](#-常见问题)
- [引用](#-引用)

---

## 🎯 框架概述

本框架由三个核心模块组成，完整实现了从数据采集到实物部署的全链路控制：

| 模块 | 功能描述 | 核心算法 |
|:----:|:---------|:---------|
| **FBPoseNet** | 全身位姿分析与特征提取 | HMDS + RDAM + M-DCN |
| **MMoE + PBM** | 前向动力学建模 | Multi-gate MoE + PBM + Encoder |
| **TD3 Agent** | 最优控制策略学习 | Twin Delayed DDPG |

---

## 📁 仓库结构

```bash
FBPoseNet/
├── fbposenet/                      # FBPoseNet 模块
│   ├── hmds.py                     # 混合多维缩放
│   ├── rdam.py                     # 区域差分注意力机制
│   └── mdcn.py                     # 多通道深度卷积网络
│
├── mmoe/                           # MMoE + PBM 模块
│   ├── encoder.py                  # 控制输入编码 + 随机掩码
│   ├── experts.py                  # 多专家网络 + 门控
│   └── pbm.py                      # 参数平衡机制
│
├── td3/                            # TD3 强化学习模块
│   ├── actor.py                    # Actor 网络
│   ├── critic.py                   # 双 Critic 网络
│   └── trainer.py                  # TD3 训练器
│
├── utils/
│   ├── bilut.py                    # 双射查找表
│   └── data_loader.py              # 数据加载与预处理
│
├── configs/
│   └── default.yaml                # 配置文件模板
│
├── scripts/
│   ├── train_fbposenet.py
│   ├── train_mmoe.py
│   └── train_td3.py
│
├── optimization/                   # 贝叶斯超参数优化
│   ├── optimizer.py
│   └── search_space.yaml
│
├── data/
│   └── README.md                   # 数据格式说明
│
├── environment.yml
└── README.md
```

---

## ⚙️ 安装与环境配置

```bash
# 克隆仓库
git clone https://github.com/your-org/FBPoseNet.git
cd FBPoseNet

# 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate fbposenet
```

**主要依赖：**

| 依赖 | 版本 |
|:-----|:----:|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.12 |
| NumPy | ≥ 1.21 |
| SciPy | ≥ 1.7 |
| scikit-learn | ≥ 1.0 |
| PyYAML | ≥ 6.0 |

---

## 📊 数据准备

### 数据格式要求

原始数据 CSV 文件格式：

| 列 1–90 | 列 91–94 |
|:--------|:---------|
| 30个标记点坐标 (x,y,z) | 4个伺服控制信号 |

### 数据预处理

```bash
# 转换为训练格式
python utils/preprocess.py \
    --input data/raw/dataset.csv \
    --output data/processed/train.npz

# 划分训练/验证/测试集 (80%/10%/10%)
python utils/split_data.py \
    --input data/processed/train.npz \
    --output_dir data/processed/
```

---

## 🔍 贝叶斯超参数优化

```bash
python optimization/optimizer.py \
    --config configs/default.yaml \
    --search_space optimization/search_space.yaml \
    --n_trials 200 \
    --output_dir optimization/results/
```

**搜索空间定义**（`optimization/search_space.yaml`）：

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

## 🚀 训练流程

### 步骤 1：训练 FBPoseNet

```bash
python scripts/train_fbposenet.py \
    --config configs/default.yaml \
    --data_dir data/processed/ \
    --epochs 100 \
    --batch_size 64 \
    --output_dir checkpoints/fbposenet/
```

| 配置项 | 值 |
|:-------|:--:|
| 输入维度 | 90 (30个标记点 × 3) |
| 损失函数 | 重构误差 + 扰动鲁棒性正则项 |
| 收敛损失 | 训练 0.23 / 测试 1.33 |

### 步骤 2：训练 MMoE

```bash
python scripts/train_mmoe.py \
    --config configs/default.yaml \
    --fbposenet_path checkpoints/fbposenet/fbposenet_best.pth \
    --data_dir data/processed/ \
    --epochs 400 \
    --batch_size 128 \
    --output_dir checkpoints/mmoe/
```

| 配置项 | 值 |
|:-------|:--:|
| 输入 | Encoder 编码后的伺服信号 |
| 损失函数 | RMSE + PBM 梯度修正 |
| 收敛损失 | 训练 0.38 mm / 测试 1.50 mm |

### 步骤 3：训练 TD3 Agent

```bash
python scripts/train_td3.py \
    --config configs/default.yaml \
    --mmoe_path checkpoints/mmoe/mmoe_best.pth \
    --sim_env isaac_sim \
    --episodes 5000 \
    --output_dir checkpoints/td3/
```

| 配置项 | 值 |
|:-------|:--:|
| 状态空间 | 位姿 + 目标 + 误差 + 障碍物距离 |
| 动作空间 | 4个伺服控制信号增量 |
| 奖励函数 | 目标接近奖励 + 碰撞惩罚 (-15) + 越界惩罚 (-20) |
| 收敛误差 | 1.83 mm (MED) |

### 推理加速部署（TensorRT）

```bash
# 导出 ONNX
python scripts/export_onnx.py \
    --checkpoint checkpoints/td3/td3_best.pth \
    --output models/policy.onnx

# 转换为 TensorRT
python scripts/convert_trt.py \
    --onnx models/policy.onnx \
    --output models/policy.plan \
    --fp16
```

> **推理性能：** 单次推理 ≤ 30 ms，满足实时控制需求。

---
## 🔄 应用于新机械臂构型

### 模块迁移需求总览

当机器人构型发生改变时（分段数量、标记点布局或驱动方式变化），各模块的迁移需求如下：

| Module | Network Architecture Reused? | Items Requiring Redesign | Hyperparameters to be Determined via Bayesian Optimization | Retraining Required? |
|:-------|:---------------------------:|:-------------------------|:-----------------------------------------------------------|:--------------------:|
| **FBPoseNet** | Partially reusable: architecture transferable, but non-metric MDS geometric constraints are morphology-specific and require redesign. | The geometric constraint formulation in HMDS and the input layer size must be re-derived and re-calculated for new manipulator morphologies. | MDS embedding dimension, number of attention heads, number of depthwise convolution channels, number of MLP layers and hidden units, learning rate | ✅ Yes |
| **Encoder** | Partially reusable: linear projection and random masking transferable, but encoding logic is servo-specific and requires redesign. | The feature encoding logic must be redesigned for different actuation mechanisms (e.g., pneumatic, hydraulic, or other transmission forms). | Encoded feature dimension, mask ratio, learning rate | ✅ Yes |
| **MMoE** | Fully reusable: expert networks, gating networks, PBM mechanism transferable. | None | Number of experts, expert hidden layer size and depth, gating network structure, task-specific loss weights, learning rate | ✅ Yes |
| **TD3 Agent** | Fully reusable: Actor/Critic MLP structure transferable. | The state and action space dimensions must be re-calculated based on the actuation mechanism and workspace of the new manipulator morphology. | Number of Actor/Critic layers and hidden units, Actor/Critic learning rates, discount factor, target network update rate, exploration noise standard deviation | ✅ Yes |
| **Re-learning MLP** | Fully reusable: linear and dropout alternating structure transferable. | None | Number of MLP layers and hidden units, dropout ratio, learning rate | ✅ Yes |

### 新构型适配步骤

#### 步骤 1：修改配置文件

```yaml
# configs/default_new.yaml
data:
  marker_count: <新标记点总数>        # 例如：原CASM为30，新构型为24
  actuator_count: <新驱动单元数量>    # 例如：原CASM为4，新构型为6

fbposenet:
  input_dim: <marker_count × 3>      # 24 × 3 = 72
  output_dim: <目标特征维数>          # 根据新构型复杂度设置，建议16–64

mmoe:
  encoder_input_dim: <actuator_count>
  num_outputs: <fbposenet.output_dim>

td3:
  state_dim: <fbposenet.output_dim + actuator_count + 目标位姿 + 障碍物>
  action_dim: <actuator_count>
```

#### 步骤 2：运行贝叶斯优化

```bash
python optimization/optimizer.py \
    --config configs/default_new.yaml \
    --search_space optimization/search_space_new.yaml \
    --n_trials 200
```

#### 步骤 3–5：依次训练

```bash
python scripts/train_fbposenet.py --config configs/default_new.yaml
python scripts/train_mmoe.py --config configs/default_new.yaml
python scripts/train_td3.py --config configs/default_new.yaml
```

> **核心要点：** 迁移至新构型时，MMoE、TD3 Agent和重新学习MLP的网络架构可直接复用；FBPoseNet的HMDS几何约束和Encoder的特征编码逻辑需根据新构型重新设计。所有超参数通过贝叶斯优化自动确定，并在新数据上重新训练。**无需人工经验调参。**

---

## 🧪 推理与评估

```bash
# 单次推理
python scripts/inference.py \
    --checkpoint checkpoints/td3/td3_best.pth \
    --target data/targets/target_pose.npy

# 批量评估
python scripts/evaluate.py \
    --checkpoint checkpoints/td3/td3_best.pth \
    --test_data data/processed/test.npz \
    --output_dir results/
```

**评估指标：** 平均欧氏距离（MED）

---

## 🧬 重新学习机制

当末端负载变化时，冻结基础网络，仅训练 MLP 适配模块：

```bash
python scripts/relearn.py \
    --base_checkpoint checkpoints/td3/td3_best.pth \
    --load_data data/load_50g/train.npz \
    --epochs 50 \
    --output_dir checkpoints/td3_load50g/
```

---

## ❓ 常见问题

<details>
<summary><b>数据采集速度为什么是 1 step/s？</b></summary>

低速采集是为了确保每个采样位形达到准静态平衡，从而获取高质量、低噪声的训练样本以准确表征机械臂的全身位姿空间，而非定义系统的运行速度边界。
</details>

<details>
<summary><b>标记点数量变化时如何处理？</b></summary>

仅需修改配置文件中 `input_dim = 新标记点数量 × 3`，网络结构无需调整。
</details>

<details>
<summary><b>电机数量变化时如何处理？</b></summary>

仅需修改配置文件中 `encoder_input_dim = 新电机数量` 和 `action_dim = 新电机数量`，网络结构无需调整。
</details>

<details>
<summary><b>训练好的策略能否直接用于不同构型？</b></summary>

不能直接使用。网络结构可复用，但网络参数（权重）必须在新的数据集上重新训练。
</details>

<details>
<summary><b>是否需要手动调整超参数？</b></summary>

不需要。所有超参数通过贝叶斯优化自动寻优。
</details>

<details>
<summary><b>仿真到实物的迁移差距如何？</b></summary>

实物与仿真误差分布高度一致，平均误差增幅仅 0.04–0.09 mm，标准偏差增幅约 0.02–0.03 mm。
</details>

---

## 📄 引用

```bibtex
@article{fbposenet2026,
  title={Data-Driven-Full-Body-Pose-Tracking-Control-of-a-Cable-Actuated-Soft-Manipulator},
  author={...},
  journal={...},
  year={2026}
}
```

---

## 📧 联系方式

如有问题，请通过 [GitHub Issues](https://github.com/your-org/FBPoseNet/issues) 联系。
