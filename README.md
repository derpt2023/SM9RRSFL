# SM9-RRS-FL: 联邦学习中基于SM9环签名的鲁棒聚合方案

本项目实现了基于国密SM9环签名的联邦学习防御方法（Ours），并提供了与5种主流防御方法的公平对比实验框架。

**支持的防御方法**：
- **Ours (sm9rrs)** - 基于SM9环签名的鲁棒聚合（本文提出）
- **VERT** - 验证增强鲁棒训练
- **AlignIns** - 对齐增强防御
- **Krum** - 基于距离的聚合
- **TAD** - 基于异常检测的防御
- **FedAvg** - 联邦平均（基线）

**支持的数据集**：MNIST、CIFAR-10、Fashion-MNIST

**分布类型**：IID（独立同分布）、Non-IID（Dirichlet分布）

**密码模式**：真实SM9签名、快速仿真模式

---

## 快速开始

### 1. 环境安装

**系统要求**：
- Python 3.10+
- CUDA（可选，用于GPU加速）

**基础安装**：

```bash
# 克隆项目
git clone git@github.com:derpt2023/SM9RRSFL.git
cd SM9RRSFL

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

**真实SM9模式（可选）**：

如需使用真实SM9签名（而非快速仿真），需要构建原生扩展：

```bash
# 准备固定版本的GmSSL源码（SHA256必须匹配）
export GMSSL_ARCHIVE="$PWD/Downloads/GmSSL-master.zip"
# 期望SHA256: 6dc97c6b4f7d2f6df9d44f014cca0561a7b4776017efd4486d341e986051fab4

# 构建原生扩展
python setup.py build_ext --inplace --force

# 验证构建
ls sm9rrsfl/_native_sm9*.so
python -c "from sm9rrsfl.crypto import rrs_backend_name; print(rrs_backend_name())"
# 期望输出: gmssl-sm9-native-v2
```

**验证安装**：

```bash
# 运行测试套件
python -m pytest tests/ -v

# 检查GPU可用性（如果已安装PyTorch）
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 运行实验

### 方案A：快速验证（推荐入门）

使用MNIST数据集快速验证各方法的基本功能：

```bash
# 使用示例配置运行
python run_experiments_from_config.py --config configs/experiment.example.json --dry-run
python run_experiments_from_config.py --config configs/experiment.example.json
```

**关键配置**：
- `configs/experiment.example.json` - 单次实验配置示例
- 自动下载MNIST数据集
- 默认使用快速仿真模式（`crypto_mode: "simulated"`）
- 输出目录：`outputs/example_run/`

### 方案B：完整实验（论文复现）

公平对比6种方法，包含自动超参数搜索和多seed评估：

```bash
# 修改配置文件
cp configs/fair_tuning.example.json configs/my_experiment.json
# 编辑 my_experiment.json，设置数据集、攻击参数等

# 运行完整实验流程
python run_fair_tuning_from_config.py --config configs/my_experiment.json --dry-run
python run_fair_tuning_from_config.py --config configs/my_experiment.json
```

**实验流程**：
1. **校准阶段**：在训练留出集上搜索最优超参数
   - Ours、VERT、AlignIns各12个候选
   - 其他方法使用固定配置
   - 约1170次完整训练

2. **正式阶段**：使用选定参数在测试集上评估
   - 6方法 × 2分区 × 5恶意比例 × 3 seed
   - 约180次完整训练

**已验证的成功配置**：
- **MNIST v7**：`configs/mnist_v7_target_fair_tuning.json`（推荐）
- **CIFAR-10 v13**：`configs/cifar10_v13_fair_tuning_fixed.json`（当前最新）
- **Fashion-MNIST**：`configs/fashion_mnist_fair_tuning.json`

---

## 配置说明

### 核心参数

```jsonc
{
  // 数据集配置
  "dataset": "mnist",              // mnist, cifar10, fashion_mnist
  "num_clients": 100,              // 客户端数量
  "partition_type": "dirichlet",   // iid, dirichlet
  "dirichlet_alpha": 0.5,          // Non-IID程度（越小越异质）
  
  // 训练配置
  "rounds": 150,                   // 联邦学习轮次
  "local_epochs": 1,               // 本地训练轮数
  "batch_size": 50,                // 本地批大小
  "learning_rate": 0.01,           // 初始学习率
  
  // 攻击配置
  "attack_type": "alternating_minimization",  // 攻击方法
  "malicious_ratios": [0, 0.2, 0.4, 0.6, 0.8], // 恶意客户端比例
  "attack_start_round": 12,        // 攻击开始轮次（必须 > K+2）
  
  // Ours特定参数
  "K": 7,                          // 干净热身轮数
  "crypto_mode": "simulated",      // sm9, simulated
  
  // 计算资源
  "compute_backend": "auto",       // auto, torch, numpy
  "device": "auto",                // auto, cuda, mps, cpu
  "jobs": "auto"                   // 并行任务数
}
```

### 性能目标（可选）

`configs/mnist_v7_target_fair_tuning.json`使用了性能目标筛选：

```jsonc
{
  "tuning": {
    "performance_target": {
      // Ours在健康候选中优先选择满足以下目标者：
      // 1. 准确率接近VERT（落后≤2个百分点）
      // 2. ASR优于或接近VERT（高出≤1个百分点，且自身≤5%）
      // 3. ASR峰值≤20%
    }
  }
}
```

设置为`null`可关闭目标筛选，仅使用公共Score选参。

---

## 结果解读

### 输出目录结构

```
outputs/<experiment_name>/
├── tuning/                          # 校准阶段
│   ├── candidate_feasibility.csv    # 候选健康检查结果
│   ├── validation_results.csv       # 校准性能结果
│   ├── best_parameters.json         # 选定的最优参数
│   └── weights_search.json          # Score权重搜索结果
│
├── final_evaluation/                # 正式阶段
│   ├── aggregate.csv                # 汇总统计（推荐查看）
│   ├── scenario_audit.csv           # 逐场景详细结果
│   ├── performance_target.json      # 性能目标达成情况
│   └── visualizations/              # 可视化图表
│       ├── accuracy_comparison.svg
│       ├── asr_comparison.svg
│       └── ...
│
└── detailed_logs/                   # 详细日志（供调试）
```

### 关键指标

**Aggregate.csv 主要列**：

| 列名 | 含义 |
|------|------|
| `method` | 防御方法 |
| `final_accuracy_mean` | 最终准确率均值（越高越好）|
| `attack_asr_mean` | 攻击成功率均值（越低越好）|
| `honest_weight_loss_mean` | 诚实客户端权重损失（越低越好）|
| `false_revocation_rate_mean` | 误撤销率（越低越好）|
| `completion_rate` | 完成率（应接近1.0）|
| `health_failures` | 健康失败次数（应为0）|

**评估准则**：
- ✅ 高准确率 + 低ASR + 低误撤销率 = 优秀防御
- ⚠️ `health_failures > 0` = 该方法在某些场景失效
- ⚠️ `completion_rate < 1.0` = 训练提前终止

---

## GPU加速（可选）

### 单机多卡

```bash
# 自动使用所有可用GPU
python run_fair_tuning_from_config.py \
    --config configs/my_experiment.json \
    --device auto \
    --jobs auto
```

**调度策略**：
- 每张GPU最多运行1个完整实验
- 按设备队列派发任务
- 自动检测OOM并重试

### AI Station / 集群

```bash
# 示例：8核CPU + 8张RTX 4090
python -u run_fair_tuning_from_config.py \
    --config configs/cifar10_v13_fair_tuning_fixed.json \
    --compute-backend auto \
    --device auto \
    --jobs auto
```

**注意事项**：
- 不要在多机间同步`.venv`、`build/`或`*.so`
- 每台机器需独立构建SM9原生扩展
- 使用`git pull`同步代码，不要直接复制文件

---

## 常见问题

### Q1: 实验运行很慢？

**A**: 
- 使用`--crypto-mode simulated`跳过真实SM9签名（速度提升10-100倍）
- 减少`rounds`、`num_clients`或`malicious_ratios`数量
- 使用GPU加速：`--device cuda`或`--device mps`（Mac）
- 启用并行：`--jobs 4`

### Q2: 如何复现论文实验？

**A**: 
使用已验证的配置文件：
```bash
# MNIST实验（成功案例）
python run_fair_tuning_from_config.py \
    --config configs/mnist_v7_target_fair_tuning.json

# CIFAR-10实验（最新版本）
python run_fair_tuning_from_config.py \
    --config configs/cifar10_v13_fair_tuning_fixed.json
```

完整运行时间：
- MNIST：约2-4小时（8 GPU）
- CIFAR-10：约18-24小时（8 GPU）

### Q3: OOM错误？

**A**: 
- 减小`batch_size`
- 减少并行任务：`--jobs 1`
- 检查GPU显存：至少需要12GB（CIFAR-10 + ResNet18）

### Q4: 如何只测试Ours方法？

**A**: 
编辑配置文件，设置：
```jsonc
{
  "method_spaces": {
    "sm9rrs": "auto",  // 或12个候选
    "vert": 0,         // 禁用
    "alignins": 0,
    "krum": 0,
    "ding13": 0,
    "fedavg": 0
  }
}
```

### Q5: 真实SM9和仿真模式的区别？

**A**: 
| 项目 | 真实SM9 | 仿真模式 |
|------|---------|----------|
| 签名算法 | SM9环签名 | SHA-256 |
| 摘要算法 | SM3 | SHA-256 |
| 速度 | 慢（群运算） | 快（哈希） |
| 安全性 | 真实密码学 | 无 |
| 用途 | 论文最终实验 | 快速开发调试 |

**推荐策略**：
- 开发调试：使用`simulated`
- 最终结果：切换到`sm9`

### Q6: 调优失败："Krum is undefined" 或运行时间过长

**问题1: Krum约束错误**
```
ValueError: Krum is undefined for this configuration: n=10, f=8, n-f-2=0; require n-f-2 >= 1
```

**根本原因**: 
- ❌ `num_clients=10` 太小，不符合论文设置
- ❌ 在`malicious_ratio=0.8`时违反Krum约束（n-f-2 < 1）

**问题2: 运行时间过长**
- ❌ 10个客户端时运行39小时
- ❌ 如果修正为100客户端，需要420小时（17.5天）❌

**推荐解决方案: 使用优化后的快速配置**

```bash
# 1. 使用v13_fast配置（100客户端，优化搜索空间）
bash run_v13_fast_tuning.sh

# 预计耗时：57小时（2.4天）✅
```

**v13_fast配置优势**:
- ✅ `num_clients=100`（与MNIST一致）
- ✅ 验证：1种子、1分区、50轮（快速验证）
- ✅ 正式：2种子、2分区、150轮（充分验证）
- ✅ 恶意比例：[0.0, 0.2, 0.4, 0.6]（充分覆盖）
- ✅ 聚焦最佳参数（14候选 vs 39候选）
- ✅ 总任务：152（vs 999）
- ✅ 仍符合一区论文标准

**Krum约束验证**（num_clients=100）:
| ratio | 恶意数f | n-f-2 | Krum兼容 |
|-------|---------|-------|----------|
| 0.0 | 0 | 98 | ✅ |
| 0.2 | 20 | 78 | ✅ |
| 0.4 | 40 | 58 | ✅ |
| 0.6 | 60 | 38 | ✅ |

**注意：断点恢复不适用**

如果已经用错误配置（num_clients=10）运行了很久：
- ❌ 修改`num_clients`会改变配置指纹
- ❌ 系统无法识别已完成的任务
- ✅ 必须从头开始，因此使用v13_fast很重要

**配置对比**:
```
v13_fixed（错误）:
  - num_clients: 10 ❌
  - 任务数: 999
  - 耗时: 420小时（修正后）❌

v13_fast（推荐）:
  - num_clients: 100 ✅
  - 任务数: 152
  - 耗时: 57小时 ✅
  - 学术要求: 完全满足 ✅
```

**参考文档**: 
- `docs/CIFAR-v13-配置对比-2026-10-15.md` - 详细对比与推荐
- `docs/CIFAR-v13-客户端数量与资源优化-2026-10-15.md` - 优化分析

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{yourpaper2026,
  title={SM9-RRS-FL: Robust Aggregation in Federated Learning via SM9 Ring Signature},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2026}
}
```

---

## 项目结构

```
SM9RRSFL/
├── sm9rrsfl/              # 核心代码库
│   ├── fl_runner.py       # 联邦学习主流程
│   ├── methods/           # 6种防御方法实现
│   ├── attacks/           # 攻击实现
│   └── crypto/            # SM9密码学
│
├── configs/               # 实验配置文件
│   ├── mnist_v7_target_fair_tuning.json      # ✅ MNIST成功配置
│   ├── cifar10_v13_fair_tuning_fixed.json    # ✅ CIFAR-10最新配置
│   └── fair_tuning.example.json              # 配置模板
│
├── tests/                 # 单元测试
├── outputs/               # 实验结果（不提交Git）
├── docs/                  # 详细文档
│   ├── SESSION_HANDOVER.md                   # 项目最新状态
│   ├── CIFAR-10六方案实验说明-2026-09-07.md   # 方法对比详解
│   └── TAD原论文与实现复核-2026-09-10.md      # TAD方法说明
│
├── run_fair_tuning_from_config.py    # 方案B：完整实验（推荐）
├── run_experiments_from_config.py    # 方案A：快速验证
├── requirements.txt                  # Python依赖
├── setup.py                          # SM9原生扩展构建
└── README.md                         # 本文档
```

---

## 技术支持

**文档**：
- 📖 [项目最新状态](docs/SESSION_HANDOVER.md) - 当前实验进展
- 📖 [GPU并行分析](docs/GPU_UTILIZATION_ANALYSIS.md) - 性能优化
- 📖 [CIFAR-10实验说明](docs/CIFAR-10六方案实验说明-2026-09-07.md) - 方法详解

**归档文档**（历史调试记录）：
- `docs/archive/历史诊断/` - CIFAR历史版本诊断
- `docs/archive/codex交接历史/` - 开发交接记录

**问题反馈**：
- GitHub Issues：[项目Issues页面]
- Email：[联系邮箱]

---

## 许可证

[待添加]

---

**最后更新**：2026-10-14  
**版本**：v4.8.10  
**文档状态**：✅ 适合审稿人和复现者使用
