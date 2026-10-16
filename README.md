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

**现有配置**：
- **MNIST v7**：`configs/mnist_v7_target_fair_tuning.json`（推荐）
- **CIFAR-10 v13 (10客户端)**：`configs/cifar10_v13_fair_tuning_fixed.json`
- **CIFAR-10 v13 (100客户端)**：`configs/cifar10_v13_100clients.json`
- **CIFAR-10 v14 (100客户端，逐轮/GPU进度)**：`configs/cifar10_v14_fair_grid.json`（本轮新增，等待 AI Station 实测）
- **Fashion-MNIST**：`configs/fashion_mnist_fair_tuning.json`

### CIFAR-10 v14：交接参数 + 全局轮数与 GPU 面板

v14 沿用最新 v13 100 客户端配置中的攻击与训练条件，独立运行六方法公平网格调优。Ours 固定 `attack_boost=20`（所有方法共享）、`detector_subspace_dim=3`、`detector_history_threshold=1.5`，只搜索 `detector_distance_threshold=2.0,2.1,…,3.1`；`K=20`，第 25 轮开始攻击。这些设置对应本地交接中的改进方向，实际 Acc/ASR 由本轮结果确认。

| 项目 | v14 配置 |
|---|---|
| 数据与模型 | CIFAR-10，当前通用实验路径的 CIFAR CNN；加载官方 50000 训练样本、10000 测试样本 |
| 训练分割 | 原训练集 80% 用于训练、15% 用于调优评估、5% 用于攻击辅助；正式测试集不参与选参 |
| 客户端与分布 | 100 客户端；IID + Dirichlet（α=0.5） |
| 恶意比例 | 0%、20%、40%、60%、80% |
| 公共训练 | 每项 150 轮，E=1，batch=50，lr=0.01，lr_decay=0.99；禁用早停 |
| 攻击 | alternating_minimization，boost=20，源标签 5 → 目标标签 7 |
| 密码模式 | 真实 SM9，DKG 3/2，sm9_workers=1 |
| 调优 | Ours / VERT / AlignIns 各 12 个候选，Krum / TAD / FedAvg 各 1 个固定配置；验证 seed=301/302/303 |
| 正式评估 | 使用选出的各方法参数，独立 seed=401/402/403 |
| 并行 | v14 全可见 GPU 池；符合单任务总显存条件的卡全部纳入，每卡最多 1 项任务；调优和正式阶段均生效 |
| 输出 | `outputs/cifar10_v14_fair_grid/` |

完整计划：**1170 项调优 × 150 轮 + 180 项正式评估 × 150 轮 = 1350 项 / 202500 轮**。正式阶段沿用当前调优器的选择、健康与目标评估规则；若原规则阻止推进，面板保留未完成状态。此处的任务完成计数表示已保存结果，健康是否通过需查看结果审计。

由用户提交并推送以下 **7 个文件** 后，在 AI Station 现有项目环境运行：

- `README.md`
- `configs/cifar10_v14_fair_grid.json`
- `run_cifar_v14_with_progress.py`
- `scripts/experiments/cifar10/fair_tuning_dashboard.py`
- `scripts/experiments/cifar10/cifar_fair_gpu_pool.py`
- `tests/test_cifar_fair_dashboard.py`
- `tests/test_cifar_fair_gpu_pool.py`

```bash
cd /3251901002/SM9RRSFL
git pull
# 先确认配置与准确工作量；不下载数据、不训练、不创建实验输出
python run_cifar_v14_with_progress.py --dry-run
# 在 AI Station 交互终端检查资源，通过后启动
python run_cifar_v14_with_progress.py --check-gpus && \
  python -u run_cifar_v14_with_progress.py
```

面板在同一终端区域刷新：大进度条累计两阶段的训练轮数，并显示任务完成数、已用时、估算剩余时间、当前阶段。每张 GPU 显示当前候选、分区、恶意比例、seed、轮次进度条，以及 Ours 阈值 / VERT 参数 / AlignIns 参数。80 列、24 行终端可同时显示 8 张 GPU；更短的终端压缩详情，建议放大窗口。`cuda:N` 是当前进程的逻辑编号；若设置 `CUDA_VISIBLE_DEVICES`，编号按该列表重新映射。

**GPU 使用策略**：v14 新入口把所有可见且总显存足够的 GPU 纳入任务池，GPU 并发数不再受 CPU 配额数直接限制。因此即使容器只分配 1 个 CPU 配额，只要主内存足够、8 卡都通过显存准入且还有至少 8 项任务，两阶段均可启动 8 项并行任务。保留主内存预算（容器可用总量的 75%）与原显存安全余量；若主内存限制并发，启动日志 `gpu_pool_plan` 明示 `host_memory_limited=true`、`host_slots` 与 `target_jobs`，不会把它误报为已用满所有卡。

任务队列每秒复查显存；起初繁忙但总显存足够的卡持续保留在池中，空闲后可加入当前阶段；任何卡完成任务后可接续下一项。断点续跑只剩少量任务时，仍可在整个可见 GPU 池中选择空闲卡。全池都暂不可用时最多等待 600 秒，仍不可用则保留已有结果并报资源错误；原有 OOM 冷却、有限重试、断点与结果提交规则继续生效。`gpu_pool_plan` 列出 `pool_devices`、`ready_now`、`waiting_for_memory`、`target_jobs`，面板显示 `gpu=正在运行/池大小 slots=并发上限`，便于远端核对。此入口要求 CUDA；CUDA 不可见时明确停止，不转到 CPU 长时间训练。`CUDA_VISIBLE_DEVICES` 仍是用户控制的可见边界。

**单卡显存查询失败的处理**：接近满载的卡可能在 `torch.cuda.mem_get_info` 时就触发 CUDA OOM。v14 按逻辑卡分别捕获错误，保留其他卡成功的显存读数；检查输出 `memory_query_status=partial` 和 `query_failures` 记录具体失败卡与原始错误。只要至少一张卡查询成功，继续按成功读数选择足够显存的卡；失败卡保持等待查询状态，在取得符合显存准入的有效读数之前不会接任务。调度每秒重查，恢复时输出 `gpu_memory_status`；`gpu_pool_plan.waiting_for_memory_query` 区分查询失败与已知空闲显存不足。全部卡查询均失败时，启动检查停止并给出逐卡原因。此处理适用于启动和运行期间，不再因一张卡查询失败阻断整个GPU池。

2026-10-16 单卡查询修复需用户提交这 **3 个文件**：`README.md`、`scripts/experiments/cifar10/cifar_fair_gpu_pool.py`、`tests/test_cifar_fair_gpu_pool.py`。实验配置、轮数、选参和训练规则保持原版本。

多卡调度利用的是独立实验任务。SM9、检测器、聚合、存盘等阶段含 CPU/I/O 工作，实验末尾也可能剩余不足 8 项任务，因此无法承诺每张 GPU 在所有时刻都达到 100% 算力利用率。

直接运行上述入口即可，无需额外管道或 `EMIT_PROGRESS_EVENTS` 环境变量。重跑同一命令沿用原调优器的断点恢复；缓存和恢复前缀计入总进度，但不当作本次新完成轮次加速 ETA。ETA 随实际吞吐估算，初始显示 `estimating`，任务耗时差异、显卡变化和调优后处理均可能影响它。若重定向到文件，自动输出普通日志，每 30 秒更新一次摘要；也可显式加 `--display-mode log`。

结果查看：`validation_results.csv`、`tuning_summary.json`、`best_parameters.json`、`final_evaluation/summary.csv` 和 `final_evaluation/rounds.csv`（均位于新输出目录）。新入口调用当前调优器，在独立作用域内补充 GPU 资源计划与任务池范围；原训练、选择、恢复和报告流程继续使用，旧配置和旧实验目录保留。

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

### Q6: "ModuleNotFoundError: No module named 'fair_tuning_progress_display'"

**问题描述**:
```
ModuleNotFoundError: No module named 'fair_tuning_progress_display'
```

**原因**: 进度显示模块不在Python搜索路径中。

**解决方案**:
```bash
# 复制模块到项目根目录
cp scripts/experiments/mnist/fair_tuning_progress_display.py .

# 验证导入
python3 -c "from fair_tuning_progress_display import progress_display; print('✅ 成功')"
```

该模块提供实时进度显示和日志管理，不影响核心训练逻辑。

### Q7: CIFAR-10 v13调优配置说明

**问题背景**: v13 fixed (10客户端)在运行39小时后失败

```
ValueError: Krum is undefined for this configuration: n=10, f=8, n-f-2=0; require n-f-2 >= 1
```

**根本原因**: 
- ❌ `num_clients=10` 不符合论文标准设置（论文用100）
- ❌ `malicious_ratio=0.8` 时违反Krum约束（n-f-2 < 1）

**正确配置: v13_100clients**

```bash
# 使用100客户端配置（与MNIST v7一致）
bash run_v13_100clients.sh

# 或直接运行
python run_fair_tuning_from_config.py \
    --config configs/cifar10_v13_100clients.json
```

**配置参数**:
- ✅ `num_clients = 100`（与论文一致）
- ✅ `ratios = [0.0, 0.2, 0.4, 0.6, 0.8]`（充分覆盖）
- ✅ `rounds = 150`（充分训练）
- ✅ `attack_boost = 20.0`（验证有效）
- ✅ 搜索空间：sm9rrs(12) + vert(12) + alignins(12) + 3基线

**任务规模**:
- 验证阶段: 585 trials (39 candidates × 5 ratios × 3 seeds)
- 最终阶段: 90 trials (6 methods × 5 ratios × 3 seeds)
- 总计: 675 trials × 150 rounds = 101,250 训练轮
- **预计耗时**: 100-150小时（7张4090D并行）

**Krum约束验证**（num_clients=100）:
| ratio | 恶意数f | n-f-2 | Krum兼容 |
|-------|---------|-------|----------|
| 0.0 | 0 | 98 | ✅ |
| 0.2 | 20 | 78 | ✅ |
| 0.4 | 40 | 58 | ✅ |
| 0.6 | 60 | 38 | ✅ |
| 0.8 | 80 | 18 | ✅ |

**注意事项**:
1. ⚠️ **不能断点恢复**: 修改`num_clients`会改变配置指纹，必须从头运行
2. ✅ **学术标准**: 完全符合一区论文要求
3. ✅ **公平对比**: 所有方法使用相同的客户端数量

**参考文档**: 
- `docs/CIFAR-v13-100clients-验证完成-2026-10-16.md` - 完整配置说明
- `docs/CIFAR-v13-Krum约束修复-2026-10-15.md` - 问题诊断

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
