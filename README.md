# SM9-RRS-FL: 联邦学习中基于SM9环签名的鲁棒聚合方案

## CP1的IID失效诊断与独立CUDA开发（2026-10-27）

固定CP1本地60项验证已完成，但尚未达标：IID60%诚实误撤销7/40=17.5%，IID60/80%末10轮ASR为93.37/92.88%。IID20/40%末10轮Accuracy为59.06/58.16%、ASR为9.96/10.36%；40%首轮攻击仍有短暂严重退化，不能只报告尾段。首轮漏入先于误撤销，恶意更新随后进入正常历史。原MPS结果和科学源码保持冻结。

新入口`run_cifar_cp1_cuda_development.py`使用`configs/cifar10_cp1_cuda_diagnosis150_v1.json`，在独立目录开展150轮开发。第一批为IID clean/60/80%各FedAvg、CP1、W85、W80，共12项1800轮；W85/W80仅改变Ours告警、历史准入及漂移容限三项阈值，其他参数（含累计撤销容忍60）不变。固定V2交替最小化攻击、100客户端、数据划分和已观察开发seed2026102201不变。此批用于检验早期漏入机制，不是最终资格或正式实验。

CUDA统一启用确定性算法，禁用TF32及cuDNN benchmark，设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`。平台/软件变化单独记录，不能将MPS与CUDA成绩混作配对基线。输入数据的全部数组与原MPS哈希核对；使用原攻击及观测代码、原健康函数。任务完整保留失败；partial和无身份残留禁止自动重试。首个可用GPU须连续3次（间隔10秒）满足空闲显存至少16GiB、利用率不高于5%，本批固定同一UUID串行执行；等候上限24小时，超时仅表示资源阻塞，不代表算法失败。此抽样准入不等于集群独占预约，其他用户进程不会被终止。

```bash
# 在已配置PyTorch CUDA和SM9 native库的解释器下执行
python3 -m unittest tests.test_cp1_cuda_development tests.test_cifar_tad_no_prior -v
python3 run_cifar_cp1_cuda_development.py          # 仅检查协议与来源
python3 -u run_cifar_cp1_cuda_development.py --batch
python3 run_cifar_cp1_cuda_development.py --summary
```

阶段完成后检查匹配clean前24轮、FedAvg攻击有效性、原健康、首轮冲击、恢复时间及尾10指标，再决定下一批。最终仍须两分布各0/20/40/60/80%及六法的完整同平台证据：方法需在同分布全部五比例健康后才进入参照集，Ours也必须健康；clean不限制名次，Accuracy/ASR距健康最优≤1/2pp，攻击场景双前二且差距≤2/5pp。六法原始结果和健康失败原因全部保留；不能以本12项开发面板宣称达标，也不自动启动正式测试集实验。

`diagnose_cifar_cp1_iid.py`仅供拥有原本地输出的环境执行，校验原64项科学来源及引用、逐客户端计数/撤销/聚合质量和时序，输出独立诊断JSON；远端不需要复制这些历史输出才能启动新CUDA面板。


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

单卡查询修复已提交到 V4.10.1；下面的资源清理、故障证据与显示补丁仍需同步。

**空闲卡与终止错误的显示**：每次资源复查会更新面板。等待显存的卡显示当前空闲量及所需空闲量，查询失败显示具体原因，满足准入但未派发的卡显示ready。`slots=8`表示并发上限，实际运行数看 `gpu=active/pool`；可见8卡不等于当前8卡都有足够空闲显存。

`cuda_worker_failed`（例如未经确认分类的 `CUDNN_STATUS_INTERNAL_ERROR`）会按原控制器规则停止派发新任务，等待其他已运行任务结束并保存结果，然后报错退出。面板此时明确显示 `DISPATCH STOPPED`、`draining_after_worker_error` 和 `eta=paused`；正常任务的结果快照不会清除该状态。原有可重试OOM仍按原规则处理；显示修正不把cuDNN内部错误自动当作OOM，也不代表该错误已解决。

**2026-10-16 代码审查与修复**：

- 已回传的异常是 cuda3 在 `local_train_delta_resident` 的 `loss.backward()` 报 cuDNN 内部错误。按现有严格失败规则，这会停止整个队列的新任务派发；其余卡即使后来空闲也不会继续接任务。`--check-gpus` 只查询显存，检查通过不代表卷积反向传播已通过。进度条在测量函数返回或抛异常时都会撤下，单凭条消失不能区分成功、终态断点重放或新错误；历史 `last_failure.json` 也不能证明重启后再次报错。
- 复现并修复闲置缓存造成的准入阻塞：原队列只在 OOM 后释放缓存，正常任务结束后仍可能由本进程的 PyTorch 闲置缓存占用显存，导致下一项任务过不了驱动空闲显存检查。v14 现在每项任务退出后释放该卡本进程的闲置缓存，保持每卡单任务及原显存门槛。此处释放缓存的作用符合 [PyTorch 2.3 显存管理说明](https://docs.pytorch.org/docs/2.3/notes/cuda.html#memory-management)，不能消除其他进程占用或保证未来峰值显存充足。
- 非 OOM 异常原来由 Future 的 traceback 一直持有失败训练的模型、梯度等局部对象，直到其他任务全部收尾。v14 现在先保存完整故障证据，再清理栈帧的局部引用；保留异常类型、消息、栈位置及原终止规则。
- 每次实际任务失败会追加到对应阶段的 `gpu_failures.jsonl`，记录时间、配置、实际 Python 轮次/客户端/批次位置（存在时）、张量形状/设备、该卡空闲/已分配/保留显存及查询错误、Torch/CUDA/cuDNN 版本与开关。原 `last_failure.json` 和断点继续保留。并发失败各自留档，诊断写盘失败不会替换训练原异常。
- 启动输出 `v14_runtime` 包含 Git HEAD 与关键源码/配置 SHA256；故障记录带同一份启动代码身份。仅在 AI Station 执行 `git pull`，不会拿到尚未由用户在本地主项目提交、推送的补丁。

当前证据尚不足以确认 cuDNN 内部错误来自显存竞争、运行库或具体算子，也未收到此次环境重启后的新异常。上述两项资源清理修复不冒充该 cuDNN 错误的因果修复；没有修改 batch、训练/攻击参数、cuDNN/TF32 开关、候选、健康/选参规则或自动重试分类。

只读提取当前失败证据与代码身份（不加载数据、不反序列化训练断点、不启动训练；可在另一终端执行）：

```bash
python run_cifar_v14_with_progress.py --diagnose
```

本轮需用户提交这 **8 个文件**（含此前待提交的显示补丁）：`README.md`、`run_cifar_v14_with_progress.py`、`scripts/experiments/cifar10/cifar_fair_gpu_pool.py`、`scripts/experiments/cifar10/fair_tuning_dashboard.py`、`scripts/experiments/cifar10/cifar_fair_diagnostics.py`、`tests/test_cifar_fair_gpu_pool.py`、`tests/test_cifar_fair_dashboard.py`、`tests/test_cifar_fair_diagnostics.py`。交接 Markdown 仅本地保留，不提交。现有训练使用启动时加载的代码；同步新补丁后，须等现有控制器退出，再用原命令续跑，保留实验输出。

将诊断输出的 `CIFAR_V14_DIAGNOSIS_BEGIN` 至 `END` 完整回传，可区分旧记录、当前失败、代码未同步与已恢复情况。屏蔽某张卡只是一种临时隔离手段；未经实际新证据，不能据此认定该卡损坏或 cuDNN 错误已经解决。

多卡调度利用的是独立实验任务。SM9、检测器、聚合、存盘等阶段含 CPU/I/O 工作，实验末尾也可能剩余不足 8 项任务，因此无法承诺每张 GPU 在所有时刻都达到 100% 算力利用率。

直接运行上述入口即可，无需额外管道或 `EMIT_PROGRESS_EVENTS` 环境变量。重跑同一命令沿用原调优器的断点恢复；缓存和恢复前缀计入总进度，但不当作本次新完成轮次加速 ETA。ETA 随实际吞吐估算，初始显示 `estimating`，任务耗时差异、显卡变化和调优后处理均可能影响它。若重定向到文件，自动输出普通日志，每 30 秒更新一次摘要；也可显式加 `--display-mode log`。

结果查看：`validation_results.csv`、`tuning_summary.json`、`best_parameters.json`、`final_evaluation/summary.csv` 和 `final_evaluation/rounds.csv`（均位于新输出目录）。新入口调用当前调优器，在独立作用域内补充 GPU 资源计划与任务池范围；原训练、选择、恢复和报告流程继续使用，旧配置和旧实验目录保留。

---

### v14 快速攻击对比：60% / 80%

用户尚无主实验选定参数时，可运行固定候选的短程预检：Ours 使用最近实际回传的 `sm9rrs-001`（距离阈值 2.0），VERT 使用原网格 `vert-001`（history=5、projection=20、epochs=5、lr=.01、top_k=5、不使用比例先验），FedAvg 作为攻击效果对照。这三组属于原主实验候选，未声称是调优后的最优参数。

```bash
.venv/bin/python run_cifar_v14_attack_screen.py \
  --config configs/cifar10_v14_attack_screen_100clients.json --dry-run
.venv/bin/python -u run_cifar_v14_attack_screen.py \
  --config configs/cifar10_v14_attack_screen_100clients.json
```

当前配置 `configs/cifar10_v14_attack_screen_100clients.json` 引用原 v14 配置并校验 SHA256。50 轮、100 客户端、单种子 301、IID/Dirichlet、恶意比例 0.6/0.8，共 **12 项、600 轮**；优先完成 Dirichlet/80% 的 Ours、VERT、FedAvg，再运行其余场景。原40,000主训练、7,500验证、2,500攻击辅助划分及其seed不变，所有比较在训练集留出验证数据上进行。原训练/攻击参数、检测/聚合/SM9代码继续使用。

本入口面向本地 Apple MPS；独立目录 `outputs/cifar10_v14_attack_screen_100clients/` 保存逐轮断点，重跑可恢复。每完成一项即更新 `summary.csv`、`rounds.csv`、`scenario_audit.csv` 与 `screen_report.json`。报告沿用主实验的攻击期均值、末10轮均值、最终轮、峰值和健康检查：准确率落后 VERT ≤2pp，ASR 高出 VERT ≤1pp，ASR ≤8%，峰值 ≤20%，非有限更新必须为0。`numerical_audit` 明示每项累计非有限更新、首次发生轮次和单轮最大次数；`run_dimensions` 记录实际客户端数、轮数和种子。全部12项完成前显示 `pending`，缺失任务不会填零或算通过。

这是固定候选的攻击短测，未执行clean对照或全网格选参；`main_experiment_target_status` 始终为 `not_evaluated`。其通过/失败只描述本次50轮、100客户端、单种子的攻击条件，不能直接认证或否定AI Station的150轮、多种子CUDA主实验。当前后台实例的日志在 `run.log`，进程与退出状态在 `controller_status.json`；同目录有排他锁，重启前先确认原控制器已退出。

之前的50客户端短测已按用户要求停止，保留 `configs/cifar10_v14_attack_screen.json` 和 `outputs/cifar10_v14_attack_screen/` 供审计。100客户端版本仅改变客户端数与输出目录，重新训练独立任务，不复用50客户端断点。入口的历史默认配置仍为50客户端，运行100客户端时必须使用上面的显式 `--config`。

### 独立攻击稳定性筛选：本地 MPS、限时两小时

`run_cifar_attack_stability.py` 使用独立协议 `configs/cifar10_attack_stability_v1.json`，先检查攻击是否数值稳定、能否在 FedAvg 上产生有效的定向投毒，再用同一攻击比较 Ours、VERT、FedAvg。此入口面向本地 Apple MPS，要求本机已有 CIFAR-10 数据及可用的 PyTorch MPS；**它不是 AI Station 的 CUDA 启动命令**。

攻击来源为 Bhagoji、Chakraborty、Mittal、Calo 的 [Analyzing Federated Learning through an Adversarial Lens（ICML 2019）](https://proceedings.mlr.press/v97/bhagoji19a.html)，作者提供了 [官方实现](https://github.com/inspire-group/ModelPoisoning)。本协议明确记录 SGD/Adam、攻击步频率和参数范围的差异，用于检查当前 CIFAR 配置的稳定性，不将这些移植变体统称为论文原始配置的精确复现。

```bash
cd /Users/derpt/PycharmProjects/SM9RRSFL
# 只读检查计划和配置，不训练
.venv/bin/python run_cifar_attack_stability.py --dry-run
# 本地执行；同一输出目录只允许一个控制器
.venv/bin/python -u run_cifar_attack_stability.py --display-mode live
# 另一个终端只读查看当前状态及已有报告
.venv/bin/python run_cifar_attack_stability.py --summary
```

公共设置为100客户端、30轮、seed301、原CIFAR CNN、E=1、batch=50、lr=.01、decay=.99、第25轮开始攻击，保留40,000主训练／7,500验证／2,500攻击辅助的原划分。官方测试集不参与选参。Ours固定`sm9rrs-001`，VERT固定`vert-001`；攻击选择完成前，不根据防御表现改变候选或阈值。

| 候选 | 恶意客户端优化器／学习率 | boost | 正常训练步分组上限 | boost与距离约束范围 |
|---|---|---:|---:|---|
| C0 | SGD／继承当轮学习率 | 20 | 1 | 全部参数 |
| C1 | SGD／继承当轮学习率 | 5 | 1 | 全部参数 |
| C2 | SGD／继承当轮学习率 | 10 | 1 | 全部参数 |
| C3 | SGD／继承当轮学习率 | 20 | 5 | 全部参数 |
| C4 | SGD／继承当轮学习率 | 20 | 10 | 全部参数 |
| C5 | SGD／继承当轮学习率 | 20 | 10 | 仅权重 |
| C6 | Adam／.001 | 10 | 10 | 仅权重 |
| C7 | Adam／.001 | 20 | 10 | 仅权重 |

每个正常训练步分组后执行一次目标步，最后不足上限的分组也执行一次。“仅权重”表示维度大于1的参数接受额外boost及距离约束，偏置仍进行普通更新。Adam的正常参考、stealth、target分别使用独立优化器状态，每个客户端调用重新初始化；目标步先执行普通Adam，再放大该步实际参数位移。不会用“把梯度乘boost”替代这一操作。诚实客户端继续使用原训练路径；异常更新不裁剪、不替换成有限数。

执行分为以下阶段：

1. 小型真实SM9基准比较1／4／8个密码工作线程；只有所检查科学结果一致时才按耗时选择。随后分别完成IID和Dirichlet的30轮clean FedAvg，保存第0、24轮公开模型锚点。
2. 从固定80%恶意客户端集合中，按数据量及源标签覆盖预先选出每分区16个代表。8候选 × 2分区 × 2锚点，最多512次CPU客户端探针；最多3进程、每进程2个Torch线程。每个合格候选再用MPS重放其权重增长最大的客户端条件，最多8次；MPS训练任务始终串行。
3. 完整通过CPU与MPS数值检查的候选，按目标命中率、置信度、耗时排序，最多取前2名完成FedAvg／Dirichlet／80%的30轮验证。只用FedAvg攻击证据选攻击，不使用Ours优劣决定攻击设置。
4. 若选出有效攻击，则固定该配置，依次比较Dirichlet／80%的Ours、VERT，以及Dirichlet／60%的FedAvg、Ours、VERT。若未选出合格或有效攻击，停止并保留证据，**不会自动进入Ours调参或150轮主实验**。

攻击有效性查看第25—30轮：训练辅助目标的平均命中率至少80%，平均目标置信度比配对clean FedAvg高至少0.2；若clean本来就满足高置信度目标错误条件，则该证据不能用来证明攻击有效。相关任务还必须完整且数值有限。全流程至多9项30轮任务，共270训练轮，另有上述客户端探针及小型SM9基准；失败或预算不足时会减少实际工作量。

报告区分三个量，均显示实际分母：

- `training_target_success/confidence`：攻击优化所用辅助实例在聚合后的全局模型上的命中率／置信度，用于确认攻击是否确实生成有效投毒。
- `source_transfer_asr/confidence`：独立训练留出验证集中全部源标签样本的迁移ASR／置信度；这些样本不参与攻击优化。
- `legacy_asr/confidence`：原协议的一张独立验证图像指标，保留供历史对照，不能与全源类别ASR混为一谈。

`outputs/cifar10_attack_stability_v1/` 单独保存配置和源码身份、任务断点、目标索引、逐轮扩展指标、客户端审计及首次失败证据。`micro_screen.json`记录探针筛选，`fedavg_selection.json`记录有效性判断；只有通过后才生成`recommended_attack.json`。`report.json`与`controller_status.json`区分完整、部分、无合格攻击、无有效攻击和执行失败。攻击适配器逐步检查前向输出、损失、梯度及参数，worker同时检查提交更新、全局参数及评估指标；发现NaN/Inf即保留首次阶段／客户端证据，相关任务不算合格。

总运行预算为7,200秒，CPU探针阶段单次最多1,800秒，并始终受总预算限制。终端大条显示已用预算、剩余预算、已完成轮数和当前任务；剩余预算不是完成全部阶段的保证或精确ETA。预算不足时保留完整检查点和实际覆盖范围，不把未运行项目当作成功。重跑同一命令会核对不可变配置／源码／数据身份，复用完整结果并恢复可恢复的部分任务；已计耗时不会重置，数值失败不会在原目录里被静默重试。重启前应确认前一个控制器已退出；改变协议或科学代码后应使用新的版本与输出目录，不能覆盖旧证据。

这是单种子、本地MPS、30轮且仅含6轮攻击的筛选。CPU探针覆盖两个分区，完整防御对比聚焦Dirichlet；未完成IID攻击对比、Ours/VERT clean健康资格、多种子或150轮尾期验证。`main_150_round_target_status`和`defense_clean_health_status`保留`not_evaluated`。即便得到推荐攻击，也不能直接声称Ours达到主实验目标，或保证AI Station上表现相同。

本地执行记录（2026-10-20）：9项任务、270轮全部完成，墙钟约85分钟（包含单独归档的预运行及暂停）；正式任务均无非有限更新。CPU检查实际449次，其中C0首客户端出现NaN后淘汰；C1–C7各64次CPU检查和一次MPS重放通过。FedAvg从C7/C5中选中C7（Adam .001、boost20、ls10、仅权重），未根据Ours表现选择攻击。

以下为同一C7条件下的实测。准确率取第30轮；三项攻击指标取第25–30轮均值，分别针对攻击使用的目标图1张、旧独立图1张和独立源类图750张：

| 恶意比例 | 方法 | 最终准确率 | 目标命中率 | 旧单图ASR | 迁移ASR |
|---|---|---:|---:|---:|---:|
| 60% | FedAvg | 13.48% | 100.00% | 100.00% | 95.73% |
| 60% | Ours | 43.67% | 0.00% | 0.00% | 10.49% |
| 60% | VERT | 24.52% | 100.00% | 100.00% | 52.51% |
| 80% | FedAvg | 13.04% | 100.00% | 100.00% | 99.02% |
| 80% | Ours | 39.69% | 16.67% | 0.00% | 13.69% |
| 80% | VERT | 26.56% | 83.33% | 16.67% | 22.24% |

Ours两比例均无诚实误撤销，六个攻击轮的恶意聚合权重均为0，最终撤销名单与恶意名单完全一致。80%下第27轮目标被误判为目标类一次，因此不能仅报告最终轮的0%；在恶意权重为0时，也不能直接将该误判归因于恶意更新渗入。C7使FedAvg正常准确率大幅下降，本轮未认证论文的隐蔽性。完整候选、首发异常、实际步骤和耗时保存在本地忽略目录的`screen_brief.json`及原始审计文件；后续先补Ours/VERT clean对照和独立seed401，再决定150轮资格，不能将本表外推到AI Station长程结果。

### C7独立复核：补clean对照与seed401

`run_cifar_c7_replication.py` 使用 `configs/cifar10_c7_replication_v1.json`，固定上一轮选定的C7，新增 **8项、240轮** 本地实验：seed301下Ours、VERT各一项clean对照；seed401下Ours、VERT、FedAvg分别运行80%和60%恶意比例。全部采用Dirichlet分区（α=.5）、100客户端、30轮，CNN、诚实训练SGD、E1、batch50、lr=.01、decay=.99、K20及第25轮攻击等参数保持原样；C7仍为Adam .001、boost20、正常步骤分组10、仅权重额外放大与距离约束。Ours与VERT候选也保持不变，本轮不继续选攻击或调防御参数。

```bash
cd /Users/derpt/PycharmProjects/SM9RRSFL
# 检查固定计划、既有参考输出与源码身份，不训练
.venv/bin/python run_cifar_c7_replication.py --dry-run
# 单个MPS任务串行运行；SM9沿用8个工作线程
.venv/bin/python -u run_cifar_c7_replication.py --display-mode live
# 另一个终端只读查看完整性、当前状态和已有比较
.venv/bin/python run_cifar_c7_replication.py --summary
```

启动要求本地保留完整的 `outputs/cifar10_attack_stability_v1/` 及原数据。控制器校验上一轮C7选择证据、参考任务、配置与源码身份，再将新任务写入独立的 `outputs/cifar10_c7_replication_v1/`；旧结果只读保留，原断点不用于初始化新实验。MPS仅运行一个训练任务，面板显示8项任务／240轮的总进度、耗时、估计剩余时间及当前方法、比例、种子和轮次。

seed401是完整的实验种子复核：按既有规则重新生成模型初始化、客户端Dirichlet划分、恶意客户端集合、攻击辅助单图及旧验证单图。数据集三分划分仍固定使用数据seed42、split seed20261013，独立源类验证集合仍为750张。同一种子下三法使用相同目标图，实际索引保存在各任务的 `targets.json`；301与401的单图指标不能解释为对同一张图的重复观测。

报告将seed301的新clean与同方法、同种子的旧C7攻击结果配对，给出最终及第25—30轮平均准确率损失、目标置信度变化和目标预测翻转，并记录误撤销、诚实权重损失、恶意聚合权重及NaN/Inf。seed401与301的三法攻击结果按比例比较指标差值和排名方向；**本轮没有seed401配对clean**，因此不以跨种子clean置信度差认证seed401的攻击有效性。

在结果尚未被后续实验冻结引用前，中断后可使用原命令续跑，完整任务复用，部分任务从最后完整轮断点恢复；运行前须确认旧控制器已退出。失败留存首次证据，失败任务不自动重试；身份或源码不匹配时不能覆盖原目录。报告要求每项有第0—30轮完整指标、有限数值及正确的1／1／750指标分母，未完成或证据异常的任务不算通过。**当前8项复核已被六方案补测冻结引用，请勿再启动这个旧控制器**，即便没有重训，刷新时间或状态文件也会破坏引用身份；查看已有结果使用上方只读 `--summary`。

本地完成记录（2026-10-20）：8项／240轮全部完成，控制器累计耗时4,738.0802秒（约79.0分钟），全部任务NaN/Inf更新为0，身份、指标及配对比较检查通过。seed301 clean最终准确率为Ours **44.3067%**、VERT **37.6400%**，原配对FedAvg为 **44.3067%**；两项新增clean均无诚实误撤销，但VERT比FedAvg低 **6.6667个百分点**，超过主协议5个百分点的clean准确率损失门槛。因此“完整、数值有限”不等于已经通过正式clean效用资格。

seed401的Ours在60%／80%下最终准确率分别为42.1600%／41.5467%，第25—30轮指定目标命中率和旧单图ASR均为0，全部诚实误撤销为0。迁移ASR分别为5.5111%／5.3556%；两比例均优于本次固定VERT与FedAvg，但VERT在seed401／80%下明显退化，不能称三法所有指标均已跨种子稳定。两种子、30轮和6轮攻击只能说明本次短程条件下的重复表现，不能证明统计显著性、150轮健康资格或AI Station长程效果。

### C7六方案补测：固定候选、同seed401

`run_cifar_c7_six_methods.py` 使用 `configs/cifar10_c7_six_methods_v1.json`，在上述8项复核全部完成、数值有限且证据有效后，补充AlignIns、Krum、Ding13各在80%和60%恶意比例下的对比，共 **6项、180轮**。继续使用seed401、Dirichlet α=.5、100客户端、30轮、相同C7及其余公共训练参数；不因希望得到某种排名而修改攻击、算法或筛掉失败任务。

```bash
cd /Users/derpt/PycharmProjects/SM9RRSFL
# 前8项未结束时也可查看计划；此命令不训练、不冻结未完成证据
.venv/bin/python run_cifar_c7_six_methods.py --dry-run
# 前8项完整结束后运行；单个MPS子进程串行执行六项
.venv/bin/python -u run_cifar_c7_six_methods.py --display-mode live
# 只读查看当前进度与六方案报告
.venv/bin/python run_cifar_c7_six_methods.py --summary
```

新结果独立保存在 `outputs/cifar10_c7_six_methods_v1/`，启动时冻结原攻击筛选与8项复核的源码、配置及公开引用证据；同一命令保留已完成结果、恢复部分任务的完整轮断点，不重试终结失败。面板显示6项／180轮、耗时、按已观测速度估计的剩余时间及当前方法、设备、比例与轮次。此入口面向本地Apple MPS，不沿用首轮两小时预算，也不作为AI Station CUDA入口。

AlignIns使用原v14第一候选 `alignins-001`（sparsity=.8、TDA半径=.05、MPSA半径=.1）；Krum和Ding13使用原无网格参数的首候选。复用的Ours与VERT也维持原首候选，**本表是固定候选对比，不是六方案各自充分调优后的SOTA比较**。VERT的history=5、projection=20、predict epochs=5、lr=.01、top-k=5尚未针对C7选优。

60%／80%属于多数恶意压力测试。Krum在n=100、f=60／80时不满足原保证条件 `2*f+2<n`；AlignIns在相同高恶意条件下的结果也不应直接解释为其论文保证范围内的表现。Ding13沿用实际恶意比例先验，可能在攻击开始前就撤销客户端；报告单独列出全体／全部诚实撤销、诚实权重饥饿、受攻击场景误撤销率超过10%的观察标记，以及真实攻击调用和目标步骤数。低ASR与数值有限均不能替代训练健康，受攻击场景标记也不冒充正式clean资格。

六方案报告核验同seed的实际数据数组、攻击辅助图、旧验证图、750张源类验证图和同恶意比例名单一致。除最终准确率及三类攻击指标外，保存第24—30轮轨迹、25—30／27—30轮的变化量、范围与描述性斜率，用于判断结束时是否仍明显变化。实际调用审计缺失时标记未知，不将名义 `attack_active` 直接当成客户端确实执行过攻击。

本地完成记录（2026-10-20）：新增6项／180轮全部完成，累计耗时1,336.4913秒（约22.3分钟），全部NaN/Inf更新为0；与前6项seed401攻击任务组成完整12项六法对比。下表准确率取第30轮，迁移ASR取第25—30轮750张独立源类验证图的均值，数值均为百分比：

| 方法 | 60%最终准确率 | 60%迁移ASR | 80%最终准确率 | 80%迁移ASR |
|---|---:|---:|---:|---:|
| Ours | 42.1600 | 5.5111 | 41.5467 | 5.3556 |
| VERT | 25.8533 | 34.8444 | 10.1067 | 84.1556 |
| AlignIns | 10.0000 | 93.7333 | 10.0000 | 93.7333 |
| Krum | 27.8400 | 24.9333 | 24.6533 | 2.3778 |
| Ding13／TAD | 10.0000 | 87.6000 | 10.0000 | 100.0000 |
| FedAvg | 13.9467 | 96.6222 | 13.9600 | 96.8667 |

Ours在两比例下均取得最高最终准确率，60%下迁移ASR也最低；**80%下Krum的迁移ASR为2.3778%，低于Ours的5.3556%**，因此不能宣称Ours六法所有指标全面最优。Ding13在60%／80%下分别从第18／10轮起撤销了全部40／20名诚实客户端，均早于第25轮攻击，是明确的训练健康失败；表中保留其原始指标，没有将其完整运行且数值有限解释为有效防御。

基线仍有需要单独验证的问题。VERT固定top-k=5，预热后每轮只接纳5个更新；其seed301 clean效用损失及seed401／80%崩溃均不支持“VERT表现稳定良好”。AlignIns这组固定半径在攻击前已有大量每轮0个接纳的记录，第24轮准确率仅23.7733%，显示当前候选存在严重过度过滤现象；没有通过反事实调参试验确定各因素的因果份额。Krum的高恶意测试仍超出原保证范围，不能用其单次高低表现代表所有条件下的优劣。

FedAvg的指定目标和旧单图在两比例下均为100%命中，迁移ASR均已超过96%，已接近饱和；其最终准确率13.9467%→13.9600%的微小回升不满足严格单调下降。实际60%与80%恶意集合交集为47人，60%名单有13人不在80%名单中，并非“原恶意客户端保留再增加20人”的嵌套设计，加上训练轨迹非线性，不能预先要求每个指标随比例严格单调或为了得到此形状改写结果。

第30轮尚不能视为收敛。seed301 clean的Ours／VERT从第24到30轮准确率仍增加1.1333／2.7867个百分点；seed401攻击条件下，Ours在第27—30轮分别下降0.7467／0.6533个百分点，VERT／60%同窗口准确率范围达15.8533个百分点，Krum／80%由20.1067%升至24.6533%。部分方法停在10%只是退化后的平坦曲线，也不能作为有用模型已收敛的证据。六个攻击轮适合发现NaN/Inf、首轮崩溃、明显漏入及撤销异常，尚不足以验证持续攻击下的恢复、后期健康、长期排名或150轮资格；seed401也缺少配对clean。所有运行证据已保留，后续若改变轮数、候选或比较设计，应建立新的协议与输出目录，现有结果查看使用 `--summary`，不要改写已被引用冻结的旧控制器状态。

### C7六方案60轮复核：先校准clean，再冻结候选

`run_cifar_c7_calibrated_60r.py` 使用独立配置 `configs/cifar10_c7_calibrated_60r_v1.json`，先完成VERT与AlignIns各4项seed301／30轮clean校准，再固定选择结果，以新seed402运行六法各一项clean及60%／80%恶意比例对比，每项60轮。总计 **26项、1,320轮**：校准8项／240轮，复核18项／1,080轮。保留100客户端、Dirichlet α=.5、完整数据划分、原CNN、诚实SGD E1／batch50／lr=.01／decay=.99、K20及第25轮攻击；C7继续使用Adam .001、boost20、正常步骤分组10、仅权重额外放大和距离约束。Ours、Krum、Ding13及FedAvg公共配置不因比较排名调整。

```bash
cd /Users/derpt/PycharmProjects/SM9RRSFL
# 只读检查计划、冻结引用及源码
.venv/bin/python run_cifar_c7_calibrated_60r.py --dry-run
# 本地单个MPS训练子进程串行执行，SM9使用8个工作线程
.venv/bin/python -u run_cifar_c7_calibrated_60r.py --display-mode live
# 另一终端只读查看校准选择、两阶段状态与已有报告
.venv/bin/python run_cifar_c7_calibrated_60r.py --summary
```

该入口要求本地CIFAR数据、可用MPS及完整的前序攻击筛选／C7复核／六方案补测输出和准备诊断JSON。它不沿用首轮两小时预算，也不是AI Station的CUDA入口。两阶段分别显示轮数、任务数、耗时、预计剩余时间及当前方法、候选、种子与比例。新输出为 `outputs/cifar10_c7_calibrated_60r_v1/` 下的 `calibration/`、`final/` 和两阶段之间冻结的 `selection.json`。

校准候选在训练前确定，两方法均为4项相同轮数预算：

| 候选 | VERT history／projection／predict epochs／lr／Top-k | 候选 | AlignIns sparsity／TDA半径／MPSA半径 |
|---|---|---|---|
| V1 | 5／20／5／.01／10 | A1 | .8／.5／.5 |
| V2 | 5／20／5／.01／20 | A2 | .8／1／1 |
| V3 | 10／256／5／.001／10 | A3 | .3／.5／.5 |
| V4 | 10／256／5／.001／20 | A4 | .3／1／1 |

VERT全部关闭恶意比例先验，选定的固定Top-k同时用于两种比例。训练前将候选限定为10／20，是因为100客户端、80%恶意时只有20个诚实客户端，Top-k40即使排序完全正确也必然混入至少20个恶意更新；原Top-k5的clean效用已不足。V3／V4的预测器参数参考[VERT论文实验设置](https://arxiv.org/html/2411.10673v1)，但projection256会触发项目已有的signed-hash投影、投影空间系数和历史存储适配，并非原论文dense投影的完整复现。AlignIns候选保留双重过滤与范数裁剪；.3／1／1为[作者代码默认值](https://github.com/JiiahaoXU/AlignIns/blob/master/src/federated.py)，.5／1半径范围参考[论文的半径对照](https://arxiv.org/html/2503.07978v2)。项目按论文Algorithm 1对选中更新等权聚合，与作者仓库部分样本加权实现存在差异，报告按项目实现解释结果。

选择只读取seed301的clean准确率及健康证据。候选须完整完成30轮、NaN/Inf为0、每个训练轮都有接纳更新、未撤销全部客户端、clean误撤销率不超过10%；最终准确率和第21—30轮平均准确率均须落后配对FedAvg不超过5个百分点。合格候选按末10轮准确率、最终准确率降序及候选ID升序选择。若某方法没有合格候选，保留其中完整且数值有限的最佳项用于压力对比，同时明确标记 `clean_qualified=false` 和失败原因；不放宽门槛，不把压力对比当成clean校准成功。若连完整有限候选都没有，则停止进入后续阶段。seed402结果、攻击ASR和官方测试集均不参与选择。

Ding13提前撤销已通过旧seed401逐轮记录与固定配额逻辑核对：攻击尚未开始时，检测器已经按传入的恶意比例逐轮指定异常名额，累计异常又触发永久撤销，造成60%／80%任务在第18／10轮撤销全部诚实客户端。该现象属于当前项目实现与“第25轮才攻击”协议的冲突，不能解释为C7攻击导致，也不能据此代表原论文的有效性。本轮保留原实现及其健康失败标记，不用已知攻击起始轮替检测器设置免责期，不将Ding13改为另一方案后混入同名对比。

seed402的六项配对clean先于十二项攻击执行；同一种子下保持各方法的数据、目标单图及相同比例恶意名单一致，仍分别记录指定训练目标、旧单图和750张独立源类图像指标。第25—60轮共有36个名义攻击轮，报告须结合实际攻击客户端调用、撤销情况及末10轮趋势解释，不能把已撤销客户端也计作持续在线攻击。60轮复核仍不等于150轮资格、跨平台结果或各方法充分攻击调优后的SOTA认证。

恢复时使用同一新入口和原配置，先确认旧进程已退出；已完成任务复用，未完成任务从最后完整轮断点继续，终结失败保留证据且不自动重试。两阶段分别校验源码和引用身份，进入final前冻结校准结果及选择。**前序六方案补测现已被本实验引用冻结，请勿再启动其旧控制器**，查看旧结果仅用 `--summary`；不要修改冻结源码、配置或诊断JSON后强行续跑原目录。

本地完成记录（2026-10-21，北京时间）：26项／1,320轮全部完成，2026-10-20 12:44:07至16:35:07 UTC，控制器累计13,859.7047秒（约231.00分钟）；所有任务NaN/Inf更新为0。校准耗时2,747.3387秒，seed402复核耗时11,112.3660秒。校准与final分别冻结45个源码／配置文件和164／216项引用，manifest指纹分别为 `03bcb4ff245f6117a7a31945358e9a8c743b9a6a630ea196b954c2d0ccf2929d` 与 `44c7724db3059b42a72bbd00b622a084e6c3cb56ee9b40eee8649bd760a64554`。

纯clean校准选定VERT V4（history10／projection256／预测5 epochs／lr .001／top-k20，无比例先验）及AlignIns A4（sparsity .3／两半径1.0），两者均合格。V4使用项目已有的signed-hash投影及投影空间预测适配，不称作者代码的完整复现；A4按预先固定的末10轮准确率选中，未依据攻击结果重选。新seed402下，两者clean最终准确率为45.9867%／46.3867%，相对FedAvg48.2800%分别低2.2933／1.8933个百分点，clean效用改善得以保留。Krum clean为27.5600%，其基础效用不足仍需单列。

以下均为seed402、100客户端、60轮结果；`末10`指第51—60轮，三项指标均为百分比。指定目标是攻击训练使用的一张辅助图，迁移ASR使用750张独立源类别验证图；两者及旧单图指标分别保存，不混作同一个ASR。

| 恶意比例 | 方法 | 最终Acc | 末10均Acc | 末10指定目标ASR | 末10迁移ASR |
|---|---|---:|---:|---:|---:|
| 60% | Ours | 46.2133 | 45.6240 | 0 | 8.4800 |
| 60% | VERT V4 | 34.7867 | 29.8600 | 100 | 15.2800 |
| 60% | FedAvg | 46.4667 | 44.3533 | 100 | 17.5067 |
| 60% | AlignIns A4 | 27.0667 | 25.9293 | 100 | 8.7200 |
| 60% | Krum | 28.7867 | 29.0400 | 0 | 4.3733 |
| 60% | Ding13/TAD | 27.1467 | 26.9293 | 0 | 9.8267 |
| 80% | Ours | 41.2133 | 40.9053 | 0 | 8.8667 |
| 80% | VERT V4 | 35.1067 | 31.6360 | 100 | 37.1733 |
| 80% | FedAvg | 46.1067 | 43.3027 | 100 | 13.8400 |
| 80% | AlignIns A4 | 19.5200 | 16.7613 | 100 | 2.4133 |
| 80% | Krum | 27.6933 | 26.9707 | 0 | 15.8000 |
| 80% | Ding13/TAD | 13.4933 | 13.6493 | 90 | 20.2933 |

Ours两比例在整个第25—60轮的平均准确率分别为44.1244%／40.4856%，均为六法最高；指定目标始终未命中，诚实误撤销为0。但它并非所有指标最优：60%最终准确率比FedAvg低0.2533个百分点，80%低4.8933个百分点；Krum60和AlignIns80的末10迁移ASR分别低于对应Ours。Ours60的旧单图末10ASR为20%（第57、58轮命中，配对clean也在第58轮命中），不能泛称所有单图攻击指标都是0。

Ding13两比例均提前永久删除98人，诚实误撤销分别为38/40、19/20，即95%。60%从第19轮起全部60名未来恶意身份已被撤销，只剩两名诚实客户端；该任务没有攻击开始时仍在线的恶意身份，不能以其0目标ASR证明成功抵御C7。80%从第11轮起仅剩一名诚实、一名恶意客户端。两项均未满足“全部诚实撤销”条件，且冻结报告的10%误撤销硬门槛只用于clean，所以`all_observed_health_pass=true`仍可能出现；这个布尔值不代表上述95%误撤销可接受，也不代表六法效用或攻击指标达标。

实际暴露也不同：Ours60只有第25轮的60次恶意调用，Ours80为第25—27轮共84次；之后固定恶意身份已撤销。FedAvg60／80持续承受2,160／2,880次调用，两者都满足预设指定目标攻击有效性条件。FedAvg后期准确率恢复，同时目标仍100%命中；末10轮大部分目标步骤因损失已为0按预定协议跳过，不能把恢复解释为攻击失效。60%与80%恶意名单不嵌套（重合49人），实际结果也没有在所有指标上单调恶化。

末10轮多项准确率仍明显上升，本轮不能认证收敛或150轮资格。Ding13／Krum从clean到攻击任务还改变了真实比例先验，攻击前轨迹已不同，因此其配对差值不是纯C7效应。完整三窗口指标、旧单图、实际调用、计时、独立审计及PNG/PDF曲线保存在本地 `outputs/cifar10_c7_60r_analysis_20261020/`，原始数据在独立实验输出目录；本轮结论限于固定协议和单个新seed402，不作为六法充分调优后的SOTA认证。

### AI Station：C7固定参数六方案150轮实验

新入口 `run_cifar_c7_station.py` 使用 `configs/cifar10_c7_station_150r_v1.json`，将本地C7、VERT V4和AlignIns A4用于独立的CUDA多卡实验。它直接执行冻结参数，不重复v14的12组网格搜索，也不根据此次验证结果重新选参。Ours、VERT、AlignIns、Krum、Ding13、FedAvg均保留。

| 项目 | 设置 |
|---|---|
| 客户端、轮数、分区 | 100客户端、150轮；IID及Dirichlet α=.5 |
| 恶意比例 | 0、.2、.4、.6、.8；0为各方法配对clean |
| 验证阶段 | seed301/302/303，180项、27,000轮 |
| 正式阶段 | seed401/402/403，180项、27,000轮 |
| 总规模 | **360项、54,000轮**；两阶段分别汇总 |
| 公共训练 | 原CNN、诚实SGD、E1、batch50、lr .01、decay .99、K20、第25轮开始攻击 |
| C7 | Adam固定lr .001、boost20、每组最多10个正常批次后一次目标步骤、攻击E1、距离系数1e-4；仅卷积/全连接权重额外放大并参与距离约束；源5→目标7、原单张辅助图 |
| Ours | 距离阈值2.0；其余v14公共检测参数保留 |
| VERT V4 | history10、projection256、预测5 epochs、lr .001、top-k20、关闭比例先验 |
| AlignIns A4 | sparsity .3、TDA/MPSA半径均1.0 |

训练集仍按同一split seed20261013拆为40,000训练图、7,500验证图、2,500攻击辅助图；验证阶段只评7,500验证图，正式阶段评官方10,000测试图。官方测试不用于本轮选择参数。上述种子按原v14保留，其中301/401/402已有本地观察，不能将整套实验描述为全部未见种子的盲测。第25—150轮共126个名义攻击轮，实际恶意调用数随撤销而变化。

**新输出为 `outputs/cifar10_c7_station_150r_v1/`**，不读取或覆盖旧v14、本地C7实验目录；AI Station无需复制本地实验outputs。源码、配置、数据划分及C7协议分别留存指纹，任务、轮次断点和逐轮指标都在新目录内。代码同步必须包含 `attack_stability_attack.py`、`attack_stability_worker.py` 两个C7依赖模块，不能只提交入口与JSON。

用户先在本地提交并推送代码，再在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
python run_cifar_c7_station.py --dry-run
python run_cifar_c7_station.py --check-gpus && \
  python -u run_cifar_c7_station.py --display-mode live
```

`--dry-run`不载入数据、不查询GPU、不创建输出。终端面板保留总进度条、任务/轮数、已用时、预计剩余时间和每张GPU正在执行的方法、参数、分区、比例、种子及轮次，并注明C7协议和当前评估集。重定向日志时使用默认`auto`或`--display-mode log`。

调度会检查全部可见CUDA卡，每卡同时一个训练任务；C7 Adam状态、实际驻留数据、VERT投影和Krum工作区分别估算显存。小显存卡可以领取能装下的方法，不必等待Krum；运行期间约每秒重新检查，资源释放后自动加入。CPU线程数不限制GPU任务数；每任务SM9工作线程为1、PyTorch CPU线程为2，减少多卡时CPU过度并发。显存估算留有余量，但共享机器仍可能发生竞争性OOM；最多自动重试一次，保留原任务身份并允许换卡续跑。

所有可见GPU暂时无法查询显存时，检查命令会显示等待状态；正式运行继续等待可用资源。没有运行任务且连续600秒无法分配新任务时，控制器保存状态并退出，重新执行相同启动命令续跑。没有可见CUDA设备则明确报错。只使用调度时足够空闲的资源，不能保证占满其他任务已使用的显卡。

恢复使用同一入口、配置和输出目录；程序阻止同目录双控制器，并核对冻结来源。已完成任务复用，未完成任务从最近已保存的轮次断点恢复；逐轮扩展指标回退到该断点后重算，尝试日志单独保留。改变协议或源码后应使用新版本输出，不能强行混入原结果。另一终端可只读检查：

```bash
python run_cifar_c7_station.py --summary
```

结果位于 `validation/`、`final_evaluation/`、`c7_runtime/` 及 `report.json`。报告保留最终轮、第25—150轮和末10轮指标，区分攻击使用的指定辅助目标、旧独立单图及全部独立源类图像ASR；迁移分母在验证/正式阶段分别为750/1,000，并记录实际分母、目标置信度、正常准确率、NaN/Inf、误撤销、诚实权重损失及实际攻击调用。跨种子统计给出实际完成数，未完成项不填0。

完整且数值有限的健康失败任务仍保留，并继续后续正式阶段；原v14健康门槛与攻击比例下高误撤销警示分别显示。Ding13原比例配额实现及Krum高恶意比例限制继续如实报告，不能以低ASR自动认定防御有效。数值异常不静默修补、减弱攻击或隐藏失败；执行失败会保留证据并停止派发新任务，提交其他正在运行任务的结果。本轮工程验证覆盖CPU小型真实训练/断点恢复及多卡调度模拟，尚未在AI Station实际执行，不预先保证150轮稳定性或六法排名。

### v14 本地完整网格（50 轮、50 客户端）

`configs/cifar10_v14_local_50r_50c.json` 从原 v14 配置派生：50 轮、50 客户端、恶意比例 `0/0.6/0.8`，验证单种子 `301`、最终测试单种子 `401`。0% 场景用于原选参流程的 clean 对照；六方法、IID/Dirichlet、三个可调方法各 12 组参数、数据划分、训练/攻击参数、健康与选择规则均沿用 v14。共 234 项验证（11,700 轮），验证选择成功后运行 36 项最终测试（1,800 轮）。50 客户端仍使用原来的全部训练数据，每位客户端平均分到的数据更多，不能按客户端数同比估算耗时。

本机使用通用入口自动选择 PyTorch 的可用设备（Apple GPU 使用 MPS，`jobs=auto` 为单设备串行）。AI Station 的多 CUDA 卡实验继续使用上方 v14 专用入口。

```bash
.venv/bin/python run_fair_tuning_from_config.py \
  --config configs/cifar10_v14_local_50r_50c.json --dry-run
EMIT_PROGRESS_EVENTS=1 .venv/bin/python -u run_fair_tuning_from_config.py \
  --config configs/cifar10_v14_local_50r_50c.json
```

独立输出为 `outputs/cifar10_v14_local_50r_50c/`，复用原调优器断点。已运行的本地实例另在该目录保存 `run.log`、`local_status.json`、配置快照及源码 SHA256；重复启动前先确认旧进程已退出。结果仍查看 `validation_results.csv`、`best_parameters.json`、`final_evaluation/summary.csv` 和 `final_evaluation/rounds.csv`。单种子、50 轮的本机结果用于短程对比，不能代替 150 轮、多种子的 AI Station 验证。

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

**Krum可执行性与理论条件**（num_clients=100）:

| ratio | 恶意数f | 可用邻居数n-f-2 | 满足原保证的数量条件2f+2<n |
|-------|---------|----------------|---------------------------|
| 0.0 | 0 | 98 | 是 |
| 0.2 | 20 | 78 | 是 |
| 0.4 | 40 | 58 | 是 |
| 0.6 | 60 | 38 | 否 |
| 0.8 | 80 | 18 | 否 |

邻居数为正只说明本实现能够计算，不代表具备鲁棒保证。60%／80%属于超出该数量条件的压力测试；数量条件满足时，原定理仍有梯度估计等其他假设。见[Krum原论文](https://papers.nips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)。

**注意事项**:
1. ⚠️ **不能断点恢复**: 修改`num_clients`会改变配置指纹，必须从头运行
2. **实验解释**：客户端数量和程序可执行性不能替代收敛、攻击有效性、基线适用条件及多种子验证。
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
