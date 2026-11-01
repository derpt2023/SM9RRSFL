# SM9-RRSFL：CIFAR-BEST 与 MNIST 固定配置复现

本仓库保留六种方法的训练实现、SM9/SM3 原生扩展，以及两套已完成实验的固定配置。新复现统一使用 `reproduce.py`，不需要运行历史调参脚本，也不需要旧 `outputs` 目录。默认只打印计划，只有 `--run` 才训练。

| 实验 | 固定种子 | 分布、恶意比例 | 训练轮数 | 六法总任务数 |
| --- | --- | --- | --- | --- |
| `cifar-best` | 2026102801、2026102802、2026102803 | IID / Dirichlet(α=0.5)，0/20/40/60/80% | 150 | 180 |
| `mnist_v7_target_fair_tuning` | 241、242、243 | 同上 | 50 | 180 |

方法名称与命令参数：Ours=`sm9rrs`、VERT=`vert`、AlignIns=`alignins`、Krum=`krum`、TAD=`ding13`、FedAvg=`fedavg`。

## 1. 现有结果与目录

```text
reproduce.py                          固定配置复现入口
configs/reproduction/                 固定参数、数据划分、来源校验清单
sm9rrsfl/                             训练、六法、检测、密码协议
scripts/experiments/cifar10/           CIFAR 攻击和观测依赖
third_party/gmssl/                    SM9 编译所需固定版源码、许可证、SHA
render_cifar_r4_formal.py              已接受 CIFAR 数据包的报告重绘
outputs/
  cifar-best/                         新 R4 30 项 + 原其他五法 150 项
  mnist_v7_target_fair_tuning/          MNIST 原始调参及最终 180 项
  old/                                历史实验、清理凭据及新复现默认位置
```

已有数据包入口：

- [CIFAR 六法报告](outputs/cifar-best/final_evaluation/report.html)，每 seed 独立 PNG/SVG；`summary.csv`、`rounds.csv`、`aggregate.csv` 保留全部原值。
- MNIST：`outputs/mnist_v7_target_fair_tuning/final_evaluation/summary.csv`、`aggregate.csv`，各 `seed_241/242/243` 的图。
- `outputs/cifar-best/provenance/task_sources.json` 逐项记录 R4 与五法来源。原 `cifar-paper` 位于 `outputs/old/cifar-paper`。

`data/` 和 `outputs/` 不随 Git 分发。克隆代码后可以下载公开数据从头训练；查看、重绘历史结果则需要上述结果包。已清理完成实验的大型断点，结果、逐轮指标和审计仍保留，不能依赖已删除断点续跑。

根目录仍保留部分旧命名模块：它们在原实验 126 个冻结来源或实际导入链中，不能仅按文件名删除。新实验只使用本页入口。

## 2. 环境与安装

训练使用 AI Station/Linux + NVIDIA CUDA；macOS 可运行计划、数据校验、测试和绘图。本轮整理没有启动训练。

原 CIFAR 环境为 Python 3.10.12、PyTorch `2.3.0a0+40ec155e58.nv24.03`、CUDA 12.4、cuDNN 9.0、NumPy 1.24.4、RTX 4090 D。NVIDIA 的 [24.03 发布说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-03.html)对应这个 PyTorch/CUDA 系列；它是搭建环境的起点，不保证自行编译出的原生二进制 SHA 与历史机器相同。

```bash
git clone --branch codex/cp1-iid-cuda-development https://github.com/derpt2023/SM9RRSFL.git
cd SM9RRSFL
```

在已有 CUDA PyTorch 的 AI Station 环境安装。若使用 NVIDIA 容器，可先在克隆目录运行：

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/workspace/SM9RRSFL -w /workspace/SM9RRSFL \
  nvcr.io/nvidia/pytorch:24.03-py3 bash
```

容器或 Station 终端内：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt setuptools wheel
export GMSSL_SOURCE="$PWD/third_party/gmssl"
python setup.py build_ext --inplace
python -c "import torch; from sm9rrsfl import _native_sm3, _native_sm9; print(torch.__version__, torch.cuda.is_available())"
```

SM9 需要 C 编译器和 Python 开发头文件；容器通常已提供。`third_party/gmssl` 包含原固定版本 `3.3.0-dev.1183` 的最小源码依赖及 Apache-2.0 许可证，不再依赖作者的 `~/Downloads`。原始压缩包 SHA 为 `6dc97c6b4f7d2f6df9d44f014cca0561a7b4776017efd4486d341e986051fab4`，逐文件 SHA 见 `third_party/gmssl/provenance.json`。不要用跳过 SM9 扩展的方式替代正式实验。

CIFAR 入口默认严格检查历史环境；其他机器需要显式加 `--allow-environment-difference`，差异会保存在每项 `environments/*.json`。该选项允许在其他环境复现固定协议，不代表结果应逐位相同。MNIST 原存档没有完整的软件/源码锁，因此只能核对固定配置与数据划分，不能承诺恢复其历史数值环境。

## 3. 下载数据、核对计划

```bash
python reproduce.py cifar-best --download
python reproduce.py mnist_v7_target_fair_tuning --download

python reproduce.py cifar-best
python reproduce.py mnist_v7_target_fair_tuning

python reproduce.py cifar-best --check-data
python reproduce.py mnist_v7_target_fair_tuning --check-data
```

已有数据可省略下载；通过 `--data-dir /absolute/path` 指定数据目录。下载只使用数据集加载器的公开 MNIST/CIFAR-10 地址。划分固定，校验不训练：

| 数据集 | 原训练样本 | 实际训练 | 训练衍生验证 | 攻击辅助集 | 官方测试 | 划分种子 |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 50,000 | 40,000 | 7,500 | 2,500 | 10,000 | 20261013 |
| MNIST | 50,000 | 45,000 | 2,500 | 2,500 | 10,000 | 20260810 |

CIFAR 核对六个数据数组和划分索引 SHA；MNIST 核对原存档记录的划分索引 SHA。两套任务均为 100 客户端，数据抽样种子 42，不将验证样本回填训练集。

## 4. 启动固定配置实验

以下命令只准备给复现者执行。`--methods` 可选一项或逗号分隔多项；省略时为六法。每个 seed、分布、比例各独立运行，无参数搜索。

### CIFAR：仅补做 R4，30 项

```bash
python reproduce.py cifar-best --methods sm9rrs --ready --allow-environment-difference
python reproduce.py cifar-best --methods sm9rrs --run \
  --gpus 0,1,2 --allow-environment-difference \
  --output outputs/old/reproductions/cifar-r4-run1
```

`cifar-best` 中其他五法的原 150 项应继续复用。下面的完整六法命令仅供独立从头复现实验，不是更新现有结果包所必需的步骤：

```bash
python reproduce.py cifar-best --run --gpus 0,1,2 \
  --allow-environment-difference \
  --output outputs/old/reproductions/cifar-six-run1
```

CIFAR 固定 R4 为恢复系数 4、裁剪因子 1，保留原 Ours 核心；CNN、batch 50、学习率 0.05、衰减 0.99、每轮本地 1 epoch。攻击从第 25 轮开始，V2 交替最小化，target 学习率 **0.01**、10 个 stealth steps、boost 15、目标数 128、source 5 → target 7，使用原 target-view 变换。VERT 使用已接受的 PaperVERT 实现，TAD 使用无比例先验的实现；不能只用通用六法入口替代这些作用域。

### MNIST：最终固定配置，180 项

```bash
python reproduce.py mnist_v7_target_fair_tuning --ready
python reproduce.py mnist_v7_target_fair_tuning --run --gpus 0,1,2 \
  --output outputs/old/reproductions/mnist-final-run1
```

本入口直接使用最终执行 manifest 的参数，不重复 v7 调参。batch 32、学习率 0.05、衰减 0.99、本地 1 epoch、检测窗口 7；攻击从第 12 轮开始，stealth steps 1、boost 15、目标数 128。Ours 固定 `sm9rrs-012`、VERT `vert-003`、AlignIns `alignins-008`。SM9 工作线程采用实际执行 manifest 的 1；原导出 summary 中的 8 是分配前配置，不能据此更改实际执行配置。

### 调度、停止与继续

`--gpus` 使用 `nvidia-smi` 的物理编号或 UUID。新复现入口每张卡一项，最多六项；每次启动前连续两次、间隔 5 秒检查可用显存至少 8 GiB，并持有本入口的 GPU 锁。原 Station 的两块隔离卡 UUID 仍禁止使用。请按自己的空闲卡修改示例中的 `0,1,2`。

任务日志在 `logs/`，实时状态在 `controller_status.json` 与 `tasks/*/status.json`。SIGINT/SIGTERM 停止新派发，等待已运行 worker 完成。失败不自动重试。

新目录默认拒绝覆盖。仅要继续尚未开始的任务，在**相同命令和参数**后加 `--continue-pending`；已有目录的完成、失败、部分任务均不重跑。若有失败，保留原证据后使用新目录单独处理，不修改原结果。运行中不要编辑来源文件。

## 5. 汇总与重绘

新训练会自动生成 `final_evaluation/summary.csv`、`rounds.csv`、`aggregate.csv` 和 `task_status.json`，也可重新汇总：

```bash
python reproduce.py cifar-best --summarize --output outputs/old/reproductions/cifar-r4-run1
python reproduce.py mnist_v7_target_fair_tuning --summarize --output outputs/old/reproductions/mnist-final-run1
```

MNIST 新运行按 seed 生成独立图。CIFAR 的官方口径表格和每 seed 图单独写入 `final_evaluation/official_test/`，口径映射见其中的 `metric_scope.json`；另有 `official_rounds.json`，其中 `accuracy` 与 `source_transfer_asr` 是官方 test 指标；通用 summary/rounds 中的 `attack_target_success_rate` 是原辅助目标口径，**不能作为 CIFAR 官方 ASR**。新运行不自动继承历史完整审计通过的结论，需独立审计后再合并正式结果。

重绘**现有已接受** CIFAR 六法数据，无需训练：

```bash
python render_cifar_r4_formal.py \
  --source outputs/cifar-best/export/combined_bundle.json \
  --baseline outputs/cifar-best/provenance/accepted_cifar_paper_bundle.json \
  --expected-sha256 7a56eec00bd2ef936f342e1f2a22088b674cff11d86bcff1981176c5bb6a6198 \
  --output outputs/old/reproductions/cifar-best-redraw
```

该重绘入口要求原始合并数据包，不能把新 worker 的任意 JSON 当作已接受 bundle。尾 10 轮与最终一轮必须分开报告；CIFAR 官方 test 的 10,000/源类 1,000 张与原验证的 7,500/源类 750 张不能混算。

## 6. 验证与结果边界

```bash
python run_reproduction_tests.py
```

原调参控制器测试仍保留冻结版本及旧证据路径，不能把全目录 discovery 当作新克隆的通过门槛；上面的入口覆盖新复现和相关科学机制。

新复现测试检查 180 项矩阵、固定参数、旧结果目录保护和无历史 outputs 的启动计划；安装后还应执行第 3 节的数据校验。测试使用小型合成数据，不运行上述正式实验。CIFAR 的原 126 个科学来源逐 SHA 保留，新增入口负责配置、环境、调度和保存，不改训练公式。

已接受 CIFAR 结果：R4 IID 尾 10 轮平均 Acc 56.9203%，与六法最高均值相差 1.1851 个百分点；15 个 IID 场景中 11 个差距 ≤1.5 个百分点，另 4 个仍超出。Non-IID Acc 平均提高 0.5452 个百分点，15 项均提高，但部分场景 ASR 上升，不能称全面改善。

R4 原验证增益为 +0.4794 个百分点，未达冻结 +0.5 门槛，`selected=null` 和历史首批 95/96 配对审计失败继续保留。本次官方测试是看过验证后由用户指定 R4；旧 test 与种子已见，不声称全新未见测试集。MNIST 原性能目标状态为 `unmet`，TAD 原失败对照仍为失败对照。复现和报告应保留所有健康、失败、逐 seed、逐比例结果，不以平均值掩盖退化。

清理前 README、移除源码与精确清单保存在 `outputs/old/reproducibility_cleanup_20261102/`。历史记录可追溯；日常使用以本 README 和 `configs/reproduction/` 为准。
