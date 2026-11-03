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
render_cifar_best.py                  已接受 CIFAR 数据包按 MNIST 格式重绘
outputs/
  cifar-best/                         新 R4 30 项 + 原其他五法 150 项
  mnist_v7_target_fair_tuning/          MNIST 原始调参及最终 180 项
  old/                                历史实验、清理凭据及新复现默认位置
```

已有数据包入口：

- [CIFAR 六法可视化](outputs/cifar-best/final_evaluation/visualizations.html)和[完整报告](outputs/cifar-best/final_evaluation/report.html)，每 seed 按 MNIST 格式分别绘制 IID / Non-IID 的准确率、ASR、总耗时、扣除密码学耗时。`summary.csv`、`rounds.csv`、`aggregate.csv` 保留全部原值。
- MNIST：`outputs/mnist_v7_target_fair_tuning/final_evaluation/summary.csv`、`aggregate.csv`，各 `seed_241/242/243` 的图。
- `outputs/cifar-best/provenance/task_sources.json` 逐项记录 R4 与五法来源。原 `cifar-paper` 位于 `outputs/old/cifar-paper`。

`data/` 和 `outputs/` 不随 Git 分发。克隆代码后可以下载公开数据从头训练；查看、重绘历史结果则需要上述结果包。已清理完成实验的大型断点，结果、逐轮指标和审计仍保留，不能依赖已删除断点续跑。

根目录仍保留部分旧命名模块：它们在原实验 126 个冻结来源或实际导入链中，不能仅按文件名删除。新实验只使用本页入口。

## 2. 环境与安装

训练使用 AI Station/Linux + NVIDIA CUDA；macOS 可运行计划、数据校验、测试和绘图。

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

### CIFAR 耗时复测：与 MNIST 使用同一计时入口

原 CIFAR 保存的 worker 总耗时包含数据加载、额外取证和断点写入，与 MNIST 的计时边界不同。2026-11-03 用户明确授权修正后完整重测六法 180 项，各 150 轮；采用独立目录，不复用上一轮 78 项其他方法结果。默认仅展示计划：

```bash
python run_cifar_timing_corrected.py
python run_cifar_timing_corrected.py --ready
python run_cifar_timing_corrected.py --batch --gpus 2,4,5,7 --max-gpus 4 \
  --output outputs/old/cifar_mnist_timing_corrected150_v1
python run_cifar_timing_corrected.py --summary \
  --output outputs/old/cifar_mnist_timing_corrected150_v1
python -m unittest tests.test_cifar_timing_corrected tests.test_cifar_mnist_timing tests.test_cifar_timing_followup
```

这是当前 Station 的独立纠错研究入口，固定上述四卡 UUID，并核验原始失败记录；普通新机器复现固定配置使用第 4 节 `reproduce.py`。`--ready` 检查软件、数据和来源，正式启动还会核对旧研究已退出、历史结果完整、新目录不存在。只在 Station 上训练；原 193 个来源不改，新入口、测试和17项失效清单独立冻结。原 R4 / PaperVERT / 无先验 TAD / V2 参数与三个种子不变。

修正后的入口在原科学作用域内，将公共计时函数绑定到当前实际训练函数。每个 checkpoint 在公共函数已扣除的 I/O 时间内核对实际检测器，记录 `training_binding.json`；TAD 必须是 `PriorFreeTAD`，VERT 必须是 `PaperVERT`。本轮不改变 MNIST 的 `run_measured_experiment`：数据加载后开始计时，扣公共 checkpoint I/O；结果保存与额外取证在计时外。去密码学耗时仅减四段互不重叠的协议墙钟时间，不减并行操作累计时间。

每个 seed、分布和恶意比例组成六法一组，在同一 GPU 上依次测量；30 组轮换方法顺序，各方法在每个位置出现5次。最多四卡、每卡一个本研究任务，8 GiB 空闲显存连续两次间隔5秒、持锁准入；不使用 GPU0/1/3/6。记录共享负载，不声称独占测速。已有目录拒绝覆盖，无自动重试；TERM/INT 停派发并自然排空。

历史保留：第一次 GPU1 OOM 曾获得一次接续授权。之后发现旧计时入口缓存训练函数，17项 TAD 实际运行旧 Ding13 检测器及比例先验，不能用于最终耗时比较。该轮已停止，95执行完成中17项TAD无效、78项其他法完成，85项未开始；累计96次实际尝试。`run_cifar_timing.py` 和 `resume_cifar_timing.py` 只保留为冻结依赖和历史证据，不应直接启动。本轮180项全部 fresh，完成后累计276次实际尝试，不删除旧失败或把旧TAD标为有效。

最终 `analysis/timing_results.json` 包含逐seed逐比例原值、均值/样本SD/n、健康问题、实际算法及单次执行审计、完整历史成本。成功临时 checkpoint 由公共入口清除，失败保留。新耗时独立报告，辅助目标ASR仅作诊断，不替换 `cifar-best` 官方Acc/ASR或覆盖原bundle。CIFAR与MNIST的模型、轮数、攻击工作量及SM9 workers仍有差异，计时边界一致不等于工作量相同。

新训练会自动生成 `final_evaluation/summary.csv`、`rounds.csv`、`aggregate.csv` 和 `task_status.json`，也可重新汇总：

```bash
python reproduce.py cifar-best --summarize --output outputs/old/reproductions/cifar-r4-run1
python reproduce.py mnist_v7_target_fair_tuning --summarize --output outputs/old/reproductions/mnist-final-run1
```

MNIST 新运行按 seed 生成独立图。CIFAR 的官方口径表格和每 seed 图单独写入 `final_evaluation/official_test/`，口径映射见其中的 `metric_scope.json`；另有 `official_rounds.json`，其中 `accuracy` 与 `source_transfer_asr` 是官方 test 指标；通用 summary/rounds 中的 `attack_target_success_rate` 是原辅助目标口径，**不能作为 CIFAR 官方 ASR**。新运行不自动继承历史完整审计通过的结论，需独立审计后再合并正式结果。

重绘**现有已接受** CIFAR 六法数据，无需训练：

```bash
python render_cifar_best.py \
  --package outputs/cifar-best \
  --expected-sha256 7a56eec00bd2ef936f342e1f2a22088b674cff11d86bcff1981176c5bb6a6198 \
  --output outputs/old/reproductions/cifar-best-redraw
```

输出目录必须是新目录。默认生成 30 幅 SVG 和每 seed 的 `visualizations.html`，配色、线型、五比例准确率 / 四比例 ASR 布局与 MNIST 一致；`runtime_visualizations.html` 集中展示各 seed 的两种耗时。需要 PNG 时追加 `--png-browser /path/to/chrome`，macOS 可用 `--png-browser "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`。

`*_runtime_without_crypto.svg` 使用各任务原始总墙钟时间减去四段互不重叠的密码协议墙钟时间；不减并行签名/验签累计时间，不代表关闭密码学重跑或独占速度测试。CIFAR 原峰值内存字段均为占位值 0，对应 `*_memory_overhead.svg` 明示“未记录”，不将其画成实际 0 MB。

该重绘入口要求原始合并数据包及配套 summary，不能把新 worker 的任意 JSON 当作已接受 bundle。尾 10 轮与最终一轮必须分开报告；CIFAR 官方 test 的 10,000/源类 1,000 张与原验证的 7,500/源类 750 张不能混算。原 `render_cifar_r4_formal.py` 作为冻结来源保留。

## 6. 验证与结果边界

```bash
python run_reproduction_tests.py
```

原调参控制器测试仍保留冻结版本及旧证据路径，不能把全目录 discovery 当作新克隆的通过门槛；上面的入口覆盖新复现和相关科学机制。

新复现测试检查 180 项矩阵、固定参数、旧结果目录保护和无历史 outputs 的启动计划；安装后还应执行第 3 节的数据校验。测试使用小型合成数据，不运行上述正式实验。CIFAR 的原 126 个科学来源逐 SHA 保留，新增入口负责配置、环境、调度和保存，不改训练公式。

已接受 CIFAR 结果：R4 IID 尾 10 轮平均 Acc 56.9203%，与六法最高均值相差 1.1851 个百分点；15 个 IID 场景中 11 个差距 ≤1.5 个百分点，另 4 个仍超出。Non-IID Acc 平均提高 0.5452 个百分点，15 项均提高，但部分场景 ASR 上升，不能称全面改善。

R4 原验证增益为 +0.4794 个百分点，未达冻结 +0.5 门槛，`selected=null` 和历史首批 95/96 配对审计失败继续保留。本次官方测试是看过验证后由用户指定 R4；旧 test 与种子已见，不声称全新未见测试集。MNIST 原性能目标状态为 `unmet`，TAD 原失败对照仍为失败对照。复现和报告应保留所有健康、失败、逐 seed、逐比例结果，不以平均值掩盖退化。

清理前 README、移除源码与精确清单保存在 `outputs/old/reproducibility_cleanup_20261102/`。历史记录可追溯；日常使用以本 README 和 `configs/reproduction/` 为准。
