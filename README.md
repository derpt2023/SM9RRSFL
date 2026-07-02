# SM9-RRS-FL 实验复现

本项目根据 `方案（计算机学报排版）.docx` 中的 SM9-RRS-FL 方案，提供一套可执行的联邦学习投毒防御实验框架，用于后续论文实验和对比研究。

## 已实现内容

- MNIST IDX gzip 与 CIFAR-10 python batches 数据加载与可选下载。
- 默认基于 NumPy 的卷积神经网络（CNN）联邦训练，并提供可选 PyTorch 后端用于 GPU 加速：MNIST 使用轻量 compact CNN，CIFAR-10 使用参考 FedAvg/CIFAR-10 经验配置的 `Conv-Conv-FC-FC-Logits` CNN 与按通道标准化输入；支持单个客户端数量或 `20/50/100` 等多客户端数量对比、本地训练轮次、批大小、学习率、学习率衰减、IID 独立同分布/Dirichlet 非 IID 划分和误差阈值早停。
- 默认实验比例：`0%`、`10%`、`20%`、`40%`、`45%`、`60%`、`80%`，其中 `0%` 用于无恶意节点收敛对比。
- SM9-RRS-FL 流程：基于 SM9 群运算的可追踪环签名、论文同款动态累加器、SM9 加密撤销陷门、可链接标签、纵向 SVD 投毒检测、审计撤销、黑名单剔除。
- 疑似恶意节点处理采用文献 [13] 同款动态降权：单次异常先降低聚合权重，后续正常则恢复权重，连续异常达到阈值后再触发撤销/剔除。
- Krum 与 FedAvg baseline，对照实验使用相同数据划分、恶意比例和攻击方式；FedAvg/加权聚合按客户端本地样本数加权，Krum 保持原始单更新选择语义。
- 文献 [13] 对照实验：复现其“奇异值轨迹差分 + Isolation Forest + 动态权重惩罚/恢复 + 连续异常剔除”的在线投毒检测流程。
- 输出 `summary.csv`、`rounds.csv`、`summary.json`，并自动生成 HTML/SVG 可视化图表，便于后续绘图和论文表格整理。

真实动态模式默认使用用户提供的 GmSSL C 源码：`sm9rrsfl/_native_rrs.c` 在 GmSSL 国标 SM9 曲线、点乘和双线性配对 API 上实现本方案原有的动态累加器、常数大小环签名 `σ=(h,R,S,T)` 与 SM9 IBE 撤销陷门，而不是用普通 SM9 身份签名替代环签名。公共环仍由 `V=[∏(v_i+s)]P1` 和见证 `W_i=[∏_{j≠i}(v_j+s)]P1` 表示，签名包不携带完整成员列表；协议公式、链接标签、审计撤销和检测规则保持不变。原生扩展不可用时会回退到原有 Python 实现。

## 目录结构

当前代码按“核心方案、训练实验、密码开销实验、文档与测试”拆分：

- `sm9rrsfl/crypto.py`、`sm9rrsfl/accumulator.py`：SM9-RRS 数据包、签名/验签、动态累加器和撤销陷门。
- `sm9rrsfl/_native_rrs.c`、`sm9rrsfl/_native_sm3.c`：基于 GmSSL 的原生环签名/IBE 后端和大更新 SM3 摘要后端。
- `sm9rrsfl/fl.py`、`sm9rrsfl/experiments.py`：联邦训练主流程和 MNIST/CIFAR-10 对比实验入口。
- `sm9rrsfl/benchmarks/`：独立微基准实验入口，不混入 CNN 训练时间；目前包含签名与验签开销测试。
- `gmssl/`：项目内置国密 SM2/SM3/SM4/SM9 实现。
- `tests/`：单元测试与流程自检。
- `docs/`：论文补充说明和机制设计材料。
- `outputs/`：实验输出目录，默认由启动命令自动创建。

## 环境说明

建议使用 Python 3.10 及以上版本。首次下载项目后，可按以下方式创建虚拟环境并安装依赖：

```bash
git clone git@github.com:derpt2023/SM9RRSFL.git
cd SM9RRSFL

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 将 GmSSL-master.zip 放在 ~/Downloads 后，原地构建原生 SM3/SM9-RRS 扩展
python setup.py build_ext --inplace
```

基础运行不强制依赖 PyTorch、torchvision 或 scikit-learn；文献 [13] 的 Isolation Forest 已用纯 NumPy 实现。项目内已经包含实验所需的 `gmssl` SM2/SM3/SM4/SM9 代码。

`setup.py` 会依次查找环境变量 `GMSSL_SOURCE`、已解压的 `~/Downloads/GmSSL-master/` 和 `~/Downloads/GmSSL-master.zip`；也可用 `GMSSL_ARCHIVE=/绝对路径/GmSSL-master.zip` 指定压缩包。原生运算会释放 Python GIL。程序启动时输出 `sm3_backend=native-extension` 和 `rrs_backend=gmssl-native` 才表示两层加速都已启用；构建失败不影响正确性，但会回退到明显更慢的 Python 实现。

可单独检查后端：

```bash
python -c "from sm9rrsfl.crypto import rrs_backend_name, sm3_backend_name; print(sm3_backend_name(), rrs_backend_name())"
```

`--crypto-mode simulated` 明确用于快速验证联邦学习、攻击和检测流程，其更新摘要使用系统库加速的 SHA-256；`--crypto-mode sm9` 才使用真实 SM3/SM9。两种模式不会写入同一个默认输出目录。

如果希望在 Mac 或 Windows 上使用 GPU 加速 CNN 本地训练，需要额外安装 PyTorch：

- macOS Apple Silicon：安装官方 PyTorch 后，可用 `--compute-backend torch --device mps` 走 Metal/MPS。
- Windows/Linux NVIDIA：按 PyTorch 官网安装与你显卡驱动匹配的 CUDA 版 PyTorch 后，可用 `--compute-backend torch --device cuda`。
- 如果不确定设备是否可用，可以使用 `--compute-backend auto --device auto`；代码会优先选择 CUDA，其次选择 MPS，否则回落到 NumPy。

示例检查命令：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("mps", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

安装完成后建议先运行测试：

```bash
python -m unittest discover -s tests
```

## 快速自检

不下载真实数据集，使用合成 MNIST 形状数据做 smoke run：

```bash
python -m sm9rrsfl.experiments \
  --dataset synthetic \
  --crypto-mode simulated \
  --num-clients 8 \
  --rounds 7 \
  --train-samples 800 \
  --test-samples 200 \
  --methods sm9rrs krum ding13 fedavg
```

真实 SM9 模式也可以运行；大规模重复实验建议先使用 `--crypto-mode simulated`，确认参数后再切换到 `--crypto-mode sm9`。

## 签名与验签开销实验

该实验用于回答“客户端数量变化时，本方案签名与验签耗时是多少”。实验入口是 `sm9rrsfl/benchmarks/crypto_overhead.py`，不会训练模型，只测密码层。

实验设计：

- 对每个客户端数量 `N` 分别创建 `N` 个客户端身份，并初始化 `SM9RRSContext`。
- 单独记录 `setup_ms`，包含 SM9 主密钥/客户端私钥提取、动态累加器和见证材料生成。
- 默认先对所有客户端各签名一次，记录为 `sign_cache_ms`，用于预热动态累加器签名中可复用的身份相关材料；如需把首次签名冷启动开销计入逐次签名，可加 `--no-precompute-sign-cache`。
- 每次迭代构造同等大小的模型更新摘要，单独计时签名算法，得到 `sign_*_ms`。
- 使用对应签名包调用公开验签流程，单独计时并校验结果，得到 `verify_*_ms`。
- 输出 `summary.csv`、`samples.csv`、`summary.json`、`visualizations.html` 和 `plots/*.svg`，其中 `summary.csv` 按客户端数量给出 mean/median/p95/std，`samples.csv` 保留每次迭代的原始样本，可视化页面展示单次签名/验签均值对比、签名耗时、验签耗时、上下文初始化和签名缓存预热开销。

快速自检可以先跑 simulated 模式：

```bash
python -m sm9rrsfl.benchmarks.crypto_overhead \
  --crypto-mode simulated \
  --accumulator-mode dynamic \
  --client-counts 20 50 100 \
  --iterations 30 \
  --warmup 3 \
  --update-size 4096 \
  --output-dir outputs/crypto_overhead_simulated
```

论文最终开销建议跑真实 SM9 模式：

```bash
python -m sm9rrsfl.benchmarks.crypto_overhead \
  --crypto-mode sm9 \
  --accumulator-mode dynamic \
  --client-counts 20 50 100 \
  --iterations 30 \
  --warmup 3 \
  --update-size 4096 \
  --output-dir outputs/crypto_overhead_sm9
```

这条命令各参数含义如下：

- `python -m sm9rrsfl.benchmarks.crypto_overhead`：以模块方式运行签名/验签开销实验脚本。
- `--crypto-mode sm9`：使用真实 SM9 密码运算；若改为 `simulated`，则只用于快速流程自检。
- `--accumulator-mode dynamic`：使用本方案的动态累加器模式，即常数大小环签名结构。
- `--client-counts 20 50 100`：分别测试客户端数量为 `20`、`50`、`100` 时的开销。
- `--iterations 30`：每种客户端数量下正式记录 `30` 次密码操作样本，用于计算平均值、P95 等；这里不是联邦学习的 `30` 轮训练。
- `--warmup 3`：正式计时前先运行 `3` 次预热，预热样本不计入最终统计。
- `--update-size 4096`：构造长度为 `4096` 的模型更新向量，用于模拟客户端上传的一次更新。
- `--output-dir outputs/crypto_overhead_sm9`：指定 CSV、JSON、SVG 和 HTML 可视化的输出目录。

结果口径说明：

- `sign_mean_ms` / `sign_p95_ms` 表示单个客户端对一次更新执行一次签名的耗时统计。
- `verify_mean_ms` / `verify_p95_ms` 表示验证方对一个客户端的一份签名包执行一次验签的耗时统计。
- `setup_ms` 单独统计上下文初始化开销，包括 SM9 参数/主密钥生成、客户端私钥提取、动态累加器和见证材料生成；它不计入 `sign_mean_ms` 或 `verify_mean_ms`。
- `sign_cache_ms` 单独统计签名缓存预热开销；它也不计入 `sign_mean_ms` 或 `verify_mean_ms`。
- `P95` 是 95 分位数：将多次耗时从小到大排序后，约 95% 的样本不超过该值，用于观察尾部开销。
- 上述单次签名/验签开销不是所有客户端总时间。若要估算一轮联邦学习中所有客户端均上传一次更新的串行密码开销，可近似使用 `单次开销 × 客户端数量`；若要估算 `30` 轮，则再乘以 `30`。
- 当 `--client-counts 100 --iterations 30` 时，只会正式记录 `30` 次单次签名/验签样本，并按客户端顺序轮换采样；如果希望 100 客户端场景中每个客户端至少进入一次正式统计，可以设置 `--iterations 100` 或更大。

如果要对比旧版抽样环模式，可以运行：

```bash
python -m sm9rrsfl.benchmarks.crypto_overhead \
  --crypto-mode sm9 \
  --accumulator-mode none \
  --ring-size 5 \
  --client-counts 20 50 100 \
  --iterations 30 \
  --warmup 3 \
  --update-size 4096 \
  --output-dir outputs/crypto_overhead_legacy_ring5
```

如果只想保存 CSV/JSON，不生成 SVG 和 HTML，可以额外加入：

```bash
--no-visualizations
```

## 主实验说明

主实验使用统一的模型接口，并分别在 MNIST 和 CIFAR-10 上运行。MNIST 仍使用轻量 compact CNN；CIFAR-10 会自动切换到更适合该数据集的 `Conv-Conv-FC-FC-Logits` CNN，并对 CIFAR-10 图像执行通道标准化。本项目不会在一个命令中同时跑两个数据集：每次启动只选择一个 `--dataset`，默认输出目录也按数据集隔离，避免 CIFAR-10 影响 MNIST 的复现实验速度。

两组主实验均对比本方案、Krum、文献 [13] 和 FedAvg，并保留原有实验变量：恶意节点比例、IID/Dirichlet 数据分布、Dirichlet 参数、客户端数量、训练轮次和目标误差阈值。

论文实验配置原则：先用 `0%` 恶意节点的 FedAvg/干净训练确认数据集和 CNN 能正常收敛，然后固定同一套模型结构、`E`、`B`、`lr`、`lr_decay` 和 `rounds`，再比较 SM9-RRS-FL、FedAvg、Krum 和文献 [13] 在攻击场景下的鲁棒性。不要为某个防御方法单独调整 CNN 或训练超参。

### 数据集模型与训练协议

| 数据集 | CNN 结构 | 参数量 | `rounds` | `E` / `--local-epochs` | `B` / `--batch-size` | `lr` | `lr_decay` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST | `Conv(8, 3x3, stride=2) -> AvgPool(2x2) -> FC(392, 10)` | 4,010 | 30 | 1 | 32 | 0.05 | 1.00 |
| CIFAR-10 | `Conv(64, 5x5) -> AvgPool(2x2) -> Conv(64, 5x5) -> AvgPool(2x2) -> FC(4096, 384) -> FC(384, 192) -> FC(192, 10)` | 1,756,426 | 300 | 5 | 50 | 0.05 | 0.99 |

`lr_decay` 是每轮通信后的学习率乘子；CIFAR-10 第 `t` 轮实际学习率为 `0.05 * 0.99^(t-1)`。MNIST 训练较容易，保持常数学习率即可。CIFAR-10 的 `E=5, B=50, lr_decay=0.99` 参考 FedAvg 在 CIFAR-10 上增加本地计算、使用学习率衰减的经验；`rounds=300` 用作正式鲁棒性对比的默认预算，若要追求更高 clean FedAvg 收敛上限，可单独增加干净基线轮数。

联邦聚合口径：

- FedAvg 使用各客户端本地样本数作为聚合权重；在 Dirichlet 非 IID 划分下，样本更多的客户端对全局更新贡献更大。
- SM9-RRS-FL 和文献 [13] 的动态权重会再乘以本地样本数，兼顾防御权重和统计样本量。
- Krum 保持原始算法语义：每轮选择一个更新，不在 `0%` 恶意节点时自动退化为 FedAvg。
- SM9-RRS-FL 在 `0%` 恶意节点诊断场景中不触发 SVD 剔除；CIFAR-10 有恶意节点时会使用更长检测窗口和更保守阈值，以降低良性客户端误剔除。

## simulated 模式快速实验

如果只是想先确认 CNN 训练流程、四组方法对比、IID/Dirichlet 划分和可视化是否能正常跑通，可以先使用 `--crypto-mode simulated` 做小规模快速实验。下面两条命令分别运行 MNIST 和 CIFAR-10，不会互相影响，也不会覆盖正式主实验输出。

MNIST 快速实验：

```bash
python -m sm9rrsfl.experiments \
  --dataset mnist \
  --download \
  --data-dir data/mnist \
  --methods sm9rrs krum ding13 fedavg \
  --ratios 0.00 0.20 0.40 \
  --partitions iid dirichlet \
  --dirichlet-alpha 0.5 \
  --num-clients 10 \
  --rounds 5 \
  --train-samples 2000 \
  --test-samples 500 \
  --target-error 0.12 \
  --crypto-mode simulated \
  --output-dir outputs/quick_mnist_simulated
```

CIFAR-10 快速实验：

```bash
python -m sm9rrsfl.experiments \
  --dataset cifar10 \
  --download \
  --data-dir data/cifar10 \
  --methods sm9rrs krum ding13 fedavg \
  --ratios 0.00 0.20 0.40 \
  --partitions iid dirichlet \
  --dirichlet-alpha 0.5 \
  --num-clients 10 \
  --rounds 5 \
  --train-samples 3000 \
  --test-samples 500 \
  --target-error 0.12 \
  --crypto-mode simulated \
  --output-dir outputs/quick_cifar10_simulated
```

快速实验只用于预跑和排查问题；正式论文结果仍建议使用下面的主实验命令，并在最终密码开销实验中切换为 `--crypto-mode sm9`。

## MNIST 主实验

运行本方案、Krum、文献 [13] 和 FedAvg 四组对照：

```bash
python -m sm9rrsfl.experiments \
  --dataset mnist \
  --download \
  --data-dir data/mnist \
  --methods sm9rrs krum ding13 fedavg \
  --ratios 0.00 0.10 0.20 0.40 0.45 0.60 0.80 \
  --partitions iid dirichlet \
  --dirichlet-alpha 0.5 \
  --client-counts 20 50 100 \
  --rounds 30 \
  --local-epochs 1 \
  --batch-size 32 \
  --lr 0.05 \
  --lr-decay 1.0 \
  --target-error 0.12 \
  --crypto-mode sm9 \
  --compute-backend auto \
  --device auto
```

真实 SM9 MNIST 主实验默认写入 `outputs/mnist/`；模拟模式默认写入 `outputs/mnist_simulated/`。

## CIFAR-10 干净基线

在跑投毒和防御对比前，建议先确认 CIFAR-10 的干净训练能正常收敛。该入口会自动设置 `--dataset cifar10`、`--methods fedavg`、`--ratios 0.00`、`--attack none`、`--partitions iid`，并使用完整 CIFAR-10 训练集/测试集。未显式传入训练参数时，会使用上表中的 CIFAR-10 论文实验协议：`--rounds 300`、`--local-epochs 5`、`--batch-size 50`、`--lr 0.05`、`--lr-decay 0.99`。

```bash
python -m sm9rrsfl.experiments \
  --cifar10-clean-baseline \
  --download \
  --data-dir data/cifar10 \
  --compute-backend auto \
  --device auto
```

默认输出目录为 `outputs/cifar10_clean_baseline/`。如果只是做快速烟测，可以显式追加 `--rounds 2 --train-samples 400 --test-samples 100 --output-dir outputs/cifar10_clean_smoke`。

## CIFAR-10 主实验

CIFAR-10 使用相同四组方法和相同实验变量，但单独启动：

```bash
python -m sm9rrsfl.experiments \
  --dataset cifar10 \
  --download \
  --data-dir data/cifar10 \
  --methods sm9rrs krum ding13 fedavg \
  --ratios 0.00 0.10 0.20 0.40 0.45 0.60 0.80 \
  --partitions iid dirichlet \
  --dirichlet-alpha 0.5 \
  --client-counts 20 50 100 \
  --rounds 300 \
  --local-epochs 5 \
  --batch-size 50 \
  --lr 0.05 \
  --lr-decay 0.99 \
  --target-error 0.12 \
  --crypto-mode sm9 \
  --compute-backend auto \
  --device auto \
  --jobs auto \
  --eval-interval 5 \
  --sm9-workers auto
```

真实 SM9 CIFAR-10 主实验默认写入 `outputs/cifar10/`；模拟模式默认写入 `outputs/cifar10_simulated/`。

上面的 `--jobs auto` 会根据系统物理内存和当前实验规模估算外层实验配置并发数；检测到单个 MPS/CUDA GPU 时自动固定为 `1`，避免配置争抢显存以及 CUDA fork 初始化错误。`--jobs 1` 只关闭配置级并发，单个配置仍使用 GPU。只有确认多 GPU 且已配置进程到设备的映射后，才建议显式提高并发数。

外层并发默认优先使用进程池，让不同实验配置互相隔离；如果当前运行环境限制系统 semaphore 或进程池初始化，程序会提示 `process_pool_unavailable=...` 并自动回退到线程池继续执行。

`--eval-interval 5` 表示每 5 轮评估一次测试集准确率，最后一轮始终评估。它不改变训练、投毒、防御、聚合和最终评估公式，但启用早停时判断粒度会变粗，可能多运行到下一个评估点；如果论文图表需要完整逐轮曲线，可改回 `--eval-interval 1`。

`--sm9-workers auto`（默认）会用最多 8 个线程并行处理每轮中相互独立的 GmSSL 封包、签名和验签；原生 SM3、SM9 点乘和配对都会释放 GIL。SVD 轨迹检测仍按客户端稳定顺序更新状态，因此并行不会改变撤销、降权和异常判定语义。

主实验运行时会显示配置级进度条和预计剩余时间，例如 `42/168 25.0% elapsed=... eta=...`。串行模式下会显示当前正在运行的配置；并发模式下会在每个配置完成后更新总进度。ETA 基于已经完成配置的平均耗时估算，前几个配置完成前会显示 `eta=estimating`。

在本机 Mac 上启用 PyTorch MPS 后端，可以在命令末尾额外加入：

```bash
--compute-backend torch --device mps
```

在 Windows 或 Linux 的 NVIDIA CUDA 环境中，可以改为：

```bash
--compute-backend torch --device cuda
```

如果希望自动选择可用 GPU，并在不可用时回落到原 NumPy 实现，可以使用：

```bash
--compute-backend auto --device auto
```

启用 PyTorch 后端时，数据集、客户端索引、全局参数、客户端更新和聚合结果会持续驻留所选设备。FedAvg/加权 FedAvg、Krum 距离、投毒变换和 Ding13 SVD 特征均直接在设备端执行；真实 SM9 只为协议要求的更新 SM3 摘要回传一次 CPU 字节。MPS 不支持的 SVD 算子会显式转到 NumPy/CPU。上述优化不改变客户端划分、攻击启动轮次、聚合公式、检测阈值或撤销规则。

运行默认启用断点续跑：第 1 轮前及每轮结束后都会原子写入 `输出目录/.checkpoints/`，每完成一个配置还会先更新事务式结果快照，再派生 CSV/JSON。训练、密码运算、检测、聚合、断电或手动终止发生在一轮中间时，保留的是上一轮完整一致状态，不会恢复“完成一半”的聚合。修复代码后使用完全相同的数据规模和参数重启命令，会从下一轮继续并跳过已完成配置；代码本身不参与运行指纹，实验参数或数据变化才会使旧结果移入 `.stale/`。以后再次执行某个旧命令时，程序也会按指纹自动找回对应 `.stale/` 结果，无需手工搬回 CSV。确实需要从零重跑时加 `--no-resume`。

交互式终端发现“完整配置字段和运行指纹都相同”的断点时，会询问 `是否沿着断点运行（Y/N）`：输入 `Y` 从下一轮继续；输入 `N` 从第 0 轮开始，并将原断点备份到 `.checkpoints/discarded/`。CI、重定向输入等非交互式环境不会等待键盘输入，默认选择 `Y`。运行指纹是将数据集元信息和整组有效实验配置规范化为 JSON 后计算的 SHA-256 摘要，用来防止不同实验误用同一结果；它不包含代码版本，因此修复实现错误不会让同参数断点失效。

如果希望先快速复现模型收敛趋势和可视化，可以把上面的 `--crypto-mode sm9` 改成 `--crypto-mode simulated`。真实 SM9 模式会显著更慢，适合用于最终密码开销实验；模拟模式适合先检查准确率、抗投毒趋势和图表生成。

如果显式传入 `--output-dir`，则以手动指定的目录为准。

运行完成后，脚本会自动生成以下可视化：

- 无恶意节点时，各方案随训练轮次增加的模型收敛/准确率对比。
- 不同恶意节点比例下，各方案随训练轮次增加的模型收敛/准确率对比。
- 各方案在不同恶意节点比例下的运行时间开销对比。
- 各方案在不同恶意节点比例下的进程峰值 RSS 内存开销对比。
- 在客户端数量为 `20`、`50`、`100` 时，各方案的最终准确率、运行时间和峰值内存横向对比。

当命令中包含 `--partitions iid dirichlet` 时，脚本会分别为 IID 独立同分布和 Dirichlet 非 IID 两种数据场景生成独立图表。图表文件名前缀分别为 `iid_` 与 `dirichlet_alpha_..._`，总览页 `visualizations.html` 会按场景分块展示。

当命令中包含 `--client-counts 20 50 100` 时，脚本会对每个数据划分场景分别运行 `20`、`50`、`100` 个客户端的实验。原有收敛曲线会按客户端数拆分，例如 `iid_clients_020_accuracy_ratio_045.svg`；同时会额外生成 `iid_client_count_accuracy_ratio_045.svg`、`iid_client_count_runtime_ratio_045.svg`、`iid_client_count_memory_ratio_045.svg` 这类客户端数量横向对比图。若只运行单个数据划分场景，文件名前缀会省略为 `client_count_...`。若不传 `--client-counts`，则保持旧行为，只运行 `--num-clients` 指定的单个客户端数量。

如果某个方法提前达到误差阈值，曲线会在真实停止轮次后用较淡的虚线平台段延伸到最大轮次，表示该方法已经停止聚合、最终准确率保持不变。这样既保留“误差小于阈值时停止聚合”的实验语义，也能让各方案在同一横轴上比较。

如果你希望各方案都强制跑满固定轮次，便于观察完整收敛曲线，可以加入：

```bash
--no-early-stop
```

如果已经跑完实验，只想根据现有 CSV 重新生成图表，不想重新训练，可以运行：

```bash
python -m sm9rrsfl.experiments \
  --dataset mnist \
  --output-dir outputs/mnist \
  --visualize-only
```

如果要重新生成 CIFAR-10 的图表，可以运行：

```bash
python -m sm9rrsfl.experiments \
  --dataset cifar10 \
  --output-dir outputs/cifar10 \
  --visualize-only
```

如果要重新生成模拟模式的图表，需要指定对应数据集或输出目录，例如：

```bash
python -m sm9rrsfl.experiments \
  --dataset mnist \
  --crypto-mode simulated \
  --visualize-only
```

如果要更贴近文献 [13] 的公开实验比例，可以额外运行：

```bash
python -m sm9rrsfl.experiments \
  --dataset mnist \
  --download \
  --data-dir data/mnist \
  --methods ding13 \
  --ratios 0.10 0.28 0.40 \
  --num-clients 50 \
  --rounds 50 \
  --target-error 0.12 \
  --crypto-mode simulated \
  --output-dir outputs/ding13_mnist
```

## 常用参数

- `--dataset mnist|cifar10|synthetic`：选择单个实验数据集；MNIST 和 CIFAR-10 请分开启动。
- `--cifar10-clean-baseline`：运行完整 CIFAR-10 干净 FedAvg 基线，默认输出到 `outputs/cifar10_clean_baseline/`。
- `--data-dir data/mnist|data/cifar10`：数据集目录；未指定时会按 `--dataset` 自动选择默认目录。
- `--methods sm9rrs krum ding13 fedavg`：选择实验方法。
- `--ratios 0.00 0.10 0.20 0.40 0.45 0.60 0.80`：设置恶意节点比例，建议保留 `0.00` 用于生成无恶意节点基线图。
- Krum 的邻居数为 `n-f-2`。若该值小于 `1`（例如 `10` 个客户端、`80%` 恶意客户端时 `f=8`），程序会在训练前跳过该无定义配置并写入 `skipped_configs.json`，不会修改 `f` 或让整个实验网格中断。即使某些高恶意比例仍可计算，超过经典 Krum 条件 `n≥2f+3` 时也只应视为压力测试，不代表理论鲁棒性保证仍成立。
- `--train-samples 10000` / `--test-samples 2000`：限制真实数据集样本量；不传时使用完整真实数据集。快速实验建议显式传入较小样本数。
- `--num-clients 20`：设置单个客户端数量；当没有传 `--client-counts` 时生效。
- `--client-counts 20 50 100`：一次运行多个客户端数量，并生成客户端数量横向对比图；该参数优先于 `--num-clients`。
- `--rounds 30|300`：通信轮数；MNIST 默认 `30`，CIFAR-10 默认 `300`。
- `--local-epochs 1|5`：每轮通信中客户端本地训练的 epoch 数，即 FedAvg 论文中的 `E`；MNIST 默认 `1`，CIFAR-10 默认 `5`。
- `--batch-size 32|50`：客户端本地 mini-batch size，即 FedAvg 论文中的 `B`；MNIST 默认 `32`，CIFAR-10 默认 `50`。
- `--attack sign_flip|gaussian|alternating|none`：设置投毒方式，默认 `alternating`；其中 `alternating` 为交替投毒攻击，会把完整攻击目标拆成多个触发器特征子集，由不同恶意客户端在不同轮次只注入其中一个小扰动分片。
- `--attack-scale 5.0`：设置投毒强度；在 `alternating` 中表示被选中触发器分片的局部扰动约为本地更新 RMS 的百分比，默认 `5.0` 即单次只注入约 `5%` 的小扰动。
- `--attack-start-round 0`：默认值 `0` 表示在检测器完成基线窗口后开始攻击。
- `--partition iid|dirichlet`：设置数据划分方式。
- `--partitions iid dirichlet`：一次运行多个数据划分场景，并分别生成可视化；该参数优先于 `--partition`。
- `--dirichlet-alpha 0.5`：非 IID 强度，值越小各客户端标签分布越偏。
- `--crypto-mode sm9|simulated`：真实 SM9 或快速模拟密码层。
- `--compute-backend numpy|auto|torch`：CNN 本地训练/评估后端；默认 `auto`，有 CUDA/MPS 时自动使用 PyTorch GPU，否则回落到 NumPy；`numpy` 和 `torch` 可强制指定实现。
- `--device auto|cpu|cuda|mps`：PyTorch 设备；Mac Apple Silicon 使用 `mps`，Windows/Linux NVIDIA 使用 `cuda`，`auto` 优先 CUDA、其次 MPS。
- `--jobs 1|auto|N`：外层实验配置并发数。`auto` 会按系统物理内存和实验规模估算上限，并在单 GPU/MPS/CUDA 后端下保守限制并发；该参数只改变 168 个配置的调度顺序，不改变单个配置内部的实验变量。
- `--eval-interval 1|5|10`：每隔多少轮评估一次测试集准确率；最终轮始终评估。`1` 会保留完整逐轮曲线，较大值可减少 CIFAR-10 正式实验中的测试集前向开销。
- `--sm9-workers 1|auto|N`：SM9-RRS 每轮封包、签名和验签 worker 数；检测器状态仍按客户端顺序更新，不改变方案中的 SVD 轨迹判定、降权或撤销规则。
- `--no-resume`：忽略输出目录中的已完成配置和逐轮检查点，从零开始本次实验；默认会安全断点续跑。
- `--no-progress`：关闭终端进度条和 ETA 输出；训练、结果文件和可视化生成不受影响。
- `--accumulator-mode dynamic|none`：默认 `dynamic`，使用动态累加器和常数大小环签名；`none` 为旧版实验抽象，会在签名包中携带抽样环成员列表。
- `--ring-size 5`：旧版 `--accumulator-mode none` 下的抽样环大小；动态累加器模式下公共环默认为当前实验的全部客户端，签名包只记录累加器摘要和公共环大小。
- `--no-early-stop`：不使用误差阈值早停，所有方法都跑满 `--rounds`。
- `--suspicion-penalty-factor 0.5`：本方案中疑似节点的权重惩罚因子。
- `--suspicion-recovery-factor 2.0`：疑似节点后续恢复正常时的权重恢复倍数。
- `--suspicion-remove-after 3`：连续疑似达到该次数后再撤销身份并剔除。
- `--no-visualizations`：只输出 CSV/JSON，不生成 HTML/SVG 图表。
- `--visualize-only`：不重新训练，只读取输出目录中的 `summary.csv` 和 `rounds.csv` 重新生成图表。
- `--lr 0.05`：CNN 本地训练初始学习率；MNIST 和 CIFAR-10 默认均为 `0.05`。
- `--lr-decay 1.0|0.99`：每轮通信后的学习率衰减乘子；MNIST 默认 `1.0`，CIFAR-10 默认 `0.99`。
- `--output-dir outputs/custom`：手动指定输出目录；未指定时，`sm9` 默认写入 `outputs/<dataset>/`，`simulated` 默认写入 `outputs/<dataset>_simulated/`，`--cifar10-clean-baseline` 默认写入 `outputs/cifar10_clean_baseline/`。

## 输出文件

默认输出目录按数据集和密码模式区分：MNIST 的 `--crypto-mode sm9` 写入 `outputs/mnist/`，CIFAR-10 的 `--crypto-mode sm9` 写入 `outputs/cifar10/`；模拟模式分别写入 `outputs/mnist_simulated/` 和 `outputs/cifar10_simulated/`；CIFAR-10 干净基线写入 `outputs/cifar10_clean_baseline/`。每个输出目录都会包含：

- `summary.csv`：每个方法和恶意比例的一行摘要，同时包含训练、攻击、摘要、封包、签名、验签、检测、聚合和评估的分阶段耗时。使用多个 `sm9-workers` 时，密码字段是各客户端操作耗时之和，用于判断热点；`runtime_seconds` 才是配置的真实墙钟时间。
- `rounds.csv`：逐轮准确率、误差、接收/拒绝更新数、黑名单数量、TP/FP 等。
- `summary.json`：与 `summary.csv` 对应的 JSON 结果。
- `run_manifest.json`：本次数据规模与完整配置指纹，用于确认检查点可以安全复用。
- `skipped_configs.json`：因算法数学条件不成立而在训练前跳过的配置及具体原因。
- `.completed_results.pickle`：已完成配置的事务式权威快照；即使断电发生在多个 CSV 替换之间，也能据此恢复并重新生成表格。
- `last_failure.json`：最近一次未预料异常、完整堆栈、出错配置和最后完成轮次；该配置后来成功完成时会标记为 `resolved`。
- `.checkpoints/*.pickle`：当前未完成配置的逐轮状态，配置成功完成后自动删除。
- `.checkpoints/discarded/*.pickle`：用户在断点询问中选择 `N` 后保留的旧断点备份，不参与自动续跑。
- `visualizations.html`：自动生成的可视化总览页面。
- `plots/*.svg`：各个图表的 SVG 文件，可直接插入论文或进一步编辑。

每轮和每个配置的结果都通过临时文件及原子替换保存。长时间实验中途退出后，既能保留已经完成的配置，也能从当前配置的下一轮继续，而不必重跑前面的数百轮。

当同时运行 IID 和 Non-IID 时，典型图表包括：

- `plots/iid_accuracy_baseline.svg`
- `plots/iid_accuracy_ratio_020.svg`
- `plots/dirichlet_alpha_0_5_accuracy_baseline.svg`
- `plots/dirichlet_alpha_0_5_accuracy_ratio_020.svg`

当同时运行多个客户端数量时，典型图表还包括：

- `plots/iid_clients_020_accuracy_ratio_045.svg`
- `plots/iid_clients_050_accuracy_ratio_045.svg`
- `plots/iid_clients_100_accuracy_ratio_045.svg`
- `plots/iid_client_count_accuracy_ratio_045.svg`
- `plots/iid_client_count_runtime_ratio_045.svg`
- `plots/iid_client_count_memory_ratio_045.svg`

时间开销使用单个配置运行的墙钟时间；内存开销使用当前 Python 进程的峰值 RSS。由于同一脚本会顺序运行多组配置，RSS 是粗粒度指标；如果需要更严格的内存隔离，可以分别运行单个 `--methods` 和单个 `--ratios` 配置后对比。

注意：Krum 在实现上需要满足 `n - f - 2 >= 1`，其中 `n` 为当前参与客户端数，`f` 为恶意客户端数。默认 `20/50/100` 客户端数量下，`80%` 恶意节点仍可运行；如果把客户端数改得很小，`60%` 或 `80%` 可能会使 Krum 无法计算。即使在 `0%` 恶意节点下，Krum 也保持“每轮选择一个更新”的原始语义，不自动替换为 FedAvg。

## 文献 [13] 复现说明

文献 [13] 为：

Ding Z, Wang W, Li X, et al. Identifying alternately poisoning attacks in federated learning online using trajectory anomaly detection method. Scientific Reports, 2024, 14: 20269. 论文链接：[Nature Scientific Reports](https://www.nature.com/articles/s41598-024-70375-w)。

本文献方法在每轮联邦学习中记录客户端模型参数轨迹，对参数代表矩阵提取奇异值，并用相邻轮次奇异值差分构造轨迹特征；随后使用 Isolation Forest 判断异常客户端，对异常客户端降低聚合权重，对恢复正常的客户端提升权重，连续异常客户端被移除。项目中的实现位于 `sm9rrsfl/ding13_detector.py`。
