# SM9-RRS-FL 实验复现

本项目根据最新版 `方案第二版（计算机学报排版）.docx` 中的 SM9-RRS-FL 方案，提供一套可执行的联邦学习投毒防御实验框架，用于后续论文实验和对比研究。

## 已实现内容

- MNIST IDX gzip 与 CIFAR-10 python batches 数据加载与可选下载。
- 默认基于 NumPy 的卷积神经网络（CNN）联邦训练，并提供可选 PyTorch 后端用于 GPU 加速：MNIST 使用轻量 compact CNN，CIFAR-10 使用参考 FedAvg/CIFAR-10 经验配置的 `Conv-Conv-FC-FC-Logits` CNN 与按通道标准化输入；支持单个客户端数量或 `20/50/100` 等多客户端数量对比、本地训练轮次、批大小、学习率、学习率衰减、IID 独立同分布/Dirichlet 非 IID 划分和误差阈值早停。
- 默认实验比例：`0%`、`10%`、`20%`、`40%`、`45%`、`60%`、`80%`，其中 `0%` 用于无恶意节点收敛对比。
- SM9-RRS-FL v2 流程：固定任务环、常数大小签名 `σ=(c,A,B,C)`、任务级匿名标签、两个签名验证等式、纵向 SVD 投毒检测、D-KGC 门限追踪、门限 Schnorr 证书确认和任务环更新。
- 疑似恶意节点处理采用动态降权：单次异常先降低聚合权重，后续正常则恢复权重；同一任务标签连续异常达到阈值后立即拒绝该触发轮梯度并请求追踪，但只在追踪证书验证成功后永久撤销身份。
- 纵向 SVD 对完整的一维模型更新按 `ceil(|G|/num_classes) × num_classes` 规范成矩阵（末尾补零），前 `K` 次观测建立基线，第 `K+1` 次开始评分；异常特征不进入正常窗口，但仍作为下一轮公式中的相邻 `r-1` 观测。
- 攻击端实现 Bhagoji 等人的带距离约束交替最小化：恶意客户端在本地训练中交替优化目标误分类损失与正常任务/距离隐蔽损失，并只对目标攻击步进行显式提升；Ding 等人的实验以该攻击为基础。旧版“轮换参数分片并注入随机扰动”的实现已经删除，不再把一般向量噪声称为交替最小化攻击。
- 联邦学习主流程通过 `ClientSigner`、`ASVerifier` 和 `AuditorService` 角色对象分别调用签名、完整验签和门限追踪。客户端对象只保存本客户端私钥和成员见证；AS 对象保存验签所需公开参数、非公开任务点、TPK、审计台账和独立的审计提交认证密钥，但不含任何客户端签名私钥或 D-KGC 追踪份额。AS 候选状态仅按不透明任务标签索引，不保存标签到真实身份的映射。
- Krum 与 FedAvg baseline，对照实验使用相同数据划分、恶意比例和攻击方式；FedAvg/加权聚合按客户端本地样本数加权，Krum 保持原始单更新选择语义。
- 文献 [13] 对照实验：复现其“奇异值轨迹差分 + Isolation Forest + 动态权重惩罚/恢复 + 连续异常剔除”的在线投毒检测流程。
- 输出 `summary.csv`、`rounds.csv`、`summary.json`，并自动生成 HTML/SVG 可视化图表，便于后续绘图和论文表格整理。

当前协议版本为 v2。任务公共环由 `RID`、`ACC` 和成员见证 `W_π` 表示；客户端生成任务级标签 `Tag_π` 与承诺 `R_tag`，AS 通过两个验证等式同时检查环签名及标签归属。同一标签连续异常达到阈值后，AS 先用独立控制面 Schnorr 票据认证其提交的精确证据；每个参与 D-KGC 逻辑端点验证该票据后，再对 `TaskID、RID、H5(E_π)` 和一次性会话标识生成独立批准。只有至少 `t` 份有效批准才能启动门限追踪并生成 `τ_trace`，节点名称或整数编号本身不构成授权。该票据用于实现论文所假设的 AS→Auditor/D-KGC 认证信道，不进入 `E_π`、`H5(E_π)` 或环签名公式。AS 仅在验证追踪证据与门限 Schnorr 证书后更新黑名单和任务环。协议中不存在加密身份陷门，也不存在独立的非交互式零知识证明对象；Fiat-Shamir 哈希挑战 `c` 是环签名本身的一部分。

AS 创建追踪证据时会把完整不可变证据、摘要和控制面授权票据自动登记到内部 pending 台账；仅调用验证函数不会清除该状态。只有匹配证据的门限追踪结果验证通过并经 `archive_trace_result` 显式归档后，对应记录才会关闭；完整证据与认证结果会继续保存在检查点中，归档存储仍须实施访问控制。追踪临时失败时，`C_tol` 触发轮更新仍按 Word 方案保持拒绝状态，检查点保存原始证据供重启后优先重试。`finalize_task` 不再接受调用者自报的布尔值，发现任何待处理审计就拒绝销毁。全部审计关闭后，系统才清零当前进程中的 `κ_t`，删除 `h_t`、任务标签缓存和环历史，并写入无秘密 tombstone 防止任务重新激活；若撤销后环为空，则直接进入该终态，不复用旧环。此前已经复制到外部存储的旧检查点仍须由其存储所有者按保留策略删除。

实现边界需要明确区分：`crypto_mode="sm9"` 的群、配对、SM3 `H_v` 和规范序列化由 GmSSL 国标 SM9 原生桥执行；Shamir/Feldman 份额关系、份额域交叉项乘法、随机盲化求逆、门限部分结果组合、逐节点追踪批准和门限 Schnorr 验证均在代码中执行。`ξ`、`P_r`、`P_r^(-1)`、`Δ_j`、`msk` 和 `β_j` 不在环建立或追踪路径中重构；求逆只开放论文允许公开的随机掩码乘积。角色对象之间采用单向下发的最小状态，AS/客户端对象图不含 D-KGC、追踪网关或其他客户端私钥。论文所述 Paillier 密文交互、独立进程和多节点认证信道仍由单进程中的份额域协议模型代替，未部署为真实跨主机网络；同一 Python 进程内的调试器、`gc` 遍历或内存读取也不属于安全边界。检查点由可信实验编排层统一保存各节点份额，因此本实现可验证协议代数、正常角色调用链和实验开销，但不能等同于具备进程/主机级信任隔离的生产 D-KGC。

## 目录结构

当前代码按“核心方案、训练实验、密码开销实验、文档与测试”拆分：

- `sm9rrsfl/crypto.py`、`sm9rrsfl/accumulator.py`、`sm9rrsfl/threshold.py`：v2 数据包、角色能力、签名/验签、任务环累加器、D-KGC 份额域协议、门限追踪及门限 Schnorr 证书。
- `sm9rrsfl/sm9_backend.py`、`sm9rrsfl/_native_sm9.c`：定长字节接口和基于 GmSSL 的国标 SM9 `G1/G2/GT`、配对及 `H_v` 原生桥。
- `sm9rrsfl/_native_sm3.c`：大更新 SM3 摘要的可选原生加速器。`_native_rrs.c` 是已停用的 v1 源码，不参与 v2 构建或运行。
- `sm9rrsfl/fl.py`、`sm9rrsfl/experiments.py`：联邦训练主流程和 MNIST/CIFAR-10 对比实验入口。
- `sm9rrsfl/benchmarks/`：独立微基准实验入口，不混入 CNN 训练时间；目前包含签名与验签开销测试。
- `gmssl/`：历史 Python 兼容代码；v2 的真实 SM9 群运算不使用其中的旧双线性曲线，而由 `_native_sm9` 调用用户提供的 GmSSL C 源码。
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

# 将 GmSSL-master.zip 放在 ~/Downloads 后，原地构建 SM3 与 SM9 v2 扩展
python setup.py build_ext --inplace
```

基础运行不强制依赖 PyTorch、torchvision 或 scikit-learn；文献 [13] 的 Isolation Forest 已用纯 NumPy 实现。`simulated` 模式无需 GmSSL 原生扩展；真实 `sm9` 模式需要用户提供的 GmSSL C 源码和成功构建的 `_native_sm9` 扩展。

`setup.py` 优先使用显式的 `GMSSL_SOURCE=/绝对路径/GmSSL-master`；否则读取 `GMSSL_ARCHIVE` 指定的压缩包，未指定时使用 `~/Downloads/GmSSL-master.zip`。默认归档固定为 GmSSL `3.3.0-dev.1183`，构建前校验 SHA-256，按摘要隔离缓存并执行受限解压；源码存在时 `_native_sm9` 编译失败会直接终止。真实模式启动时应输出 `rrs_backend=gmssl-sm9-native-v2`。若该桥未构建，`crypto_mode="sm9"` 会明确报错，不会静默退回非国标参考曲线；`simulated` 模式仍可运行。`_native_sm3` 构建失败时，更新摘要会回退到 OpenSSL 或 Python SM3。

原生桥会严格检查定长编码、曲线、无穷点和素数阶子群，并在耗时群运算中释放 GIL。需要注意，当前 GmSSL z256 底层并未声明为恒定时间实现，因此此桥用于本地论文实验，不应直接作为具备侧信道防护的生产密码模块。

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
- 单独记录 `setup_ms`，包含 D-KGC 门限参数生成和客户端签名私钥提取。
- 默认注册任务并为全部成员生成非公开 `h_t`、`RID`、`ACC`、见证、D-KGC 内部的追踪份额、`g1/g2` 和任务标签材料，记录为 `task_precompute_ms`；如需把任务首次建立开销留在在线路径，可加 `--no-task-precompute`。
- 每次迭代构造同等大小的模型更新摘要，单独计时签名算法，得到 `sign_*_ms`。
- 使用对应签名包调用 AS 的两个验证等式，单独计时并校验结果，得到 `verify_*_ms`。
- 输出 `summary.csv`、`samples.csv`、`summary.json`、`visualizations.html` 和 `plots/*.svg`，其中 `summary.csv` 按客户端数量给出 mean/median/p95/std，`samples.csv` 保留每次迭代的原始样本，可视化页面展示单次签名/验签均值、签名耗时、验签耗时、上下文初始化和任务材料预计算开销。

快速自检可以先跑 simulated 模式：

```bash
python -m sm9rrsfl.benchmarks.crypto_overhead \
  --crypto-mode simulated \
  --dkg-threshold 2 \
  --dkg-nodes 3 \
  --task-id crypto-overhead \
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
  --dkg-threshold 2 \
  --dkg-nodes 3 \
  --task-id crypto-overhead \
  --client-counts 20 50 100 \
  --iterations 30 \
  --warmup 3 \
  --update-size 4096 \
  --output-dir outputs/crypto_overhead_sm9
```

这两个命令所用参数及其默认值统一列在下文“命令行参数说明（集中索引）”的“签名与验签开销入口”小节中。

结果口径说明：

- `sign_mean_ms` / `sign_p95_ms` 表示单个客户端对一次更新执行一次签名的耗时统计。
- `verify_mean_ms` / `verify_p95_ms` 表示验证方对一个客户端的一份签名包执行一次验签的耗时统计。
- `setup_ms` 单独统计 D-KGC 门限参数和客户端私钥材料的初始化开销；它不计入 `sign_mean_ms` 或 `verify_mean_ms`。
- `task_precompute_ms` 单独统计任务环、成员见证、签名公共量和任务标签材料的预计算开销；它也不计入 `sign_mean_ms` 或 `verify_mean_ms`。
- `P95` 是 95 分位数：将多次耗时从小到大排序后，约 95% 的样本不超过该值，用于观察尾部开销。
- 上述单次签名/验签开销不是所有客户端总时间。若要估算一轮联邦学习中所有客户端均上传一次更新的串行密码开销，可近似使用 `单次开销 × 客户端数量`；若要估算 `30` 轮，则再乘以 `30`。
- 当 `--client-counts 100 --iterations 30` 时，只会正式记录 `30` 次单次签名/验签样本，并按客户端顺序轮换采样；如果希望 100 客户端场景中每个客户端至少进入一次正式统计，可以设置 `--iterations 100` 或更大。

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
- SM9-RRS-FL 严格使用第 4.3.3 节归一化后的动态权重 `w_i`，不再二次乘本地样本数；文献 [13] 对照实现仍保持其原实验聚合口径。
- Krum 保持原始算法语义：每轮选择一个更新，不在 `0%` 恶意节点时自动退化为 FedAvg。
- SM9-RRS-FL 对所有通过密码验证的更新统一执行相同的纵向 SVD 检测；代码不读取“是否恶意”的实验真值来改变判定，也不按数据集暗改窗口或阈值。

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
  --attack alternating_minimization \
  --attack-boost 10 \
  --attack-epochs 10 \
  --attack-stealth-steps 10 \
  --attack-distance-weight 1e-4 \
  --attack-source-label 5 \
  --attack-target-label 7 \
  --attack-target-count 1 \
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

`--sm9-workers auto`（默认）会用最多 8 个线程调度每轮中相互独立的封包、签名和验签。GmSSL v2 原生桥在点乘、配对和目标群运算期间释放 GIL；实际加速比仍取决于 CPU 核数和配对开销。SVD 轨迹检测按任务标签的稳定顺序更新状态，因此并行不会改变追踪、降权和异常判定语义。

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

启用 PyTorch 后端时，数据集、客户端索引、全局参数、客户端更新和聚合结果会持续驻留所选设备。FedAvg/加权 FedAvg、Krum 距离、投毒变换和 Ding13 SVD 特征均直接在设备端执行；真实 SM9 只为协议要求的更新 SM3 摘要回传一次 CPU 字节。MPS 不支持的 SVD 算子会显式转到 NumPy/CPU；CPU 路径使用缩放后的 float64 等价分解，并在 LAPACK 不收敛时依次回退特征值分解和 Jacobi/幂迭代。任何包含 `NaN/Inf` 的客户端更新都会在摘要、检测和聚合前拒绝，避免污染全局模型。上述优化不改变客户端划分、攻击启动轮次、聚合公式、检测阈值或撤销规则。

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

## 命令行参数说明（集中索引）

README 示例中使用的项目参数均集中列在本节。主实验与签名/验签开销实验是两个独立入口；同名参数应按其所在入口理解。正确的目标误差参数名是 `--target-error`，不是 `--error-target`。

### 主实验入口

以下参数适用于：

```bash
python -m sm9rrsfl.experiments
```

#### 数据集、实验网格与输出

- `--dataset mnist|cifar10|synthetic`：选择单个实验数据集，默认 `mnist`；MNIST 和 CIFAR-10 请分开启动。
- `--cifar10-clean-baseline`：强制使用 CIFAR-10、FedAvg、无攻击、恶意比例 `0` 和 IID 划分；默认使用完整数据集、`20` 个客户端并写入 `outputs/cifar10_clean_baseline/`。它不会自动关闭早停，若要固定跑满轮数还需加入 `--no-early-stop`。
- `--data-dir data/mnist|data/cifar10`：数据集目录；未指定时会按 `--dataset` 自动选择默认目录。
- `--download`：在本地文件缺失时下载 MNIST 或 CIFAR-10；已存在的文件不会重复下载。`synthetic` 不需要该参数。
- `--train-samples 10000` / `--test-samples 2000`：对 MNIST/CIFAR-10 限制载入的训练集/测试集样本数，对 `synthetic` 指定生成的样本数；真实数据集不传时使用完整数据。
- `--methods sm9rrs krum ding13 fedavg`：选择一个或多个实验方法，默认运行全部四种方法。
- `--ratios 0.00 0.10 0.20 0.40 0.45 0.60 0.80`：设置一个或多个恶意节点比例，取值范围为 $[0,1)$；建议保留 `0.00` 用于生成无恶意节点基线图。
- `--num-clients 20`：设置单个客户端数量，默认 `20`；当没有传 `--client-counts` 时生效。
- `--client-counts 20 50 100`：一次运行多个客户端数量，并生成客户端数量横向对比图；该参数优先于 `--num-clients`。兼容旧参数名 `--num-clients-list`。
- `--partition iid|dirichlet`：设置单个数据划分方式，默认 `iid`。
- `--partitions iid dirichlet`：一次运行多个数据划分场景，并分别生成可视化；该参数优先于 `--partition`。
- `--dirichlet-alpha 0.5`：Dirichlet 非 IID 划分参数，默认 `0.5`；值越小，各客户端的标签分布越偏。
- `--output-dir outputs/custom`：手动指定输出目录；未指定时，`sm9` 写入 `outputs/<dataset>/`，`simulated` 写入 `outputs/<dataset>_simulated/`，CIFAR-10 干净基线写入 `outputs/cifar10_clean_baseline/`。
- `--seed 42`：设置随机种子，默认 `42`；影响样本抽取、客户端划分、恶意客户端选择、本地训练和攻击等随机过程。

Krum 的邻居数为 $n-f-2$。若该值小于 $1$（例如 $10$ 个客户端、$80\%$ 恶意客户端时 $f=8$），程序会在训练前跳过该无定义配置并写入 `skipped_configs.json`，不会修改 $f$ 或让整个实验网格中断。即使某些高恶意比例仍可计算，超过经典 Krum 条件 $n\geq 2f+3$ 时也只应视为压力测试，不代表理论鲁棒性保证仍成立。

#### 本地训练、评估与停止条件

- `--rounds 30|300`：最大通信轮数；MNIST/Synthetic 默认 `30`，CIFAR-10 默认 `300`。
- `--local-epochs 1|5`：每轮通信中客户端本地训练的 epoch 数，即 FedAvg 论文中的 $E$；MNIST/Synthetic 默认 `1`，CIFAR-10 默认 `5`。
- `--batch-size 32|50`：客户端本地 mini-batch size，即 FedAvg 论文中的 $B$；MNIST/Synthetic 默认 `32`，CIFAR-10 默认 `50`。
- `--lr 0.05`：CNN 本地训练初始学习率；三个数据集默认均为 `0.05`。
- `--lr-decay 1.0|0.99`：每轮通信后的学习率衰减乘子；MNIST/Synthetic 默认 `1.0`，CIFAR-10 默认 `0.99`。
- `--target-error 0.12`：目标测试误差阈值，默认 `0.12`。程序仅在测试集评估轮检查 $1-\mathrm{Accuracy}\leq 0.12$，即准确率不低于 $0.88$ 时允许早停；含恶意客户端的配置不会在实际攻击开始轮之前早停。
- `--eval-interval 1|5|10`：每隔多少轮评估一次测试集准确率，默认 `1`，最终轮始终评估。该参数也决定 `--target-error` 的检查频率；较大值可减少测试集前向开销，但早停可能推迟到下一个评估轮。
- `--no-early-stop`：完全禁用 `--target-error` 早停条件，使所有方法跑满 `--rounds`；误差阈值仍会写入实验配置和结果。

#### 攻击、检测、降权与追踪

- `--attack none|sign_flip|gaussian|alternating_minimization`：设置投毒方式，默认 `alternating_minimization`。兼容名称 `alternating` 会在解析后规范为 `alternating_minimization`。该攻击发生在恶意客户端本地优化过程中，不能通过训练完成后修改更新向量来替代。
- `--attack-scale 5.0`：仅用于训练后攻击。对 `sign_flip` 表示更新翻转倍数，对 `gaussian` 表示噪声标准差倍数；交替最小化命令若显式传入该参数会直接报错，防止把增大向量扰动强度误当成 Bhagoji 攻击，应改用 `--attack-boost`。
- `--attack-boost 10.0`：交替最小化攻击的显式提升因子 $\lambda$，默认 `10.0`。每个目标攻击步产生的参数位移会乘以 $\lambda$，正常任务和距离隐蔽步骤不提升。
- `--attack-epochs 10`：恶意客户端每轮用于交替优化的正常任务/隐蔽训练 epoch 数，默认 `10`，对应 Bhagoji 官方实现中的 `mal_E`。这与普通客户端的 `--local-epochs` 相互独立。
- `--attack-stealth-steps 10`：每个目标攻击步之间执行的正常任务/距离隐蔽优化步数，默认 `10`，对应 Bhagoji 官方实现中的 `ls`。
- `--attack-distance-weight 1e-4`：隐蔽目标中距离约束的权重 $\rho$，默认 $10^{-4}$。项目先从当前全局模型独立执行一次正常本地训练得到 $w_{\mathrm{ben}}$，再优化
  $L_{\mathrm{stealth}}(w)=L(D_m;w)+\frac{\rho}{2}\lVert w-w_{\mathrm{ben}}\rVert_2^2$。
- `--attack-source-label 5` / `--attack-target-label 7`：目标攻击的真实类别和错误目标类别，默认沿用 Bhagoji 实验的类别编号 $5\rightarrow7$。原论文使用 Fashion-MNIST；本项目运行 MNIST 时，这两个编号对应数字类别而非原论文中的鞋类类别。二者必须不同且处于数据集类别范围内。
- `--attack-target-count 1`：目标辅助样本数 $r$，默认 `1`。程序用随机种子从测试集中真实标签为 `attack-source-label` 的样本中确定性选取，作为威胁模型中的同分布辅助集 $D_{\mathrm{aux}}$；若限定测试集后没有足够样本，程序会在训练前明确失败。
- `--attack-start-round 0`：设置所有方法共同使用的攻击起始轮。默认 `0` 是特殊值，表示从第 $K+2$ 轮开始，使前 $K$ 轮为无攻击观察期，并保留第 $K+1$ 轮作为首次正常评分；正整数表示绝对通信轮号。
- `--K 3`：论文 4.3.3 节中的滑动窗口容量和初始观察轮数 $K$，默认 `3` 且不得小于 `2`。前 $K$ 次观测仅建立每个 $Tag_{\pi}$ 的纵向 SVD 基线，第 $K+1$ 次观测首次计算 Z-Score。兼容参数名 `--k` 和 `--detector-window`。
- `--z-threshold 3.0`：论文中的 Z-Score 容忍阈值 $\theta$，默认 `3.0`，即采用 $3\sigma$ 准则。
- `--suspicion-penalty-factor 0.5`：SM9RRS 中疑似节点的聚合权重惩罚因子，默认 `0.5`。
- `--suspicion-recovery-factor 2.0`：疑似节点后续恢复正常时的权重恢复倍数，默认 `2.0`。
- `--C_tol 3`：论文中的连续异常阈值 $C_{\mathrm{tol}}$，默认 `3`。同一任务标签连续疑似达到该次数后，SM9RRS 请求门限追踪；只有追踪证书通过后才撤销身份并剔除。该参数不控制 Ding13，兼容参数名 `--c-tol` 和 `--suspicion-remove-after`。

#### 密码、计算后端与并发

- `--crypto-mode sm9|simulated`：选择 GmSSL 国标 SM9 原生密码层或快速同构标量模拟层，默认 `sm9`；真实模式要求先构建 `_native_sm9`。
- `--dkg-threshold 2`：执行门限操作所需的最少 D-KGC 节点数，默认 `2`。
- `--dkg-nodes 3`：D-KGC 节点总数，默认 `3`；门限至少为 $1$，且不得大于节点总数。
- `--compute-backend numpy|auto|torch`：CNN 本地训练/评估后端，默认 `auto`；有 CUDA/MPS 时自动使用 PyTorch GPU，否则回落到 NumPy，`numpy` 和 `torch` 可强制指定实现。
- `--device auto|cpu|cuda|mps`：PyTorch 设备，默认 `auto`；Mac Apple Silicon 可用 `mps`，Windows/Linux NVIDIA 可用 `cuda`，`auto` 优先 CUDA、其次 MPS。
- `--jobs 1|auto|N`：外层实验配置并发数，默认 `1`。`auto` 会按系统物理内存和实验规模估算上限；当前 CUDA/MPS 后端会保守限制为 `1`。该参数只改变当前实验网格中各配置的调度和总墙钟时间，不改变单个配置的实验变量。
- `--sm9-workers 1|auto|N`：单个 SM9RRS 配置每轮内部的摘要、构包、签名和验签线程数，默认 `auto`；自动值不超过 `8`、CPU 逻辑核数和最大客户端数。SVD 检测器状态仍按客户端稳定顺序更新，不改变检测、降权或撤销规则。

#### 断点、进度与可视化

- `--no-resume`：忽略输出目录中的已完成配置和逐轮检查点，从零开始本次实验；默认会安全断点续跑。
- `--no-progress`：关闭终端进度条和 ETA 输出；训练、结果文件和可视化生成不受影响。
- `--no-visualizations`：只输出 CSV/JSON，不生成 HTML/SVG 图表。
- `--visualize-only`：不重新训练，只读取输出目录中的 `summary.csv` 和 `rounds.csv` 重新生成图表。

### 签名与验签开销入口

以下参数适用于：

```bash
python -m sm9rrsfl.benchmarks.crypto_overhead
```

- `--client-counts 20 50 100`：设置要分别测试的任务环规模，默认依次测试 `20`、`50`、`100` 个客户端。
- `--iterations 20`：设置每种环规模正式记录的签名/验签样本数，默认 `20`；它不是联邦学习训练轮数，也不表示每个客户端都执行 `20` 次。
- `--warmup 3`：正式计时前执行的预热次数，默认 `3`；预热样本不进入最终统计。
- `--update-size 4096`：构造的模型更新向量长度，默认 `4096`。
- `--crypto-mode sm9|simulated`：选择真实 SM9 群运算或快速模拟层，默认 `sm9`。
- `--dkg-threshold 2` / `--dkg-nodes 3`：设置 D-KGC 门限与节点总数，默认采用 `2-of-3` 配置。
- `--task-id crypto-overhead`：设置任务标识，默认 `crypto-overhead`；任务环、非公开 $h_t$ 和任务标签均与其绑定。
- `--no-task-precompute`：关闭默认的任务材料显式预计算，改为首次构包时惰性生成；该首次生成通常发生在预热路径，不计入正式签名/验签样本。
- `--output-dir outputs/crypto_overhead`：设置结果目录，默认 `outputs/crypto_overhead/`。程序不会自动按 `sm9` 和 `simulated` 分目录，连续运行两种模式时应显式指定不同目录以免覆盖。
- `--no-visualizations`：只输出 `summary.csv`、`samples.csv` 和 `summary.json`，不生成 HTML/SVG 图表。
- `--seed 42`：设置构造基准更新和密码上下文所用的随机种子，默认 `42`。

## 输出文件

默认输出目录按数据集和密码模式区分：MNIST 的 `--crypto-mode sm9` 写入 `outputs/mnist/`，CIFAR-10 的 `--crypto-mode sm9` 写入 `outputs/cifar10/`；模拟模式分别写入 `outputs/mnist_simulated/` 和 `outputs/cifar10_simulated/`；CIFAR-10 干净基线写入 `outputs/cifar10_clean_baseline/`。每个输出目录都会包含：

- `summary.csv`：每个方法和恶意比例的一行摘要，同时包含最终目标攻击成功率 `final_attack_target_success_rate`、平均目标类别置信度 `final_attack_target_confidence`，以及训练、攻击、摘要、封包、签名、验签、检测、聚合和评估的分阶段耗时。使用多个 `sm9-workers` 时，密码字段是各客户端操作耗时之和，用于判断热点；`runtime_seconds` 才是配置的真实墙钟时间。
- `rounds.csv`：逐轮准确率、误差、目标攻击成功率/置信度、接收/拒绝更新数、黑名单数量、TP/FP 等。交替最小化的无恶意客户端对照组也会在测试集样本充足时记录同一目标指标；非交替最小化配置的目标指标为空值。
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

Ding 等人在实验部分说明其投毒方法基于参考文献 [8]，并对攻击作交替式修改，但正文没有给出独立的攻击目标函数、伪代码或完整超参数。因此，本项目不声称恢复了未公开的 Ding 攻击源码；攻击端按其引用的 Bhagoji 等人交替最小化目标结构实现，并采用官方代码 `distance-constrained/self-reference` 配置中的距离锚点、交替比例与关键默认系数，作为 Ding 检测实验所针对的交替投毒攻击基础。模型、数据集和优化器仍沿用本项目实验配置，因此这不是原仓库运行环境的逐比特复现：

Bhagoji A N, Chakraborty S, Mittal P, et al. Analyzing Federated Learning through an Adversarial Lens. ICML, 2019: 634-643. 论文及官方代码：[PMLR](https://proceedings.mlr.press/v97/bhagoji19a.html)、[ModelPoisoning](https://github.com/inspire-group/ModelPoisoning)。

Bhagoji 论文第 3.4 节按“目标步骤后接隐蔽步骤”描述单个 epoch，而官方仓库 `alternate_train` 的可执行循环按 `ls` 个正常/距离隐蔽步骤后接一个提升后的目标步骤。本项目明确采用官方可执行实现的顺序；论文实验部分应据此写为“Bhagoji 官方代码的 `distance-constrained/self-reference` 交替最小化变体”，避免声称同时逐行复现两种不同顺序。

对每个交替块，恶意客户端先执行若干正常任务/距离隐蔽步骤，再执行一个目标误分类步骤，并只提升该目标步骤：

$$
L_{\mathrm{stealth}}(w)
=L(D_m;w)+\frac{\rho}{2}\lVert w-w_{\mathrm{ben}}\rVert_2^2,
\qquad
w\leftarrow w-\lambda\eta\nabla_w L(D_{\mathrm{aux}}^{\tau};w).
$$

默认参数为 $\lambda=10$、$\rho=10^{-4}$、`attack_epochs=10`、`attack_stealth_steps=10` 和单个 $5\rightarrow7$ 辅助目标。交替攻击由 `sm9rrsfl/model.py` 与 `sm9rrsfl/torch_backend.py` 在本地训练阶段执行；`sm9rrsfl/attacks.py` 会主动拒绝把该攻击当作训练后的更新向量变换。
