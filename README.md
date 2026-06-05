# SM9-RRS-FL 实验复现

本项目根据 `方案（计算机学报排版）.docx` 中的 SM9-RRS-FL 方案，提供一套可执行的联邦学习投毒防御实验框架，用于后续论文实验和对比研究。

## 已实现内容

- MNIST IDX gzip 数据加载与可选下载。
- 基于 NumPy 的联邦 Softmax 回归训练，支持单个客户端数量或 `20/50/100` 等多客户端数量对比、本地训练轮次、批大小、学习率、IID 独立同分布/Dirichlet 非 IID 划分和误差阈值早停。
- 默认实验比例：`0%`、`10%`、`20%`、`40%`、`45%`、`60%`、`80%`，其中 `0%` 用于无恶意节点收敛对比。
- SM9-RRS-FL 流程：基于 SM9 群运算的可追踪环签名、论文同款动态累加器、SM9 加密撤销陷门、可链接标签、纵向 SVD 投毒检测、审计撤销、黑名单剔除。
- 疑似恶意节点处理采用文献 [13] 同款动态降权：单次异常先降低聚合权重，后续正常则恢复权重，连续异常达到阈值后再触发撤销/剔除。
- Krum 对照实验，使用相同数据划分、恶意比例和攻击方式。
- 文献 [13] 对照实验：复现其“奇异值轨迹差分 + Isolation Forest + 动态权重惩罚/恢复 + 连续异常剔除”的在线投毒检测流程。
- 输出 `summary.csv`、`rounds.csv`、`summary.json`，并自动生成 HTML/SVG 可视化图表，便于后续绘图和论文表格整理。

项目中已有的 `gmssl.sm9` 提供真实 SM9 签名、验签、加密和解密能力。当前默认启用 `--accumulator-mode dynamic`：`sm9rrsfl/accumulator.py` 按《基于国密算法 SM9 的可追踪环签名方案》实现双线性动态累加器，使用 `V=[∏(v_i+s)]P1` 和签名者见证 `W_i=[∏_{j≠i}(v_j+s)]P1` 表示公共环；`sm9rrsfl/crypto.py` 输出常数大小的环签名 `σ=(h,R,S,T)`，数据包中不再携带完整环成员列表，因此签名数据大小不会随环成员数量线性增长。为保持本项目原有审计流程，代码仍保留 SM9 加密撤销陷门，确认恶意后可直接追溯客户端身份。

## 环境说明

建议使用 Python 3.9 及以上版本。首次下载项目后，可按以下方式创建虚拟环境并安装依赖：

```bash
git clone git@github.com:derpt2023/SM9RRSFL.git
cd SM9RRSFL

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

本实现没有依赖 PyTorch、torchvision 或 scikit-learn；文献 [13] 的 Isolation Forest 已用纯 NumPy 实现。项目内已经包含实验所需的 `gmssl` SM2/SM3/SM4/SM9 代码。

安装完成后建议先运行测试：

```bash
python -m unittest discover -s tests
```

## 快速自检

不下载 MNIST，使用合成 MNIST 形状数据做 smoke run：

```bash
python -m sm9rrsfl.experiments \
  --dataset synthetic \
  --crypto-mode simulated \
  --num-clients 8 \
  --rounds 7 \
  --train-samples 800 \
  --test-samples 200 \
  --methods sm9rrs krum ding13
```

真实 SM9 模式也可以运行；大规模重复实验建议先使用 `--crypto-mode simulated`，确认参数后再切换到 `--crypto-mode sm9`。

## MNIST 主实验

运行本方案、Krum、文献 [13] 三组对照：

```bash
python -m sm9rrsfl.experiments \
  --download \
  --data-dir data/mnist \
  --methods sm9rrs krum ding13 \
  --ratios 0.00 0.10 0.20 0.40 0.45 0.60 0.80 \
  --partitions iid dirichlet \
  --dirichlet-alpha 0.5 \
  --client-counts 20 50 100 \
  --rounds 30 \
  --target-error 0.12 \
  --crypto-mode sm9
```

如果希望先快速复现模型收敛趋势和可视化，可以把上面的 `--crypto-mode sm9` 改成 `--crypto-mode simulated`。真实 SM9 模式会显著更慢，适合用于最终密码开销实验；模拟模式适合先检查准确率、抗投毒趋势和图表生成。

运行完成后，脚本会自动生成以下可视化：

- 无恶意节点时，三个方案随训练轮次增加的模型收敛/准确率对比。
- 不同恶意节点比例下，三个方案随训练轮次增加的模型收敛/准确率对比。
- 三个方案在不同恶意节点比例下的运行时间开销对比。
- 三个方案在不同恶意节点比例下的进程峰值 RSS 内存开销对比。
- 在客户端数量为 `20`、`50`、`100` 时，三个方案的最终准确率、运行时间和峰值内存横向对比。

当命令中包含 `--partitions iid dirichlet` 时，脚本会分别为 IID 独立同分布和 Dirichlet 非 IID 两种数据场景生成独立图表。图表文件名前缀分别为 `iid_` 与 `dirichlet_alpha_..._`，总览页 `visualizations.html` 会按场景分块展示。

当命令中包含 `--client-counts 20 50 100` 时，脚本会对每个数据划分场景分别运行 `20`、`50`、`100` 个客户端的实验。原有收敛曲线会按客户端数拆分，例如 `iid_clients_020_accuracy_ratio_045.svg`；同时会额外生成 `iid_client_count_accuracy_ratio_045.svg`、`iid_client_count_runtime_ratio_045.svg`、`iid_client_count_memory_ratio_045.svg` 这类客户端数量横向对比图。若只运行单个数据划分场景，文件名前缀会省略为 `client_count_...`。若不传 `--client-counts`，则保持旧行为，只运行 `--num-clients` 指定的单个客户端数量。

如果某个方法提前达到误差阈值，曲线会在真实停止轮次后用较淡的虚线平台段延伸到最大轮次，表示该方法已经停止聚合、最终准确率保持不变。这样既保留“误差小于阈值时停止聚合”的实验语义，也能让三组方案在同一横轴上比较。

如果你希望三组方案都强制跑满固定轮次，便于观察完整收敛曲线，可以加入：

```bash
--no-early-stop
```

如果已经跑完实验，只想根据现有 CSV 重新生成图表，不想重新训练，可以运行：

```bash
python -m sm9rrsfl.experiments \
  --output-dir outputs/mnist \
  --visualize-only
```

如果要更贴近文献 [13] 的公开实验比例，可以额外运行：

```bash
python -m sm9rrsfl.experiments \
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

- `--methods sm9rrs krum ding13`：选择实验方法。
- `--ratios 0.00 0.10 0.20 0.40 0.45 0.60 0.80`：设置恶意节点比例，建议保留 `0.00` 用于生成无恶意节点基线图。
- `--num-clients 20`：设置单个客户端数量；当没有传 `--client-counts` 时生效。
- `--client-counts 20 50 100`：一次运行多个客户端数量，并生成客户端数量横向对比图；该参数优先于 `--num-clients`。
- `--attack sign_flip|gaussian|alternating|none`：设置投毒方式，默认 `alternating`。
- `--attack-start-round 0`：默认值 `0` 表示在检测器完成基线窗口后开始攻击。
- `--partition iid|dirichlet`：设置数据划分方式。
- `--partitions iid dirichlet`：一次运行多个数据划分场景，并分别生成可视化；该参数优先于 `--partition`。
- `--dirichlet-alpha 0.5`：非 IID 强度，值越小各客户端标签分布越偏。
- `--crypto-mode sm9|simulated`：真实 SM9 或快速模拟密码层。
- `--accumulator-mode dynamic|none`：默认 `dynamic`，使用动态累加器和常数大小环签名；`none` 为旧版实验抽象，会在签名包中携带抽样环成员列表。
- `--ring-size 5`：旧版 `--accumulator-mode none` 下的抽样环大小；动态累加器模式下公共环默认为当前实验的全部客户端，签名包只记录累加器摘要和公共环大小。
- `--no-early-stop`：不使用误差阈值早停，所有方法都跑满 `--rounds`。
- `--suspicion-penalty-factor 0.5`：本方案中疑似节点的权重惩罚因子。
- `--suspicion-recovery-factor 2.0`：疑似节点后续恢复正常时的权重恢复倍数。
- `--suspicion-remove-after 3`：连续疑似达到该次数后再撤销身份并剔除。
- `--no-visualizations`：只输出 CSV/JSON，不生成 HTML/SVG 图表。
- `--visualize-only`：不重新训练，只读取输出目录中的 `summary.csv` 和 `rounds.csv` 重新生成图表。

## 输出文件

默认输出目录为 `outputs/mnist/`：

- `summary.csv`：每个方法和恶意比例的一行摘要。
- `rounds.csv`：逐轮准确率、误差、接收/拒绝更新数、黑名单数量、TP/FP 等。
- `summary.json`：与 `summary.csv` 对应的 JSON 结果。
- `visualizations.html`：自动生成的可视化总览页面。
- `plots/*.svg`：各个图表的 SVG 文件，可直接插入论文或进一步编辑。

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

注意：Krum 在实现上需要满足 `n - f - 2 >= 1`，其中 `n` 为当前参与客户端数，`f` 为恶意客户端数。默认 `20/50/100` 客户端数量下，`80%` 恶意节点仍可运行；如果把客户端数改得很小，`60%` 或 `80%` 可能会使 Krum 无法计算。

## 文献 [13] 复现说明

文献 [13] 为：

Ding Z, Wang W, Li X, et al. Identifying alternately poisoning attacks in federated learning online using trajectory anomaly detection method. Scientific Reports, 2024, 14: 20269. 论文链接：[Nature Scientific Reports](https://www.nature.com/articles/s41598-024-70375-w)。

本文献方法在每轮联邦学习中记录客户端模型参数轨迹，对参数代表矩阵提取奇异值，并用相邻轮次奇异值差分构造轨迹特征；随后使用 Isolation Forest 判断异常客户端，对异常客户端降低聚合权重，对恢复正常的客户端提升权重，连续异常客户端被移除。项目中的实现位于 `sm9rrsfl/ding13_detector.py`。
