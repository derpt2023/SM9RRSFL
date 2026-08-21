# SM9-RRS-FL 实验复现

本项目根据 2026-08-21 的 `方案第三版（计算机学报排版）.pdf` 中 SM9-RRS-FL 方案，提供一套可执行的联邦学习投毒防御实验框架，用于论文实验和对比研究。

## 已实现内容

- MNIST IDX gzip 与 CIFAR-10 python batches 数据加载与可选下载。
- 默认基于 NumPy 的卷积神经网络（CNN）联邦训练，并提供可选 PyTorch 后端用于 GPU 加速：MNIST 使用轻量 compact CNN，CIFAR-10 使用参考 FedAvg/CIFAR-10 经验配置的 `Conv-Conv-FC-FC-Logits` CNN 与按通道标准化输入；支持单个客户端数量或 `20/50/100` 等多客户端数量对比、本地训练轮次、批大小、学习率、学习率衰减、IID 独立同分布/Dirichlet 非 IID 划分和误差阈值早停。
- 默认实验比例：`0%`、`10%`、`20%`、`40%`、`45%`、`60%`、`80%`，其中 `0%` 用于无恶意节点收敛对比。
- SM9-RRS-FL v3 流程：固定任务环、常数大小签名 `σ=(c,A,B,C)`、任务级匿名标签、两个签名验证等式、可信锚定双参考纵向 SVD 检测、D-KGC 门限追踪、门限 Schnorr 证书确认和任务环更新。
- 纵向检测把完整一维模型更新按 `ceil(|G|/num_classes) × num_classes` 重构并补零，提取前 `q` 维对数奇异谱、主左奇异子空间及第 `q+1` 个奇异值形成的相对谱间隙。每个 `Tag_π` 同时维护最近观测与仅含 `R=0` 更新的可信历史；相邻参考捕获单轮突变，可信锚点与累计漂移捕获彼此相似但持续偏离正常轨迹的 `attack→attack` 更新。实现仅保存薄基 `U_q`，通过严格等价的 Gram/主角度公式计算投影矩阵距离，不会构造不可执行的 `m×m` 投影矩阵。
- 四类非负距离采用单侧 MAD 稳健标准化；方向证据按相对谱间隙和当前更新尺度得到的 `χ` 相乘降权。最终复合变量 `R` 同时控制动态降权与异常计数：`R=1` 时权重乘惩罚因子且 `Count=min(C_max,Count+1)`，`R=0` 时恢复权重且 `Count=floor(Count/2)`。达到 `C_tol` 后立即拒绝触发轮更新并请求门限追踪，只有证书验证成功后才永久撤销身份。
- 攻击端实现 Bhagoji 等人的带距离约束交替最小化：恶意客户端在本地训练中交替优化目标误分类损失与正常任务/距离隐蔽损失，并只对目标攻击步进行显式提升；Ding 等人的实验以该攻击为基础。旧版“轮换参数分片并注入随机扰动”的实现已经删除，不再把一般向量噪声称为交替最小化攻击。
- 联邦学习主流程通过 `ClientSigner`、`ASVerifier` 和 `AuditorService` 角色对象分别调用签名、完整验签和门限追踪。客户端对象只保存本客户端私钥和成员见证；AS 对象保存验签所需公开参数、非公开任务点、TPK、审计台账和独立的审计提交认证密钥，但不含任何客户端签名私钥或 D-KGC 追踪份额。AS 候选状态仅按不透明任务标签索引，不保存标签到真实身份的映射。
- Krum 与 FedAvg baseline，对照实验使用相同数据划分、恶意比例和攻击方式；FedAvg/加权聚合按客户端本地样本数加权，Krum 保持原始单更新选择语义。
- TAD（Trajectory Anomaly Detection，文献 [13]）对照实验：复现其“奇异值轨迹差分 + Isolation Forest + 动态权重惩罚/恢复 + 连续异常剔除”的在线投毒检测流程。
- 输出 `summary.csv`、`rounds.csv`、`sm9rrs_diagnostics.csv`、`summary.json`，并自动生成 HTML/SVG 可视化图表，便于后续绘图、检测诊断和论文表格整理。

### 第三版公式实现说明

当前异常判定公式中的 `∨` 是逻辑“或”：当 `S_adj>theta_adj`、`S_anc>theta_anc` 或 `Q>h` 任意一个条件成立时，均令 `R=1`；只有三个条件全部不成立时才令 `R=0`。代码默认使用 `detector_decision_rule="any"`，直接执行这一公式，因此可信锚点或累计漂移可以在相邻分数已经下降的连续 `attack→attack` 场景中独立触发检测。

代码不提供三条件同时成立的 AND 变体；配置若传入 `all` 会在训练开始前直接报错。复合分数中的 `chi` 按正文所述作为方向证据可靠性权重，代码实现为 `max{Z_lambda, chi*Z_U}`；若把 `chi` 独立放入 `max`，它将无法降低不可靠的方向证据。

当前密码协议实现版本为 v2；本文所称“第三版”是论文方案与纵向检测算法的修订版本，二者不是同一个版本号。任务公共环由 `RID`、`ACC` 和成员见证 `W_π` 表示；客户端生成任务级标签 `Tag_π` 与承诺 `R_tag`，AS 通过两个验证等式同时检查环签名及标签归属。同一标签的异常证据计数达到阈值后，AS 先用独立控制面 Schnorr 票据认证其提交的精确证据；每个参与 D-KGC 逻辑端点验证该票据后，再对 `TaskID、RID、H5(E_π)` 和一次性会话标识生成独立批准。只有至少 `t` 份有效批准才能启动门限追踪并生成 `τ_trace`，节点名称或整数编号本身不构成授权。该票据用于实现论文所假设的 AS→Auditor/D-KGC 认证信道，不进入 `E_π`、`H5(E_π)` 或环签名公式。AS 仅在验证追踪证据与门限 Schnorr 证书后更新黑名单和任务环。协议中不存在加密身份陷门，也不存在独立的非交互式零知识证明对象；Fiat-Shamir 哈希挑战 `c` 是环签名本身的一部分。

AS 创建追踪证据时会把完整不可变证据、摘要和控制面授权票据自动登记到内部 pending 台账；仅调用验证函数不会清除该状态。只有匹配证据的门限追踪结果验证通过并经 `archive_trace_result` 显式归档后，对应记录才会关闭；完整证据与认证结果会继续保存在检查点中，归档存储仍须实施访问控制。追踪临时失败时，`C_tol` 触发轮更新仍按 Word 方案保持拒绝状态，检查点保存原始证据供重启后优先重试。`finalize_task` 不再接受调用者自报的布尔值，发现任何待处理审计就拒绝销毁。全部审计关闭后，系统才清零当前进程中的 `κ_t`，删除 `h_t`、任务标签缓存和环历史，并写入无秘密 tombstone 防止任务重新激活；若撤销后环为空，则直接进入该终态，不复用旧环。此前已经复制到外部存储的旧检查点仍须由其存储所有者按保留策略删除。

实现边界需要明确区分：`crypto_mode="sm9"` 的群、配对、SM3 `H_v` 和规范序列化由 GmSSL 国标 SM9 原生桥执行；Shamir/Feldman 份额关系、份额域交叉项乘法、随机盲化求逆、门限部分结果组合、逐节点追踪批准和门限 Schnorr 验证均在代码中执行。`ξ`、`P_r`、`P_r^(-1)`、`Δ_j`、`msk` 和 `β_j` 不在环建立或追踪路径中重构；求逆只开放论文允许公开的随机掩码乘积。角色对象之间采用单向下发的最小状态，AS/客户端对象图不含 D-KGC、追踪网关或其他客户端私钥。论文所述 Paillier 密文交互、独立进程和多节点认证信道仍由单进程中的份额域协议模型代替，未部署为真实跨主机网络；同一 Python 进程内的调试器、`gc` 遍历或内存读取也不属于安全边界。检查点由可信实验编排层统一保存各节点份额，因此本实现可验证协议代数、正常角色调用链和实验开销，但不能等同于具备进程/主机级信任隔离的生产 D-KGC。

## 目录结构

当前代码按“核心方案、训练实验、密码开销实验、文档与测试”拆分：

- `sm9rrsfl/crypto.py`、`sm9rrsfl/accumulator.py`、`sm9rrsfl/threshold.py`：v2 数据包、角色能力、签名/验签、任务环累加器、D-KGC 份额域协议、门限追踪及门限 Schnorr 证书。
- `sm9rrsfl/sm9_backend.py`、`sm9rrsfl/_native_sm9.c`：定长字节接口和基于 GmSSL 的国标 SM9 `G1/G2/GT`、配对及 `H_v` 原生桥。
- `sm9rrsfl/_native_sm3.c`：大更新 SM3 摘要的可选原生加速器。`_native_rrs.c` 是已停用的 v1 源码，不参与 v2 构建或运行。
- `sm9rrsfl/fl.py`、`sm9rrsfl/experiments.py`：联邦训练主流程和 MNIST/CIFAR-10 对比实验入口。
- `sm9rrsfl/vert.py`、`sm9rrsfl/fedredefense.py`：VERT 纵向历史梯度预测基线和 FedREDefense 更新重构误差基线。
- `sm9rrsfl/benchmarks/`：独立微基准实验入口，不混入 CNN 训练时间；目前包含签名与验签开销测试。
- `gmssl/`：历史 Python 兼容代码；v2 的真实 SM9 群运算不使用其中的旧双线性曲线，而由 `_native_sm9` 调用用户提供的 GmSSL C 源码。
- `tests/`：单元测试与流程自检。
- `docs/`：论文补充说明和机制设计材料。
- `configs/experiment.json`：可直接编辑的 JSON 实验参数配置。
- `configs/fair_tuning.example.json`：六种方案统一公平调参与独立最终评估的示例配置。
- `run_experiments_from_config.py`：读取默认 JSON 配置并启动实验的入口文件。
- `run_fair_tuning_from_config.py`：带资源规划、进度条和 GPU/MPS/CPU 并发的公平调参入口。
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

# 真实 SM9 模式：准备项目固定版本的 GmSSL-master.zip，显式指定其实际位置。
# 示例假定压缩包放在当前项目的 Downloads/ 目录；不要把这里的路径省略。
export GMSSL_ARCHIVE="$PWD/Downloads/GmSSL-master.zip"
sha256sum "$GMSSL_ARCHIVE"  # macOS 可改用：shasum -a 256 "$GMSSL_ARCHIVE"
python setup.py build_ext --inplace --force

# 构建后必须确认原生桥已生成且可被 Python 加载。
ls sm9rrsfl/_native_sm9*.so
python -c "from sm9rrsfl.crypto import rrs_backend_name; print(rrs_backend_name())"
```

基础运行不强制依赖 PyTorch、torchvision 或 scikit-learn；TAD 的 Isolation Forest 已用纯 NumPy 实现。`FedREDefense` 的服务端更新重构依赖 PyTorch 高阶自动微分，因此即使客户端训练选择 `--compute-backend numpy`，运行该方法也必须安装 PyTorch；按论文默认迭代数执行时，其运行时间会显著长于其余方法。`simulated` 模式无需 GmSSL 原生扩展；真实 `sm9` 模式需要用户提供的 GmSSL C 源码和成功构建的 `_native_sm9` 扩展。

`GmSSL-master.zip` 必须是项目固定的 GmSSL `3.3.0-dev.1183` 归档，其 SHA-256 为 `6dc97c6b4f7d2f6df9d44f014cca0561a7b4776017efd4486d341e986051fab4`；不要以当前最新版 GmSSL 替代。`setup.py` 优先使用显式的 `GMSSL_SOURCE=/绝对路径/GmSSL-master`，其次读取 `GMSSL_ARCHIVE=/绝对路径/GmSSL-master.zip`；只有两者都未设置时才会尝试 `~/Downloads/GmSSL-master.zip`。其中 `~` 是运行命令用户的主目录（例如 root 用户为 `/root`），**不是项目内的 `Downloads/`**。将归档放在项目内的 `Downloads/` 时，应在已进入项目根目录的前提下使用 `GMSSL_ARCHIVE="$PWD/Downloads/GmSSL-master.zip"`；若放在其他位置，则改为该文件的绝对路径。构建前会校验摘要并执行受限解压；源码存在时 `_native_sm9` 编译失败会直接终止。真实模式启动时应输出 `rrs_backend=gmssl-sm9-native-v2`。若命令输出 `unavailable` 或找不到 `sm9rrsfl/_native_sm9*.so`，说明归档路径错误、归档版本/摘要不符或编译失败；应先修复构建问题，不能将 `--crypto-mode simulated` 的输出当作真实 SM9 实验结果。

GitHub 的源码 ZIP 与 `git clone` 都只分发可移植的 C 源和 Python 源，故不会包含针对某台机器编译的 `_native_sm9*.so`；Linux x86_64、Mac ARM、Python 版本不同的扩展二进制不能互相复制。每个新环境都应使用上面的固定 GmSSL 归档在本机重新构建，然后执行后端检查。

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
  --methods sm9rrs vert fedredefense krum ding13 fedavg \
  --fedre-initial-iterations 2 \
  --fedre-max-iterations 2 \
  --fedre-synthetic-steps 1
```

真实 SM9 模式也可以运行；大规模重复实验建议先使用 `--crypto-mode simulated`，确认参数后再切换到 `--crypto-mode sm9`。

## 使用配置文件启动实验

项目提供了示例配置 `configs/experiment.json`。修改其中 `parameters` 对象后，可以通过以下两种方式启动同一组实验。

方式一：直接执行项目根目录的启动文件：

```bash
./run_experiments_from_config.py
```

启动文件会优先自动使用项目 `.venv` 中可执行且与当前系统兼容的 Python；若该虚拟环境不可执行或来自另一操作系统，脚本会输出警告并自动回退到当前命令中的 `python`。虚拟环境不可跨 macOS/Linux 复制，向 AI Station 覆盖项目时应排除 `.venv/`，并在目标机器上重新创建。也可以使用：

```bash
python run_experiments_from_config.py
```

方式二：在终端中显式指定任意配置文件：

```bash
python -m sm9rrsfl.config_runner --config configs/experiment.json
```

正式启动前，建议先进行只校验、不训练的预检查。它会打印转换后的原始实验命令、解析参数，以及把兼容哨兵解析后的 `effective_detector_parameters`（两个实际阈值、`C_max` 和实际起攻轮）；`jobs=auto`、`sm9_workers=auto` 等依赖实际 CPU 配额、内存和 CUDA 空闲显存的值会在正式启动、数据集加载后写入 `resource_plan=...`：

```bash
python -m sm9rrsfl.config_runner \
  --config configs/experiment.json \
  --dry-run
```

配置文件规则：

- 顶层必须包含 `"schema_version": 1` 和 `"parameters": { ... }`。
- 参数名对应主实验长参数去掉 `--` 后将连字符改成下划线，例如 `--vert-top-k` 写成 `"vert_top_k"`，`--client-counts` 写成 `"client_counts"`。
- 论文符号可直接写成 `"K"`、`"q"`、`"g0"`、`"theta_adj"`、`"theta_anc"`、`"beta"`、`"kappa"`、`"h"`、`"C_tol"` 和 `"C_max"`；它们分别映射到对应的 `detector_*`、`suspicion_remove_after` 与 `suspicion_count_max` 字段。同一份配置不能同时写别名与规范名。
- 旧字段 `z_threshold` 仍可作为 `theta_adj` 与 `theta_anc` 的共同兼容值；它不能与任一显式新阈值同时配置，避免覆盖顺序不透明。
- `methods`、`ratios`、`client_counts`、`partitions` 使用 JSON 数组；普通数值和字符串直接填写。
- 布尔参数可直接填写 `true/false`。配置入口额外支持较直观的 `early_stop`、`visualizations`、`progress` 和 `resume`；例如 `"early_stop": false` 等价于命令行 `--no-early-stop`。`"progress_mode"` 可取 `"auto"`、`"live"` 或 `"log"`；本示例使用 `"live"`，以兼容 PyCharm 等不报告 TTY 的本地控制台。
- 未填写的参数继续使用原命令行默认值和数据集训练预设；未知参数、空数组、非法取值或冲突组合会在加载数据和开始训练前报错。

配置入口最终仍调用 `sm9rrsfl.experiments`，因此命令行模式原有的参数校验、数据集预设、并行执行、断点恢复、可视化和输出格式全部保持一致。每次实际生效的完整参数仍会写入输出目录的 `run_manifest.json`、`summary.csv` 和 `summary.json`。

## 分数据集的统一公平调参和主实验

允许 MNIST 与 CIFAR-10 使用不同的最优防御参数，因为模型维度、更新范数、局部训练轮数与数据异质性不同；但不能在看到最终测试结果后只为某一个方案手工修改参数。项目提供 `sm9rrsfl.fair_tuning`，在一份数据集配置中同时编排六种方案，并强制执行以下协议：

1. 只从训练集按类别分层划出验证集。官方测试集不参加参数选择；选参结束后，最终主实验重新使用完整训练集和官方测试集。
2. 六种方法共享数据集、客户端数量、IID/Non-IID 划分、恶意比例、攻击及其绝对起始轮、轮数、本地 epoch、batch size、学习率、学习率衰减、提前停止规则和评价函数。方法搜索空间只能包含该方法专属的防御参数，不能把 `lr`、攻击强度或训练轮数按方法分别调整。若搜索 `detector_window`，配置必须显式固定共同的 `attack_start_round >= max(K候选)+2`；否则较大 K 会更晚受攻击并获得不公平优势，程序会拒绝该搜索。
3. Ours、VERT 和 FedREDefense 等确有暴露超参数的方法必须具有完全相同的网格候选数 `trials_per_tunable_method`。FedAvg、Krum 和当前 TAD 实现没有额外的可调防御参数，因此各自只有一个空候选，但仍完整参加验证和最终主实验。
4. 所有候选使用同一组 `validation_seeds` 和场景，最终主实验使用与之不重叠的 `final_seeds`。配置强制 `early_stop=false` 和 `eval_interval=1`，避免某个候选少跑轮次，并完整审计非有限更新。
5. 默认 `require_finite_updates=true`：只要候选产生任何 NaN/Inf 更新，该候选即无效，不能因恶意更新被自动丢弃而获得虚假的高分。如果某方法全部候选无效，搜索会直接失败并要求先修正共享训练/攻击配置。
6. 默认 `require_clean_acceptance=true`：每个候选必须在每个 `0%` 恶意验证场景中至少有一次实际聚合。像 FedREDefense 第一轮拒绝并永久屏蔽全部正常客户端的候选会直接判为无效，不能再从一组全部崩溃的参数中强行选出“最优值”。
7. 选参前预先声明统一目标：干净准确率、攻击场景准确率、目标攻击成功率和误撤销率的固定权重。最终论文表格使用独立 `final_seeds` 的均值/标准差，不从最终测试集反向选择候选。
8. 参数选择阶段把“候选 × 验证种子 × 场景”展开为独立配置，复用主实验的资源规划：CUDA/MPS 使用线程队列和设备端 Torch 训练，多个 CUDA 设备自动轮转；NumPy/Torch CPU 使用多进程。进度条按实际配置数显示完成比例、已用时间和 ETA。最终论文计时阶段固定 `jobs=1` 且不跨 GPU 轮转，避免并发竞争或不同 GPU 性能污染时间公平性；单个配置仍可使用 CUDA/MPS 加速。

示例配置为 `configs/fair_tuning.example.json`。先做只校验不训练的检查：

```bash
python run_fair_tuning_from_config.py \
  --config configs/fair_tuning.example.json \
  --dry-run
```

正式运行：

```bash
python run_fair_tuning_from_config.py \
  --config configs/fair_tuning.example.json
```

也可直接使用模块入口：

```bash
python -m sm9rrsfl.fair_tuning \
  --config configs/fair_tuning.example.json
```

调参配置直接复用主实验的计算和进度参数。示例使用以下自动模式：

```json
{
  "compute_backend": "auto",
  "device": "auto",
  "jobs": "auto",
  "resume": true,
  "progress": true,
  "progress_mode": "live"
}
```

- `compute_backend=auto, device=auto`：优先使用 CUDA，其次使用 Apple MPS，否则回落到 NumPy；也可显式指定 `compute_backend=torch, device=cuda|mps`。
- `jobs=auto`：根据 CPU 配额、主存、CUDA 空闲显存、数据集和更新规模决定验证配置并发数；可填写正整数限制并发。
- `progress=true`：启用两段进度条，先显示验证搜索，选参完成后显示独立最终评估。`progress_mode=live` 适合 PyCharm/IDE 控制台，`auto` 只在 TTY 原地刷新，`log` 输出离散状态行。
- `resume=true`：默认启用调参专用断点恢复。轮次状态按 `checkpoint_interval` 原子保存，每完成一个配置就先更新阶段快照，再更新公开 CSV，最后删除该配置的终态检查点。重启完全相同的调参配置时，会跳过已经提交的候选场景，并从未完成配置最近一次耐久化轮继续；自动间隔大于 1 时，末尾尚未落盘的少量轮次会重算。
- Ours 的 `sm9_workers=auto` 会按外层并发数重新分配 CPU 槽，避免网格并发与签名/验签线程相互过度订阅。

搜索阶段的并发墙钟时间不进入选参目标，也不应作为论文方法耗时；论文时间统一来自串行的 `final_evaluation/`。

每个数据集应复制一份配置并分别声明搜索空间，例如 MNIST 与 CIFAR-10 各自一份；不要在同一次搜索中混合两个数据集。网格必须为有限的显式数组，三种可调方法的笛卡尔积大小必须等于相同的 `trials_per_tunable_method`。当前允许的专属参数如下：

- Ours：`detector_window`、`detector_subspace_dim`、`detector_gap_threshold`、`detector_adjacent_threshold`、`detector_anchor_threshold`、`detector_drift_memory`、`detector_drift_allowance`、`detector_drift_threshold`、`suspicion_penalty_factor`、`suspicion_recovery_factor`、`suspicion_remove_after`、`suspicion_count_max`；旧 `z_threshold` 仅作共同阈值兼容字段。`detector_decision_rule` 固定为 `any`，对应第三版公式的逻辑“或”，不能作为验证集调参轴。
- VERT：`vert_history_window`、`vert_projection_dim`、`vert_predict_epochs`、`vert_predict_lr`、`vert_top_k`、`vert_use_ratio_prior`。
- FedREDefense：`fedre_threshold`、各重构迭代数/步数/图像数及 `fedre_*_lr`。
- Krum、TAD、FedAvg：当前没有项目级可调防御参数，搜索空间必须写成 `{}`。

第三版只明确给出 `q=2` 及其他符号的取值范围，没有给出 `g0、theta_adj、theta_anc、beta、kappa、h` 的实验数值。下面是用于启动验证搜索的工程起始配置，不得称为“论文官方参数”或未经验证直接作为最终最优参数：

```json
{
  "K": 10,
  "q": 2,
  "g0": 0.1,
  "theta_adj": 3.0,
  "theta_anc": 3.0,
  "beta": 0.9,
  "kappa": 1.0,
  "h": 5.0,
  "detector_decision_rule": "any",
  "C_tol": 3,
  "C_max": 3,
  "suspicion_penalty_factor": 0.01,
  "suspicion_recovery_factor": 10.0,
  "attack_start_round": 12,
  "checkpoint_interval": 0
}
```

其中 `"any"` 对应公式中的逻辑“或”，是第三版方案的固定语义，而不是待搜索超参数。首轮公平搜索建议固定 `q=2、g0=0.1、beta=0.9、kappa=1.0`，仅使用 `detector_anchor_threshold=[2.5,3.0] × detector_drift_threshold=[4.0,6.0]` 的四个候选；待验证连续攻击召回率和 `0%` 诚实客户端误报率后，再决定是否扩大网格。MNIST 与 CIFAR-10 可分别在各自训练验证集上选择参数，但同一数据集内选中的参数必须固定用于 IID/Non-IID 及所有恶意比例的最终主实验。

示例中的 MNIST FedREDefense 网格不再使用 `teacher_lr=0.1/0.2`。历史 `outputs/` 结果表明，共享客户端 `lr=0.05` 时，`teacher_lr=5.0` 能恢复训练但仍会误屏蔽约 `19%–51%` 的正常客户端；而旧的低尺度候选会开局拒绝全部客户端。因此默认网格以历史可工作点为中心，搜索 `fedre_teacher_lr=[5.0, 6.0]` 与 `fedre_threshold=[0.6, 0.7]`，其中保留官方阈值 `0.6`，并让统一验证目标决定是否需要较温和的 `0.7`。这组数值只适用于该 MNIST、`lr=0.05` 配置；CIFAR-10 必须使用自己的搜索文件，不能直接搬用。

调参目录会输出：

- `tuning_trials.csv`：每个候选的统一验证分数、四项选参指标、干净场景接受率、有效性与非有限更新数。
- `validation_results.csv`：所有候选、种子和场景的原始验证摘要。
- `tuning_progress.json`：可随时读取的阶段进度，分别记录验证/最终复验的指纹、`completed/total/pending` 和状态；不再需要等整个验证阶段结束才知道 `x/总数`。
- `best_parameters.json` 与 `tuning_manifest.json`：共享协议、数据隔离声明、候选预算和各方法入选参数。
- `final_evaluation/summary.csv`、`rounds.csv`：独立最终种子的完整主实验结果；逐轮文件带 `seed`，不会把不同重复实验混在一起。
- `final_evaluation/aggregate.csv`：论文表格应使用的多种子均值/标准差，同时给出端到端时间、扣除密码协议后的时间和密码协议墙钟时间。
- `final_evaluation/seed_<seed>/visualizations.html`：每个最终种子的独立图表。多种子统计以 `aggregate.csv` 为准，避免同名曲线覆盖。

恢复所需的权威状态保存在 `.tuning_state/<phase>/<fingerprint>/`。其中 `.checkpoints/` 保存正在运行配置的逐轮状态，`.completed_results.pickle` 保存已经完成配置的事务式快照；公开的 `validation_results.csv` 和 `final_evaluation/` 都可由该快照重建。修改候选、种子、共享参数、数据规模或有效设备分配后指纹会变化，旧状态不会被错误复用。显式设置 `"resume": false` 可以从零运行，但原状态仍保留，避免不可逆删除。

这一功能不替代攻击有效性预检查：应先确认无防御 FedAvg 在攻击场景中能稳定产生有限更新并达到预期 ASR，再启动防御参数搜索。

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

MNIST 主实验对比本方案、VERT、FedREDefense、Krum、TAD 和 FedAvg；CIFAR-10 统一主实验按下文失效说明排除 FedREDefense，比较其余五种方法。两组实验均保留恶意节点比例、IID/Dirichlet 数据分布、Dirichlet 参数、客户端数量、训练轮次和目标误差阈值。

论文实验配置原则：先用 `0%` 恶意节点的 FedAvg/干净训练确认数据集和 CNN 能正常收敛，然后固定同一套模型结构、`E`、`B`、`lr`、`lr_decay` 和 `rounds`，再比较纳入该数据集主实验的方法在攻击场景下的鲁棒性。各防御算法自身的论文参数单独记录，但不要为某个方法调整共享 CNN 或客户端训练超参；被排除的方法必须另行给出统一协议下的失效证据。

### 数据集模型与训练协议

| 数据集 | CNN 结构 | 参数量 | `rounds` | `E` / `--local-epochs` | `B` / `--batch-size` | `lr` | `lr_decay` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST | `Conv(8, 3x3, stride=2) -> AvgPool(2x2) -> FC(392, 10)` | 4,010 | 30 | 1 | 32 | 0.05 | 1.00 |
| CIFAR-10 | `Conv(64, 5x5) -> AvgPool(2x2) -> Conv(64, 5x5) -> AvgPool(2x2) -> FC(4096, 384) -> FC(384, 192) -> FC(192, 10)` | 1,756,426 | 300 | 5 | 50 | 0.05 | 0.99 |

`lr_decay` 是每轮通信后的学习率乘子；CIFAR-10 第 `t` 轮实际学习率为 `0.05 * 0.99^(t-1)`。MNIST 训练较容易，保持常数学习率即可。CIFAR-10 的 `E=5, B=50, lr_decay=0.99` 参考 FedAvg 在 CIFAR-10 上增加本地计算、使用学习率衰减的经验；`rounds=300` 用作正式鲁棒性对比的默认预算，若要追求更高 clean FedAvg 收敛上限，可单独增加干净基线轮数。

联邦聚合口径：

- FedAvg 使用各客户端本地样本数作为聚合权重；在 Dirichlet 非 IID 划分下，样本更多的客户端对全局更新贡献更大。
- SM9-RRS-FL 严格使用第 4.3.3 节归一化后的动态权重 `w_i`，不再二次乘本地样本数；TAD 对照实现仍保持其原实验聚合口径。
- Krum 保持原始算法语义：每轮选择一个更新，不在 `0%` 恶意节点时自动退化为 FedAvg。
- VERT 前两轮使用 FedAvg 建立纵向历史，此后在每个全局轮次重新初始化共享三层预测器与两个集成系数，按客户端历史依次训练并计算预测/实际更新余弦相似度，最后对入选更新执行等权 FedAvg。默认按论文的无先验策略对当轮相似度执行 `K=2` 的 K-means 并选择高相似度簇，不读取实验配置中的恶意节点比例；`--vert-use-ratio-prior` 可显式复现项目旧版的已知比例自动规则，也可用正整数 `--vert-top-k` 固定保留人数。
- FedREDefense 为每个客户端维护合成图像、软标签和合成学习率，以归一化模型更新重构误差过滤客户端；被判为恶意的客户端沿用官方实现语义，在后续轮次不再参与。
- SM9-RRS-FL 对所有通过密码验证的更新统一执行相同的双参考纵向 SVD 检测；最近观测可包含异常，可信窗口、四类正常距离和锚点只接纳 `R=0` 更新。代码不读取“是否恶意”的实验真值来改变判定，也不按数据集暗改窗口或阈值。

## simulated 模式快速实验

如果只是想先确认 CNN 训练流程、IID/Dirichlet 划分和可视化是否能正常跑通，可以先使用 `--crypto-mode simulated` 做小规模快速实验。MNIST 快速命令把 FedREDefense 的合成优化迭代数临时降到 `2`，只用于检查流程，不能作为论文结果；CIFAR-10 快速命令遵循五方法主实验口径。

MNIST 快速实验：

```bash
python -m sm9rrsfl.experiments \
  --dataset mnist \
  --download \
  --data-dir data/mnist \
  --methods sm9rrs vert fedredefense krum ding13 fedavg \
  --ratios 0.00 0.20 0.40 \
  --partitions iid dirichlet \
  --dirichlet-alpha 0.5 \
  --num-clients 10 \
  --rounds 5 \
  --train-samples 2000 \
  --test-samples 500 \
  --target-error 0.12 \
  --fedre-initial-iterations 2 \
  --fedre-max-iterations 2 \
  --fedre-synthetic-steps 1 \
  --crypto-mode simulated \
  --output-dir outputs/quick_mnist_simulated
```

CIFAR-10 快速实验：

```bash
python -m sm9rrsfl.experiments \
  --dataset cifar10 \
  --download \
  --data-dir data/cifar10 \
  --methods sm9rrs vert krum ding13 fedavg \
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

运行本方案、VERT、FedREDefense、Krum、TAD 和 FedAvg 六组对照：

```bash
python -m sm9rrsfl.experiments \
  --dataset mnist \
  --download \
  --data-dir data/mnist \
  --methods sm9rrs vert fedredefense krum ding13 fedavg \
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
  --attack-start-round 12 \
  --K 10 \
  --detector-subspace-dim 2 \
  --detector-gap-threshold 0.1 \
  --detector-adjacent-threshold 3.0 \
  --detector-anchor-threshold 3.0 \
  --detector-drift-memory 0.9 \
  --detector-drift-allowance 1.0 \
  --detector-drift-threshold 5.0 \
  --detector-decision-rule any \
  --C_tol 3 \
  --C_max 3 \
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

CIFAR-10 按统一协议比较本方案、VERT、Krum、TAD 和 FedAvg 五种方法，并把 FedREDefense 的失效/排除证据单独报告：

```bash
python -m sm9rrsfl.experiments \
  --dataset cifar10 \
  --download \
  --data-dir data/cifar10 \
  --methods sm9rrs vert krum ding13 fedavg \
  --ratios 0.00 0.10 0.20 0.40 0.45 0.60 0.80 \
  --partitions iid dirichlet \
  --dirichlet-alpha 0.5 \
  --client-counts 20 50 100 \
  --rounds 300 \
  --local-epochs 5 \
  --batch-size 50 \
  --lr 0.05 \
  --lr-decay 0.99 \
  --attack-start-round 12 \
  --K 10 \
  --detector-subspace-dim 2 \
  --detector-gap-threshold 0.1 \
  --detector-adjacent-threshold 3.0 \
  --detector-anchor-threshold 3.0 \
  --detector-drift-memory 0.9 \
  --detector-drift-allowance 1.0 \
  --detector-drift-threshold 5.0 \
  --detector-decision-rule any \
  --C_tol 3 \
  --C_max 3 \
  --target-error 0.12 \
  --crypto-mode sm9 \
  --compute-backend auto \
  --device auto \
  --jobs auto \
  --eval-interval 5 \
  --checkpoint-interval 0 \
  --sm9-workers auto
```

真实 SM9 CIFAR-10 主实验默认写入 `outputs/cifar10/`；模拟模式默认写入 `outputs/cifar10_simulated/`。

`--jobs auto` 会根据进程实际可用的 CPU 核（含 Linux 容器的 CPU 配额/亲和性）、主存、模型参数量、待运行配置数，以及 CUDA 当前可用显存估算配置级并发数。例如可用 `4` 核的环境有 60 个配置且内存充足时，自动计划可同时推进 4 个配置；完成一个后会继续领取剩余配置。`--jobs 1` 可显式恢复完全串行；`--jobs N` 则由用户明确指定最多 `N` 个并发配置。

NumPy 及 CPU Torch 后端优先使用多进程，使独立配置能够占用多个 CPU 核；如果当前运行环境禁止系统 semaphore 或进程池初始化，会提示 `process_pool_unavailable=...` 并回退到线程池。CUDA/MPS 后端在同一进程中使用线程队列共享设备和驻留数据集，避免 CUDA fork/MPS 子进程初始化问题。CUDA 的 `auto` 会先以显存安全上限约束并发数，再以 CPU/主存预算约束；当进程可见多张 CUDA 卡且 `device=auto` 时，配置会轮转分配到各卡。单卡上的多个小型 CNN 配置可以交叠 CPU 准备与设备计算，但并发不是无限增益；运行日志中的 `resource_plan=...` 会给出实际 CPU 槽、设备、并发数和 SM9 线程数，出现显存不足时应显式降低 `--jobs`。

`--eval-interval 5` 表示每 5 轮评估一次测试集准确率，最后一轮始终评估。它不改变训练、投毒、防御、聚合和最终评估公式，但启用早停时判断粒度会变粗，可能多运行到下一个评估点；如果论文图表需要完整逐轮曲线，可改回 `--eval-interval 1`。

`--sm9-workers auto`（默认）会用最多 8 个线程调度每轮中相互独立的封包、签名和验签，但会按 `jobs` 将实际 CPU 预算分给每个并发配置：例如 4 个 CPU 槽、`jobs=4` 时每个配置使用 1 个 SM9 线程，避免 `4 × 8` 的超额竞争。GmSSL v2 原生桥在点乘、配对和目标群运算期间释放 GIL；实际加速比仍取决于 CPU 核数和配对开销。SVD 轨迹检测按任务标签的稳定顺序更新状态，因此并行不会改变追踪、降权和异常判定语义。

主实验运行时会显示配置级进度条和预计剩余时间，例如 `42/168 25.0% elapsed=... eta=...`。`progress_mode=auto` 时，交互式终端中的独立刷新线程每秒在同一行重绘一次；`progress_mode=live` 可强制 IDE 控制台也使用同样的单行刷新；`log` 则只在启动、配置完成和结束时写入状态行，适合重定向日志。配置文件启动器与直接输入参数共用同一个标准输出流。第一个配置完成前仍显示 `eta=estimating`，之后根据本次运行已完成配置的吞吐量建立并持续倒计时。串行模式显示当前配置，并发模式显示待运行配置数和 worker 数。

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

启用 PyTorch 后端时，数据集、客户端索引、全局参数、客户端更新和聚合结果会持续驻留所选设备。FedAvg/加权 FedAvg、Krum 距离、投毒变换和 Ding13 SVD 特征均直接在设备端执行；真实 SM9 只为协议要求的更新 SM3 摘要回传一次 CPU 字节。Ours 只求前 `q+1` 个真实奇异值及前 `q` 维薄左子空间，CUDA 使用缩放后的列 Gram 小矩阵分解，MPS 不支持时显式回退 NumPy/CPU；可信历史以可移植的 float32 薄基保存在主存。`jobs=auto` 会按 `max(num_clients) × (K+1) × ceil(d/c) × q × 4 bytes` 估计检测器薄基状态并计入并发预算。任何包含 `NaN/Inf` 的客户端更新都会在摘要、检测和聚合前拒绝，避免污染全局模型，同时累计到输出中的 `nonfinite_updates`，不得把这类运行当作防御有效结果。上述优化不改变客户端划分、攻击启动轮次、聚合公式、检测阈值或撤销规则。

运行默认启用断点续跑：第 1 轮前和每个有效检查点间隔结束后会原子写入 `输出目录/.checkpoints/`，每完成一个配置还会先更新事务式结果快照，再派生 CSV/JSON。训练、密码运算、检测、聚合、断电或手动终止发生在区间中间时，只恢复最近一次完整耐久化状态，不会恢复“完成一半”的聚合；自动间隔为 `N` 时最多重算 `N-1` 轮。同一 checkpoint schema、相同数据规模和参数下重启会继续并跳过已完成配置；第三版检测状态使用 schema 11，旧 schema 不会被加载，必须换新输出目录从第 0 轮重跑 Ours。以后再次执行同 schema 的旧命令时，程序可按指纹找回对应 `.stale/` 结果。确实需要从零重跑时加 `--no-resume`。

交互式终端发现“checkpoint schema、完整配置字段和运行指纹都相同”的断点时，会询问 `是否沿着断点运行（Y/N）`：输入 `Y` 从下一轮继续；输入 `N` 从第 0 轮开始，并将原断点备份到 `.checkpoints/discarded/`。CI、重定向输入等非交互式环境不会等待键盘输入，默认选择 `Y`。运行指纹是将数据集元信息和整组有效实验配置规范化为 JSON 后计算的 SHA-256 摘要，用来防止不同实验误用同一结果；它不包含代码版本，因此只有 schema 未变化的兼容修复才可能沿用同参数断点，检测状态结构变化必须提升 schema。

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

如果要更贴近 TAD（文献 [13]）的公开实验比例，可以额外运行：

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
- `--methods sm9rrs vert fedredefense krum ding13 fedavg`：选择一个或多个实验方法，默认运行全部六种方法。结果文件保留稳定的内部标识；可视化图例将 `sm9rrs` 显示为 `Ours`，将 `ding13` 显示为 `TAD`。
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
- `--K 3`：可信历史窗口容量和初始观察数 $K$，默认 `3` 且不得小于 `2`。每个任务标签的前 $K$ 次观测建立可信轨迹，第 $K+1$ 次观测首次评分；推荐配置使用 `K=10`，并将所有方法共同的绝对起攻轮固定为至少 `max(K候选)+2`。兼容参数名 `--k` 和 `--detector-window`。
- `--detector-subspace-dim 2`：前导奇异子空间维数 $q$，默认采用第三版给出的 `2`。检测同时需要 $\lambda_{q+1}$ 计算相对谱间隙，因此十分类数据必须满足 `1 <= q <= 9`。
- `--detector-gap-threshold 0.1`：方向可靠度中的谱间隙尺度 $g_0$；`0.1` 是工程起始值，不是论文给出的实验最优值。
- `--detector-adjacent-threshold 3.0` / `--detector-anchor-threshold 3.0`：相邻复合分数 $S_{\mathrm{adj}}$ 与可信锚点复合分数 $S_{\mathrm{anc}}$ 的阈值 $\theta_{\mathrm{adj}}$、$\theta_{\mathrm{anc}}$。四类距离各自使用单侧中位数/MAD 标准化，而不是均值标准差的普通 $3\sigma$ 检验。
- `--detector-drift-memory 0.9` / `--detector-drift-allowance 1.0` / `--detector-drift-threshold 5.0`：累计漂移递推中的 $\beta$、$\kappa$ 与报警阈值 $h$，即 `Q=max(0,beta*Q_prev+S_anc-kappa)`；三者均为待验证的工程起始值。
- `--detector-decision-rule any`：固定实现公式中的逻辑“或”，允许相邻突变、可信锚点偏移或累计漂移任一触发，并可识别相邻变化已经变小的连续 `attack→attack`。该参数省略时同样默认为 `any`；传入其他值会在训练前报错。
- `--z-threshold 3.0`：仅用于旧配置兼容；未显式设置两个新阈值时，同时作为 $\theta_{\mathrm{adj}}$ 和 $\theta_{\mathrm{anc}}$。不得与任一显式新阈值混用。
- `--suspicion-penalty-factor 0.5`：SM9RRS 中疑似节点的聚合权重惩罚因子，默认 `0.5`。
- `--suspicion-recovery-factor 2.0`：疑似节点后续恢复正常时的权重恢复倍数，默认 `2.0`。
- `--C_tol 3` / `--C_max 0`：异常证据阈值与计数上限。`C_max=0` 会解析为 `C_tol`，避免无意义的额外调参轴；显式值必须不小于 `C_tol`。复合判定 `R=1` 时同轮执行降权并令 `Count=min(C_max,Count+1)`；`R=0` 时恢复权重并令 `Count=floor(Count/2)`。达到 `C_tol` 后触发轮更新权重立即置零并请求门限追踪，只有追踪证书通过后才永久撤销。`C_tol` 兼容参数名 `--c-tol`、`--suspicion-remove-after`，`C_max` 兼容 `--c-max`、`--suspicion-count-max`；二者均不控制 TAD。

#### VERT 与 FedREDefense 参数

- `--vert-history-window 10`：VERT 使用的历史投影更新轮数 $H$，默认采用论文设置 `10`。前两轮只建立历史并执行 FedAvg，从第 3 轮开始训练预测器和筛选更新。
- `--vert-projection-dim 128`：MNIST 低维投影输出长度，默认采用论文设置 `128`。MNIST 使用固定随机全连接投影器；若模型对应的稠密投影矩阵超过 `256 MiB`，实现会切换到固定稀疏符号哈希投影，以避免 CIFAR 大模型出现不可执行的内存占用，该工程适配必须在论文复现说明中披露。
- `--vert-predict-epochs 5` / `--vert-predict-lr 0.01`：VERT 每个全局轮次重新初始化共享三层预测器和两个逐元素集成系数，再按客户端历史依次执行 Adam 训练。训练轮数 `5` 与论文一致；程序当前的预测器学习率默认值 `0.01` 跟随 VERT 官方仓库当前 `conf.json`，但论文第 6.1 节报告的预测器 Adam 学习率是 `0.001`，两者并不一致。严格按论文参数复现时应显式传入 `--vert-predict-lr 0.001`。该参数只控制 VERT 服务器端预测器，不是客户端本地训练的 `--lr`；公平主实验可固定所有方法共享的客户端 `--lr 0.05`，同时为 VERT 使用 `--vert-predict-lr 0.001` 或仅在训练集验证划分上公平调节该内部参数。固定线性投影的输出和预测器最后一层均不施加 Softmax，预测器只在前两层线性层后使用 ReLU。
- `--vert-top-k 0`：VERT 每轮进入聚合的客户端选择策略。正整数表示与恶意比例无关的显式 $k$；默认 `0` 对当轮预测余弦相似度执行论文第 VI-C3 节的 K-means（$K=2$），保留中心值较高的簇。相似度完全相同等无法形成两个有效簇时保留全部客户端。该规则不读取真实恶意节点比例。
- `--vert-use-ratio-prior`：显式启用项目旧版的 VERT 恶意比例先验自动接口。每个配置会自动使用自己的恶意比例 $r$ 和当轮活跃客户端数 $n$；$r=0$ 时保留全部客户端，否则取 $k=\max(1,\lceil(1-r)n\rceil-1)$。该开关可直接配合多个 `--ratios` 和 `--client-counts`，无需逐项手算 $k$；不能与正整数 `--vert-top-k` 同时使用。论文主实验使用的是若干显式 $k$ 值，并非这个自动公式；若要逐项精确复现论文表格，应使用正整数 `--vert-top-k`。三种选择模式共用完全相同的投影、历史替换、预测器训练、余弦评分和等权 FedAvg 代码，只在获得相似度分数后分叉。
- `--fedre-threshold 0.6`：FedREDefense 的归一化更新重构误差阈值，默认采用官方实现的 `0.6`；超过阈值的客户端被过滤，并按官方 `clients_flags` 语义在后续轮次保持不可用。
- `--fedre-initial-iterations 800` / `--fedre-max-iterations 2000`：每个客户端第一次和后续轮次优化持久合成数据的最大迭代次数，默认采用官方 Fashion-MNIST 配置。FedREDefense 本身计算开销很大，不应为了缩短论文主实验时间而无说明地降低该参数。
- `--fedre-synthetic-steps 5` / `--fedre-images-per-class 1`：每次重构的可微合成 SGD 步数，以及每类持久合成图像数，默认采用官方配置。
- `--fedre-image-lr 0.5`、`--fedre-label-lr 0.2`、`--fedre-teacher-lr 0.1`、`--fedre-teacher-lr-lr 5e-6`：合成图像、软标签、初始合成训练步长及该步长自身的优化学习率，默认采用官方 Fashion-MNIST 配置。

#### 密码、计算后端与并发

- `--crypto-mode sm9|simulated`：选择 GmSSL 国标 SM9 原生密码层或快速同构标量模拟层，默认 `sm9`；真实模式要求先构建 `_native_sm9`。
- `--dkg-threshold 2`：执行门限操作所需的最少 D-KGC 节点数，默认 `2`。
- `--dkg-nodes 3`：D-KGC 节点总数，默认 `3`；门限至少为 $1$，且不得大于节点总数。
- `--compute-backend numpy|auto|torch`：CNN 本地训练/评估后端，默认 `auto`；有 CUDA/MPS 时自动使用 PyTorch GPU，否则回落到 NumPy，`numpy` 和 `torch` 可强制指定实现。
- `--device auto|cpu|cuda|mps`：PyTorch 设备，默认 `auto`；Mac Apple Silicon 可用 `mps`，Windows/Linux NVIDIA 可用 `cuda`，`auto` 优先 CUDA、其次 MPS。
- `--jobs 1|auto|N`：外层实验配置并发数，默认 `auto`。CPU/NumPy 使用多进程，CUDA/MPS 使用同进程线程队列。自动值会综合进程可用 CPU、主存、每张 CUDA 卡的实时空闲显存和待运行配置数；`device=auto` 时仅在满足单实验显存估计及 `25%` 安全余量的 CUDA 卡之间轮转，并默认每张可用卡最多放置一个并发配置。其他进程已经占满的可见 GPU 不再被错误计为一个可用槽。`resource_plan` 会同时打印 `visible_cuda_devices`、`usable_cuda_devices`、各卡 `cuda_free_mb` 和单配置显存估计。该参数只改变实验网格的调度和总墙钟时间，不改变单个配置的实验变量；手动指定过大的并发数仍可能导致显存不足。
- `--sm9-workers 1|auto|N`：单个 SM9RRS 配置每轮内部的摘要、构包、签名和验签线程数，默认 `auto`；自动值不超过 `8`、每个并发配置分得的 CPU 槽和最大客户端数。SVD 检测器状态仍按客户端稳定顺序更新，不改变检测、降权或撤销规则。

#### 断点、进度与可视化

- `--no-resume`：忽略输出目录中的已完成配置和逐轮检查点，从零开始本次实验；默认会安全断点续跑。长时间 AI Station 实验不建议设置该选项，应在 JSON 中使用 `"resume": true`。自动调度因某张 GPU 被其他进程占用而改用另一 CUDA 编号时，程序会把它视为等价的运行时放置，继续复用已完成配置，而不会仅因 `cuda:2` 改为 `cuda:3` 就重跑整个网格。
- `--checkpoint-interval 0|N`：耐久化轮次检查点间隔。默认 `0` 自动按检测器状态大小选择：不超过 `256 MiB` 每轮保存，`256–1024 MiB` 每 5 轮，`1–2 GiB` 每 10 轮，超过 `2 GiB` 每 25 轮；正整数强制精确间隔。CIFAR-10、`K=10`、`q=2`、100 客户端时可信薄基约 1.4 GiB，自动值为 10，避免每轮同步重写约 1.4 GiB。间隔为 10 时意外中断最多重算 9 个已完成但尚未耐久化的轮次；若必须每轮恢复可显式设为 `1`，但不得把大量写盘时间算作防御算法开销。实际值会写入 `resource_plan` 的 `checkpoint_intervals`。

原子替换写盘时旧检查点与新临时文件会短暂同时存在，因此单个大 Ours 配置至少应预留约“两份当前检测状态 + 256 MiB”的输出盘空间；多个并发配置还需按同时写入数增加。写入前代码会串行化同进程的检查点写操作、复用上次中断留下的固定临时文件，并检查可用空间；不足时保留旧的完整检查点并明确报错，不会生成半份权威状态。
- `--no-progress`：关闭终端进度条和 ETA 输出；训练、结果文件和可视化生成不受影响。
- `--progress-mode auto|live|log`：`auto` 仅在 TTY 中实时覆写；`live` 强制实时覆写，适合 PyCharm 等本地 IDE 控制台；`log` 不创建刷新线程，只保留关键状态行。
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

- `summary.csv`：每个方法和恶意比例的一行摘要，同时包含配置值 `attack_start_round`、实际起攻轮 `effective_attack_start_round`、最终目标攻击成功率 `final_attack_target_success_rate`、平均目标类别置信度 `final_attack_target_confidence`、累计非有限更新数 `nonfinite_updates`，以及训练、攻击、摘要、封包、签名、验签、检测、聚合和评估的分阶段耗时。使用多个 `sm9-workers` 时，`hash_seconds`、`sign_seconds`、`verify_seconds` 等字段是各客户端操作耗时之和，只用于判断热点，不能从墙钟时间直接相减。`runtime_seconds` 是包含完整密码协议、但排除实验编排层耐久化写盘的计算墙钟时间；本次执行段中已完成的周期性写盘另记为 `checkpoint_io_seconds`，不会只因 Ours 的可信历史更大而污染算法时间比较。断点续跑后的该字段不追溯已经结束的旧进程写盘时间，因此只用于 I/O 诊断，不作为跨次累计总量。`crypto_setup_wall_seconds`、`crypto_packet_wall_seconds`、`crypto_audit_wall_seconds` 和 `crypto_finalize_wall_seconds` 是互不重叠的密码协议墙钟区间，其和为 `crypto_wall_seconds`；`runtime_without_crypto_seconds = max(0, runtime_seconds - crypto_wall_seconds)`，用于与不含密码层的防御做公平算法时间对比。
- `rounds.csv`：逐轮准确率、误差、目标攻击成功率/置信度、`attack_active`、接收/拒绝更新数、该轮非有限更新数 `nonfinite_updates`、黑名单数量、TP/FP 和随机种子 `seed` 等。`attack_active` 直接标记本轮是否已经进入有效攻击阶段，避免再把配置中的特殊值 `attack_start_round=0` 误读为第 1 轮起攻。交替最小化的无恶意客户端对照组也会在测试集样本充足时记录同一目标指标；非交替最小化配置的目标指标为空值。
- `sm9rrs_diagnostics.csv`：Ours 的逐客户端逐轮诊断记录，包括四类距离、四类单侧 MAD 分数、`S_adj`、`S_anc`、累计漂移 `Q`、相对谱间隙、方向可靠度 `chi`、三个阈值越界标记、可信历史长度、`attack_active`、异常原因、惩罚/恢复前权重、归一化前权重、实际聚合权重、`Count` 前后值以及追踪、待处理和撤销状态。兼容列 `z_sigma`、`z_direction`、`sigma_delta` 和 `cosine_similarity` 仍保留，但不应替代上述第三版精确字段做机理分析。`client_id` 与 `is_malicious` 仅作为实验真值写入结果，不参与服务器检测或聚合；客户端被撤销后不再提交更新，因此后续轮次不会再产生新的诊断行。
- `summary.json`：与 `summary.csv` 对应的 JSON 结果。
- `run_manifest.json`：本次数据规模与完整配置指纹，用于确认检查点可以安全复用。
- `skipped_configs.json`：因算法数学条件不成立而在训练前跳过的配置及具体原因。
- `.completed_results.pickle`：已完成配置的事务式权威快照；即使断电发生在多个 CSV 替换之间，也能据此恢复并重新生成表格。
- `last_failure.json`：最近一次未预料异常、完整堆栈、出错配置和最后完成轮次；该配置后来成功完成时会标记为 `resolved`。

诊断记录、最近观测、可信历史、四组正常距离历史、累计漂移和动态计数均随逐轮检查点保存和恢复。第三版检测状态使用 checkpoint schema 11，旧版二进制检查点不会被加载，因为旧状态无法无损迁移为可信双参考轨迹；升级后应使用新输出目录从第 0 轮重跑 Ours。旧 `summary.csv`/`rounds.csv` 仍可读取并采用明确的兼容默认值，但不能事后恢复当时未记录的第三版诊断量。
- `.checkpoints/*.pickle`：当前未完成配置的逐轮状态，配置成功完成后自动删除。
- `.checkpoints/discarded/*.pickle`：用户在断点询问中选择 `N` 后保留的旧断点备份，不参与自动续跑。
- `visualizations.html`：自动生成的可视化总览页面。
- `plots/runtime_without_crypto.svg`：扣除 Ours 密码协议墙钟区间后的公平算法时间对比，应作为论文“防御算法时间”主图。
- `plots/runtime_overhead.svg`：包含密钥初始化、签名、验签、追踪等协议成本的端到端时间，应作为系统总体成本补充图。
- `plots/*.svg`：其余图表，可直接插入论文或进一步编辑。

每个耐久化轮和每个配置的结果都通过临时文件及原子替换保存。长时间实验中途退出后，既能保留已经完成的配置，也能从最近一次耐久化轮继续；自动间隔大于 1 时只需重算该区间尚未保存的轮次。

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

时间开销同时报告含密码协议的计算墙钟时间和扣除密码协议后的计算墙钟时间；二者均排除实验编排层的检查点写盘，周期性写盘另记为 `checkpoint_io_seconds`。扣密时间只减去外层测得的非重叠密码墙钟区间，不会错误减去并行客户端操作耗时之和；训练、攻击、SVD/重构检测和聚合均保留。内存开销使用当前 Python 进程的峰值 RSS。由于同一脚本会顺序运行多组配置，RSS 是粗粒度指标；如果需要更严格的内存隔离，可以分别运行单个 `--methods` 和单个 `--ratios` 配置后对比。

旧结果没有保存密钥生成、任务环更新、追踪和销毁的墙钟边界，无法从已有 `summary.csv` 精确反推出 `runtime_without_crypto_seconds`。准确率/ASR 结论不因此前的纯计时代码改变而必须重跑，但若论文要使用新的公平时间列或两张时间图，需要重跑相应的计时实验。本次第三版检测器则改变了 Ours 的算法状态与判定语义，因此 Ours 的新准确率、ASR 和检测结论必须从第 0 轮重跑，不能继承旧二进制检查点。

注意：Krum 在实现上需要满足 `n - f - 2 >= 1`，其中 `n` 为当前参与客户端数，`f` 为恶意客户端数。默认 `20/50/100` 客户端数量下，`80%` 恶意节点仍可运行；如果把客户端数改得很小，`60%` 或 `80%` 可能会使 Krum 无法计算。即使在 `0%` 恶意节点下，Krum 也保持“每轮选择一个更新”的原始语义，不自动替换为 FedAvg。

## VERT 与 FedREDefense 复现说明

VERT 文献与官方实现：

Wang J, Wang R, Zhang F. How to Defend Against Large-Scale Model Poisoning Attacks in Federated Learning: A Vertical Solution. IEEE Transactions on Dependable and Secure Computing, 2026. 论文与代码：[arXiv](https://arxiv.org/abs/2411.10673)、[VERT](https://github.com/mylab426/VERT)。

VERT 的公开参数存在需要披露的论文/仓库差异：论文第 6.1 节将客户端本地 SGD 学习率和服务器端预测器 Adam 学习率都写为 `0.001`；官方仓库当前 `conf.json` 则只有一个 `lr=0.01`，并在 `client.py` 与 `defenses.py` 中同时用于客户端 SGD 和预测器 Adam。本项目将两者拆分为客户端 `lr` 和 VERT 专属 `vert_predict_lr`，避免为了复现预测器参数而迫使所有对比方法采用较小的客户端学习率。主实验若使用共享 `lr=0.05`，应将其描述为统一训练协议下的公平比较，而不是 VERT 官方训练参数的逐项复现；严格官方复现结果应单独报告并注明选择的是论文参数还是仓库参数。

本项目的 VERT 位于 `sm9rrsfl/vert.py`，保留两轮历史建立、冻结的随机线性投影、每个全局轮次重新初始化并按客户端顺序训练的共享三层预测器与集成系数、原始线性投影特征、余弦相似度排序、被排除更新以全局更新替换历史，以及入选更新等权 FedAvg 的语义。默认的无先验模式仅把论文已知 $k$ 的 Top-k 选择替换为论文提出的 `K=2` K-means 高相似度簇选择；显式 `--vert-use-ratio-prior` 和正整数 `--vert-top-k` 与它共用同一评分核心。MNIST 参数向量规模允许直接使用固定随机全连接投影；只有预计稠密投影器超过 `256 MiB` 时才使用稀疏符号哈希投影，该路径是面向大模型内存约束的工程适配。所有被 VERT 排除的客户端仅在当前轮不聚合，其更新历史由本轮全局更新替换，不会像 FedREDefense 一样永久进入黑名单。

当 `--compute-backend torch --device cuda`（或可用的 `auto`）启用时，VERT 的固定投影、三层预测器、集成系数训练和余弦评分会使用 Torch 设备张量；历史记录和检查点仍为可移植的 NumPy 数据。无 CUDA/MPS、未安装 Torch 或显式 `--compute-backend numpy` 时，自动保持原有 NumPy 实现，因此 Mac CPU 环境可正常运行。SM9 签名验签仍为 CPU 原生密码学计算。

FedREDefense 文献与官方实现：

Xie Y, Fang M, Gong N Z. FedREDefense: Defending against Model Poisoning Attacks for Federated Learning using Model Update Reconstruction Error. ICML, 2024: 54460-54474. 论文与代码：[PMLR](https://proceedings.mlr.press/v235/xie24c.html)、[FedREDefense](https://github.com/xyq7/FedREDefense)。

本项目的 FedREDefense 位于 `sm9rrsfl/fedredefense.py`，按照官方 `image_synthesizer.py` 为每个客户端持续维护合成图像、软标签和可训练合成步长，从当前全局参数出发执行可微 SGD，使用“重构参数平方误差 / 实际客户端更新平方范数”作为归一化重构误差，并采用阈值 `0.6`。通过筛选的客户端执行官方等客户端 FedAvg；超过阈值的客户端在后续轮次保持屏蔽。

官方仓库公开的是 Fashion-MNIST、CIFAR-10 和 CINIC-10 脚本，没有单独的 MNIST 参数文件。本项目在 MNIST 上沿用 Fashion-MNIST 的同形状配置，因此论文应写成“基于官方代码迁移到 MNIST”，而不是声称运行了作者发布的 MNIST 脚本。FedREDefense 论文的核心可分性针对“真实本地训练更新与手工构造模型投毒更新”；当前交替最小化攻击本身也通过训练过程生成，所以它可能得到较低重构误差。这属于威胁模型匹配结果，不应通过读取恶意客户端真值或修改阈值来人为强化。

官方 CIFAR-10 脚本的完整协议为：`100` 个客户端、每轮 `10%` 参与、`300` 轮、`local_epochs=1`、`batch_size=32`、客户端 SGD `lr=0.001`、Dirichlet `alpha=0.1`，并使用 Scaling 攻击；FedREDefense 内部参数为 `threshold=0.6`、`synthetic_steps=5`、`images_per_class=1`、`image_lr=0.1`、`label_lr=0.05`、`teacher_lr=0.05`、`teacher_lr_lr=5e-5`、`initial_iterations=500` 和 `max_iterations=1500`。只复制最后一组重构参数，却把共享客户端学习率改为 `0.05`、让全部客户端每轮参与、改用交替最小化攻击或改变数据异质性，属于统一主实验下的 FedREDefense 参数迁移，不应标为官方 CIFAR-10 逐项复现。

在统一共享 `lr=0.05` 的公平主实验中，只能依据训练集验证划分调节 FedREDefense 专属参数，并必须同时检查：干净场景连续多轮的正常客户端接受率、攻击场景的误接收率、NaN/Inf 以及实际重构迭代耗时。首轮接受并不代表参数可用，因为官方 `clients_flags` 会将后续任一轮被拒绝的客户端永久屏蔽；若放宽阈值后恶意更新的重构误差反而低于正常更新，则不存在能解决该候选的静态阈值，应如实报告威胁模型不匹配，不能按客户端真值选阈值。

本项目的 CIFAR-10 统一主实验明确不再纳入 FedREDefense。其原因不是预先认定该算法在所有 CIFAR-10 场景中无效，而是实测表明：在本项目固定的共享客户端 `lr=0.05`、全客户端参与、IID/Dirichlet 划分和交替最小化攻击协议下，官方 CIFAR-10 重构参数会拒绝正常客户端；围绕合成学习率、重构迭代数和阈值进行参数迁移后，仍出现正常/恶意重构误差重叠或顺序反转，以及正常客户端在后续干净轮次被永久屏蔽的问题。放宽阈值只能掩盖首轮拒绝，不能同时保证连续正常接受和恶意拒绝。因此 CIFAR-10 结果应表述为“FedREDefense 在本项目统一实验协议下失效/不适用”，并将其排除原因、官方协议差异和校准失败证据单独报告；MNIST 或 FedREDefense 官方协议下的既有结果不由该结论否定。

## TAD（文献 [13]）复现说明

TAD 指本文复现的 Trajectory Anomaly Detection 方法，对应文献 [13]：

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
