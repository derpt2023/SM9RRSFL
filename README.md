# SM9-RRS-FL 实验复现

六种方法：Ours（内部名 `sm9rrs`）、VERT、AlignIns、Krum、TAD（`ding13`）、FedAvg。支持 MNIST/CIFAR-10、IID/Dirichlet Non-IID、NumPy/PyTorch，以及真实 SM9 或快速仿真密码模式。

## 2026-09-03：正常状态检测器与激进撤销规则

Ours 现在采用 **K-means 学习正常状态，历史偏离识别异常**。它不是“把本轮客户端聚成两个簇，再把小簇判为攻击者”。每个匿名任务标签独立学习一个或多个正常模式，所有簇均作为正常参照，但新更新仍须通过偏离检测；0% 恶意场景也不会被强行分出恶意簇。

撤销规则统一改为：**严重偏离立即发起撤销；一般可疑使嫌疑值加 1，正常轮使嫌疑值折半，达到 C_tol 当轮发起撤销**。不新增新旧策略切换开关。永久撤销仍须验证身份追溯证书，两个触发路径都在本轮聚合前执行。

本次是算法和配置的破坏性更新：配置/统一调参 schema 为 **3**，Ours 工件 schema 为 **4**，训练检查点 schema 为 **15**，Ours 算法版本为 `ours-normal-states-v2-aggressive-revocation`。包括上一版保守正常状态检测器在内，旧参数工件、1170 配置验证结果和轮次检查点不能作为新算法结果续用，必须重新校准；程序按新指纹隔离，现有 outputs 不会被本次代码修改删除。新的默认输出目录使用 `mnist_v6_aggressive_revocation*`。

已移除旧的双参考四距离 Z 检测、谱间隙门控 `g0`、`theta_adj/theta_anc`、`z_threshold`、`C_max`、`detector_decision_rule`、clean-shadow 探测及多轮阈值膨胀。旧字段会报错，不会被悄悄映射到新算法。FedREDefense 和前三攻击轮召回率硬门槛仍不启用。

这会改变论文的检测特征、正常状态建模、历史准入、权重和撤销公式；不能继续描述为第三版 PDF 原检测公式的逐项复现。SVD、K-means 和新颖性检测本身不是本项目的新发明，论文需要分别交代已有技术和组合后的机制贡献。新实现也不承诺在 80% 恶意比例下必然有效。

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

基础运行不强制依赖 PyTorch、torchvision 或 scikit-learn；TAD 的 Isolation Forest 和 AlignIns 均提供纯 NumPy 路径。使用 Torch 训练时，AlignIns 会在原设备上流式计算符号、Top-k、余弦和范数，不会为检测额外复制完整的 `客户端数 × 参数量` CPU 更新矩阵。`simulated` 模式无需 GmSSL 原生扩展；真实 `sm9` 模式需要用户提供的 GmSSL C 源码和成功构建的 `_native_sm9` 扩展。

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

## AI Station 启动与资源配置

先确认容器里的 CUDA 版 PyTorch 可看到所分配显卡：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -u run_fair_tuning_from_config.py --config configs/fair_tuning.example.json --dry-run
python -u run_fair_tuning_from_config.py --config configs/fair_tuning.example.json
```

这是推荐的**方案 B 完整公平实验**。修改 `configs/fair_tuning.example.json` 的公共数据、训练、攻击和资源参数即可；`method_spaces.sm9rrs="auto"` 不需要手填 Ours 检测参数。默认仍是 1170 次验证训练 + 180 次正式训练，不再有前置的 Ours 影子训练。每个配置是一次完整的联邦训练，不是一次测试集推理。

8 核 CPU / 8 张 RTX 4090：

- 保持 `compute_backend="auto", device="auto", jobs="auto", tuning.final_jobs="auto", sm9_workers="auto"`。自动并发受容器 CPU 配额、主机内存、GPU 空闲显存共同限制，资源够时最多同时使用 8 个工作槽；不要为了分配所有卡改成 `device="cuda"`，后者固定默认卡。
- CUDA 任务按设备队列派发，每卡最多一个完整实验。8 槽/8 卡时快卡可继续自己队列中的下一配置，不会因普通线程队列抢占忙卡；断点恢复只剩部分显卡上的任务时，也不会把原来的 8 槽挤到一张卡。jobs 是全局并发上限，设得超过可用卡数也不会突破每卡一个的限制。
- 每个配置中的联邦轮次必须顺序执行；并行的是相互独立的参数、比例、分区和 seed。SVD/K-means 使用小矩阵 CPU 运算，不另训练神经网络；本地 CNN、攻击优化和推理继续使用所分配 GPU。
- 包初始化在 NumPy/Torch 导入前将未设置的 `OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS` 默认置为 1，避免每个 GPU 作业再创建大量 CPU 线程；明确提供的环境变量不会被覆盖。
- `progress_mode="live"` 每秒刷新进度、elapsed、ETA；当前配置未完成时 ETA 只是估计，并不证明卡住。非交互日志可用 `"log"`。主实验和验证都保留每配置/每轮断点；重启使用相同配置、输出目录，保留隐藏的 `.tuning_state` 目录。
- 并发所得 runtime 是并发负载下的时间，不是独占设备基准。论文要比较独占耗时可设 `tuning.final_jobs=1`，并明确披露计时环境。
- 不要同步 Mac 的 `.venv`、`build/` 或 `*.so` 到 AI Station；在 Linux 自己的 Python 环境安装依赖、重新构建原生 SM9。

另一个入口：

```bash
python -u run_experiments_from_config.py --dry-run
python -u run_experiments_from_config.py
```

它读取 `configs/experiment.json`，是 **Ours 单独自动校准 + 已配置基线正式运行**，不是六方法等预算的方案 B。此处的 Ours 同样自动学习正常状态、选择并冻结参数，但 VERT/AlignIns 参数来自配置，不能称为与方案 B 等价的公平调参。默认 3 个独立校准 seed，12 个 Ours 候选和匹配的干净 FedAvg 对照；有进度与断点，不依赖历史输出。

## 数据、训练、选参和正式评估

1. **只划分一次原始训练集。** 方案 B 默认 50000 条训练样本分为约 45000 条联邦训练、2500 条验证、2500 条攻击辅助样本，按类别分层取整。校准和正式实验共享完全相同的训练数组与攻击辅助数组；不会再把 45000 条训练数据二次切成 40499 条。各 seed 仍可产生不同客户端分区，但同一 seed 的所有方法使用同一分区、模型初始化、本地批次和学习率计划。
2. `tuning.validation_fraction` 现在是占**原始训练集**的比例，默认 0.05；攻击辅助比例固定 0.05，训练比例为 `0.95-validation_fraction`。改变它会同时改变所有阶段的共同训练集。固定参数入口也使用一次 90/5/5 训练集划分。
3. 攻击者只从训练来源的攻击辅助集选择目标样本。验证 ASR 用训练留出集；正式 accuracy/ASR 用官方测试集。已删除固定参数旧路径“缺少辅助集就用官方测试集训练攻击”的回退。直接调用 `run_experiment` 时，交替最小化攻击也必须提供 `x_attack/y_attack`。
4. 默认 `ratio_range=[0,0.8,5]` 对应正式比例 0/20/40/60/80%，校准比例 0/10/30/50/70%；正式 seed 141/142/143 与验证 seed 41/42/43 分离。0% 是两边共享的必要干净对照。比例范围、攻击类型、模型和干净前缀属于公开研究条件，分离比例不能消除这些先验，论文应披露。
5. 方案 B 三个可调方法各 12 个候选；其余方法保持单一公开算法配置。39 候选 × 2 分区 × 5 校准比例 × 3 seed = 1170 次完整训练。Ours 每次训练自行学习该客户端的正常中心、尺度和半径；并不提前读取正式客户端的数据。
6. 统一 Score 权重通过留一攻击比例验证选择，四项权重均至少 0.05，步长 0.05。每折的候选筛选只看非留出比例和干净对照；留出比例用于外层检验。权重搜索复用每折候选的充分统计，不重复训练或 969 次重新扫描全部轮次。
7. 选出每种方法唯一的固定超参数集，才运行 6 方法 × 2 分区 × 5 正式比例 × 3 独立 seed = 180 次正式实验。同一数据集不同恶意比例不能分别调一套参数。换 CIFAR-10 后需重新校准，不能套 MNIST 工件。

## Ours 一轮实际如何工作

**前提：前 K 轮全部诚实。** K 是公共实验参数，不是恶意比例。默认示例 K=7、attack_start_round=12；若设 attack_start_round=0，统一解析为 K+2。Ours 拒绝攻击开始早于或等于 K 的配置。新加入、未拥有足够干净历史的标签不能在攻击期间重新领取“无条件热身期”。

1. 客户端按统一规则训练，返回更新；先验证 SM9 标签和签名，再检测。检测器不接收恶意比例、客户端好坏真值或攻击目标类别。
2. 用本轮学习率归一化更新，减少单纯 lr 衰减造成的尺度漂移。完整更新补零重排为 `ceil(P/C) × C` 矩阵，保留范数、前 q 个奇异值和紧凑的**右子空间投影**；这是新特征，不声称与旧左子空间距离等价。
3. 同时提取全模型确定性有符号投影，以及真正分类头各类别的有符号权重均值和 bias 变化。这样更新取反不再因奇异值不变而不可见。所有类别同时检测，不偷看 source/target 标签。投影维数有限、类别摘要也有信息损失，不能宣称消除了全部盲区。
4. 每个标签用前 K 轮特征拟合标准化后的正常 K-means。最多允许指定数量的模式，每个模式至少 3 个样本；样本不足或出现单点簇则退化到一个正常模式。尺度为 `max(1.4826 × MAD, 0.15)`；0.15 是明确的正则化下限，1.4826 是正态一致性换算常数，不是由本次数据学出的常数。半径使用干净簇内最大训练距离，下限 1；小 K 下不声称具有统计误封率保证。
5. 对每个正常模式计算谱、有符号、类别证据的分组距离，取同一模式下最大组距离，再找最接近的正常模式。最终新颖度 D 取“动态正常模型距离”和“冻结正常锚点距离 / reference_budget”的较大值；累积漂移为 `Q=max(0,beta*Q+min(D,reject_threshold)-kappa)`。`D > reject_threshold` 为严重偏离，立即拒绝更新并发起撤销，不等待 C_tol，也不另设有符号/类别证据必须同时越界的条件。这两类证据仍参与 D 的计算。其余 `D > distance_threshold` 或 `Q > drift_threshold` 的情况为一般可疑。
6. **聚合许可不等于历史许可。** 只有低于预警阈值一半、冻结锚点距离不超过 1、漂移足够小、无需范数裁剪，并连续满足 history_confirm 次的更新，才有资格写入动态可信窗口/恢复可靠度；还必须通过整轮安全检查。冻结锚点、原始尺度约束和干净范数上界永不从攻击期更新中重学。
7. 可疑更新使可靠度乘 penalty；仅可信资格确认后才乘 recovery 恢复，最大为 1。嫌疑值 C 初始为 0：有正常参照且本轮被判可疑时 `C=min(C_tol,C+1)`；本轮检测正常时 `C=0.5*C`。C 使用浮点数，不取整、不清零；正常折半无需等待历史/恢复准入确认。缺席、无效更新或缺少干净参照不冒充正常观察，不触发折半。一般可疑达到 `C >= C_tol` 时，当轮发起撤销，本轮更新也不再参与聚合；严重偏离直接绕过此阈值。
8. 新增嫌疑超过当前可见标签一半的“分布突变轮”仍暂停可信历史写入和权重恢复，但**不暂停正常嫌疑值折半，也不延迟任一撤销路径**。删除每轮约 10% 撤销配额和至少保留两名身份的限制。若所有客户端被撤销，记录实际终止轮次并安全关闭任务，不填充虚假后续轮次；校准仍按完成率约束判定该候选是否有效。
9. 两条撤销路径都请求 D-KGC 追踪；仅证书验证成功才能永久撤销、更新任务环。“立即”表示在本轮聚合前完成，不代表跳过追溯认证。证书失败时保留原始证据和零聚合状态，即使不是定期检查点轮次也强制保存，中断后恢复运行优先重试；等待证书期间不能恢复权重或重复累计。

例如 C_tol=3，连续三轮一般可疑时 C 为 `1 → 2 → 3`，第 3 次当轮撤销；若依次为“可疑、正常、可疑、可疑、可疑”，C 为 `1 → 0.5 → 1.5 → 2.5 → 3`，第 5 轮撤销（存储值封顶于 C_tol）。任意一次严重偏离均立即走撤销路径。折半只作用于 C，不改变 Q 的 beta/kappa 更新规则。这里的“正常”要求本轮检测整体正常：即使瞬时 D 较小，若 Q 仍超过漂移阈值，本轮仍是可疑，不能折半。

这是更激进的安全/可用性取舍：检测错误可能造成不可逆误撤销或全部客户端退出，不能声称一定提升准确率。误撤销、诚实权重损失、干净准确率和完成率仍需如实报告，不能删除失败场景来美化结果。

**限制漏检者的绝对聚合影响：** 令 `b_i=n_i/sum_all_clients(n_j)`，`r_i∈[0,1]` 为可靠度，拒绝者 `u_i=0`，其他 `u_i=b_i*r_i`。实际更新系数为

`a_i = min(u_i/sum_j u_j, weight_cap*u_i) * clip_i`。

`clip_i=min(1,clip_factor*clean_norm_limit/current_norm)`；全零 u 时本轮不更新。**裁剪/封顶后不重新归一化**，系数和小于 1 意味着服务器步长缩小。干净、未裁剪时退化为按样本数的 FedAvg，而不是旧 Ours 的等客户端权重。默认 weight_cap=2 时，100 个等样本客户端中 7 个完全漏检者的绝对系数最多 0.14（尚未计范数裁剪），不再因其他节点被撤销自动放大到 0.26；它们在“剩余更新”中的相对比例仍可能很高，这并非检测正确性的替代。

## 自动参数与验证报告

“自动”区分两件事：正常中心、MAD 尺度、半径由各次训练的干净前缀学习；检测阈值、C_tol、penalty/recovery 等超参数从有界候选中用训练留出结果选择，正式实验统一冻结。正常轮嫌疑值乘 0.5 是本方案指定的固定规则，不参与搜索，也不新增手动参数。不是所有安全常数都由数据学习，更不是读取正式测试结果调参。

默认 12 个候选覆盖 q=1/2/3，正常模式上限 1/2，预警阈值 1.5/2/2.5/3，拒绝阈值 3/4/5/6，beta=0.8/0.9/0.95，kappa=0.75/1/1.25，h=4/6/8，可信确认次数 2/3/4，冻结锚点预算 1.5/2/2.5，系数放大上限 1.5/2/2.5，C_tol=2/3/5，penalty=0.02/0.1/0.25，recovery=1.1/1.25/1.5，范数倍率 1.5/2/3。它是预先公布的 12 个联合设计，不是所有组合的笛卡尔积，也不能证明找到全局最优。候选预算允许 1–36；方案 B 同时要求三个方法预算相等。

本次撤销修改不改变选参指标，公共 Score 仍为：

`S = w_c*A_clean + w_r*A_robust + w_a*(1-ASR) + w_h*(1-H_loss)`。

- A_clean：干净场景最终准确率。
- A_robust、ASR：**每个攻击场景从攻击开始到结束的逐轮平均**，再场景等权平均；不再只看最后一轮，避免漏掉前期模型崩溃。
- H_loss：每场景逐轮平均的诚实名义 FedAvg 权重损失，再场景等权平均，包含干净场景。标签只用于事后统计，不进入检测。
- 硬门槛：通信轮完成率（默认 1）、非有限更新数量（默认 0），以及可调方法相对同分区/Dirichlet α/客户端数/seed 的 FedAvg 干净最终准确率下降（默认 ≤0.05）。Krum/TAD/FedAvg 是固定对照，干净结果差仍应报告，不因此删掉该对照。高 ASR 不再导致整项研究中止，但会影响 Score。
- 不设置前三攻击轮召回率硬门槛。初期恶意聚合质量、误撤销、最差准确率、峰值 ASR 都另行报告；这些诊断不是隐含的新硬门槛。
- 缺失 ASR、未完成或 NaN/Inf 不冒充正常结果；没有合格候选时明确停止，不自动放宽阈值。先看已写出的 `candidate_feasibility.csv` 和 `validation_results.csv`。同一新版本验证已完成后可复用其断点，不必重训已有配置。
- `best_parameters.json` 报告 Score 定义、选择余量、各场景结果离散程度、权重选择模式数。若 `weights_identifiable=false`，代表多组权重选中了同样的候选，不能把那组数字解释为唯一学出的“真实权重”。

硬门槛接口：公共 CLI `--min-round-completion`、`--max-nonfinite-updates`（JSON 对应 `calibration_min_round_completion_rate`、`calibration_max_nonfinite_updates`）；方案 B 的干净下降限制为 `tuning.max_clean_accuracy_drop`。删除的 `--ASR`/前三轮召回率等旧接口不可再用。

当前自动 Score 包含定向 ASR，因此自动校准/方案 B 要求 `attack=alternating_minimization`；sign_flip/gaussian 等非定向攻击可在 fixed 模式运行，但不能无定义地套用定向 ASR 选参。启动前会检查攻击辅助、验证和正式评价分区是否有足够的 source-label 样本，避免长时间训练后才发现 ASR 缺失。

### Ours 固定参数/消融接口

正式 auto 模式不填下表；手填会被拒绝，避免“部分自动、部分偷偷固定”。需要消融时设 `ours_parameter_mode="fixed"` 并用下列规范 JSON 键；CLI 把下划线换成连字符。K/窗口独立由 `--K` 设置。

| 参数 | 固定模式默认值 | 含义 |
|---|---:|---|
| detector_subspace_dim | 2 | q，谱/子空间维度 |
| detector_normal_clusters | 2 | 每客户端正常模式数上限 |
| detector_distance_threshold | 2 | 预警距离 |
| detector_reject_threshold | 4 | 严重偏离立即撤销距离，必须大于预警 |
| detector_drift_memory | 0.9 | beta，漂移记忆 |
| detector_drift_allowance | 1 | kappa，正常漂移扣除量 |
| detector_drift_threshold | 6 | h，漂移预警界限 |
| detector_history_confirm | 3 | 历史/恢复准入连续确认次数 |
| detector_reference_budget | 2 | 冻结锚点偏离预算 |
| detector_clip_factor | 2 | 干净范数上限倍率 |
| detector_weight_cap | 2 | 可靠度调整后名义系数放大上限 |
| suspicion_penalty_factor | 0.1 | 嫌疑可靠度乘子 |
| suspicion_recovery_factor | 1.25 | 可信恢复乘子 |
| suspicion_remove_after | 3 | C_tol，正常轮折半的累计嫌疑值撤销阈值 |

JSON 保留有实际含义的别名 K、q、beta、kappa、h、C_tol；CLI 保留 `--C_tol/--c-tol`。新特征不再使用 g0 或旧 theta_adj/theta_anc。所有当前公共、训练、攻击、基线和运行参数可用下面的命令查询，避免在 README 维护另一份容易过期的重复列表：

```bash
python -m sm9rrsfl.experiments --help
python -u run_fair_tuning_from_config.py --help
```

## 快速自检与 CIFAR-10

```bash
python -m unittest discover -s tests
python -u -m sm9rrsfl.experiments --dataset synthetic --crypto-mode simulated \
  --methods sm9rrs fedavg --num-clients 8 --rounds 8 --K 3 --attack sign_flip \
  --ratios 0 0.5 --train-samples 800 --test-samples 200 --no-early-stop \
  --compute-backend numpy --jobs 1 --output-dir outputs/normal_state_smoke
```

切换 CIFAR-10：复制方案 B 配置，修改 dataset、data_dir、output_dir 及公共训练超参数；不要修改各方法的数据/优化器口径。MNIST 使用 compact CNN；CIFAR-10 使用 Conv-Conv-FC-FC-Logits CNN 和按通道标准化。应先检查相同公共参数下的干净 FedAvg 收敛，再开启整个调参网格。K、总轮数、lr、batch 等属于实验条件，自动检测参数不能弥补欠训练的基础模型。

## 输出、同步与代码位置

- 普通实验：`summary.csv/json`、`rounds.csv`、`sm9rrs_diagnostics.csv`、`visualizations.html` 和 SVG 图。
- 方案 B：`validation_results.csv`（每配置/seed 与攻击全过程诊断）、`candidate_feasibility.csv`（学权重前的可行性）、`tuning_trials.csv`、`best_parameters.json`、`tuning_manifest.json`、`tuning_progress.json`；正式结果和 HTML 在各 seed 子目录。验证阶段未完成时没有正式曲线是正常的。
- 新客户端诊断包含 novelty/anchor/signed/class 分数、clip_factor、实际 aggregation_weight、aggregation_accepted、history_eligible、history_admitted、history_frozen、immediate_revocation、浮点 count_before/count_after 与证书状态，且包含 seed。`immediate_revocation=true` 表示严重偏离触发，不表示证书已经通过；永久撤销以 `revoked` 为准。验证报告分别统计 `immediate_revocation_requests` 和 `cumulative_revocation_requests`，不再输出已移除的 revocation_guarded 字段。
- `ours_calibration.json` 在方案 B 中仅声明候选空间（`candidate_space_only`），不假称已冻结；正式选中的唯一参数以 `best_parameters.json` 为准。普通 auto 入口工件状态为 `frozen`。
- `svd_detector.py`：正常状态建模与两阶段历史准入；`ours_policy.py`：唯一参数定义和有界设计；`weighting.py`：可靠度、两级撤销与嫌疑值折半、实际聚合系数；`fl.py`：训练与 SM9 接入；`ours_calibration.py/fair_tuning.py`：校准、冻结和报告；`execution.py`：GPU 分队列；`experiments.py`：资源、进度和检查点。
- 同步代码应包含上述已改源文件、两个 configs、README 和 tests。不要提交数据、历史 outputs、虚拟环境、编译产物或本机缓存。新环境没有任何历史结果也可以从头运行。

## 密码协议与实现边界

密码协议仍为 v2，与本次正常状态检测器的版本号无关。任务公共环由 `RID`、`ACC` 和成员见证 `W_π` 表示；客户端生成任务级标签 `Tag_π` 与承诺 `R_tag`，AS 通过两个验证等式同时检查环签名及标签归属。同一标签的异常证据计数达到阈值后，AS 先用独立控制面 Schnorr 票据认证其提交的精确证据；每个参与 D-KGC 逻辑端点验证该票据后，再对 `TaskID、RID、H5(E_π)` 和一次性会话标识生成独立批准。只有至少 `t` 份有效批准才能启动门限追踪并生成 `τ_trace`，节点名称或整数编号本身不构成授权。该票据用于实现论文所假设的 AS→Auditor/D-KGC 认证信道，不进入 `E_π`、`H5(E_π)` 或环签名公式。AS 仅在验证追踪证据与门限 Schnorr 证书后更新黑名单和任务环。协议中不存在加密身份陷门，也不存在独立的非交互式零知识证明对象；Fiat-Shamir 哈希挑战 `c` 是环签名本身的一部分。

AS 创建追踪证据时会把完整不可变证据、摘要和控制面授权票据自动登记到内部 pending 台账；仅调用验证函数不会清除该状态。只有匹配证据的门限追踪结果验证通过并经 `archive_trace_result` 显式归档后，对应记录才会关闭；完整证据与认证结果会继续保存在检查点中，归档存储仍须实施访问控制。追踪临时失败时，无论严重偏离还是 C_tol 触发，触发轮更新均保持拒绝状态，检查点保存原始证据供重启后优先重试。`finalize_task` 不再接受调用者自报的布尔值，发现任何待处理审计就拒绝销毁。全部审计关闭后，系统才清零当前进程中的 `κ_t`，删除 `h_t`、任务标签缓存和环历史，并写入无秘密 tombstone 防止任务重新激活；若撤销后环为空，则直接进入该终态，不复用旧环。此前已经复制到外部存储的旧检查点仍须由其存储所有者按保留策略删除。

实现边界需要明确区分：`crypto_mode="sm9"` 的群、配对、SM3 `H_v` 和规范序列化由 GmSSL 国标 SM9 原生桥执行；Shamir/Feldman 份额关系、份额域交叉项乘法、随机盲化求逆、门限部分结果组合、逐节点追踪批准和门限 Schnorr 验证均在代码中执行。`ξ`、`P_r`、`P_r^(-1)`、`Δ_j`、`msk` 和 `β_j` 不在环建立或追踪路径中重构；求逆只开放论文允许公开的随机掩码乘积。角色对象之间采用单向下发的最小状态，AS/客户端对象图不含 D-KGC、追踪网关或其他客户端私钥。论文所述 Paillier 密文交互、独立进程和多节点认证信道仍由单进程中的份额域协议模型代替，未部署为真实跨主机网络；同一 Python 进程内的调试器、`gc` 遍历或内存读取也不属于安全边界。检查点由可信实验编排层统一保存各节点份额，因此本实现可验证协议代数、正常角色调用链和实验开销，但不能等同于具备进程/主机级信任隔离的生产 D-KGC。


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

完整参数见 `python -m sm9rrsfl.benchmarks.crypto_overhead --help`。

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


## VERT 与 AlignIns 复现说明

VERT 文献与官方实现：

Wang J, Wang R, Zhang F. How to Defend Against Large-Scale Model Poisoning Attacks in Federated Learning: A Vertical Solution. IEEE Transactions on Dependable and Secure Computing, 2026. 论文与代码：[arXiv](https://arxiv.org/abs/2411.10673)、[VERT](https://github.com/mylab426/VERT)。

VERT 的公开参数存在需要披露的论文/仓库差异：论文第 6.1 节将客户端本地 SGD 学习率和服务器端预测器 Adam 学习率都写为 `0.001`；官方仓库当前 `conf.json` 则只有一个 `lr=0.01`，并在 `client.py` 与 `defenses.py` 中同时用于客户端 SGD 和预测器 Adam。本项目将两者拆分为客户端 `lr` 和 VERT 专属 `vert_predict_lr`，避免为了复现预测器参数而迫使所有对比方法采用较小的客户端学习率。主实验若使用共享 `lr=0.05`，应将其描述为统一训练协议下的公平比较，而不是 VERT 官方训练参数的逐项复现；严格官方复现结果应单独报告并注明选择的是论文参数还是仓库参数。

本项目的 VERT 位于 `sm9rrsfl/vert.py`，保留两轮历史建立、冻结的随机线性投影、每个全局轮次重新初始化并按客户端顺序训练的共享三层预测器与集成系数、原始线性投影特征、余弦相似度排序、被排除更新以全局更新替换历史，以及入选更新等权 FedAvg 的语义。默认的无先验模式仅把论文已知 $k$ 的 Top-k 选择替换为论文提出的 `K=2` K-means 高相似度簇选择；显式 `--vert-use-ratio-prior` 和正整数 `--vert-top-k` 与它共用同一评分核心。MNIST 参数向量规模允许直接使用固定随机全连接投影；只有预计稠密投影器超过 `256 MiB` 时才使用稀疏符号哈希投影，该路径是面向大模型内存约束的工程适配。所有被 VERT 排除的客户端仅在当前轮不聚合，其更新历史由本轮全局更新替换，不进入永久黑名单。

当 `--compute-backend torch --device cuda`（或可用的 `auto`）启用时，VERT 的固定投影、三层预测器、集成系数训练和余弦评分会使用 Torch 设备张量；历史记录和检查点仍为可移植的 NumPy 数据。无 CUDA/MPS、未安装 Torch 或显式 `--compute-backend numpy` 时，自动保持原有 NumPy 实现，因此 Mac CPU 环境可正常运行。SM9 签名验签仍为 CPU 原生密码学计算。

AlignIns 文献与官方实现：

Xu J, Zhang Y, Hu J. Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection. CVPR, 2025. 论文与代码：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Detecting_Backdoor_Attacks_in_Federated_Learning_via_Direction_Alignment_Inspection_CVPR_2025_paper.html)、[AlignIns](https://github.com/JiiahaoXU/AlignIns)。

本项目的实现位于 `sm9rrsfl/alignins.py`。每轮首先计算更新与当前全局参数的余弦方向一致性 TDA；再对所有客户端更新逐坐标投票得到主符号，只在每个客户端更新绝对值最大的 `sparsity` 比例坐标上计算 MPSA。两组分数分别以中位数为中心、总体标准差为尺度，两个绝对标准化分数均严格小于各自半径的客户端才入选。入选更新随后按其范数中位数裁剪，并严格执行论文的 `1/|S|` 等客户端线性和；裁剪因子不会被聚合器再次归一化。

AlignIns 防御接口只接收当前全局参数和本轮更新映射，不接收真实恶意比例、恶意数量或客户端标签。NumPy 和 Torch 路径均采用流式累计主符号/线性和；Torch 路径在训练设备上完成 Top-k、余弦和范数计算。方法是无状态逐轮过滤，因此检查点无需保存额外防御器历史。

论文和官方代码的默认起点为 `sparsity=0.3`、两个半径均为 `1.0`；方案 B 的 12 候选网格包含这个点，并在相同训练留出场景、候选数和验证 seed 下与 Ours、VERT 共同学习 Score。恶意比例达到 `60%–80%` 时，基于客户端多数统计的主符号与中位数可能已由攻击者主导，因此这些点应明确报告为超出常规诚实多数假设的压力测试，不能表述为 AlignIns 在该区间仍有理论保证。

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
