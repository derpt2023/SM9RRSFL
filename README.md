# SM9-RRS-FL 实验复现

文档同步约定：仅根目录 `README.md` 随 Git 同步；其余 Markdown 为本地工作文档。下文标注的本地参考不包含在 Git 克隆中，运行所需说明以本 README 和仓库配置为准。

六种方法：Ours（内部名 `sm9rrs`）、VERT、AlignIns、Krum、TAD（`ding13`）、FedAvg。支持 MNIST/CIFAR-10、IID/Dirichlet Non-IID、NumPy/PyTorch，以及真实 SM9 或快速仿真密码模式。

## 2026-09-06：正常状态 v3 与当轮隔离

Ours 现在采用 **K-means 学习正常状态，历史偏离识别异常**。它不是“把本轮客户端聚成两个簇，再把小簇判为攻击者”。每个匿名任务标签独立学习一个或多个正常模式，所有簇均作为正常参照，但新更新仍须通过偏离检测；0% 恶意场景也不会被强行分出恶意簇。

**所有可疑更新当轮零权隔离**，无需等待 C_tol；C_tol 只控制永久撤销身份的时机。严重偏离立即发起撤销，一般可疑使嫌疑值加 1，正常轮折半，达到 C_tol 当轮追踪撤销。C_tol=1 即首次可疑请求永久撤销，已纳入候选。永久撤销仍须验证身份追溯证书。VERT 的逐轮拒绝不是永久撤销身份。

本次是算法和配置的破坏性更新：配置/统一调参 schema 为 **3**，Ours 工件 schema 为 **5**，训练检查点 schema 为 **16**，Ours 算法版本为 `ours-normal-states-v3-quarantine`。包括上一版保守正常状态检测器在内，旧参数工件、1170 配置验证结果和轮次检查点不能作为新算法结果续用，必须重新校准；程序按指纹隔离，现有 outputs 不会被代码修改删除。方案 B 当前默认输出目录为 `outputs/mnist_v7_target_fair_tuning`；新增目标选参不改变检测器公式，但改变候选空间和最终选择策略。

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
4. 默认 `ratio_range=[0,0.8,5]` 对应正式比例 0/20/40/60/80%，校准比例 0/10/30/50/70%；正式 seed 241/242/243 与验证 seed 71/72/73 分离。0% 是两边共享的必要干净对照。比例范围、攻击类型、模型和干净前缀属于公开研究条件，分离比例不能消除这些先验，论文应披露。
5. 方案 B 三个可调方法各 12 个候选；其余方法保持单一公开算法配置。39 候选 × 2 分区 × 5 校准比例 × 3 seed = 1170 次完整训练。Ours 每次训练自行学习该客户端的正常中心、尺度和半径；并不提前读取正式客户端的数据。
6. 统一 Score 权重通过留一攻击比例验证选择，四项权重均至少 0.05，步长 0.05。每折的候选筛选只看非留出比例和干净对照；留出比例用于外层检验。权重搜索复用每折候选的充分统计，不重复训练或 969 次重新扫描全部轮次。
7. 选出每种方法唯一的固定超参数集，才运行 6 方法 × 2 分区 × 5 正式比例 × 3 独立 seed = 180 次正式实验。同一数据集不同恶意比例不能分别调一套参数。换 CIFAR-10 后需重新校准，不能套 MNIST 工件。

当前方案 B 配置另启用 `tuning.performance_target`，对应“准确率尽量接近 VERT，ASR 接近或优于 VERT”的目标。先按公共 Score 独立选出所有方法，冻结 VERT；再仅用训练留出数据，在健康合格的 Ours 候选中优先选择满足下列目标者：

- 每个种子、分区、恶意比例均检查，不只比较总平均。
- 干净最终准确率，以及攻击全程、最后10轮、最终轮准确率：落后 VERT 不超过2个百分点。
- 攻击全程、最后10轮、最终轮 ASR：不高于 VERT 超过1个百分点，且本身不超过5%。单轮ASR峰值不超过20%，避免短时严重污染被均值隐藏。
- 多个候选达标时优先低ASR，再比较准确率；没有候选达标时，选择偏离目标最少的健康候选继续完整实验，同时明确记录 `unmet`，不把它写成成功。

这些数值是显式研究目标，不是成功保证。该二次选择是**针对VERT的目标选参**，不能再将最终Ours选择描述为纯方法中立Score；VERT和其他基线不会因此被削弱或重选。三个可调方法仍各12个候选。设 `performance_target: null` 可关闭二次选择，仅用原公共Score。目标在训练留出阶段选择后冻结，正式测试只做只读达标审计，不重选参数。

启动入口不变，仍修改 `configs/fair_tuning.example.json` 后运行 `run_fair_tuning_from_config.py`。查看 `performance_target_validation.json` 的 `status` 判断校准是否达标；正式结果看 `final_evaluation/performance_target.json`。`best_parameters.json` 和 `tuning_manifest.json` 保存独立选择、目标选择与所用阈值。`unmet` 不会自动删除结果或跳过正式比较；Ours/VERT 无健康候选仍停止。其他方法若仅算法健康/干净效用不合格，则保留失败标签继续六方案比较；结果不完整或指标不可用仍不能放行。详见 本地文档《TAD失败后的续跑与兼容性说明》（`docs/TAD失败后的续跑与兼容性说明-2026-09-10.md`，不随 Git 同步）。

2026-09-10：统一调参现已支持更换CUDA卡号/密码线程数后复用原验证与正式缓存，恢复时重新绑定VERT/TAD的设备；CUDA任务每卡最多一个，派发前复查显存，OOM最多额外重试一次，并保留其他工作线程的完成结果。训练、攻击、评估批大小及配置均不变。补丁文件清单、限制GPU的启动命令和恢复数量检查见 本地文档《AI Station换卡续跑与OOM恢复》（`docs/AIStation换卡续跑与OOM恢复-2026-09-10.md`，不随 Git 同步）。

本次100客户端、单开发种子、非IID、boost15补测满足上述接近目标；它不是多种子正式结论。配置含义、全部实测数值及启动说明见 本地文档《性能目标选参与启动说明》（`docs/性能目标选参与启动说明-2026-09-07.md`，不随 Git 同步）。

## Ours 一轮实际如何工作

**前提：前 K 轮全部诚实。** K 是公共实验参数，不是恶意比例。默认示例 K=7、attack_start_round=12；若设 attack_start_round=0，统一解析为 K+2。Ours 拒绝攻击开始早于或等于 K 的配置。新加入、未拥有足够干净历史的标签不能在攻击期间重新领取“无条件热身期”。

1. 客户端按统一规则训练，返回更新；先验证 SM9 标签和签名，再检测。检测器不接收恶意比例、客户端好坏真值或攻击目标类别。
2. 用本轮学习率归一化更新，减少单纯 lr 衰减造成的尺度漂移。完整更新补零重排为 `ceil(P/C) × C` 矩阵，保留范数、前 q 个奇异值和紧凑的**右子空间投影**；这是新特征，不声称与旧左子空间距离等价。
3. 同时提取全模型确定性有符号投影，以及真正分类头各类别的有符号权重均值和 bias 变化。这样更新取反不再因奇异值不变而不可见。所有类别同时检测，不偷看 source/target 标签。投影维数有限、类别摘要也有信息损失，不能宣称消除了全部盲区。
4. 每个标签用前 K 轮特征拟合标准化后的正常 K-means。最多允许指定数量的模式，每个模式至少 3 个样本；样本不足或出现单点簇则退化到一个正常模式。尺度为 `max(1.4826 × MAD, 0.15)`；0.15 是明确的正则化下限，1.4826 是正态一致性换算常数，不是由本次数据学出的常数。每个特征块的半径使用该块簇内最大 RMS 训练距离，下限 1；拟合半径和在线评分使用相同分组定义；小 K 下不声称具有统计误封率保证。
5. 对每个正常模式计算谱、有符号、类别、独立对数范数四块证据的分组距离（范数在线只检测超过冻结干净期最大值的增长；正常收敛和低于该上界的回升不作为幅度攻击），取同一模式下最大组距离，再找最接近的正常模式。最终新颖度 D 取“动态正常模型距离”和“冻结正常锚点距离 / reference_budget”的较大值；累积漂移为 `Q=max(0,beta*Q+min(D,reject_threshold)-kappa)`。`D > reject_threshold` 为严重偏离，立即拒绝更新并发起撤销，不等待 C_tol，也不另设有符号/类别证据必须同时越界的条件。这两类证据仍参与 D 的计算。其余 `D > distance_threshold` 或 `Q > drift_threshold` 的情况为一般可疑，同样拒绝当轮更新。kappa 的候选范围提升为 1.75–3.0，作为正常距离包络联合校准；未实现额外的阶段预测网络或声称学出了逐轮正常均值。
6. **聚合、历史写入、可靠度恢复分别判断。** 历史要求 D <= history_threshold、原始冻结锚点分数 <= reference_budget、Q <= kappa、无需裁剪、检测正常并连续确认 history_confirm 次；还有整轮冻结和实际正系数检查。动态重拟合使用冻结的干净坐标位置/尺度，不随短窗口收缩；每次准入使中心每块最多移动 0.25 个干净半径，动态半径不低于干净半径。冻结锚点和范数上界不重学。0.25 是固定的正则化选择，不是统计保证。
7. 一般可疑当轮隔离且可靠度乘 penalty；正常、无需裁剪、D/Q <= kappa 连续 recovery_confirm 次后可独立恢复，无需达到历史写入资格。恢复公式为 `omega += (1-omega)*(1-1/recovery_factor)`，有界渐进接近 1，不再乘法缓慢脱离极小值。嫌疑值 C 初始为 0：可疑时 `C=min(C_tol,C+1)`；有效正常观察时 `C=0.5*C`。C 为浮点数，不取整、不清零；缺席、无效、缺少干净参照不触发折半。达到 C_tol 或严重偏离同轮请求永久撤销，等待证书的标签不能恢复。
8. 新增嫌疑超过当前可见标签一半的“分布突变轮”仍暂停可信历史写入和权重恢复，但**不暂停正常嫌疑值折半，也不延迟任一撤销路径**。删除每轮约 10% 撤销配额和至少保留两名身份的限制。若所有客户端被撤销，记录实际终止轮次并安全关闭任务，不填充虚假后续轮次；校准同时检查完成率、全员/全部诚实撤销和持续权重饥饿。
9. 两条撤销路径都请求 D-KGC 追踪；仅证书验证成功才能永久撤销、更新任务环。“立即”表示在本轮聚合前完成，不代表跳过追溯认证。证书失败时保留原始证据和零聚合状态，即使不是定期检查点轮次也强制保存，中断后恢复运行优先重试；等待证书期间不能恢复权重或重复累计。

例如 C_tol=3，连续三轮一般可疑时 C 为 `1 → 2 → 3`，第 3 次当轮撤销；若依次为“可疑、正常、可疑、可疑、可疑”，C 为 `1 → 0.5 → 1.5 → 2.5 → 3`，第 5 轮撤销（存储值封顶于 C_tol）。任意一次严重偏离均立即走撤销路径。折半只作用于 C，不改变 Q 的 beta/kappa 更新规则。这里的“正常”要求本轮检测整体正常：即使瞬时 D 较小，若 Q 仍超过漂移阈值，本轮仍是可疑，不能折半。

这是更激进的安全/可用性取舍：检测错误可能造成不可逆误撤销或全部客户端退出，不能声称一定提升准确率。误撤销、诚实权重损失、干净准确率和完成率仍需如实报告，不能删除失败场景来美化结果。

**限制漏检者的绝对聚合影响：** 令 `b_i=n_i/sum_all_clients(n_j)`，`r_i∈[0,1]` 为可靠度，拒绝者 `u_i=0`，其他 `u_i=b_i*r_i`。实际更新系数为

`a_i = min(u_i/sum_j u_j, weight_cap*u_i) * clip_i`。

`clip_i=min(1,clip_factor*clean_norm_limit/current_norm)`；全零 u 时本轮不更新。**裁剪/封顶后不重新归一化**，系数和小于 1 意味着服务器步长缩小。干净、未裁剪时退化为按样本数的 FedAvg，而不是旧 Ours 的等客户端权重。默认 weight_cap=2 时，100 个等样本客户端中 7 个完全漏检者的绝对系数最多 0.14（尚未计范数裁剪），不再因其他节点被撤销自动放大到 0.26；它们在“剩余更新”中的相对比例仍可能很高，这并非检测正确性的替代。

## 自动参数与验证报告

“自动”区分两件事：正常中心、MAD 尺度、半径由各次训练的干净前缀学习；检测阈值、C_tol、penalty/recovery 等超参数从有界候选中用训练留出结果选择，正式实验统一冻结。正常轮嫌疑值乘 0.5 是本方案指定的固定规则，不参与搜索，也不新增手动参数。不是所有安全常数都由数据学习，更不是读取正式测试结果调参。

默认12个联合候选的前4个固定为已在开发留出实验测过的检测组合（q2、模式2、warning3、severe6、drift_allowance2.5、history_threshold2、reference_budget3.5），分别采用C_tol=2/3/1/5；其余保留q、阈值、漂移、恢复和裁剪的多种组合。实际范围和组合以 `sm9rrsfl/ours_policy.py:bounded_candidates` 为准。候选空间受前期训练留出实验启发，不使用正式测试集；不是笛卡尔积，不能证明全局最优。预算允许1–36，方案B三个可调方法预算相同。

独立选参阶段的公共 Score 为：

`S = w_c*A_clean + w_r*A_robust + w_a*(1-ASR) + w_h*(1-H_loss)`。

- A_clean：干净场景最终准确率。
- A_robust、ASR：**每个攻击场景从攻击开始到结束的逐轮平均**，再场景等权平均；不再只看最后一轮，避免漏掉前期模型崩溃。
- H_loss：每场景逐轮平均的诚实名义 FedAvg 权重损失，再场景等权平均，包含干净场景。标签只用于事后统计，不进入检测。
- 硬门槛：通信轮完成率（默认1）、非有限更新数量（默认0），以及可调方法相对同分区/Dirichlet α/客户端数/seed 的FedAvg干净最终准确率下降（当前方案B配置≤0.03，旧配置缺省值仍为0.05）。Krum/TAD/FedAvg是固定对照，干净结果差仍应报告，不因此删掉该对照。高ASR影响Score，并使新增性能目标审计记录 `unmet`；单纯目标未达不隐藏整项研究结果。
- 新增固定健康门槛：任意轮（含最后一轮）全员或全部诚实者永久撤销、干净场景累计误撤销率 >10%，以及 Ours 连续 5 轮诚实系数损失 >=99% 均使候选失效。最后一项只适用于 Ours 未重新归一化的系数，不把 Krum 的逐客户端权重缺口误当全局步长。阈值是公开的开发选择，不是鲁棒性定理。客户端真值仅在离线评价使用。
- 不设置前三攻击轮召回率硬门槛；初期恶意聚合质量、最差准确率和峰值 ASR 另行报告。
- 缺失 ASR、未完成或 NaN/Inf 不冒充正常结果；Ours/VERT 没有合格候选时明确停止。AlignIns/Krum/TAD/FedAvg 优先选健康候选；若全部候选仅违反健康/干净效用门槛，则按同一 Score 选出完整、有限的失败对照继续，保持 `valid=False`。固定TAD仍只有1候选，不因此新增调参或修改算法。先看已写出的 `candidate_feasibility.csv` 和 `validation_results.csv`；不删除失败场景。
- 自动权重学习仍优先使用三个可调方法。联合留一比例拟合不可行时，先只用 Ours/VERT 重试；仍不可行则使用预先声明的默认权重。每次回退记录原因与参与方法；随后对 Ours/VERT 的全部验证场景继续执行原健康门槛。回退不使用正式测试数据，也不减少任何方法的候选训练预算。
- `best_parameters.json` 的 `selection_policy` 说明进入下一阶段的规则；失败对照带 `selection_status=comparison_only_failed`、`validation_valid=false` 和 `invalid_reasons`，`validation_score=null`。`comparison_selection_score` 仅解释失败对照之间的排序，不代表健康通过。正式 `aggregate.csv` 增加健康失败次数/原因，`scenario_audit.csv` 保留每次运行的诊断。
- `best_parameters.json` 报告 Score 定义、选择余量、各场景结果离散程度、权重选择模式数。若 `weights_identifiable=false`，代表多组权重选中了同样的候选，不能把那组数字解释为唯一学出的“真实权重”。

硬门槛接口：公共 CLI `--min-round-completion`、`--max-nonfinite-updates`（JSON 对应 `calibration_min_round_completion_rate`、`calibration_max_nonfinite_updates`）；方案 B 的干净下降限制为 `tuning.max_clean_accuracy_drop`。删除的 `--ASR`/前三轮召回率等旧接口不可再用。

当前自动 Score 包含定向 ASR，因此自动校准/方案 B 要求 `attack=alternating_minimization`；sign_flip/gaussian 等非定向攻击可在 fixed 模式运行，但不能无定义地套用定向 ASR 选参。启动前会检查攻击辅助、验证和正式评价分区是否有足够的 source-label 样本，避免长时间训练后才发现 ASR 缺失。

### Ours 固定参数/消融接口

正式 auto 模式不填下表；手填会被拒绝，避免“部分自动、部分偷偷固定”。需要消融时设 `ours_parameter_mode="fixed"` 并用下列规范 JSON 键；CLI 把下划线换成连字符。K/窗口独立由 `--K` 设置。

| 参数 | 固定模式默认值 | 含义 |
|---|---:|---|
| detector_subspace_dim | 2 | q，谱/子空间维度 |
| detector_normal_clusters | 2 | 每客户端正常模式数上限 |
| detector_distance_threshold | 2.5 | 预警距离 |
| detector_reject_threshold | 5 | 严重偏离立即撤销距离，必须大于预警 |
| detector_drift_memory | 0.8 | beta，漂移记忆 |
| detector_drift_allowance | 2 | kappa，正常漂移扣除量 |
| detector_drift_threshold | 6 | h，漂移预警界限 |
| detector_history_confirm | 2 | 历史准入连续确认次数 |
| detector_history_threshold | 1.75 | 历史准入距离阈值，严格低于预警阈值 |
| detector_recovery_confirm | 2 | 独立恢复连续确认次数 |
| detector_reference_budget | 3 | 冻结锚点偏离预算 |
| detector_clip_factor | 2 | 干净范数上限倍率 |
| detector_weight_cap | 2 | 可靠度调整后名义系数放大上限 |
| suspicion_penalty_factor | 0.5 | 嫌疑可靠度乘子 |
| suspicion_recovery_factor | 1.25 | 渐进恢复速率参数，rho=1-1/factor |
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

### CIFAR-10 当前推荐：750 组验证，仅增加四套 Ours 候选（2026-09-18）

按“规则不变、小幅增加候选、Ours尽量最优”的要求，新增 `configs/cifar10_six_mnist_gate_compact_plus_v4.json`，在630组紧凑版基础上保留全部21套候选及原顺序，再追加4套Ours；五个基线的候选和固定回退项完全不变。新预算为 **Ours14、VERT4、AlignIns4、Krum/TAD/FedAvg各1，共25套 × 30 = 750组验证 / 75000训练轮**；达标后正式仍180组/18000轮，总上限930组/93000轮。相对630组多120组验证、12000轮（+19.05%），新增全部属于Ours，实际耗时增幅不保证等于任务数增幅。

新增四套完整候选均来自原完整v4空间，不拼接历史赢家的部分参数：

| 新增Ours候选 | 增加理由 | 已知限制 |
|---|---|---|
| `sm9rrs-v10-010` | MNIST验证Score第三；与已有012形成不同子空间、惩罚、撤销组合的完整策略对照 | MNIST成绩不能直接当作CIFAR成绩 |
| `sm9rrs-v10-011` | MNIST验证Score第二；加上010、已有012，覆盖历史验证前三套完整策略 | 必须重新通过本轮完整验证 |
| `sm9rrs-v10-016` | 相对CIFAR锚点015只把撤销阈值3→5，考察延后撤销的影响 | 旧CIFAR对应004曾有1次非有限更新，虽raw Score与ASR更好，历史仍不健康 |
| `sm9rrs-v10-039` | 相对015只把可信历史连续确认2→3，考察延后历史准入的影响 | 尚无本轮成绩，不能保证阻止持续低异常攻击 |

**保持原规则：** 公共模型/训练/攻击、数据划分、seed、场景、每组100轮、健康和MNIST性能门槛、Score权重、基线失败回退全部不变。`require_mean_dual_best`仍为false；在达标Ours中优先选验证均值双优，再按原Score等顺序选择，目标未达仍停止。更多候选增加找到好解的机会，不能保证存在合格解或正式结果最佳，不能通过削弱基线、放宽门槛或使用正式成绩选参来保证排名。各方法候选预算仍不相等，需如实披露。

```bash
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_mnist_gate_compact_plus_v4.json --plan-only
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_mnist_gate_compact_plus_v4.json \
  --devices auto --progress-mode live
```

生成器为 `cifar_compact_plus_candidates.py`，独立输出 `outputs/cifar10_six_mnist_gate_compact_plus_v4/`。源码/配置保留630紧凑版、2610完整备选及旧实验身份；不向已冻结任务中追加候选。进度、多卡、断点与正式完成后的HTML/均值SVG/PDF继续由原包装入口处理。新配置生成与运行无需历史outputs；历史验证只作开发候选设计依据，不转移为新验证资格。

### CIFAR-10 基础紧凑方案：630 组验证（保留）

为减少 2610 组完整搜索的开销，新增 `configs/cifar10_six_mnist_gate_compact_v4.json`，独立输出 `outputs/cifar10_six_mnist_gate_compact_v4/`。它复用现有 v4 入口和门槛，保留每候选 **3 seed × 10 场景 × 100 轮**，仅把候选压缩为 **Ours 10、VERT 4、AlignIns 4、Krum/TAD/FedAvg 各1**。因此验证为 **630 组 / 63000 训练轮**，Ours 达标后正式仍为 **180 组 / 18000 轮**，合计最多810组/81000轮。相对完整87候选，验证轮次减少 **75.86%**，含正式阶段总轮次减少 **70.97%**。第0轮是初始评估，不额外算一轮训练。

```bash
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_mnist_gate_compact_v4.json --plan-only
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_mnist_gate_compact_v4.json \
  --devices auto --progress-mode live
```

完整搜索配置和旧630组v3均原样保留，不混用输出目录。紧凑版生成器为 `cifar_compact_candidates.py`，JSON记录每个候选的 `shortlist_reason`，不依赖本地历史outputs即可在新服务器生成、运行。沿用v4种子、Score、健康/性能门槛、公共训练和攻击设置，不启用新的算法变体；仍自动生成正式汇总HTML与均值SVG/PDF。

**减法依据仅来自历史验证记录与参数覆盖，不采用正式结果筛选，也不把历史Score当作本轮资格。** Ours保留CIFAR健康最高Score锚点、两个低阈值高raw Score但不健康的方向、三个完整MNIST代表策略（含验证最佳），再保留子空间维数、漂移、惩罚和固定锚点预算四个单参数探测。VERT保留CIFAR健康最佳和raw Score最佳，增加低预测学习率的对应配置及MNIST验证最佳；删除成本高的50预测epoch和未测history20扩展。AlignIns保留CIFAR raw Score前二、29/30组独立健康的候选，以及MNIST按当前CIFAR权重重评分最佳，移除新增半径1.5探索。

| 方法 | 保留的完整v4候选ID后缀（方法前缀与 `-v10-` 不变） |
|---|---|
| Ours | 015、013、014、002、007、012、023、029、034、041 |
| VERT | 014、016、015、003 |
| AlignIns | 008、012、009、001 |
| Krum / TAD / FedAvg | 各001 |

旧CIFAR验证中，VERT history10/20epochs的30任务共记录10.19任务小时，而50epochs为20.64小时，前者同时有更高最终准确率和更低攻击窗口ASR；这支持优先删去50epochs，不能证明新seed永远如此。两者均未通过整体健康，50epochs的非有限更新更少（2次对4次），所以这是预算取舍，并非所有指标都更差。旧Ours六候选都未满足新绝对目标：最低平均ASR仍约24.48%，且该候选不健康。保留候选是新的开发优先级，不是已经找到低于5% ASR的CIFAR最优解。

轮次数减少不等于耗时同比减少：Ours在旧日志中每30任务约18任务小时，AlignIns约1.36小时。紧凑版比旧v3同为630组但有更多Ours，按历史同类任务粗估约221任务小时，甚至高于旧v3的176.57任务小时；这些含并发、设备与未测参数估计，不能换算成承诺的墙钟工期。降低的是相对2610组扩展方案的预算。

**基线回退的准确触发方式：** 对五种基线分别执行“本方法存在健康候选→最高健康Score；本方法没有健康候选→最高完整可评分raw Score；全部不可评分→固定首候选”。不要求五种一起失败，有健康候选的方法不会被失败候选替代。选择器先计算基线参照，再检查Ours，所以Ours未达时报告也可能出现回退建议；只有状态为 `qualified_for_final` 才冻结和执行180组。Ours未通过时不会因基线回退而启动正式训练。这里基线的健康要求不等同于Ours专属ASR性能目标；VERT完全不可评分时的明确未评估分支等细则，继续遵循下文v4政策。

### CIFAR-10 完整候选备选：2610 组验证与 MNIST 逐场景目标（v4，保留）

新配置为 `configs/cifar10_six_mnist_gate_v4.json`，独立入口 `run_cifar_six_mnist_gate.py`，独立输出 `outputs/cifar10_six_mnist_gate_v4/`。旧 v2/v3 的入口、配置和输出保留供续跑与追溯。用户自行提交、推送 GitHub，再由 AI Station `git pull` 更新；不要用新配置覆盖旧实验目录。

先安装报告依赖并检查计划（不加载数据、不检查 GPU、不训练、不创建输出目录），再通过包装入口运行：

```bash
python -m pip install -r requirements.txt
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_mnist_gate_v4.json --plan-only
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_mnist_gate_v4.json \
  --devices auto --progress-mode live
```

**为什么扩大候选：** 旧 CIFAR 网格是有界的人工预设筛查空间，Ours 只搜索告警阈值 1.25/1.75/2.5 × 撤销阈值 3/5，共 6 套；它没有覆盖 MNIST 已执行的 12 套完整策略。旧配置也未证明其余固定参数已经最优。因此新实验完整保留两套历史参数空间，增加有明确含义的敏感性探测；所有候选在本轮验证集重新训练、评分，不复用旧研究的 Score 或正式测试成绩。

| 方法 | 候选数 | 新空间 |
|---|---:|---|
| Ours | 42 | MNIST 实际 12 套完整策略 + CIFAR v3 的 6 套 + 围绕 CIFAR 锚点的 24 个单参数探测，覆盖子空间维数、正常簇数、告警/强拒绝、漂移、历史准入、恢复、裁剪、权重上限及撤销 |
| VERT | 24 | 历史窗口 5/7/10/20 × 预测训练轮数 5/20/50 × 预测学习率 0.0005/0.001；投影维数、top-k、比例先验不变 |
| AlignIns | 18 | 历史 12 套 + 分别单独放宽 TDA 或 MPSA 半径至 1.5 的 6 套，保留稀疏率 0.1/0.3/0.5 |
| Krum、TAD、FedAvg | 各 1 | 当前实验接口未暴露可搜索的专属参数，保持原算法；重复相同配置不算新候选 |

候选生成器为 `cifar_expanded_candidates.py`，每套参数记录 `origin`。**87 套候选 × 3 seed × 10 场景 = 2610 组验证**，达标后再运行六方案 **180 组正式实验**，最多 2790 组，每组 100 轮。相比旧 630 组验证，任务数约为 4.14 倍；VERT 部分候选训练更重，耗时不能直接按任务数同比估计。各方法搜索预算不等，论文须披露，不能称为等预算调参或穷举全局最优。

**实际推进条件：** Ours 必须健康，并通过以下与 MNIST 相同的性能目标数值和窗口。这里把目标设为新 CIFAR 的推进条件；历史 MNIST 入口曾允许目标未达时退回 nearest-healthy 继续运行，本 v4 不采用该回退。

- 对每个 seed、每个场景分别判断，不用跨场景均值掩盖局部失败。受攻击任务分别检查第 25–100 轮均值、最后 10 轮均值和最终轮；三个窗口的 ASR 均须 ≤ **5%**，整个攻击窗口峰值须 ≤ **20%**。
- 选定 VERT 可评分时，上述三个窗口分别要求 Ours 准确率落后 ≤ **2 个百分点**、ASR 高出 ≤ **1 个百分点**；干净任务检查最终准确率相对差距。VERT 健康失败但完整可评分时仍参与比较，并记录 `reference_health_qualified=false`。
- Ours 保留原健康检查（完整轮次、零非有限更新、误撤销等）；健康的 FedAvg 干净对照可用时仍检查干净准确率下降不超过 3 个百分点。缺失的健康对照明确标为未评估，不当作已通过。
- **对照方案不因健康失败阻断正式阶段：** 有健康候选时选择公共 Score 最高者；无健康候选时，从本轮全部历史候选的 30 组完整、可评分验证记录中选 `raw_score` 最高者，保留 `valid=false`、失败原因和 `best_scored_unqualified`。只有全部不可评分时使用预先声明的第一候选，不凭空生成分数。
- 如果 VERT 全部无法评分，健康且绝对 ASR 目标达标的 Ours 仍可推进；相对比较为 `unassessed`，完整目标为 `full_target_passed=false`，推进分支明确为 `mnist_absolute_target_without_scorable_vert`。这只适用于已有身份与训练上下文证据的算法数值失败；待执行、损坏缓存、OOM、环境或数据缺失会先返回 `validation_evidence_incomplete`，必须排除原因并续跑，不能归为算法失败。
- 默认 `promotion.require_mean_dual_best=false`：在上述合格 Ours 中优先选验证均值双优者，否则取合格者的最高公共 Score。可在**创建新实验前**改成 `true`，把均值双优设为附加硬条件；比较范围为健康 Ours 候选及五个对照选定的可评分候选，缺失参照不算被战胜，报告明确比较范围。准确率为全部 30 组最终轮均值，ASR 为 24 组受攻击任务各自攻击窗口均值的等权平均。两项必须同时最优，仅允许 `1e-12` 数值容差；同分按 Score、最坏 ASR、候选 ID 排序。

公共 Score 保持 CIFAR v3 的原公式：`0.25 × clean_accuracy + 0.50 × robust_accuracy + 0.20 × (1 − attack_success_rate) + 0.05 × (1 − honest_weight_loss)`，没有混入 MNIST 的历史学习权重。公共模型、数据划分、训练/攻击优化器、100 客户端、batch 50、预热 K=20、攻击起点 25 和 boost=5 均保持原 CIFAR 协议；改变的是候选参数与选参/推进政策。

验证 seed 为 1001/1002/1003，比例 0/0.1/0.3/0.5/0.7；正式 seed 为 1101/1102/1103，比例 0/0.2/0.4/0.6/0.8；均含 IID/Dirichlet。没有 Ours 达标时停在 `needs_ours_development` 或 `needs_ours_target_development`，不自动放宽目标。达标后冻结六个候选，不继承验证模型权重，全部进入正式阶段。续跑核验源码、配置、数据和冻结计划身份，不根据正式结果重新选参。

`validation_summary.json` 的 `mnist_target_gate` 和 `ours_target` 保存真正生效的逐场景检查、各候选失败原因、参照范围和选择。正式摘要复制到 `validation_mnist_target_gate` / `validation_ours_target`，HTML 同时展示。通过上述包装入口完成正式阶段后，自动生成汇总 HTML、各 seed 图、**12 张均值 SVG/PNG 和 12 页矢量 PDF**；种子从新计划读取。单独补报：

```bash
python -u run_cifar_six_with_progress.py --report-only \
  --output outputs/cifar10_six_mnist_gate_v4
```

直接运行底层 `run_cifar_six_mnist_gate.py` 时只保存 CSV/JSON，完成后可用补报命令。更广的搜索只能提供更多可验证的选择，不能保证 Ours 必定达标或正式测试最优。新 seed 仍使用已有研究历史的同一个官方测试集，不把它声称为从未查看过的测试集。

### CIFAR-10 历史流程：630 组验证、均值最优或接近 VERT（2026-09-16，v3）

上一轮配置为 `configs/cifar10_six_630_mean_v3.json`，入口为 `run_cifar_six_630.py`；继续旧实验时通过下面的显示包装启动。它恢复旧版验证规模和 Ours 原策略网格，按当时确认的近似门槛和失败候选择优规则，独立输出到 `outputs/cifar10_six_630_near_vert_v3/`，保留所有旧实验。

```bash
cd /3251901002/SM9RRSFL && \
git pull --ff-only origin main && \
env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_630_mean_v3.json \
  --devices auto --progress-mode live
```

包装根据这份 v3 配置启动新入口，使用全部通过初始化、型号和空闲显存检查的卡，每卡同时一个独立任务；保留动态进度、原始日志、OOM 排除换卡及断点续跑。旧 v2 配置仍启动原入口。不要将新协议写入旧输出目录，也不要为本轮重复执行旧 `prepare_cifar_final_recovery.py`。

验证规模为 **21 候选 × 3 种子 × 2 分区 × 5 比例 = 630 组**，每组 100 轮。种子 401/402/403；IID 与 Dirichlet 都验证恶意比例 0/0.1/0.3/0.5/0.7。Ours、VERT、AlignIns 各 6 候选，其余各 1 个。Ours 恢复原策略的告警阈值 1.25/1.75/2.5 × 撤销阈值 3/5，不启用弱告警隔离变体；公共训练、攻击优化器和基线算法保持原实现。

用户选定的“平均双指标最优或非常接近 VERT”采用以下固定口径：

- 准确率：每个候选 30 组验证任务的第 100 轮准确率等权平均，包含 6 组干净场景。
- ASR：仅使用 24 组受攻击任务；各任务先对第 25–100 轮 ASR 求均值，再对 24 个任务等权平均。干净场景的 ASR 不计入。
- 先按原完整性、零非有限更新、误撤销等健康约束筛选；健康的 FedAvg 干净参考可用时仍检查准确率下降不超过 3 个百分点。
- 健康 Ours 满足任一分支即可推进：①在健康 Ours 候选和五个对照方案选定的可评分候选中，同时达到最高平均准确率、最低平均 ASR；②相对选定 VERT，平均准确率落后不超过 **0.005（0.5 个百分点）**，平均 ASR 高出不超过 **0.01（1 个百分点）**。分支②允许其他方案表现更好；两个差距必须同时满足，不相互补偿。分支①仅用 `1e-12` 数值容差。
- Ours 优先选择分支①达标者，再考虑分支②达标者；每档采用公共 Score、较低最坏 ASR、候选 ID 的既有同分规则。所有 Ours 候选仍须通过原健康检查，不能用不健康 Ours 的有限成绩放行。
- **五个对照方案统一选参**：有健康候选时取公共 Score 最高者；无健康候选时，从 30 组完整验证、全部评分指标有效的失败候选中取原公式 `raw_score` 最高者，标记 `best_scored_unqualified`，保留健康失败原因。只有所有候选都无法评分时才使用固定第一候选，标记 `fixed_fallback_unqualified`。按全部验证任务汇总，不能挑单个种子最好的一次。
- 健康失败的原 `score` 仍为空，`valid` 仍为 false；`raw_score` 只是失败候选之间的观察成绩排名。客户端非有限更新计数、误撤销或权重饥饿等健康失败不会被抹去。缺任务、不满100轮、配置不匹配、NaN/无穷或缺失的评分指标不具备评分资格；不采用哨兵值代替真实指标。
- **VERT 没有健康候选时，使用最高 `raw_score` 的完整失败候选作为近似参照**；明确记录 `reference_health_qualified=false`。这是与该实现的已观察成绩比较，不能描述成健康基线验证通过。VERT 连可评分候选也没有时，近似分支为未评估；Ours 仍可通过分支①对其余可评分参照的比较推进。完全没有可评分基线时，仅能评估 Ours 候选内部双优，不能宣称六方案双优。
- 旧 `performance_target` 的逐场景差距、绝对 ASR、峰值 ASR 均保留为描述性报告；新的 `mean_dual_gate` / `near_vert_gate` 记录实际推进分支。缺失指标不会自动视为达到近似目标。

Ours 健康且满足上述任一性能分支后，**全部六方案自动进入 180 组正式实验**：各方法冻结一个候选，IID/Dirichlet、比例 0/0.2/0.4/0.6/0.8、种子 901/902/903，各 30 组。对照方案的健康失败本身不阻断，但已选 VERT 的可评分观察成绩仍用于已确认的近似比较。新种子用于新协议下的重复试验，仍使用同一官方 CIFAR-10 测试集，不代表获得了一个从未查看过的新测试集；论文应保留此前实验历史。正式指标不参与本轮参数选择。

若固定六个 Ours 候选中没有达标者，会停在 `needs_ours_development`（健康未通过）或 `needs_ours_dual_development`（健康通过、两个性能分支均未通过），保存所有候选指标和差距，不自动降低门槛或反复重跑挑结果。630 组扩大筛查覆盖，不能保证一定出现合格候选，也不能保证新正式种子上永不出现 NaN。

OOM、初始化/数据准备失败、损坏缓存和仍待执行的验证任务属于证据不完整，单独返回 `validation_evidence_incomplete`；不能把它们作为基线算法失败而冻结备用参数。已进入正式阶段后，再次执行相同命令会跳过验证训练、核验已有验证与冻结计划并续跑正式任务。已有正式计划不会被新候选静默覆盖。

`validation_summary.json` 中的 `mean_dual_gate` 保存均值、raw Score、参照范围、每个 Ours 的差距、推进分支及最终选择。`final_summary.json` 的方法明细继续保留验证选参来源、失败原因和 `selected_without_valid_validation`；正式训练后来健康也不会抹去失败候选的来源。`final_results/aggregate.csv` 保存正式结果。终端结尾的 `FINAL_EXECUTION_SUMMARY` 分别显示执行是否完成和 Ours 是否健康：`completed_with_health_failures` 表示已有结果中存在健康失败，不能解释为没有进入正式阶段；基线失败不会取消其他正式任务。

### 正式实验完成后的 HTML 与均值图（2026-09-18）

通过 `run_cifar_six_with_progress.py` 启动的流程，在本次正式阶段成功结束后自动生成离线报告。包含健康失败但已完整执行的正式结果也会报告，所有失败 seed 保留并明确标注；仅完成验证、任务缺失或轮次不完整时不会伪造完整均值图。安装或更新依赖：

```bash
python -m pip install -r requirements.txt
```

已有完整结果可直接补报，不需要 CUDA，不会启动训练或读取模型 pickle：

```bash
python -u run_cifar_six_with_progress.py --report-only \
  --output outputs/cifar10_six_630_near_vert_v3
```

也可运行 `python experiment_reporting.py --output outputs/cifar10_six_630_near_vert_v3`。报告入口为输出目录下的 `final_results/visualizations.html`，同时提供 `visualized.html` 跳转入口及各 `seed_*/visualizations.html`。每个分区生成六类均值图：逐轮 Accuracy、逐轮 ASR、最终与攻击窗口指标、扣除密码时段的耗时、总耗时及进程峰值 RSS；本配置共 12 张。SVG 和 PNG 位于 `final_results/mean_plots/svg/`、`png/`，12 页矢量图册为 `mean_plots/mean_figures.pdf`。HTML 可直接离线打开。

均值按相同方法、分区、恶意比例和轮次跨正式 seed 等权计算，阴影/误差棒为样本标准差（`ddof=1`，不是置信区间）；单 seed 不画标准差。原 `aggregate.csv` 的总体标准差（`ddof=0`）保持原样，新统计另写 `scenario_mean_sd.csv`、`curve_mean_sd.csv`、`per_run.csv`、`health_failures.csv`，来源 SHA-256 和审计说明见 `data_audit.json`。ASR 攻击窗口先在单任务内求均值，不把每轮当作独立重复；0% 的目标误分类率仅是无攻击背景。

绘图配色与原 seed 页面一致。Word 可从文件插入 SVG；LaTeX 使用矢量 PDF，例如 `\includegraphics[page=1,width=\textwidth]{mean_figures.pdf}`，无需截图。计时是并行运行日志，不是受控性能基准；扣除密码时段不等于关闭密码重跑，RSS 不是 GPU 显存。

原 CIFAR v2/v3 底层入口只写 CSV/JSON，没有接入旧 `fair_tuning` 的 HTML 收尾，这是本次缺页面的原因。自动报告挂在不参与训练源码指纹的包装入口；直接调用底层 `run_cifar_six_630.py` / `run_cifar_six_from_scratch.py` 后需执行上述补报命令。没有修改底层训练入口、算法或已冻结的 manifest，所以已有断点仍保持相同源码身份。

报告异常单独输出 `REPORT_FAILED` 和补报命令；包装入口将状态写入 `report_generation_status.json`，不会因此重训或触发 OOM 换卡。更新已在运行的旧包装脚本不会给该进程补上新收尾逻辑；待其训练完成后执行一次 `--report-only` 即可。

### CIFAR-10 旧 84 组验证流程（2026-09-15，v2，保留续跑）

代码更新统一通过 GitHub 同步，在 AI Station 执行 `git pull --ff-only origin main`；不再使用上传代码包的方式。

从零重训使用独立配置 `configs/cifar10_six_original_v2.json`，不依赖旧630组结果：

```bash
env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_original_v2.json \
  --devices auto --progress-mode live
```

`--devices auto` 自动发现容器内可见的 CUDA 卡，每卡同时运行一组独立实验；已有输出只选取与记录的型号、计算能力和显存容量兼容的卡。每张卡在独立进程中检查 CUDA 初始化及空闲显存，默认至少需要 4096 MiB 空闲（显示入口参数 `--min-free-gpu-memory-mib` 可调整）；不满足条件的卡会列明原因并排除，显式指定的卡首次预检不满足则报错。这个阈值是资源准入条件，不修改 batch、模型、训练或攻击参数，也不保证训练全程没有显存竞争。上述 `env -u` 解除先前手动设置的 `CUDA_VISIBLE_DEVICES=7` 限制，不改变容器的 GPU 分配。

启动时打印实际选卡清单；动态进度显示当前阶段完成组数、轮次进度、耗时、预计剩余时间及各卡状态。多卡同时打印时，显示入口先拆分拼接的日志，再把 `ROUND` 合并到进度条，不逐轮追加到终端。逐轮原始输出仍保存在 `progress_display_logs/` 和各任务的 `worker.log`，真实异常保留。终端重定向到普通日志时可用 `--progress-mode log`，这种模式按间隔追加摘要。

如果训练期间出现 CUDA OOM，显示入口会输出 `RESOURCE_OOM`，中止该次调度并保留断点，明确报告排除的设备，再预检其余已选设备、从同一输出目录续跑。每张 OOM 卡在本次启动中最多排除一次；全部卡不可用时以错误状态退出，用户主动按 `Ctrl+C` 不会触发自动重启。正式阶段的资源重试使用 `--phase final`，跳过验证重跑，保留冻结的候选。已有成功快照复用，失败记录保留；资源失败不代表算法在该场景下验证失败，`settled` 包含失败任务，不能作为有效实验完成数。

这层显示包装调用原 `run_cifar_six_from_scratch.py`，不改训练源码、配置或断点身份。原单卡任务切换到多卡时，先 `Ctrl+C` 并等待进程退出，再拉取更新、运行上述命令；完整任务复用，未完成任务从最近成功保存的轮次恢复。若仅使用指定卡，可设置 `CUDA_VISIBLE_DEVICES=6,7` 并继续使用 `--devices auto`。

先运行84组、每组100轮验证；只有Ours验证健康才控制是否自动进入180组正式评估。ASR相差1个百分点等性能目标只用于报告，不再阻断。基线没有健康候选时使用预先声明的固定备用候选，并明确保留验证失败；单任务异常不取消其他任务。六方案结果可以包含数值失败行，不能把缺失指标伪装成正常准确率或ASR。

本入口保留原训练/攻击优化器和基线实现，不启用上一版的步长回退、VERT数值修复或强制确定性覆盖。Ours原策略与弱告警隔离变体按配置显式比较。NaN是该实现及环境下的实验现象，不能单独证明原论文算法普遍不健壮。

相同命令支持断点续跑，输出位于 `outputs/cifar10_six_original_v2/`。协议、Ours健康门槛、未评估参考和复现边界见 本地文档《CIFAR六方案从零重训》（`docs/CIFAR六方案从零重训-2026-09-15.md`，不随 Git 同步）。MNIST继续使用原入口。

#### 验证通过后提示 `final_plan.json` 身份冲突

`VALIDATION_STATUS qualified_for_final` 表示 Ours 满足推进条件，不表示所有基线候选健康。若随后提示 `immutable experiment identity changed: .../final_plan.json`，本次正式阶段尚未启动；已有正式计划与这次验证推导出的计划不同。旧运行可能在部分验证任务资源失败时使用备用候选，补齐验证后选择发生变化，具体差异须读取计划确认。不能删除旧计划或绕过身份检查后直接混用结果。

`prepare_cifar_final_recovery.py` 默认只读审计，核对源码、配置、任务与快照身份，并用原规则重新计算候选选择。提供 `--apply` 时，在全新的输出目录复制经核验的验证缓存和环境记录，记录来源与计划差异；旧输出及全部旧正式产物保留。新目录不复制正式计划或正式结果，正式阶段按当前验证选择重新开始，验证无需重训。

本次进程已退出、代码经 GitHub 更新后，可执行：

```bash
python -u prepare_cifar_final_recovery.py \
  --source outputs/cifar10_six_original_v2 \
  --destination outputs/cifar10_six_original_v2_recovered \
  --apply && \
env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_original_v2.json \
  --output outputs/cifar10_six_original_v2_recovered \
  --phase final --devices auto --progress-mode live
```

目标目录必须尚不存在；恢复创建成功后再中断训练，只需执行第二条训练命令续跑，不能再次向同一目录执行准备操作。审计发现来源损坏、身份不一致、验证仍待运行或 Ours 未通过健康检查时会拒绝准备，不通过复制绕过原验证政策。恢复依据固定验证规则，不根据旧正式结果选择参数；旧正式结果与本次新计划的结果应分别报告，保留 `recovery_manifest.json` 的来源记录。

### CIFAR-10 旧v7协议（保留用于追溯）

旧 CIFAR 配置为 `configs/fair_tuning.cifar10.json`，MNIST 配置为 `configs/fair_tuning.example.json`：

```bash
python -u run_fair_tuning_from_config.py --config configs/fair_tuning.cifar10.json --dry-run
python -u run_fair_tuning_from_config.py --config configs/fair_tuning.cifar10.json
```

CIFAR 配置为 100 客户端、100 轮、每轮本地 1 epoch、batch 50、K20、第25轮攻击；三个可调方法各6候选，共630次验证和180次正式训练。目标为准确率落后VERT不超过0.5个百分点、ASR高于VERT不超过1个百分点，并检查绝对ASR与峰值；目标未达到仍报告 `unmet`。这不是已验证的双指标保证。

可先用 `python -u run_attack_screen.py --config configs/attack_screen.cifar10.json --clean-only` 检查训练留出的干净 FedAvg 收敛；去掉 `--clean-only` 才是可选的 Ours/VERT 攻击参数探索。这个探索脚本默认串行、模拟密码，正式六方案使用自动GPU调度与SM9。已占满资源的 MNIST 作业完成后再启动 CIFAR，或明确分配独立空闲GPU。

完整数据划分、CIFAR候选网格、实测限制与启动步骤见 本地文档《CIFAR-10六方案实验说明》（`docs/CIFAR-10六方案实验说明-2026-09-07.md`，不随 Git 同步）。MNIST 使用 compact CNN；CIFAR-10 使用 Conv-Conv-FC-FC-Logits CNN 和按通道标准化。K、总轮数、lr、batch 等属于公共实验条件，六种方法保持一致。

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

2026-09-10原论文复核发现，现有TAD在输入使用增量、固定异常比例配额、权重归一化及移除判据方面存在差异或论文歧义，不能称为严格原文复现。论文第10页承认半数恶意节点、特别是大量串通时可能面临困难，但没有证明50%是失效阈值。详见 本地文档《TAD原论文与实现复核》（`docs/TAD原论文与实现复核-2026-09-10.md`，不随 Git 同步）。本次仅修复选参阶段的失败对照放行和报告，未改TAD算法，因此不使旧结果变成新的论文复现实验。

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


## Ours / VERT 攻击参数探索

```bash
python -u run_attack_screen.py --config configs/attack_screen.local.json
python -u run_attack_screen.py --config configs/attack_screen.full.json
```

这是单独的探索流程，不使用官方测试集选参。local 使用 10,000 原始训练样本，按分层取整分为 8,998 本地训练、501 验证和 501 攻击辅助，20 客户端、50 轮、CPU、模拟密码；full 使用 50,000 原始训练样本、100 客户端、128 个攻击/评价目标、真实 SM9。探索入口支持 jobs 个独立进程，local/intermediate 配置为 2 个 CPU 进程；full 默认串行，完整六方法的 8 GPU 并行仍使用 `run_fair_tuning_from_config.py`。

预先列出的攻击为 boost=3/15/30，其中 boost=3 使用 stealth_steps=3、distance_weight=.001，其余为 1/.0001；它们是三种联合攻击条件，不能把差异全归因于 boost。每种条件对两种防御相同。Ours 比较同参数的 C_tol=3/1，VERT 比较历史窗口 5/7，各 2 个候选；每个方法在全部探索场景上选一套参数，固定后跑独立确认 seed。效用为攻击期平均准确率与 1-ASR 各 .5，同时要求相对干净 FedAvg 下降 <=.03 和健康门槛。所有失败、完整逐轮记录和客户端诊断保存在带源码指纹的输出目录；不只展示有利攻击强度。确认 seed 仍是探索验证，不是六方法正式结果，不能据此声称普遍超过 VERT。

新版正式配置保留原攻击 boost=15、50 轮作为可比较参照，校准 seed 改为 71/72/73，正式 seed 改为 241/242/243；不会自动把探索最有利的攻击覆盖为唯一正式场景。

中等强度补充探索：`configs/attack_screen.intermediate.json` 使用 boost=5/8、stealth_steps=1、distance_weight=.0001、80% 恶意，探索 seed=84，确认 seed=85/86。`configs/attack_screen.100client_probe.json` 是 50,000 样本、100 客户端、Dirichlet、40% 恶意、单 seed 的规模核查，CPU/模拟密码，不是独立正式实验。


本轮实测见 本地文档《实验改进与参数筛选-2026-09-06》（`docs/实验改进与参数筛选-2026-09-06.md`，不随 Git 同步）。共完成181个实验任务（含提前终止失败）及279项测试。部分20客户端弱攻击场景出现双指标数值优势，但C_tol=1的独立干净确认误撤销10%–20%，且有全员退出，未通过确认健康检查；100客户端C_tol=3抽查也没有复现优势。不能将这些结果概括为稳定超过VERT。`confirmation_audit.json`记录确认状态；`comparison.json`的`strictly_better_both`仅表示数值领先，`validated_better_both`还要求双方的完整确认场景通过健康和干净精度检查。使用`--report-dir 已有运行目录`可从原始实验JSON重建这两份报告，不重新训练。

2026-09-07 复核与新结果见 本地文档《代码复核与VERT参数探索》（`docs/代码复核与VERT参数探索-2026-09-07.md`，不随 Git 同步）。282项完整测试通过，筛选器新增缓存配置身份和完整确认矩阵检查，脚本本身加入指纹；无攻击期不会生成伪造的攻击尾段指标。固定原C_tol=1的100客户端对照仍领先原VERT配置，但干净误撤销7%–12%，未全部通过健康门槛。更关键的是，20客户端下将VERT预测训练从5增至20次后，原四个弱攻击运行ASR全部为0，Ours原双指标优势消失。后续方案B预测训练网格改为5/20次，仍为12候选；完整规模筛选也包含20次候选。新配置不代表正式实验已运行。

补充实验可运行：

```bash
python -u run_attack_screen.py --config configs/attack_screen.scale100_data10000.json
python -u run_attack_screen.py --config configs/attack_screen.scale100_data50000.json
python -u run_attack_screen.py --config configs/attack_screen.vert_epoch_probe.json
python -u run_attack_screen.py --config configs/attack_screen.K10_search.json
python -u run_attack_screen.py --config configs/attack_screen.C2_search.json
```

K10搜索虽在部分确认场景双指标领先并通过现有健康门槛，但ASR仍很高；`validated_better_both`没有约束绝对ASR，不能把它等同于模型已得到充分保护。完整失败结果与C_tol=2对照均在新报告中披露。
