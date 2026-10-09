# SM9-RRS-FL 实验复现

文档同步约定：仅根目录 `README.md` 随 Git 同步；其余 Markdown 为本地工作文档。下文标注的本地参考不包含在 Git 克隆中，运行所需说明以本 README 和仓库配置为准。

六种方法：Ours（内部名 `sm9rrs`）、VERT、AlignIns、Krum、TAD（`ding13`）、FedAvg。原入口支持 MNIST/CIFAR-10、NumPy/PyTorch；新增独立Fashion-MNIST入口使用PyTorch。支持IID/Dirichlet Non-IID，以及真实 SM9 或快速仿真密码模式。

## 2026-10-09最新：首次分歧在单样本尾批梯度，进行局部确定性对照

三个局部探针已完整回传：同一GPU、每次执行原初始化/round0评估/客户端0–84，均在聚合前停止。
本地HEAD为482f0fe，原9文件已提交；回传78源码map与本地一致，但附件没有远端Git日志，不推断远端HEAD。
三组两两比较均首先在client-19的第15批（索引14、1个样本）发现梯度差，client-84的第7批（索引6、1个样本）也相同。
此前批次、实际输入索引/特征/标签、前向参数、logits和loss均一致；旧上下文及非目标更新复现。

- 差异只涉及`conv1_w`/`conv1_b`；其余8个参数块没有数值差异。
- 最大全梯度绝对差约`2.8983e-6`，相对L2最大约`6.5724e-7`；最终客户端更新最大绝对差约`1.4156e-7`。
- 这是首次**已观测**反向梯度边界，尚非具体CUDA算子的证明；上游输入梯度也可能传播至conv1。
  这组短探针不能证明微小扰动造成了Ours长期性能下降，更不能据此判定防御已修好。

新增独立入口`run_cifar_tail_determinism.py`，预定六个fresh子进程按A1/B1/A2/B2/A3/B3串行：
A保留原设置，B仅在client-19和84两个单样本尾批的原`Tensor.backward`调用期间临时设置
`torch.backends.cudnn.deterministic=True`，完成后立即恢复。A/B均使用同一观测器与flags核验。
不改forward、数据/批次、SGD、TF32、cuDNN enabled/benchmark或全局deterministic_algorithms。
这是显式数值策略诊断变体，不是对历史实验或公共训练协议的静默修改。
[PyTorch 2.3卷积反向源码](https://github.com/pytorch/pytorch/blob/v2.3.0/aten/src/ATen/native/Convolution.cpp#L1877-L1943)
在backward时读取确定性设置；远端NVIDIA定制构建仍需实测，不能用上游源码或flag getter替代具体内核证据。

每个进程仍保留完整原config（horizon=3），执行第1轮0–84号客户端、在85入口停止，
共510次客户端训练调用、0个完整训练轮、无聚合。每worker记录两次作用域前/内/后flags，其他边界校验原flags；
末批公开数值快照约56MB/worker、六份约340MB，不含SM9秘密或检查点。
保留原78源码，四个新模块另冻结为82来源；原3个step、6个prefix、90项历史引用均只读。
新输出为`outputs/cifar_v8_diagnostic_v1/tail_determinism_v1`，不覆盖任何旧目录。
固定原GPU UUID，每次准入仍要求空闲显存≥16384MiB、利用率≤5%，每10秒查询、最多等600秒。
完成项复用，失败保留并停止；检查失败后才显式`--retry-failed`启动新的fresh attempt。

用户自行提交推送README、4个tail_determinism模块及4个对应测试，共9文件后，在AI Station运行：

```bash
cd /3251901002/SM9RRSFL
git pull
python run_cifar_tail_determinism.py
```

完成或停止后运行：

```bash
python run_cifar_tail_determinism.py --summary > /tmp/cifar_tail_determinism_summary.txt
cat /tmp/cifar_tail_determinism_summary.txt
```

回传完整`CIFAR_TAIL_DETERMINISM_BEGIN`至`CIFAR_TAIL_DETERMINISM_END`，失败时另附控制台报错和提示的worker.log末尾。
`--summary`仅读取、不查询GPU/加载数据/训练/写实验目录；`--plan-only`只显示计划。
`--step-output`可指定已完成局部探针路径，`--output`指定独立新目录，`--data-dir`指定原数据位置。
续跑不能改变冻结引用；`--wait-seconds`0–600、`--poll-seconds`大于0且≤60只调整等待时间。

判读按每个目标客户端分别进行：先核全部干预前边界、旧上下文与非目标更新，再比较组内三对及跨组三对。
原A复现同边界差异、B三次一致且干预前状态相同，才支持当前环境下局部策略消除所观测差异；
A也一致时标为本批未复现，B仍不同则标为该策略不足，干预前不同时不作局部因果解释。
即使成功也只验证这两次backward，之后仍须逐批决定短训练复现与防御机制对照；不自动推进NoPermanent、TPE或150轮。
新增观测/同步可能改变时序，三次一致也不是长期确定性保证。

本地55项新增＋22项原观测/报告回归，共77项通过。覆盖真实小型CPU双卷积十参数CNN的原分支逐位一致与RNG不变、
仅两次原backward开关生效/所有forward保持原flags、异常恢复、原99项序列化引用链、坏快照及篡改policy拒绝发布/复用、
六fresh交错同UUID控制，以及独立客户端判读/只读报告。旧78源码逐SHA不变。
这些测试不是实际CIFAR/GPU运行；本批六项仍待AI Station实测。

## 2026-10-09历史步骤：首轮差异仅在两名单样本尾批客户端，进入逐批次定位

本节三项局部探针已完成，结果及当前命令见最上方；无需重跑本节。


用户已提交上一批3文件，AI Station pull到474bb3e，本地主项目同HEAD且本轮起始干净。
只读客户端明细217条JSON完整，200客户端、六组首轮配对、全部分母及来源校验通过，六份原artifact身份未变。

| 第1轮观测 | IID | Dirichlet |
| --- | --- | --- |
| 三次已记录输入 | 全100人一致 | 全100人一致 |
| 三次平均本地loss | 全100人一致 | 全100人一致 |
| 任意一对更新不同的客户端 | 0/100 | 仅client-19、client-84，2/100 |
| 最后一批恰为1个样本 | 0人 | 仅19（701样本）、84（301样本） |

Dirichlet三组两两比较均只有这两人更新不同，其他98人均一致；尾批1组为2/2不同，其他批次形状合计0/98不同。
本地真实标签重算与远端200人的样本/批次大小一致；此次回传不含远端分区索引SHA，不能扩大为本地和远端索引逐位核验。
第2轮开始已使用不同全局模型，后续大范围差异需与首轮定位分开。平均loss一致与最后一次反向/更新才分歧相容，
但尚未证明最后批次、梯度或某个CUDA算子就是原因；更新SHA也不能量化误差。

新增独立入口`run_cifar_client_step_probe.py`：三次fresh进程、原物理GPU UUID串行，保持原Dirichlet H0配置、
初始化和round0评估，执行第1轮客户端0–84，在85开始训练前停止；共255次客户端训练调用，不完成聚合或一个完整训练轮。
仅目标19/84逐批次细查：实际执行索引、特征/标签、进入批次的参数、原forward输出/loss、原backward梯度、
原SGD后的参数，以及最终客户端更新。使用原训练函数，不重写SGD、不重新forward，不改数值flags或随机数。
每批次保留指纹；两目标末批和最终delta额外保存公开数值数组，用于max-absolute/relative-L2/不等元素比例比较。
其他批次没有完整数值快照时只定位指纹边界，不补造误差幅度。三份快照按当前CNN大小合计约170MB，不含SM9秘密或检查点。

新增观测会增加CPU读回/同步，可能扰动执行时序；报告另核旧前缀上下文及非目标客户端更新。
如原差异不再复现，只能说明当前观测条件下未复现，不能据此宣称问题解决。
不设置deterministic、不关闭TF32、不丢弃尾批、不改变baseline/public训练参数。
此批不评估防御健康/性能资格，也不推进NoPermanent、TPE或150轮实验。

新输出独立为`outputs/cifar_v8_diagnostic_v1/client_step_probe_v1`，原六项prefix及90项引用只读。
原72源码不改；新协议独立冻结78来源（增加reader、GPU等待适配器和4个step模块）。
固定原UUID，显存仍≥16GiB、利用率≤5%，每次准入最多等待600秒；无自动换卡/降低门槛。
完成项复用，失败attempt保留且默认停；只有检查失败原因后显式`--retry-failed`才从头运行新attempt。

用户自行提交推送README、4个step模块和4个对应测试共9文件后，在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
python run_cifar_client_step_probe.py
```

执行完成或停止后：

```bash
python run_cifar_client_step_probe.py --summary > /tmp/cifar_client_step_probe_summary.txt
cat /tmp/cifar_client_step_probe_summary.txt
```

回传完整`CIFAR_CLIENT_STEP_PROBE_BEGIN`至`CIFAR_CLIENT_STEP_PROBE_END`；失败时附控制台报错和提示的worker.log末尾。
`--summary`只读、无GPU查询/数据加载；`--plan-only`只显示计划。`--prefix-output`和`--output`可指定原引用及新输出位置，
但两者及原五个参考目录必须分离，续跑不能改变已冻结引用；可用`--data-dir`指定同一数据。
`--wait-seconds`范围0–600，`--poll-seconds`大于0且≤60，均不改变准入阈值。

本地46项新增和22项原观测/等待回归共68项通过；包含真实小型CPU十参数CNN的有/无观测更新一致、
随机数状态不变、实际批次捕获、异常恢复，以及原96任务引用链、坏快照拒绝复用和同UUID控制流程。
旧72源码逐SHA不变；这些检查不是实际CIFAR/GPU执行，三次远端局部探针仍待运行和回传。

## 2026-10-09历史步骤：六项短探针完成，只读定位首轮客户端差异

本节只读结果已回传，当前按最上方逐批次探针操作，无需重新运行本节诊断。

新回传已确认6/6完成，旧两项完成记录保留，72来源、原90项引用链及摘要读取前后证据核验通过。
本地主项目HEAD为5b0e683；此次附件只有摘要，没有远端Git或续跑控制日志，因此不据此推断远端HEAD或适配器执行过程。

- IID三次的全部已观测边界一致，与旧H0第0–3轮的科学标量也一致。
- Dirichlet两组比较（repeat2/3分别对repeat1）的首次观测差异都是第1轮`client-19`返回的更新SHA，三次SHA各不相同。
- 这早于检测、聚合和第25轮历史冻结；固定同一物理GPU仍未得到逐位一致结果。
- Dirichlet三次第2轮准确率为23.28%、23.16%、23.12%，旧H0为22.92%。准确率变化是后续观测，不能定位具体计算算子。

更新SHA不同不能说明误差大小，也不能把三轮合计的202/177个更新字段差异当作第1轮异常客户端数。
原观测中的epoch索引是独立同seed重建，尚非实际执行批次索引。本轮不据此认定cuDNN、TF32或某个算子为根因。

本地按真实CIFAR标签与原split/partition函数只读重算：Dirichlet的client-19为701个样本（14×50＋1），
client-84为301个样本（6×50＋1），只有这两名客户端的最后一批为1；IID全部为450个样本（9×50）。
这是本地标签层面的定位线索，尚未与远端完整快照逐项核对，不能证明差异来自尾批，也不能推断client-84已经发生差异。

新增独立只读入口`diagnose_cifar_prefix_clients.py`，从现有六份完整快照提取：

- IID/Dirichlet各100个第1轮客户端的样本数、批次大小、三次更新SHA与原始平均loss，以及已记录输入的一致性。
- 三组两两比较（1/2、1/3、2/3）的不同更新/损失客户端清单；按最后一批大小统计所有客户端的分母和差异数。
- 第2、3轮单独汇总，用于观察差异传播，不混作新的独立起因。

该入口复用原完整性、来源和环境审计，缺项、坏证据或读取过程中证据变化时拒绝生成成功明细。
不查询GPU、不加载数据或检查点、不训练、不写实验目录；原72源码、manifest和六项结果均保留。
结果只用于确定下一次局部计算探针的对象，暂不继续NoPermanent、TPE或150轮训练。

用户自行提交推送本批README、新入口和对应测试共3文件后，在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
python diagnose_cifar_prefix_clients.py > /tmp/cifar_prefix_clients.txt
```

执行结束后：

```bash
cat /tmp/cifar_prefix_clients.txt
```

回传完整`CIFAR_PREFIX_CLIENTS_BEGIN`到`CIFAR_PREFIX_CLIENTS_END`。这一步仅读取已经完成的结果，无需重新运行六项探针。
可用`--output`指定同一已完成probe的目录；不通过更换目录或修改旧证据绕过完整性检查。

本地14项新增与26项原报告/协议回归共40项通过，冻结72来源逐SHA与HEAD及远端摘要一致。
验证使用合成快照和真实序列化引用链，不是新CIFAR训练；远端实际客户端明细仍待上述命令回传。

## 2026-10-09历史步骤：两项短探针完成，等待原GPU并续跑剩余四项

本节是2/6时的恢复记录；最新六项已完成，当前只执行上节只读命令，无需再次续跑。

AI Station与本地均已同步至86c85bc。短探针2/6完成，两个IID任务各3轮/300次客户端观测完整，
两次所有已观测边界完全一致；两次均与旧H0的第0–3轮标量记录相同。
IID第三次和三项Dirichlet尚未启动，本次还不能判断Dirichlet训练差异的源头。
原90项引用链、72来源和摘要读取前后证据核验通过，完成项无需重训。

中断发生在第三项START之前：原控制器单次筛选固定GPU未通过后立即退出。
当时没有保存free/util快照，无法确定是哪项条件未满足，不能断言OOM、算法失败或其他进程抢占。
错误文字“no training started”只适用于这次未启动的下一项，不表示已完成两项没有运行。

新增独立 `resume_cifar_prefix_probe.py`，只在父控制器中等待原准入条件：

- 固定原manifest中的同一GPU UUID，空闲显存仍≥16384MiB、利用率仍≤5%；仅显存/利用率暂不合格时等待。
- 默认每10秒采样，单次准入最多等待600秒；打印实际UUID、显存、利用率和未通过条件。
- UUID缺失、型号或可见性掩码不符、坏库存或查询错误明确停止，不自动换卡或放宽条件。
- 超时保留结果并输出原摘要；达到期限后即使迟到的采样变为可用，也不启动该worker。
- 原72来源、manifest、参数、已完成结果均不改；原worker入口、串行执行和失败检查沿用。
  完成项直接REUSE；该次未启动任务不需要 `--retry-failed`，本适配器不自动重试失败worker。

适配器先核验既有研究及完成证据，不创建新的训练计划；新控制侧记录写入原输出的
`controller_resumes/`，独立保存适配器SHA和等待采样，不覆盖科学文件。
程序设置和恢复仅限父进程准入处理，不修改训练的确定性、TF32或随机数设置。

用户提交推送README、新续跑入口和对应测试共3文件后，在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
python resume_cifar_prefix_probe.py
```

它会复用已有两项，只补剩余四项，共12轮训练。等待超时也不要删除已有结果。
执行结束后（含等待超时），输入：

```bash
python run_cifar_prefix_probe.py --summary > /tmp/cifar_prefix_probe_summary.txt
cat /tmp/cifar_prefix_probe_summary.txt
```

回传完整 `CIFAR_PREFIX_PROBE_BEGIN` 到 `CIFAR_PREFIX_PROBE_END`；如果仍中断，附最后几条
`PREFIX_RESUME` 的等待/超时/错误行。`resume_cifar_prefix_probe.py --summary`也可只读输出同一摘要。
适配器可接受 `--output`和`--data-dir`，原五个参考目录自动从manifest读取。
`--wait-seconds`允许0–600（0为不等待），`--poll-seconds`允许大于0且不超过60；两者均不改变GPU准入阈值。

本地14项新增与38项原prefix回归共52项通过，冻结72来源SHA与86c85bc及远端回传一致；
适配器尚待AI Station实际续跑，不能把本地模拟worker检查当成剩余四项已经完成。

## 2026-10-09历史步骤：机制取证与同卡三轮重复性探针

本节保留探针设计；最新六项已完成，当前操作以最上方只读定位步骤为准。

AI Station已同步至9dc52ca，上一轮5文件的只读取证完整通过：12项机制与12个前缀配对完整，
原始90项引用链/66源码与读取前后证据核验通过；未训练或改写原输出。

现在可以确认：H0/H1诚实永久撤销总数为19/137，全部由累计可疑计数达到阈值触发；撤销当轮全部为
warning越界，没有诚实客户端由strong单次异常即时撤销，诚实更新的drift越界总数也为0。
H1 Dir clean的49次误撤销发生在52–150轮；Dir70的25次发生在32–58轮。
这指向“正常更新反复触发warning，随后被永久删除”的失效路径，而非仅调小κ/h就能修复。
同时恶意更新仍可低分通过：H1 Dir70最后30轮接纳267/270个恶意在线更新。关闭永久撤销即使能减少误伤，
也不自动解决这类漏检，因此本批尚不实施NoPermanent。

重复性仍需定位：IID的所有前24轮客户端诊断一致；Dir三种比例的准确率均从第2轮出现差异，
第21轮开始检测分数和聚类数量不同，随后接纳/权重/历史准入改变，早于第25轮冻结。
历史环境deterministic/cudnn_deterministic=false、TF32=true，配对分配在不同逻辑GPU且未记录UUID；
这些是待检验因素，不能直接宣称CUDA是根因。

新增独立 `run_cifar_prefix_probe.py`：IID/Dirichlet clean各3次新进程重复，每次3轮，共6任务/18训练轮。
只使用原H0 Ours、CNN E1、seed2026093001、100客户端、batch50、lr .05/decay .99、K20/start25。
任务保留原完整150轮配置作来源，新配置仅rounds改为3；不修改原数值flags、算法、数据、健康或正式门槛。
因为只到第3轮，不评估K20后的防御、完整健康或正式资格，亦不启动TPE、H1、NoPermanent或后续实验。

启动时只读审计原90项，在兼容且利用率≤5%、空闲显存≥16GiB的卡中选一张，以完整NVIDIA UUID固定；
六项串行，后续任务和续跑均不自动换卡。原数据/环境/源码契约核验不通过会阻止训练。
输出独立为 `outputs/cifar_v8_diagnostic_v1/prefix_probe_v1`，不复用旧模型或checkpoint。
完成任务复用；失败attempt和日志保留并停止，检查原因后才可显式 `--retry-failed` 生成新attempt。
这类三轮探针不从中间轮恢复，以保证每次比较都是独立新进程的相同前缀。

记录初始化、数据和客户端划分、每客户端输入模型/更新/索引和本地损失、实际聚合系数和顺序、聚合向量、
聚合后模型，以及原评估forward的logits/预测指纹；不导出模型向量或SM9秘密。
批次排列指纹由同一seed重建，明确不是新增训练RNG观测。指纹复制可能同步设备、扰动调度；
本次相等不能证明原多卡长实验逐位可复现，也不把检测到差异自动归因某个CUDA算子。

用户提交推送本批文件后，在AI Station运行：

```bash
cd /3251901002/SM9RRSFL
git pull
python run_cifar_prefix_probe.py --gpu auto
```

执行完后输入以下命令并回传 `CIFAR_PREFIX_PROBE_BEGIN` 到 `CIFAR_PREFIX_PROBE_END` 全部内容：

```bash
python run_cifar_prefix_probe.py --summary > /tmp/cifar_prefix_probe_summary.txt
cat /tmp/cifar_prefix_probe_summary.txt
```

`--summary`只读、不探测GPU或加载训练数据。`--plan-only`只显示6×3轮计划。
如无空闲兼容卡，命令会停止；可用 `--gpu GPU-完整UUID` 指定一张符合条件的卡。
外层 `CUDA_VISIBLE_DEVICES` 如果已设，须为完整UUID列表；数字掩码会明确拒绝，以免混淆物理与逻辑编号。
输出或原五个目录移位时传对应 `--output/--history-output/--threshold-output/--timing-output/--clean-output/--matched-output`。
六目录须分离，旧证据保留原样。失败时一并回传报错及提示的worker.log末尾，不自行删除重跑。

本地46项新增与29项旧取证回归共75项通过；真实小型CPU观测前后参数/逐轮结果/RNG一致，
原66来源及其中v8的49来源SHA不变。真实CIFAR/GPU探针尚待AI Station执行；合成6任务摘要13JSONL约6.6KB。

## 2026-10-09历史步骤：冻结历史结果与只读机制诊断

本节取证已完成，保留结果和复现命令；当前只需执行最上方只读定位步骤，无需再次取证或重跑12项。

12项历史对照已全部完成，11项健康；原66来源、参考链、数值环境与观测覆盖通过核验，manifest为
`3c6566877fe4ea96ec9a6d464828c6d4c050937e1b1cf7dea82b2eb4db640eed`。
H1第25–150轮共有56517条有效在线观测，历史准入全部0、观测缺口0，干预实际执行。

| 指标 | H0原Ours | H1冻结历史 |
| --- | --- | --- |
| 健康 | 6/6 | 5/6 |
| 六场景最终均Acc | 59.06% | 57.24% |
| 四攻击场景最终均Acc | 57.53% | 55.28% |
| 四攻击场景最终均ASR | 21.50% | 28.50% |
| clean Dirichlet诚实误撤销 | 5/100 | 49/100 |
| Dirichlet70 Acc / ASR | 47.80% / 52.00% | 41.36% / 82.50% |
| Dirichlet70诚实误撤销 | 10/30 | 25/30 |

H1唯一健康失败是Dirichlet clean误撤销49%，全部12项nonfinite为0；clean效用仍在3pp内不能抵消健康失败。
冻结未解决聚合漏入：IID70恶意全期接纳6917/7195→6723/7147，ASR仍15.5%，诚实误撤销2→7；
Dir70恶意权重均值.10330→.10584，诚实权重损失.39134→.82346。停止历史准入不等于停止恶意聚合。
本批不采用H1作为修复，不追加TPE，也暂不启动关闭永久撤销或表示变体。

需先分清重复性与机制：Dir clean/10%从第2轮已有H1−H0准确率差+0.28/−0.20pp，尚未开始第25轮冻结，
不能将最终Dir差距全部归因干预。IID70第24轮仅约1e-16权重摘要差，属于不同层级的证据。
原聚合后诊断对恶意客户端集合求和，CPU最小复现实测不同PYTHONHASHSEED能在相同系数下产生同量级末位差；
该统计不回流训练，不能解释Dir第2轮准确率差。真实更新聚合保持客户端插入顺序，未定位随机SM9标签改变聚合排序的路径。
不据环境相符或未强制确定性就断言CUDA为根因，不静默修改数值策略。

新增只读入口 `diagnose_cifar_history.py` 与纯统计模块 `cifar_history_forensics.py`，原66来源不改。
入口审计现有90项引用链，在内存中读取原始完成快照和记录的环境；不训练、不下载、不探测GPU、不打开checkpoint、
不修改任何实验文件，不创建新训练manifest。新reader自身另记SHA；12项history与24项threshold文件全程前后SHA复核，
上游54项按原嵌套审计检查。现场环境不替代历史环境，缺失字段不假装已记录。

诊断包括：

- H1/H0和H0/旧P0的逐字段前缀差异：首次精确差、首次超过显示阈值的差、最大差及轮次；同时报告每侧真实观测覆盖。
- 1e-12仅用于分开展示极小差，不四舍五入原值、不修改算法/健康容差；标量或诊断一致不等于模型、更新或RNG状态逐位相同。
- warning、漂移、二者共同触发及severe重叠计数，历史可准入与实际准入分开；永久撤销的即时/累计路径与逐轮FP/TP、最终黑名单交叉核对。
- 12任务均给汇总；重点6项（两组IID70、Dir clean、Dir70）补25–29及121–150轮的novelty/anchor/drift分布和最早一个诚实撤销案例的最后5次观测。该案例不是代表性抽样。

新增29项测试和43项既有历史实验回归共72项通过，原66来源及原v8的49项来源SHA保持不变。

用户提交推送本轮文件后，在AI Station只运行下面的取证命令，**本轮无需启动训练**：

```bash
cd /3251901002/SM9RRSFL
git pull
python diagnose_cifar_history.py > /tmp/cifar_history_forensics.txt
cat /tmp/cifar_history_forensics.txt
```

回传 `CIFAR_HISTORY_FORENSICS_BEGIN` 到 `CIFAR_HISTORY_FORENSICS_END` 全部内容；报错时附末尾异常。
stdout由shell写入/tmp，程序不把诊断存入旧实验目录。自定义路径沿用 `--output`、`--threshold-output`、
`--timing-output`、`--clean-output`、`--matched-output`，五目录保持分离，默认位置与已完成实验相同。
下一步按这批证据决定是否先做带更新指纹的短前缀重复性探针，或开展永久撤销单因素消融；不自动执行后续实验。

## 2026-10-08历史步骤：阈值面板结果与12项冻结历史对照

本节保留已完成12项的设计与复现命令；当前结论及下一步以上节为准，无需重跑。


第三阶段24项全部完成，23项健康，62项来源、参考链和数值环境核验通过；manifest为
`53b480049d8f18a81fb45f518dfbbe8a88cded441cf66f9256e8dda2f522e0ff`。
以下来自完整远端摘要；原78项任务的快照尚未导入本地，新入口会在AI Station启动时再做只读审计。

| 候选 | 健康任务 | 六场景均Acc | 四攻击均Acc | 四攻击均ASR |
| --- | --- | --- | --- | --- |
| P0：原014 | 6/6 | 59.033% | 57.58% | 20.875% |
| P1：提高warning | 6/6 | 58.807% | 57.24% | 22.625% |
| P2：增强漂移检测 | 5/6 | 58.460% | 56.84% | 23.500% |
| P3：另一折中点 | 6/6 | 58.620% | 56.91% | 23.875% |

唯一健康失败是P2 Dirichlet clean误撤销11/100，全部24项非有限更新为0。clean效用在3pp内并不抵消误撤销健康失败。
P1–P3均未在整体最终Acc/ASR上优于P0，本轮暂停原计划后4个TPE提案，转入阶段四的第一个机制对照。
四点不足以证明整个阈值空间无解；这里是调整下一批投入顺序，不声称已完成8候选搜索。

P0并未修好：四攻击任务首轮接纳7/160个恶意在线更新，全期接纳8104/8806、历史准入6870次；
Dirichlet70最终Acc47.68%、ASR50.50%，诚实误撤销9/30。P1/P2/P3在该场景ASR为58/61.5/63%，
诚实误撤销18/30、15/30、19/30。历史准入次数不是历史中恶意样本的占比，也不足以证明历史更新是唯一原因。
P2虽降低部分恶意输入，仍同时伤害诚实信息，不能只以恶意接纳数判成功。

P0与旧C的IID最终指标和计数一致，权重摘要有约1e-16末位差；Dirichlet最终Acc差最大0.4pp、攻击ASR差最大2.5pp，
没有健康变号。单次重复不构成噪声范围或置信区间，数值路径差异原因仍未定位。
为观察同期重复性，下一批保留六项原Ours真实重跑，不复制旧P0/C结果。

新增独立入口 `run_cifar_cnn_history_panel.py`，配置 `configs/cifar10_cnn_history_panel_v1.json`，
协议 `cifar-cnn-history-ablation-v1`，输出 `outputs/cifar_v8_diagnostic_v1/history_cnn_e1_k20`。
每组IID/Dirichlet(.5)×恶意比例0/.1/.7，各6项，共12项：

| 组别 | 实现 | 变化 |
| --- | --- | --- |
| H0 | 原Ours | 全流程原样，作为本批新对照 |
| H1 | Ours-FrozenHistory-v1 | 从预定第25轮起停止动态历史准入与live normal重拟合 |

两组都固定CNN E1、lr .05/decay .99、K20/start25、150轮、100客户端/batch50、开发seed2026093001及P0原014全部参数；
攻击boost5/epochs1/stealth1/distance .0001、源5目标7/200目标，数据45k/2.5k/2.5k不变，官方test不参与选择。

H1保留**第24轮末的滚动历史和动态normal**，不是退回K20时的anchor；无攻击任务也从第25轮开始冻结。
anchor和norm_limit原本就不在预热后更新，仍按原规则。漂移、连续清洁/恢复计数、临时拒绝、权重惩罚/恢复、
SM9验证、永久撤销以及已撤销tag清理均保持原逻辑；被原聚合保护拒绝时仍按原规则重置clean_streak。
干预只作用于提交历史的阶段，不直接改变当轮已算出的聚合。相同前缀下第25轮决定应相同，后续模型才可能分歧。
这是利用公开预定时点的机制实验，不能据其效果声称已经解决未知攻击起点的部署问题。

新模块在独立worker的运行上下文内接入变体，结束/异常后恢复原方法，原62项来源文件完全不改。
H0/H1配置相同但变体与冻结时点进入新任务指纹，不得交换检查点；恢复时仍进入同一变体上下文。
原始诊断字段 `history_frozen` 仍表示权重管理器原有保护，不冒充本次干预标记；报告另核查H1第25–150轮
实际 `history_admitted` 全为false（clean也核查），缺字段、缺观测与实际非零准入不会当作干预成功。
报告还比较H0/H1攻击前科学轨迹以及H0对旧P0，保留差异、健康失败和实际分母。

本地43项新增测试和116项既有回归通过；旧62项来源及原v8的49项来源SHA保持不变。本批尚未远端训练。

本批不自动启动TPE、取消永久撤销变体、正式实验或后续阶段，不改变原正式健康和最终双≤2pp规则。
原24+4+26+24＝78项只读审计，保留P2等历史失败，不重训、不改旧输出。

用户在本地主项目提交推送后，在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
# 应显示12项、H0/H1、CNN E1、K20/start25、150轮
python run_cifar_cnn_history_panel.py --plan-only
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
python -u run_cifar_cnn_history_panel.py --devices auto --max-gpus 6
```

plan-only仅展示计划；实际启动审计原78项及环境后才创建新任务。自动选择当前可见且通过16GiB空闲显存准入的同型卡，
最多6张；准入并非独占或峰值保证。保留15秒进度、逐轮检查点、完成身份复用、中断清理及故障卡暂停派发。
同目录用同一命令续跑；数值失败保留，不自动改batch或删除证据。
若旧目录移动，启动和摘要均传相同的 `--threshold-output`、`--timing-output`、`--clean-output`、`--matched-output`；
自定义新目录用 `--output`，五目录须两两独立且互不包含。

完成或异常停止后，只读汇总并回传：

```bash
python run_cifar_cnn_history_panel.py --summary > /tmp/cifar_history_summary.txt
cat /tmp/cifar_history_summary.txt
```

复制 `CIFAR_HISTORY_BEGIN` 到 `CIFAR_HISTORY_END` 完整内容；如报错，再附末尾异常。
摘要不训练、不下载或修改实验文件；shell只写上述/tmp文本，控制器结束另存新目录 `history_summary.json`。
先核对完成、健康、冻结实现和重复性，再看最终Acc/ASR、诚实误伤及持续漏检。冻结改善将支持历史适配参与放大；
若clean显著受损或攻击未改善，不把H1直接替代原算法。下一批依据回传再决定，不预先承诺达到MNIST效果。

## 2026-10-08历史步骤：第二阶段完整结果与第三阶段24项固定参数对照

本节保留已完成批次的设计和复现命令；当前结果及下一批以上节为准，不需要重跑这24项。

紧凑回传已补齐：第二阶段26项全部完成150轮，21项健康，原24项clean、4项CNN配对及26项时点实验的
身份、来源和数值环境核验通过。原26项manifest为
`20837b56e9a8d35ed0922a0e2ce41c821a1fcbebf8f609f21e04a47bfc479a60`，58项原科学来源保持冻结。
此前“只收到22条完整TASK、全局状态待核验”已由本次完整回传覆盖，不需要重复训练第二阶段。

| 条件 | 健康任务 | clean误撤销：IID/Dirichlet | 四个攻击任务均Acc | 四个攻击任务均ASR |
| --- | --- | --- | --- | --- |
| A：K10、start12 | 3/6 | 14% / 31% | 56.49% | 13.625% |
| B：K10、start25 | 4/6 | 14% / 24% | 57.74% | 10.00% |
| C：K20、start25 | 6/6 | 0% / 2% | 57.53% | 21.75% |
| FedAvg start12 | 4/4 | 本批未新增clean | 57.40% | 16.75% |
| FedAvg start25 | 4/4 | 本批未新增clean | 57.65% | 20.00% |

五项健康失败为：A的IID/Dirichlet clean、B的IID/Dirichlet clean均因永久误撤销超过10%；
A的IID70因7次非有限更新失败。表中攻击均值保留这项已完整但不健康的结果，没有将它移除后重算有利均值。
整个26项只有这7次非有限更新，总计约15.399 worker小时，不是并行墙钟时长。
上述ASR均值只包含四个攻击任务，不混入clean背景；A的正确均值为13.625%。

**下一批以C作为健康开发条件，不将C称为最佳或正式合格。** C恢复了clean健康，但Dirichlet70的最终
Acc/ASR为47.80%/53.00%，相比B为−5/+46个百分点。C的四攻击任务首轮仅接纳7/160个已验证有限恶意更新，
全攻击期却接纳8266/8964、历史准入7081；其中IID70历史准入6076、Dirichlet70为1005。
因此下一步要检查持续漏检和历史污染，不能只凭首轮低接纳率判断防御成功。

clean A/B只改变攻击起点，记录的规范化环境一致。IID的151轮、所比较12项科学指标相同；
Dirichlet首次在第3轮出现目标类置信度差异（约.093634与.093961），最终Acc相差0.4个百分点。
该轮仍在共同K10可信预热期，无实际攻击，不能归因为延后攻击的收益；来源/环境相符也没有定位差异原因，
不能直接断言CUDA非确定性。保留该重复性问题，本批不静默调整PyTorch确定性标志、优化器或密码随机性。

新增独立入口 `run_cifar_cnn_threshold_panel.py`，配置 `configs/cifar10_cnn_threshold_panel_v1.json`，
协议 `cifar-cnn-fixed-threshold-panel-v1`，默认输出 `outputs/cifar_v8_diagnostic_v1/threshold_cnn_e1_k20`。
固定原CNN、E1、lr=.05、lr_decay=.99、K20、首攻击轮25、150轮、100客户端、batch50，
开发seed `2026093001`，每候选IID/Dirichlet(.5)×恶意比例0/.1/.7，共6项。
数据保持45k训练/2.5k校准/2.5k攻击辅助，官方test不参与开发选择。

| 候选 | warning | κ | h | 目的 |
| --- | --- | --- | --- | --- |
| P0 | 1.25 | 1.25 | 6 | 原014在C条件下重新运行，同期基准和重复性检查 |
| P1 | 1.50 | 1.25 | 6 | 仅提高瞬时阈值，观察误伤与漏检取舍 |
| P2 | 1.50 | .85 | 1.5 | 相对P1调整漂移，检查持续亚阈值偏离 |
| P3 | 1.75 | 1.00 | 2 | 第二组瞬时阈值与漂移折中 |

共 **4候选×6场景＝24项Ours**。除表中三个参数外，其他均沿用014：q2、clusters2、漂移记忆.8、
severe6、历史阈值1/确认2、恢复确认2、参考预算3.5、clip2、weight cap2、penalty.5、recovery1.25、remove_after5。
攻击固定boost5、attack_epochs1、stealth_steps1、distance_weight=.0001、source5/target7、target_count200。
本批不修改Ours核心实现、基线算法、公共优化器或原正式健康及最终双≤2pp规则，不运行其他数据集。

P0六项在新研究内全部真实训练；旧C六项仅为只读重复性参照，不能复制或改名充作本批结果。
原24＋4＋26＝54项任务均只读审计、不重训、不修补；新身份冻结参考来源、原始结果和数值环境。
原58项科学来源保持不变，新增模块不会改写旧manifest或重置旧预算。
**本入口只跑上述24项，不自动启动TPE、正式实验或下一阶段。** 收到结果后先比较P0与旧C，
再检查候选相对同期P0的健康、clean效用、最终Acc/ASR和全攻击期接纳/历史准入。
若P0的健康或关键机制明显漂移，应先核查重复性再继续选参；微小单seed改善不能当作因果或统计显著性结论。

仍由用户在本地主项目提交并推送代码，随后在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
# 仅展示计划，应显示24项、P0–P3、CNN E1、K20/start25、150轮
python run_cifar_cnn_threshold_panel.py --plan-only
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
# 自动选择通过初始化和空闲显存准入的同型卡，最多6张
python -u run_cifar_cnn_threshold_panel.py --devices auto --max-gpus 6
```

原目录不在默认位置时，启动和摘要均传相同的`--timing-output 原26项目录`、`--clean-output 原24项目录`、
`--matched-output 原4项目录`；自定义新目录用`--output`，四个目录必须互不包含。
`auto`使用当前CUDA可见逻辑卡，默认空闲显存准入16GiB；准入不是独占资源或峰值保证。
保留15秒任务/轮次进度、逐轮检查点及故障卡暂停派发；中断后用相同启动命令在本批新目录续跑。
已完成任务按身份复用，数值失败保留，不自动改batch或删目录重训。

运行完成或异常停止后，执行紧凑只读摘要：

```bash
python run_cifar_cnn_threshold_panel.py --summary > /tmp/cifar_threshold_summary.txt
cat /tmp/cifar_threshold_summary.txt
```

复制 `CIFAR_THRESHOLD_BEGIN` 至 `CIFAR_THRESHOLD_END` 的全部内容；命令报错时另附末尾错误。
摘要不下载数据、不探测GPU、不训练、不修补任何实验目录；shell只将回传文本写入上述临时文件。
报告保留24项逐场景结果、实际健康/完整数、同期P0对照及P0对旧C的重复性证据，
首1/5轮和全攻击期计数继续使用已验证有限在线观测分母，缺失不填0、不把背景混淆当攻击ASR。
控制器结束时另在本批新目录保存`threshold_summary.json`；启动尚未创建manifest就失败时，
摘要会明确尚无有效研究，应同时回传启动末尾错误。

## 2026-10-08历史步骤：第二阶段回传截断后的只读紧凑摘要

本节保留首次截断回传的处理过程；证据现已补齐，当前结论和下一批命令以上节为准。

最新回传只含22条完整TASK；开头的全局身份/来源/环境信息、2条C0参考及A组前4条任务未完整收到。
这不代表远端丢失或少跑4项；当前先补齐只读证据，不重训、不开始下一批搜索。
可见B/C全部6项中，B两clean因误撤销14%/24%失败，C降至0%/2%且6项健康；
但Dirichlet70的最终Acc/ASR由B的52.80%/7.00%变为C的47.80%/53.00%。
C该场景全攻击期恶意接纳1349/1669、历史准入1005，说明首轮少量漏入并不代表持续防御有效。
附件缺失部分及全局状态待下列命令核验；不能把可见22项统计当成26项总体结论。

新增独立只读入口 `summarize_cifar_timing.py`，复用冻结报告的实际快照核验，输出紧凑JSONL：
来源与三个研究身份、实际完整/健康数、C0参考、全部26项结果及首1/5轮与全攻击期机制计数。
窗口中恶意/诚实接纳和历史准入的分母仍为实际已验证有限在线更新，保留存活/缺观测原计数；
省去逐轮coverage长列表与重复解释，不舍入原指标，不以0代替缺失，不生成新赢家。
FedAvg客户端诊断保持不可用，微小浮点越界原值与冻结报告相同。

工具另审计两分区clean A/B的实际配置差异、规范化数值环境、round0指标和首个科学指标差异轮。
ratio0、同K10时，攻击起点没有已知改变训练/检测/RNG的路径；可见Dir clean有0.4pp差异，
不能解释为延后攻击的clean收益，也不能未经证据直接归因GPU非确定性。
该审计只比较科学指标，不比较密码随机字节或耗时；审计不推断原因。

用户提交并推送本批3文件（本README、独立入口及`tests/test_cifar_timing_brief.py`）后，在AI Station运行：

```bash
cd /3251901002/SM9RRSFL
git pull
python summarize_cifar_timing.py > /tmp/cifar_timing_brief.txt
cat /tmp/cifar_timing_brief.txt
```

复制 `CIFAR_TIMING_BRIEF_BEGIN` 至 `CIFAR_TIMING_BRIEF_END` 的完整内容；若命令报错，另附末尾错误。
工具不下载、不探测GPU、不训练、不修补、不修改任何实验输出；shell重定向只写上述临时文本。
默认原24、4及26项目录不变，自定义时仍传`--clean-output`、`--matched-output`、`--output`。
来源不符或无法审核明确保留错误，先根据回传解决；勿改manifest或删除实验目录。
原58项来源文件保持不变，新摘要入口不影响已有任务身份和断点续跑。

## 2026-10-08：CNN配对结果回传后的第二阶段 K/攻击时点诊断

当前下一批采用 **原CNN、lr=.05、local_epochs=1、lr_decay=.99、150轮**。
新增4项C3回传为完整健康、非有限更新0，摘要中的来源、参考和数值环境检查均通过。
两批clean结果如下，时间为每任务worker平均耗时：

| 设置 | 最终平均Acc | 最终校准交叉熵 | 平均耗时 |
| --- | --- | --- | --- |
| C0：CNN，E1 | 61.64% | 1.1055 | 2.45分钟 |
| C3：CNN，E2 | 61.67% | 1.2754 | 4.56分钟 |
| R3：ResNet，E2 | 61.60% | 1.0652 | 31.21分钟 |

同E2下，R3较C3均Acc低0.07个百分点，最差配对低2.96个百分点，耗时为C3约6.85倍；
增加CNN本地训练至E2只提高0.03个百分点，耗时增至约1.86倍，且C3校准交叉熵在100至150轮上升。
因此本阶段选择C0的CNN E1条件，先排查检测窗口和攻击时点。该选择属于开发阶段投入判断，
不是防攻击能力结论，也不将客户端局部训练loss直接解释为全局模型过拟合证据。
本地收到的是远端摘要；新入口会在AI Station从实际快照核验上述参考，不只依赖摘要文本。

新增独立入口 `run_cifar_timing_diagnostic.py`，配置 `configs/cifar10_cnn_timing_v1.json`，
协议 `cifar-cnn-timing-diagnostic-v1`，默认输出 `outputs/cifar_v8_diagnostic_v1/timing_cnn_e1`。
固定开发seed `2026093001`、IID/Dirichlet(.5)、100客户端、batch50；
数据保持45k训练/2.5k校准/2.5k攻击辅助，官方test不参与选择。
Ours采用原014参数；基线算法、检测实现、优化器、攻击强度和原正式双≤2pp门槛均不改。

| 方法/条件 | K | 首攻击轮 | 恶意比例 | 分区 | 任务数 |
| --- | --- | --- | --- | --- | --- |
| Ours014 A | 10 | 12 | 0/.1/.7 | IID、Dirichlet | 6 |
| Ours014 B | 10 | 25 | 0/.1/.7 | IID、Dirichlet | 6 |
| Ours014 C | 20 | 25 | 0/.1/.7 | IID、Dirichlet | 6 |
| FedAvg FA12 | 10 | 12 | .1/.7 | IID、Dirichlet | 4 |
| FedAvg FA25 | 10 | 25 | .1/.7 | IID、Dirichlet | 4 |

共 **18项Ours＋8项FedAvg＝26项**，每项150轮。恶意比例0时没有实际攻击，源5→目标7只是背景混淆。
clean A/B虽预期等价，仍分别运行并保留真实身份，不复制成绩充作新任务。
**A→B同时延后攻击并把攻击暴露从139轮降为126轮，不能仅凭ASR下降断言检测修复**；
需结合FedAvg FA12→FA25和攻击早期机制。B→C保持攻击起点与暴露长度相同，仅将K从10增至20。
这里只有一个开发seed，过程指标用于诊断，不替代后续完整验证或统计结论。

启动前只读审计原24项clean与4项CNN E2，包括冻结manifest、逐任务快照/身份、观测/成本SHA、
源码、数据契约和数值环境。两个原目录均不重训、不重写；参考核验不符时停止，不更改指纹绕过。
新任务继承参考的数值环境要求，允许同型GPU逻辑/物理编号改变；本入口不自动进入TPE、正式实验或下一阶段。

用户提交推送本批文件后，在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
# 只展示计划，不读旧结果、不下载数据、不探测GPU；应显示26项、CNN E1、150轮
python run_cifar_timing_diagnostic.py --plan-only
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
# 最多6张通过初始化的同型GPU，默认空闲显存准入16GiB
python -u run_cifar_timing_diagnostic.py --devices auto --max-gpus 6
```

旧结果不在默认目录时，分别增加 `--clean-output 原24项目录`、`--matched-output 原4项目录`；
新增输出可指定`--output`，三个目录必须互不包含。`auto`只选择当前`CUDA_VISIBLE_DEVICES`可见卡，
`cuda:N`为逻辑编号；显式卡可用`--devices cuda:0 cuda:1`。准入不等于独占显存或峰值保证。
每15秒显示任务/轮次；逐任务日志在`tasks/<task_id>/worker.log`。执行故障暂停该卡本次派发，
其余正常卡继续；数值失败保留证据，不盲目重试。中断后在原目录使用同一启动命令续跑，
已完成快照复用、逐轮检查点恢复；不删除输出，不向旧24/4项目录写入新任务。

运行结束或异常停止后，执行只读摘要并复制BEGIN至END的全部输出：

```bash
python run_cifar_timing_diagnostic.py --summary
```

标记为 `CIFAR_TIMING_BEGIN/END`。若使用自定义目录，摘要传入相同`--clean-output`、`--matched-output`、`--output`。
摘要不下载数据、不探测GPU、不训练、不修补任何目录；控制器结束时另写本阶段的 `timing_summary.json`。
启动在manifest创建前失败时，摘要只会说明尚无有效研究，请同时保留并回传启动末尾错误。

报告包含实际完整/健康数、50/100/150轮Acc、攻击最终ASR与clean背景混淆、累计误撤销、
首攻击轮及首5轮的聚合接纳/历史准入/权重、完整攻击期汇总、A/B/C配对变化和worker成本/峰值显存。
**客户端接纳率和历史准入率以实际有限且通过验证的在线更新观测为分母**，并列原始客户端数、
此前已撤销数及剩余但未观测数；不能把缺少诊断记录等同于检测拒绝，分母0记为不可计算。
history准入使用`history_admitted`；攻击前预指定恶意身份仍在诚实训练，不把其warmup历史称为攻击历史污染。
`malicious_weight_mass`是恶意客户端实际聚合系数之和，并非重新归一后的恶意占比。
诚实误撤销率使用原始诚实客户端数，累计计数不按轮求和。

clean Ours相对同seed/分区C0的最终Acc下降是否≤3个百分点作为**单列效用诊断**，
不新增健康门槛、不改变健康标签或正式资格。完整健康失败照实保留；缺失最终轮不填0、不沿用最后可见轮作150轮结果。
本阶段不套用clean FedAvg每轮45000样本的loss observer，因为攻击训练和撤销改变该观测口径；
直接读取原始RoundRecord/客户端诊断，不增加训练步骤。收到这26项结果后再判断下一批应做什么。

## 2026-10-08：第一阶段回传后的4项CNN E2配对补充

本批24项回传摘要显示全部完成且健康：C0最终均Acc61.64%、R0 51.66%、R3 61.60%；
R3将本地训练从1增至2个epoch，比R0提高9.94个百分点，但与C0公共训练量不同，不能直接完成模型选择。
新增独立 `run_cifar_matched_cnn.py`，只补 **C3＝旧CNN、lr=.05、local_epochs=2、lr_decay=.99**，
同IID/Dirichlet(.5)×两个开发seed2026093001/02，共4项、150轮。
每项公共配置完整复制对应R3，只更换模型和任务身份；沿用冻结的训练、loss观测、校准集和检查点实现。

新输出为 `outputs/cifar_v8_diagnostic_v1/cnn_e2_match`。原24项只读核验、不重训、不重新写入。
配置绑定本次原manifest `96cac3f9131837d9e21f7b4a1bda7775c35d8d60f8acbf5dc3937c46d4f8e3c5`，
启动前从实际快照重新检查24项完整健康、R3选择、源码、数据契约，并核对逐任务数值环境；
把参考快照、loss、身份、环境和成本记录的SHA冻结到新manifest。不会只相信缓存摘要或跨目录拷贝成绩。
新worker沿用原R3的数值库/设备类型要求，可选择同型GPU的其他逻辑/物理编号。

用户提交推送新文件后，在AI Station执行：

```bash
cd /3251901002/SM9RRSFL
git pull
# 只展示固定计划，不需要GPU/数据；应显示4项、150轮、C3、local_epochs=2
python run_cifar_matched_cnn.py --plan-only
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
# 默认参考原24项目录；最多4张同型卡，空闲显存准入仍16GiB
python -u run_cifar_matched_cnn.py --devices auto --max-gpus 4
```

原24项位于其他位置时，增加 `--reference-output 原24项目录`；自定义新增目录用`--output`。
两目录必须互不包含，不能将补充实验写入旧目录。计划展示中的`reference_audited=false`仅说明plan-only不读原结果；
真实启动和摘要均会做完整只读审计。参考不符时应保留错误证据，不修改指纹绕过检查。

运行结束或中断后，执行并复制BEGIN至END的全部输出：

```bash
python run_cifar_matched_cnn.py --summary
```

标记为 `CIFAR_CNN_MATCH_BEGIN/END`，包含原C0/R3的8项及新增C3的4项、实际完成/健康数、
50/100/150轮Acc、loss、背景5→7混淆、R3−C3逐配对差值、C3−C0变化和worker成本比。
若使用自定义目录，启动/摘要均传相同`--reference-output`和`--output`。
摘要不下载数据、不探测GPU、不训练、不修补任一目录；控制器正常结束时另保存 `matched_summary.json`。
源码、参考或环境变化、任务缺失/不健康时不给模型判断；未结束/缺失的成本记录不被当作精确耗时。

该补充沿用15秒进度、逐轮检查点、已完成复用和故障卡暂停派发。中断后可用相同启动命令恢复，
不用重跑旧24项。结果回来后，按同E2条件比较ResNet与CNN：ResNet均Acc至少高2pp且每项不低于CNN超过1pp，
再审阅成本是否值得继续。这仍是开发投入参考，不是正式资格或统计显著性结论；本入口不自动启动攻击、Ours或TPE。

## 2026-10-07：CIFAR 分阶段诊断，第一阶段 clean 学习对照

当前新增入口 `run_cifar_diagnostic.py` 只执行第一阶段：24项无攻击 FedAvg，
用于判断旧CNN与ResNet18-GN2的公共学习条件。独立协议 `cifar-v8-clean-diagnostic-v1`，
默认输出 `outputs/cifar_v8_diagnostic_v1/clean`，不复用或覆盖原v7/v8实验。
这批不是六方法正式实验，不自动启动Ours/TPE、第二阶段或模型回调。

| 设置 | 模型 | lr | local_epochs | lr_decay |
| --- | --- | --- | --- | --- |
| C0 | 原v7 CNN | .05 | 1 | .99 |
| R0 | ResNet18-GN2 | .05 | 1 | .99 |
| R1 | ResNet18-GN2 | .02 | 1 | .99 |
| R2 | ResNet18-GN2 | .10 | 1 | .99 |
| R3 | ResNet18-GN2 | .05 | 2 | .99 |
| R4 | ResNet18-GN2 | .05 | 1 | .995 |

每设置×IID/Dirichlet(.5)×开发seed `2026093001/2026093002`，均150轮、100客户端、
batch50、全量45k训练/2.5k校准/2.5k攻击辅助划分。沿用原实现的认证、优化器与初始化seed策略；
不加入数据增强、momentum或weight decay。开发评估只使用校准集，官方test不用于选择。
`malicious_ratio=0`，配置保留原攻击字段以复现背景源5→目标7的混淆，实际没有恶意客户端。

本地由用户提交推送后，AI Station在原项目目录操作：

```bash
cd /3251901002/SM9RRSFL
git pull
# 无数据下载、GPU探测或训练；应显示24项/150轮
python run_cifar_diagnostic.py --plan-only
# 当前物理卡占用只读核查
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
# 自动选择通过CUDA初始化、至少16GiB空闲、同型的显卡，最多6张
python -u run_cifar_diagnostic.py --devices auto --max-gpus 6
```

`auto`在当前`CUDA_VISIBLE_DEVICES`可见集合内选择；日志`cuda:N`是逻辑编号。
可通过外层`CUDA_VISIBLE_DEVICES=GPU-当前UUID,...`限定已确认可用的物理卡，再运行同一命令。
不要照搬历史卡号或UUID。若需显式逻辑卡，可用`--devices cuda:0 cuda:1`，显式卡未通过准入会拒绝启动。
16GiB为本阶段FedAvg的保守准入阈值，不是峰值保证或独占预留；运行中执行故障会暂停该卡本次派发，
其他正常卡继续。不会因OOM改batch、轮数或自动进入无限重试。无可用卡时保留输出并停止。

每15秒显示总进度及每卡任务/完成轮数；逐轮日志在`tasks/<task_id>/worker.log`。
Ctrl+C尝试先保存当前轮，最多等待30秒再终止；原样重跑启动命令可从最后持久化检查点恢复。
完整快照按身份复用，数值失败保留为失败证据；损坏检查点、缺失身份和源码改变不会悄悄重训。
成功任务先写完整结果，再完成检查点提交。完整结果已存在但缺诊断证据时，摘要会报错，需检查而非删目录重跑。

运行完成或异常停止后，执行以下只读回传命令，把BEGIN至END之间的全部内容发回：

```bash
python run_cifar_diagnostic.py --summary --output outputs/cifar_v8_diagnostic_v1/clean
```

摘要命令不下载数据、不探测GPU、不训练、不写报告或修补结果，默认读取实际快照而非缓存摘要。
正常结束还会保存`diagnostic_summary.json`；未完成/证据异常退出码为2，仍打印可复制摘要。
若在manifest创建前就报错，摘要只能显示尚无实验身份，需一并提供启动命令末尾报错。

记录第50/100/150轮Acc、最终背景混淆、loss、worker耗时、CUDA峰值已分配显存和nonfinite。
`local_train_loss`是客户端报告的minibatch训练loss按客户端样本数加权，不是最终全局模型在45k上的loss；
`calibration_loss`是全局模型在2.5k校准集上的交叉熵，复用原accuracy前向，不增加训练步骤或随机抽样。
每轮loss与模型一起进入检查点；观察文件保留完整曲线。worker耗时包含该次数据载入/训练/结果写出，
包含失败尝试；被强杀而无结束记录的尝试单独标记，不能把不完整耗时当精确总成本。

先按四任务最终Acc均值选择完整健康的ResNet设置；任何设置仍有执行缺失时不确定赢家。
若赢家不是R0，摘要建议下一步为CNN补同一公共设置4项，**不会自动启动**。
若R0胜出，可直接配对C0：平均至少高2个百分点、每个seed/分区不低于CNN超过1个百分点，
再结合实测成本决定是否保留ResNet。这只是开发投入参考，不是统计显著性或原六方法双≤2pp资格。
下一步需结合这批实测结果决定；本轮不改变旧协议、旧选择和MNIST/Fashion科学源码。

## 2026-10-07：v8已完成实验的独立报告恢复

如果60个正式worker均正常退出，但控制器最后在报告阶段报
`ReportIntegrityError: finished result contains invalid malicious_weight_mass`，
先检查实际值。冻结报告要求聚合权重诊断严格≤1，而原健康检查允许≤`1+1e-9`；
浮点求和产生的`1.0000000000000002`会因此被误拒。该报错本身不能证明越界只有浮点误差，
也不能证明任务均已完整、健康或通过双指标门槛。

新增独立恢复工具，支持CIFAR-10和Fashion-MNIST的schema 8。用户提交并推送本次文件、
AI Station在原项目`git pull`后执行（自定义输出目录须保留原路径）：

```bash
# 默认仅审计：不写输出、不占GPU、不训练、不重新选参
python recover_adaptive_report.py --output outputs/cifar10_resnet18_gn_tpe_v8
# 审计通过后，只重建已有60项正式结果的报告
python recover_adaptive_report.py --output outputs/cifar10_resnet18_gn_tpe_v8 --write
# Fashion-MNIST的对应用法
python recover_adaptive_report.py --output outputs/fashion_mnist_resnet18_gn_tpe_v8 --write
```

恢复前核验冻结manifest和科学源码SHA、原始验证证据、冻结参数、正式计划与全部60个完成快照，
未达标手动继续的实验还须存在对应的历史Y记录。正在运行的控制器持锁时拒绝恢复。
只恢复既有正式结果的报告无需再次授权训练；**原训练入口每次续跑仍询问Y/N**。
任务缺失、损坏、身份不符、非有限值或真正超出容差都会拒绝，不能用此命令绕过训练失败。

仅`honest_weight_loss`和`malicious_weight_mass`两项诊断在校验时接受`1 < value ≤ 1+1e-9`。
原快照、逐任务CSV、参数和检查点不改；导出的报告CSV也保留原始数值。
Accuracy、ASR、健康判断、150轮、K+2和双≤2pp规则不变。
报告写入原`final_results/visualizations.html`和`final_summary.json`，
已有这两项会整体备份到`report_recovery_backups/<时间与唯一标识>/`。
`final_results/report_recovery.json`记录接受的原值、偏差及输入/恢复代码SHA；
生成失败不替换原报告。恢复成功不等于Ours达标，报告仍显示实际健康失败与原验证未达标状态。

本补丁不改CIFAR的49项及Fashion的54项冻结科学源码，因此能核验并复用原实验身份。
**原控制器的自动报告路径仍冻结**；遇到上述旧校验问题使用此独立入口，无需重复训练。
本次提交文件：`README.md`、`recover_adaptive_report.py`、`adaptive_report_validation.py`、
`tests/test_adaptive_report_recovery.py`、`tests/test_adaptive_report_validation.py`。
本地docs/AGENTS不提交。

## 2026-10-01：三数据集进度显示修复

MNIST继续使用原入口 `python -u run_fair_tuning_from_config.py`，并保留原配置参数。
修复动态进度与阶段、CUDA及恢复日志粘连、窄终端折行残影；终端按宽度裁剪，重定向日志时自动使用普通文本。
MNIST仍按完成的实验配置计数，不将此修复描述为新增逐轮进度。

新版CIFAR-10和Fashion-MNIST使用统一显示入口，原控制器、训练函数及检查点身份保持不变：

```bash
# CIFAR-10：独立v8双种子搜索 → 单种子六法正式
python -u run_adaptive_with_progress.py \
  --config configs/cifar10_resnet18_gn_tpe_v8.json --devices auto
# Fashion-MNIST：独立选参及正式实验
python -u run_adaptive_with_progress.py \
  --config configs/fashion_mnist_resnet18_gn_tpe_v8.json --devices auto
```

正式阶段显示完整任务数（默认60）、已保存轮数（默认9000）、失败/暂停/排队任务、每张卡当前轮次、
本次耗时及近似ETA。搜索显示当前公共条件/波次和累计已评估任务，不把自适应搜索虚构为固定总量；
ETA只估当前波次或正式计划，恢复的历史轮次不当作本次吞吐。
`done`表示完整结果，不表示通过健康或2pp门槛；数值失败、预算暂停、尚未核验的快照单独计数。

无换行Y/N提示立即显示，等待输入期间停止重绘；输入仍由原控制器接收，每次续跑重新询问的规则保留。
Ctrl+C或终止信号先交给原控制器清理worker和保存检查点，再退出。
`--progress-mode auto`默认终端动态刷新、非终端每15秒追加摘要；可用`--progress-interval 1`
调整终端刷新、`--progress-log-interval 15`调整日志间隔，或显式`--progress-mode log`。
原始控制台日志在身份匹配的输出目录下另存`progress_display_logs/adaptive_*.log`；逐轮worker日志保持原位置。
可选进度日志创建或追加失败时提示并停用该日志，训练继续；训练自身的存储错误仍由原控制器处理。

继续原实验时，保留原`--config`、`--output`、`--data-dir`和设备参数，仅替换显示入口。
支持`--phase search`、`--phase final`、`--plan-only`；后者只转交计划检查，不创建显示日志或启动训练。
直接执行`run_cifar_adaptive.py`/`run_fashion_adaptive.py`仍保持原任务日志形式；要使用新增进度视图，使用上述命令。
旧`run_cifar_six_with_progress.py --config <v8配置>`也会转交统一入口，schema2–7的旧实验仍沿原路径运行。

本次进度补丁的提交文件为README.md、`run_fair_tuning_from_config.py`、`run_cifar_six_with_progress.py`、
`fair_tuning_progress_display.py`、`adaptive_progress.py`、`run_adaptive_with_progress.py`，以及
`tests/test_fair_tuning_progress_display.py`、`tests/test_adaptive_progress.py`、
`tests/test_adaptive_progress_dispatch.py`、`tests/test_adaptive_progress_monitor.py`。
Fashion实现尚未提交时，下节新增的数据集模块、配置和三份测试也须一并由用户提交推送；docs/AGENTS不提交。

## 2026-10-01：新增Fashion-MNIST独立选参与正式实验

Fashion-MNIST使用新入口 `run_fashion_adaptive.py` 和独立配置
`configs/fashion_mnist_resnet18_gn_tpe_v8.json`。由用户提交并推送本次文件后，在AI Station项目根目录运行：

```bash
git pull
python -u run_adaptive_with_progress.py \
  --config configs/fashion_mnist_resnet18_gn_tpe_v8.json --devices auto
```

默认输出 `outputs/fashion_mnist_resnet18_gn_tpe_v8`，数据缓存 `data/fashion_mnist`。
首次运行下载作者官方四个IDX gzip文件，后续复用缓存；每次加载校验官方MD5，manifest另记录SHA256。
Fashion与MNIST的原始文件名相同，必须分开缓存；发现内容不符会报错并保留原文件，不静默覆盖。
离线环境可预先将作者官方四个gzip文件放入该目录，再用相同命令；自定义路径用 `--data-dir /实际Fashion目录`。
不需要新增Python依赖。可指定 `--devices cuda:0 cuda:1 ...`；最多7卡、每卡一个任务，规则与CIFAR新入口相同。

```bash
# 不下载数据、不占GPU，只检查配置与展示计划
python run_fashion_adaptive.py --plan-only
# 仅完成独立搜索
python -u run_adaptive_with_progress.py --config configs/fashion_mnist_resnet18_gn_tpe_v8.json --devices auto --phase search
# 搜索结束后进入正式阶段，或恢复正式任务
python -u run_adaptive_with_progress.py --config configs/fashion_mnist_resnet18_gn_tpe_v8.json --devices auto --phase final
```

完整命令再次运行会复用本Fashion目录的结果和逐轮检查点；原来指定过自定义config/output/data-dir时保持一致。
搜索未结束时`--phase final`拒绝提前选参，应使用原完整命令继续。所有Ours未达标但有健康候选时，
按原Score选择最佳健康候选并询问Y/N；每次正式续跑仍需本次Y，N/EOF停止，旧Y仅冻结选择。
没有健康候选、证据缺失/损坏或身份不匹配时，不以Y覆盖。

该实验沿用下文CIFAR新协议的搜索规则，具体为：

- 验证seed **2026100101/2026100102**，正式seed **2026100111**；验证每候选20任务，正式六法共60任务。
- IID/Dirichlet α=.5，验证恶意比例0/10/30/50/70%，正式0/20/40/60/80%，100客户端、150轮、攻击开始K＋2。
- K、warning等防御参数、学习率、batch、本地epoch、boost、攻击epoch均在声明范围内独立搜索。
  沿用预先声明的搜索范围和起点，重新评估Fashion验证结果；不导入CIFAR的已选参数、Score、TPE拟合状态或模型权重。
  每组公共条件下也为基线重新选参，同条件下三种可调方法的候选预算匹配。
- 以Ours相对优势选择公共条件的研究背景保留；两验证seed选定参数后冻结，正式结果不回流选参。
- 最终第150轮逐场景 `max(可用六法Acc)−Ours Acc≤0.02`；受攻击场景还须
  `Ours ASR−min(可用六法ASR)≤0.02`。按离散样本率精确比较，含等号；原健康和clean效用条件保持。
- 本Fashion搜索单独累计48活动小时，含五个基线评估，正式任务另计时；每轮保留检查点。
  截止只从连续完整、公平匹配的波次前缀选参，部分波次不参与选择。没有足够观察时可能尚未进入TPE学习阶段。
  48小时不保证全局最优、达到2pp或完成指定数量的公共配置；运行中的任务保存下一完整轮可能使实际墙钟超出预算。

Fashion模型为**原生1×28×28输入的ResNet-18＋GN2**：与CIFAR新模型保持相同骨干，首卷积适配单通道，
共11,172,810参数；没有缩放到32×32、通道复制、数据增强或预训练权重。六法使用完全相同的Fashion模型。
图像固定转换为float32后除以255，与旧MNIST采用相同像素缩放；不从正式测试集估计预处理统计量。

完整60,000训练样本按固定seed20261001分层划分为 **54,000联邦训练＋3,000验证＋3,000攻击辅助**，
三个部分不重叠；官方10,000测试样本只用于正式评估。验证Accuracy门槛据3,000样本计算，
正式Accuracy据10,000样本计算，不沿用CIFAR的2,500验证样本分母。
固定定向攻击为标签 **5（Sandal）→7（Sneaker）**，ASR使用200个源类评估样本；
数字与CIFAR相同、类别语义不同，数据身份和报告中明确记录。
数据规模、形状、标签和官方文件校验值参见[Fashion-MNIST作者仓库](https://github.com/zalandoresearch/fashion-mnist)。

正式结束生成 `final_results/visualizations.html`、CSV、PNG/SVG及来源SHA审计，图文明确标注Fashion-MNIST、
实际搜索预算和类别语义。只统计完整150轮结果；健康失败的完整结果保留，缺失不补0/不取早期轮，
缺任务方法不算总体，单正式seed不报告跨seed标准差。

兼容边界：本次仅新增根目录Fashion模块、独立配置与测试，原CIFAR v8的49份科学源码和配置、
旧MNIST/CIFAR v7的43份冻结源码保持原字节。共用现有通用TPE/健康/门槛和底层训练函数，
数据、模型适配器、调度入口与报告分别保存，避免改变已运行实验的源码身份。
Fashion和CIFAR的manifest、搜索状态、正式计划与缓存不能混用，两种入口会拒绝对方的协议配置。
旧实验继续使用原命令及原输出；添加Fashion代码不会自动开始、重跑或改写它们。

## 2026-09-30：双种子自适应搜索 → 单种子六法正式实验（独立v8）

新增 `run_cifar_adaptive.py` 与 `configs/cifar10_resnet18_gn_tpe_v8.json`。这是独立的新实验，
使用 **CIFAR版ResNet-18＋GroupNorm（每层2组）**，六法统一模型、公共训练和攻击条件；
每任务150轮，攻击从所选K＋2轮开始。旧v7、MNIST的入口、配置、43项科学源码与输出身份不变，
旧实验仍用原命令。不要用新入口指向旧输出，也不要把新结果写成旧CNN实验的复现。

由用户提交并推送本次文件后，在AI Station项目根目录执行：

```bash
git pull
python -u run_adaptive_with_progress.py \
  --config configs/cifar10_resnet18_gn_tpe_v8.json --devices auto
```

默认输出 `outputs/cifar10_resnet18_gn_tpe_v8`。自动选取最多7张空闲CUDA GPU、每卡1个任务；
需要明确卡号时用 `--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6`。
使用现有PyTorch/NumPy/Matplotlib环境，无需新增Optuna依赖。新模型不允许回退为NumPy或旧CNN。
如数据不在现有默认缓存位置，追加 `--data-dir /实际数据目录`，续跑保持同一路径。
请在可输入Y/N的终端或tmux会话运行，保留整个输出目录及隐藏文件。

```bash
# 只核查协议/搜索范围，不加载数据、不占GPU、不开始训练
python run_cifar_adaptive.py --plan-only
# 只搜索并保存best_parameters.json，不启动正式训练
python -u run_adaptive_with_progress.py --config configs/cifar10_resnet18_gn_tpe_v8.json --devices auto --phase search
# 搜索结束后，或正式阶段中断后，使用原目录继续
python -u run_adaptive_with_progress.py --config configs/cifar10_resnet18_gn_tpe_v8.json --devices auto --phase final
```

验证seed为2026093001、2026093002；正式seed为2026093011，均在配置中预先声明、互不重叠。
验证覆盖IID/Dirichlet α=.5下0/10/30/50/70%恶意，候选每套20任务；
正式覆盖两种分区下0/20/40/60/80%恶意，共60任务。采用原CIFAR-10划分：45,000训练、
2,500验证、2,500攻击辅助数据，官方10,000测试仅进入正式阶段。正式训练重新初始化，不继承验证模型权重，
正式结果不回流选参。此前已观察过官方测试结果，换seed不等于获得全新的未见测试集。

### 学习什么，如何对六法保持同等条件

搜索器是本项目实现的条件化product-KDE TPE：先评估声明默认值和一个随机起点，
有至少两个完整可评分观察后，用好/差结果分别拟合密度，按密度比提出下一组参数。
它属于基于历史实验结果的自适应超参数优化，不是穷举网格，也不是在前K轮把所有超参数训练出来。
原理参考[Bergstra等，2011](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)。
每次提案、随机数状态、拟合样本数、损失及失败原因保存到 `search_state.json`，不会把随机起点标成学习所得。

| 层次 | 搜索参数与范围 |
| --- | --- |
| 公共条件 | K=6…20；学习率0.01…0.1（对数尺度）；batch=32/50/64；本地epoch=1/2/3；boost=2…15（对数尺度）；攻击epoch=1/2/3 |
| Ours | warning、severe、漂移记忆/allowance/threshold、历史准入阈值及确认次数、恢复确认次数、正常簇数、子空间维数、参考预算、裁剪、权重上限、降权/恢复系数及撤销次数 |
| VERT | 历史窗口5/7/10/15/20；预测epoch=5/10/20/30；预测学习率0.0001…0.003；投影128、无恶意比例先验保持固定 |
| AlignIns | sparsity=0.1…0.5；TDA/MPSA radius各0.5…1.5 |
| Krum/TAD/FedAvg | 当前原始实现无独立候选调参项，每种公共条件下仍重新评估；不重复运行完全相同固定候选 |

精确范围、先验和条件转换见JSON的 `search_spaces`；Ours约束severe>warning、allowance≤warning、
history阈值<warning，避免生成无效组合。初始公共值为K10/lr0.05/batch50/local epoch1/boost5/attack epoch1，
Ours初始防御值来自014，但旧Score或旧结果不作为本次新模型资格。
学习率衰减0.99、stealth步数1、distance权重0.0001、源类别5→目标7、200攻击评估样本等固定项仍显式保存。

外层按Ours相对五个固定选定基线的逐场景Acc/ASR差距优化，包含健康惩罚；内层每种方法按原健康与Score规则调参。
**每种公共条件都重新建立Ours、VERT、AlignIns各自的TPE**，每法最多4套候选，
Krum/TAD/FedAvg各1套，共300个验证任务/公共条件。每一波同时增加三种可调方法各1套候选；
第1波还评估三个固定方法。基线不能沿用另一公共条件下的验证资格或Score。
所有方法共用所选K、学习率、batch、epoch、boost和攻击起始轮，
新模型参数量11,173,962，CIFAR 3×3 stride1 stem、无ImageNet maxpool、无BatchNorm运行统计。

这是**在验证集上搜索有利于Ours的公共条件后的比较**，不声称为任意条件下的普遍优势。
K变化同时改变攻击起始轮和受攻击轮数；boost/epoch/batch同时影响实际攻击优化强度和计算量，
必须随结果报告。五个基线各按原最终Score选定固定候选，不能逐场景挑不同候选。

### 48小时预算、门槛及续跑

默认搜索累计活动墙钟预算48小时，最多16组公共条件；停机期间不计时，重启不会重新获得48小时。
预算覆盖五个基线评估、Ours评估、调度和选参；正式60任务另计时。
截止后停止派发新搜索任务，运行中的任务在下一完整轮保存检查点后退出，可能因该轮耗时超过48小时。
每轮保存检查点，完成结果缓存复用；OOM/环境/IO问题最多原配置重试一次，之后阻塞并保留原始失败，
不能把它们作为算法差或自动缩小batch来改变协议。已确认的数值失败保留为失败证据。

**48小时是预算，不保证找到全局最优、达到2pp，或完成足够多试验进入外层TPE。**
只有两个完整公共条件评估后才有外层第3次自适应提案；ResNet-18在本项目七卡上的实际吞吐尚未实测。
预算截止只从已完整尝试的连续波次前缀选参，部分波次整体排除，即使其中个别Ours已经训练完成。
每个可用前缀内三种可调方法的候选数相同；不同公共条件的实际搜索次数可能不同，报告明确披露。
若连第1波完整证据都没有，或者没有健康Ours，则停止且不生成正式资格。
`search_summary.json`报告实际完成波次与TPE提案数，`best_parameters.json`给出预算内所选明确参数。

正式自动准入沿用逐场景最终轮双指标：`max(六法可用Acc)−Ours Acc≤0.02`，
攻击场景还须 `Ours ASR−min(六法可用ASR)≤0.02`，含等号，按离散样本率进行精确比较。
仍要求Ours健康及原clean效用门槛；过程指标只参与健康/诊断，不替代第150轮性能。
完整可评分的健康失败基线也参与参照；缺失仅逐任务排除并显式标注比较不完整，沿用原规则。
若所有Ours未达标但有健康候选，询问是否采用原Score最好的健康候选进入主实验：Y继续，N/EOF停止。
**每次正式续跑仍要重新输入Y**；过去的Y只冻结选择。达标路径自动推进，不询问。

原命令再次运行即恢复：核对模型/源码/配置/数据身份、验证结果和固定计划，复用完成任务并从有效检查点继续。
预算已耗尽时不继续搜索，只评估已完成前缀；冻结正式选参后禁止扩搜或回流正式结果。
配置、代码、数据发生科学身份变化时拒绝复用原输出，应保留原版本/目录，另建新实验。
末波保存与选参之间中断也可恢复，不会多生成提案。不要手改 `search_state.json` 或继续决定文件。

正式阶段自动生成 `final_summary.json` 和 `final_results/visualizations.html`，包含CSV、PNG/SVG、
参数来源和SHA审计。只使用完整150轮结果；失败任务不补0、不用早期轮次代替，健康失败的完整结果保留。
单正式seed明确n=1，不能报告跨seed标准差；缺任务的方法不计算总体指标。新报告不改旧报告器。

### 旧输出归档

提供 `archive_experiment_outputs.py`，默认只读，`--apply`才移动；先检查进程和活动锁、记录全文件SHA，
再在同文件系统重命名进入 `outputs/old`，不删除结果。2026-09-30本地已保留最新
`cifar10_six_relative_best_five_day_v7 2` 和 `mnist_v7_target_fair_tuning`，另22目录完成校验归档。
映射记录在本地 `outputs/old/archive_manifest_20260930T024556_709097.json`。
这些本地移动不会随Git同步，AI Station输出不受影响；旧实验若在本地续跑应使用移动后的实际完整路径。

## 2026-09-29：已结束但有任务失败的实验也生成可视化

报告层现在支持“所有任务已尝试，但部分任务有明确失败记录”的正式实验。原先只接受
`full_execution_completed=true`，会使179/180完成的实验在自动报告阶段报 `Formal execution is incomplete`。
现在仍逐项检查冻结身份、原始CSV、完整轮次和指标一致性；有明确失败证据的缺失任务在报告单独列出，
不会把损坏的已完成任务静默排除。训练、选参、性能门槛、完成/健康状态和43项科学源码保持原样。

HTML首页和各图明确标注不完整状态。受影响场景按可用完整seed计算描述性均值并标注实际n，例如TAD
Non-IID 80%只有两个完整seed时显示n=2/3；这不是完整三seed结果，可能有存活样本偏差。
单seed页面对应任务缺失时留空（n=0/1），不补零、不用中止前结果替代最终值。存在缺失的方法不计算总体指标。
所有完整但健康失败的运行仍纳入统计；恢复成功后保留的历史failure.json也不再被误判为当前任务未完成。

原包装入口正常结束正式调度时自动生成报告；部分报告会额外打印 `REPORT_INCOMPLETE`，
`report_generation_status.json`记录 `completed_partial`，不把原实验改成完整成功。
既有输出通过下面命令补图，无须重新训练或再次确认Y：

```bash
python -u run_cifar_six_with_progress.py --report-only \
  --output outputs/cifar10_six_relative_best_five_day_v7
```

路径包含空格时用引号包围，例如本地导入副本 `--output "outputs/cifar10_six_relative_best_five_day_v7 2"`。
产物位于 `final_results/visualizations.html`、`final_results/mean_plots/`及各 `seed_*/`，包含SVG/PNG、均值PDF、
实际样本数与缺失seed的统计CSV、`missing_tasks.csv`和源文件SHA审计。报告生成仅写派生文件，保留原实验CSV/JSON和检查点。

## 2026-09-28：v7磁盘写入失败后的任务初始化恢复

若日志出现 `task artifacts exist without their immutable identity`，表示任务目录存在残留文件但缺少
`task.json`，worker在训练前拒绝启动。ENOSPC可能使身份临时文件未提交，并留下失败记录；这些记录又会阻止后续初始化。
这类报错不代表算法性能或健康失败。`completed_snapshots`仅统计文件存在，不能单独证明快照完整或任务健康。

先确认存储写入恢复，并停止本输出目录的runner和worker。通过GitHub更新后使用独立工具；它不修改43项冻结科学源码、
配置、验证结果、候选选择或正式计划，不重新训练、不创建任务身份，也不加载结果pickle。

```bash
# 默认只检查；原来指定过自定义输出时，两条命令都加同一个 --output 路径
python repair_cifar_orphan_tasks.py \
  --config configs/cifar10_six_relative_best_five_day_v7.json

# 仅在检查得到 status=ready 后处理可安全隔离的启动残留
python repair_cifar_orphan_tasks.py \
  --config configs/cifar10_six_relative_best_five_day_v7.json --apply
```

工具校验manifest/源码/配置、验证与正式计划、原用户继续决定，并取得原runner锁、检查遗留worker进程。
只处理缺少 `task.json` 且仅有 `worker.log`、失败JSON或身份/失败临时JSON的正式任务目录；已有完整临时身份必须匹配原计划，
可解析失败记录必须属于该任务且没有训练上下文，日志不能包含已执行轮次。含检查点、结果、训练上下文、未知文件、
符号链接、损坏或冲突的正式 `task.json` 时输出 `blocked`，整批不移动；请保留输出进一步诊断。

`--apply`将符合条件的目录整体原子移动到原输出内的 `orphan_recovery/<批次>/<原task_id>/`，
保留所有原字节，并写入文件SHA、冻结元数据SHA和逐任务移动记录 `audit.json`。没有删除实验数据或回填身份文件。
备份占用仍计入原存储空间，工具不解决底层容量/配额/存储池问题。若工具中途失败，已移动目录仍在备份、其余留在tasks，
排除原因后可重新检查/执行；不要删除备份或使用旧v2的另建输出恢复脚本。

得到 `status=quarantined`（或无需处理的 `no_action`）后，按原命令、原输出续跑：

```bash
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_best_five_day_v7.json --devices auto
```

本次仍输入Y；原runner会再次验证正式身份及快照，复用有效完整任务、恢复有身份的有效检查点，
为被隔离的启动残留对应任务按原冻结计划重新初始化。恢复工具的统计不代替原runner的完整性与健康检查。
如果再次ENOSPC，保留日志并排查实际写入限制；小文件写入成功不保证并发大检查点可持续写入。

## 2026-09-27：v7未达标后的交互式正式实验与续跑

原逐任务最终Accuracy、ASR均距可用六法最优值≤2个百分点的规则不变。满足原门槛时自动进入正式阶段；
若所有Ours均未达标，但存在健康候选，控制台显示最佳健康候选（原Score最高，沿用原平分排序）并询问`Y/N`。
输入`Y`后，使用该候选和原验证已选出的五个基线运行180个正式任务；输入`N`停止并保留结果。
没有健康Ours、验证证据缺失/损坏或实验身份不符时，不能通过Y覆盖这些问题。

在AI Station项目根目录，通过Git更新后，继续使用原推荐命令：

```bash
git pull
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_best_five_day_v7.json --devices auto
```

使用原输出目录`outputs/cifar10_six_relative_best_five_day_v7`。本轮1290个验证任务已完成时，程序核查保存结果后直接询问，
不会重新训练验证；本轮最佳健康候选为`sm9rrs-v10-014`。如以前指定了`--output`或`--data-dir`，继续带相同路径。
首次Y冻结正式选参，**以后每次启动/断点续跑仍重新询问Y/N**；续跑Y复用完整任务与逐轮检查点，不重新选参。
已完成全部正式任务后重跑也会询问，Y只重建汇总。`--phase validation`只做验证，不询问或启动正式训练。
OOM恢复保持原机制；包装器自动重新启动控制器时，也需要再次输入Y。

请在可输入的交互终端中运行（例如tmux会话）；提示没有超时默认选择。非法输入会重问，EOF或无交互终端停止，
不会把管道中的Y或上次的Y当成本次许可。`--plan-only`仍只读展示任务计划；`--report-only`只重建报告，不训练、不询问。

新增控制器`run_cifar_six_interactive.py`由进度包装在schema7下调用；原`run_cifar_six_relative_best.py`及43项冻结科学源码不变，
继续承担旧协议与worker执行。旧manifest、配置、数据划分、任务指纹和检查点格式均不变。
直接调用旧`run_cifar_six_relative_best.py`仍是原严格门槛行为；交互续跑请使用上面的包装命令。
schema2–6、历史MNIST入口和训练逻辑不变。本轮尚未增加Fashion-MNIST加载器或实验配置。

用户继续的决定写入独立`continuation_decision.json`，每次回答写入`continuation_responses/`；
它们与`final_plan.json`共同固定来源和参数，不修改原manifest。
原`validation_summary.json`仍记录未达标，正式JSON/HTML也明确标注“用户确认继续；原验证目标未通过”。
正式结果仅用于评估，不能反过来选择候选。迁移或备份时保留完整输出目录，包括隐藏的`.completed_results.pickle`、检查点和新增决定文件。

## 2026-09-21：Accuracy与ASR均距六法最优值≤2个百分点（当前推荐v7）

用户最新要求：每个分区×恶意比例×seed任务只比较第100轮，
`max(可用六法Accuracy) − Ours Accuracy ≤ 0.02`；受攻击任务还须满足
`Ours ASR − min(可用六法ASR) ≤ 0.02`。**等于2个百分点也通过**，例如最高Accuracy80%时Ours至少78%；最低ASR10%时Ours至多12%。
这不是乘以最优值的2%。两项最优值可来自不同方法，Accuracy不再只与VERT比较；无攻击ASR仍仅作诊断。

使用 `configs/cifar10_six_relative_best_five_day_v7.json`，原训练入口为 `run_cifar_six_relative_best.py`（当前包装控制层见上节），
独立输出 `outputs/cifar10_six_relative_best_five_day_v7`。下面历史v6扩展配置的43套候选完全不变，仍1290验证＋达标后180正式。
Score及均值双优优先排序、基线选参/fallback、完整健康、数据、公共参数、seed及检查点均保持。
每个基线仍使用按验证Score选定的固定候选，不逐任务挑不同候选；其完整、数值有效的健康失败任务也参与比较。
缺失只排除对应任务参照，明确标记比较不完整，不冒称完整六法通过。旧v6门槛与输出保留历史语义。

已取消120小时强制截止，五天仅为估时；检查点、续跑、OOM恢复和自动HTML/SVG/PDF保留。
新门槛决定自动推进资格；9月27日起额外支持用户确认后继续。不能保证新正式seed或更高恶意比例下仍在2个百分点内；不根据正式结果回选参数。

```bash
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_best_five_day_v7.json --plan-only
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_best_five_day_v7.json --devices auto
```

`cifar_budgeted_search.py`现在默认生成上述v7配置，`build_spec_v6()`保留旧版构建逻辑。
本轮新增 `cifar_relative_best_gate.py`、`run_cifar_six_relative_best.py`、
`configs/cifar10_six_relative_best_five_day_v7.json`、`tests/test_cifar_relative_best.py`；还须一并提交此前尚未提交的v5/v6依赖及包装/报告/生成器等修改。

## 2026-09-21：五天预算的定向扩展搜索（历史v6规则）

使用 `configs/cifar10_six_relative_asr_five_day_v6.json`，独立输出到
`outputs/cifar10_six_relative_asr_five_day_v6`。原25套候选及各方法内部顺序保留，
新增14套Ours、2套VERT、2套AlignIns：**28/6/6/1/1/1，共43套、1290组验证；达标后180组正式，最多1470组/147000轮**。
训练/攻击公共参数、数据划分、种子、最终轮选参、健康检查、Score、基线fallback及逐任务ASR差值严格<1pp规则均不变。
不使用旧输出作为本轮资格缓存；旧配置和实验目录保留。

新增Ours为`sm9rrs-v10-101`至`114`，以014为基础：

- 101：漂移容忍量κ从1.25降至.85，同时漂移阈值h从6降至1。
- 102：历史准入阈值从1降至.8，连续确认从2增至3。
- 103：合并101与102；104：再将单轮告警阈值从1.25降至1.10。
- 105–114：围绕上述组合探测κ=.75/.95、h=.75/1.5、告警1.20、历史阈值.70/.90、历史确认4、漂移记忆.90，及永久撤销容忍次数7；每套只改其声明的局部参数。其余新Ours保留撤销容忍次数5。
- VERT新增015附近的历史窗口7和预测训练10轮两套；AlignIns新增半径均为.75、稀疏率.1/.3两套。只调整已有超参数。Krum/TAD/FedAvg无可搜索的专属参数，仍各一套。

这检验现有检测公式中的漂移累积和历史污染问题，未改核心算法。更严格的阈值也可能增加诚实误报，仍须通过完整健康和准确率检查。
各方法搜索额度不相等，不能在论文中称为等调参算力。新候选尚未训练，不能保证Ours达标或正式最优。
验证seed仍为1001–1003，已有开发历史；正式1101–1103保持独立于本轮选参，不把重复验证称为全新独立证据。

**时间估算：** 上次750验证实际运行43.28小时，使用7张`NVIDIA GeForce RTX 4090 D`。
逐任务计入`runtime_seconds + checkpoint_io_seconds`；已测候选使用实测均值，新候选按同方法最慢候选均值预留，
另计观测调度开销、180正式任务、20%波动余量及1小时准备/报告时间。
本轮预计验证87.80小时、正式预留5.33小时，训练合计93.13小时；含余量约**112.75小时（4.70天）**。
证据摘要在`configs/cifar10_five_day_timing_reference.json`，计算和配置生成器为`cifar_budgeted_search.py`；运行不依赖本地历史outputs。

按用户最新要求，**已移除120小时强制截止及其7卡/4090D启动限制**；五天只用于规划候选数量，不控制训练或报告退出。
仍使用原有GPU兼容性/可用显存预检，可按实际设备数量启动；上述112.75小时仅适用于参考条件，不是完成保证。
**检查点和断点续跑保留**：`checkpoint_interval=1`，逐轮保存；相同配置和输出目录重启会复用已完成任务及检查点，OOM仍走原恢复流程。
程序不再创建或读取walltime_budget计时JSON/锁；已有实验输出和检查点不会因移除截止功能被删除。

最终性能规则再次核对并按用户确认保持：受攻击任务第100轮ASR距固定入选六法最低值严格<1个百分点；
Accuracy仍相对VERT落后≤2个百分点（含无攻击任务），不新增“距六法最高Accuracy<1个百分点”的硬门槛。
候选须先满足健康和逐任务目标，之后优先最终指标均值双优，再按原Score、最差最终ASR等排序；不承诺训练前即可保证两项均最优。

用户自行提交/推送后，AI Station执行`git pull`，在项目根目录运行：

```bash
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_asr_five_day_v6.json --plan-only
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_asr_five_day_v6.json --devices auto
```

`--plan-only`不启动GPU或训练。正式实验完整结束后自动生成HTML、均值SVG/PDF。
Ours未达标则仍停止在验证阶段；报告依赖缺失等情况保留训练数据，可随后单独补报告（此命令不训练）：

```bash
python run_cifar_six_with_progress.py --report-only \
  --output outputs/cifar10_six_relative_asr_five_day_v6
```

本次新增需提交：`cifar_budgeted_search.py`、
`configs/cifar10_six_relative_asr_five_day_v6.json`、`configs/cifar10_five_day_timing_reference.json`、
`tests/test_cifar_budgeted_search.py`；本次继续修改`README.md`、`run_cifar_six_with_progress.py`。
若上轮v5/v6改动尚未提交，还需一并提交两版gate/runner/reanalysis/config/test及`experiment_reporting.py`，否则新配置缺少依赖。
本地docs、AGENTS和outputs不提交。

## 2026-09-21：CIFAR 相对最低最终ASR门槛（基础v6）

用户确认替换5%绝对ASR限制：每个“分区×恶意比例×seed”的受攻击任务，第100轮必须满足
`Ours ASR − min(该任务可用六方法的最终ASR) < 0.01`，即**严格小于1个百分点**。
200个目标样本时，差1个样本（0.5个百分点）通过，差2个样本（1个百分点）不通过。
比较会恢复float32输出对应的离散样本比例并用有理数判断，避免恰好1个百分点被浮点误差放行。

新入口 `run_cifar_six_relative_asr.py`，配置 `configs/cifar10_six_relative_asr_v6.json`，独立输出 `outputs/cifar10_six_relative_asr_v6`。
完整健康检查、最终Score及权重、基线选参回退、相对VERT最终Accuracy落后≤2个百分点/ASR高出≤1个百分点，以及均值双优仅排序的规则不变。
无攻击场景不新增ASR门槛。全部25候选和公共训练/攻击参数不变，仍750验证＋达标后180正式。

各基线使用验证Score选定的固定候选，不在每个场景另选参数。其完整且数值有效的健康失败任务也进入最低ASR比较。
候选整套不完整时，已完成的任务仍可逐任务作参照：本轮TAD的29组可用，只有缺失的那1组被排除。
缺失基线不单独阻止Ours推进，但明确记录`missing_methods`；部分比较通过不能写成完整六方法目标通过。
资源故障、损坏缓存和身份不符仍由原执行完整性规则阻塞。

无需重训即可重算现有验证：

```bash
python reanalyze_cifar_relative_asr.py \
  --source outputs/cifar10_six_mnist_gate_compact_plus_v4 \
  --output outputs/cifar_relative_asr_reanalysis
```

这是独立事后审计，不覆盖原输出或自动创建正式计划。输出目录须新建或为空。
2026-09-21重算：014满足5/24个攻击任务的ASR要求，其余健康Ours均为0/24；仍无候选满足全部要求。

按新协议启动时使用进度包装以自动生成最终HTML、均值SVG与PDF：

```bash
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_asr_v6.json --plan-only
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_relative_asr_v6.json --devices auto
```

v2–v5入口和输出保留历史语义，不要对旧目录启动v6。本次没有改核心检测算法、追加候选或启动训练。

## 2026-09-21：CIFAR 最终轮选参与性能门槛（历史v5）

上一版入口为 `run_cifar_six_final_metrics.py`，配置为
`configs/cifar10_six_final_metrics_v5.json`，独立输出 `outputs/cifar10_six_final_metrics_v5`。
根据用户确认，**每个任务仅用第100轮 Accuracy / ASR 决定性能达标**；攻击窗口均值、末10轮与峰值保留为诊断。
跨 seed、场景仍可对这些最终值等权汇总，这与对训练过程逐轮求平均不同。

- Score 保留权重 `.25/.50/.20/.05`：干净任务最终 Accuracy、攻击任务最终 Accuracy、`1−攻击任务最终 ASR`、`1−诚实权重损失`；最后一项仍按原时间/场景口径。ASR 的平分排序也改用最差任务的最终值。
- Ours 须通过完整训练健康检查；每个攻击场景、每个 seed 最终 ASR ≤5%。有完整可评分 VERT 时，相同任务最终 Accuracy 落后 ≤2 个百分点、最终 ASR 高出 ≤1 个百分点；干净场景比较最终 Accuracy。原 `max_peak_asr` / `tail_rounds` 配置仅保留作过程诊断，不再决定性能通过。
- 健康检查仍检查全过程的非有限更新、完整性、撤销状态，以及原有的干净精度检查；最终值正常不能掩盖健康失败。均值双优仍仅为达标 Ours 的优先排序，`require_mean_dual_best=false`。
- 基线继续逐方法执行：健康最高 Score → 完整可评分失败候选最高 raw Score → 固定首候选；保留失败标签。Ours 不达标仍不创建正式计划。OOM、缺失、损坏证据不能冒充算法失败。
- 25套候选、公共训练/攻击参数、验证1001–1003和正式1101–1103种子均不变，仍为750验证＋达标后180正式。没有为改善排名调整算法、评分权重或候选。
- 历史 v2/v3/v4 入口及其训练源码保持原样，供旧实验复核/续跑；不要用旧配置启动并期望自动采用新口径，也不要向旧输出目录写入 v5。

已有 v4 验证可单独重算，无需重新训练750个任务：

```bash
python reanalyze_cifar_final_metrics.py \
  --source outputs/cifar10_six_mnist_gate_compact_plus_v4 \
  --output outputs/cifar_final_metric_reanalysis
```

该命令只读取匹配身份的 CSV/JSON，复核健康指标与最终值，输出选参审计 JSON、逐任务/无攻击/场景/总体 CSV。
输出目录必须新建或为空，不能是原实验目录及其父子目录。重算明确标注为事后分析，不覆盖旧结论、不创建正式计划、不把旧实验改称事先采用新规则。
本地2026-09-21重算显示：12套健康 Ours 仍全部未满足最终 ASR 门槛，因此单改指标口径不会启动正式实验。

需要按新协议开展独立实验时，使用进度入口以便正式实验完成后自动生成报告：

```bash
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_final_metrics_v5.json --plan-only
python -u run_cifar_six_with_progress.py \
  --config configs/cifar10_six_final_metrics_v5.json --devices auto
```

完整正式实验后的 `final_results/visualizations.html` 总体 ASR 表使用最终轮值，均值曲线仍呈现各轮走势；
`mean_plots/` 同时输出 SVG、PNG、PDF 合集及可复核 CSV。旧实验 HTML 保持其原指标口径。
正式任务未执行或缺失时，不生成伪装成正式结果的均值报告。

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
