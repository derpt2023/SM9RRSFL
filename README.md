## 已接受快照与独立 Ours 论文参数验证（2026-10-29）

用户确认保留当前六方法数据，已下载为 `outputs/cifar-paper/`。入口 `outputs/cifar-paper/report/report.html`；`accepted_snapshot.json` 标记用户接受并记录逐文件SHA/远端来源，`download_verification.json` 核验3484文件、4788168621字节，当前180科学单元与原文核心VERT导出SHA `2fd0d9cdad96b9cfacb6ed69da0b482dce4bfab537c118e03b9111810714f1a1`保持。原始结果、逐轮指标、模型锚点、审计与105源码快照已保留；211个完整训练断点共167142587720字节超过本机剩余空间，仅保留在AI Station，准确路径和大小列入清单，不能将本地快照称为完整断点备份。新图没有旧Top20记录；历史科学原件仅在raw中溯源。用户接受不等于原比较标准全部通过，后续开发不覆盖此快照。

用户进一步授权在AI Station提升Ours的IID精度并保持Non-IID表现，强调尊重原论文。已只读核第五版（SHA7902b5ab2817f826ed635eaaaefd9461211d8775ee4def3a1de2816aecd17318）和第六版（SHA002fbb514b2eb414ade2bdd4ce27d39c0c979e97344937143e6494487c100ace）第4.3.3节，两版核心相同。新研究仅调整既有公式中的恢复因子rω与范数裁剪倍数cn；同轮隔离、固定参照/范数上界、可信历史与漂移、恢复资格/确认、可靠度上界1、绝对系数上限、裁剪后不归一化、证书撤销和SM9代码保持不变。裁剪参数也会影响历史准入/恢复资格，不能称纯聚合单因素。沿用已接受的150轮/V2攻击/40000训练＋2500辅助和公共CNN；这与文稿旧100轮及旧选参安排不同，明确披露，不声称实验协议逐字复现。

独立入口 `run_cifar_ours_paper_refinement.py`、配置 `configs/cifar10_ours_paper_refinement150_v1.json`、输出 `outputs/cifar_ours_paper_refinement150_v1`。首批为原seed2026102801/02/03×IID/Dirichlet(.5)×0/20/40/60/80%×6臂=180项、各150轮：H84、FedAvg、原文核心VERT、R2（恢复1.25→2）、C125（裁剪1→1.25）、R2C125（组合）。只有训练衍生7500验证图/source5的750图参与选择，官方test不参与；原测试成绩与seed已见、验证集此前用于开发的事实保留，不称独立未见holdout。所有seed与场景使用同一全局候选参数，不按名次、比例或分布挑参数，不删除失败。

预先固定经验约束：全部180完整finite、逐seed前缀/r24配对、24 FedAvg攻击门槛与执行审计；候选及配对H84全场景健康；每个Non-IID场景/seed尾10 Acc下降≤1pp、ASR上升≤2pp，攻击期均值同样限制；IID尾10ASR上升≤2pp，clean Acc下降≤0.5pp，候选各clean距FedAvg≤5pp。IID五比例×3seed等权平均Acc至少提升0.5pp才合格；合格者取IID均值最高，其后Non-IID最坏ASR增量最小、名称序。接近VERT的目标为平均差≤1pp且最坏场景差≤2pp；这些是经验约束，不是统计显著性或跨运行保证。只有完整验证后固定候选，再另建同3seed的30项正式Ours输出；正式成绩不回流调参。若首批无合格候选，先分析实际机制，最多另建预声明R4/C150/R2C150三个候选（最多90项），不无限扩展、不自动重试。本批没有自动启动后续正式或后备搜索。

训练与CUDA就绪检查硬限制AI Station `/3251901002/SM9RRSFL`，本机仅协议测试、归档与绘图。共享GPU准入仍8192MiB、两次间隔5秒、锁内复查、利用率仅记录，每卡一项最多6，持续隔离GPU3/GPU6旧不稳定UUID。TERM/INT停派发并排空；partial/错误保留、无自动重试。每轮指标与实际系数完整，checkpoint仅r0/r24/r150；旧105科学来源不改，新110来源。无新正式结果时不得声称优化已达标。

```bash
python run_cifar_ours_paper_refinement.py --ready
python run_cifar_ours_paper_refinement.py --batch --max-gpus 6
```

入口结束自动生成 `analysis/refinement_results.json` 与执行审计；先读实时controller状态和进程，`panel_terminal`且主进程存在可能仍CPU审计，禁止重复启动。只在全部训练终态且主进程异常退出时使用`--finish`恢复派生审计，不重训。所有训练、下载、候选决策和后续正式结果需在实际交接记录中更新。

## 当前六法最终结果：原文核心 VERT 三 seed 补跑完成（2026-10-29）

原 seed 2026102801/02/03 × IID/Dirichlet(.5) × 0/20/40/60/80%，30项新版VERT全部完成150轮及最终审计；本批30次执行、0次失败、0次重试。北京时间16:39:11启动，19:32:57训练终态，主入口自动审计后正常退出。当前六法报告已全部替换这30项VERT，原其他五法150项成绩和轨迹逐值不变；没有根据正式test成绩继续调参、换seed或择优替换。

[当前完整六法报告](outputs/cifar_h84_formal150_v1/report/report.html)包含180项原值、60组跨seed均值/样本SD及实际n、尾10/最终轮/全攻击期/峰值、健康失败与全部原始排名、原严格及既有Krum IID clean效用独立例外。每seed四组图（尾10、IID全程、Dirichlet全程、含/扣crypto耗时），共12 PNG＋12 SVG，已逐张目视核验，使用MNIST固定颜色。尾10主图：[seed2026102801](outputs/cifar_h84_formal150_v1/report/tail10_s2026102801.png)、[seed2026102802](outputs/cifar_h84_formal150_v1/report/tail10_s2026102802.png)、[seed2026102803](outputs/cifar_h84_formal150_v1/report/tail10_s2026102803.png)。新图没有旧Top20曲线、数值或附图。

六法攻击总览：先对每seed的八个攻击场景等权平均，再计算三seed均值±样本SD，n=3；单位%，尾10为141..150轮。所有健康失败结果仍纳入原始总览，不等于健康资格筛选后的排名或统计显著性结论。

| 方法 | Accuracy | ASR |
|---|---:|---:|
| Ours (H84) | 54.571±0.453 | 8.563±0.999 |
| TAD | 49.334±0.751 | 50.798±1.760 |
| VERT (paper core + K-means) | 49.236±0.554 | 47.321±3.045 |
| AlignIns | 43.619±0.639 | 78.788±0.657 |
| Krum | 37.125±0.839 | 12.298±4.637 |
| FedAvg | 42.188±0.853 | 91.604±1.619 |

新版VERT比FedAvg平均Accuracy高7.048个百分点、ASR低44.283个百分点。改善主要集中在IID：四攻击场景跨seed平均Accuracy57.609%、ASR7.636%；Dirichlet为40.864%/87.006%，仍存在明显防御失败，不能概括为各场景均优于FedAvg。按全部八攻击场景均值，新VERT的Accuracy和ASR均列六法第三，但该观察不是调参目标，不能取代逐场景、健康和方差分析。

新版VERT逐场景尾10结果（三seed均值±样本SD，n=3，单位%）：

| 分布 | 恶意比例 | Accuracy | ASR |
|---|---:|---:|---:|
| IID | 0% | 60.093±0.868 | 7.223±0.471 |
| IID | 20% | 59.753±0.659 | 7.027±0.197 |
| IID | 40% | 59.021±1.024 | 7.687±0.065 |
| IID | 60% | 57.554±0.625 | 7.740±0.501 |
| IID | 80% | 54.106±0.552 | 8.090±0.660 |
| Dirichlet(.5) | 0% | 58.139±1.303 | 8.480±0.325 |
| Dirichlet(.5) | 20% | 48.101±6.009 | 63.527±28.993 |
| Dirichlet(.5) | 40% | 36.247±3.396 | 97.190±1.914 |
| Dirichlet(.5) | 60% | 39.473±1.889 | 94.447±2.054 |
| Dirichlet(.5) | 80% | 39.633±1.727 | 92.860±1.623 |

30项新版VERT和原30项H84均通过单元健康检查，六组VERT clean效用均通过；这不表示所有防御或比较标准通过。新版VERT成为更强的健康参照后，H84在原冻结描述性标准下重新计算如下，原始H84成绩没有改变：

| seed | 原严格口径 | 既有Krum IID clean效用例外口径 |
|---|---:|---:|
| 2026102801 | 4/10 | 4/10 |
| 2026102802 | 5/10 | 5/10 |
| 2026102803 | 9/10 | 8/10 |
| 合计 | 18/30 | 17/30 |

13项原TAD单元健康失败、Krum的clean效用失败和全部比较失败仍完整保留。Krum IID的既有独立效用例外与原严格结果并列，不扩展为其他豁免；较低ASR伴随低Accuracy不能单独解释为有效防御。H84最初开发8/10、严格9/10及用户接受记录仍保留，不能把接受写成原标准全通过。

本次固定参数依据独立seed2026102901的7500张训练衍生验证集预注册整体规则，验证30/30及全部规则通过后才补三seed官方test；没有执行后备学习率搜索。此前已见旧官方test的后续开发如实披露，不声称全流程未见测试。恢复的论文核心为dense256、原梯度空间A/B、每轮新projector/predictor、随机参与者顺序、history10、5epoch、Adam.01；筛选采用作者VI-C3未知比例K-means(K=2)分支，论文VI-A主比较仍为Top-κ。公共CNN/本地训练/V2定向攻击保持当前六法一致，仍不同于作者原模型与基准，不能称完整作者基准逐位复现。

科学提交`8788f102f740f67402659bee7bf998a958832337`的105项冻结源与Git树一致；2178份合并输入及274份验证来源逐SHA核验。新VERT4500轮实际选择与真实聚合权重、151200次实际攻击target views/SGD/恶意覆盖、30项初始化和24组攻击前前缀/r24模型配对全部通过。两轮预热外K-means实际选择数14–96，没有固定Top20。合并782593次攻击调用和原其余方法审计保留；执行审计逐项核30次启动/成功退出和共享GPU准入，没有新失败或恢复。

本次VERT累计任务墙钟55071.145秒（15.298小时），并行批次从启动至训练终态10426.491秒（约2小时53分46秒）。H84原累计总墙钟39024.610秒，四个不重叠crypto区间20662.371秒，扣除后18362.239秒。耗时包含加载、评估、观测、审计和I/O；不是关闭密码学重跑，不减并行sign/verify累计时间，不能把共享GPU/CPU耗时当独占算法速度。原H84初始化OOM的5.421214秒和旧K-means初始化OOM的2.178092秒分别保留在历史恢复记录，并非本批失败。

完整新导出SHA：`2fd0d9cdad96b9cfacb6ed69da0b482dce4bfab537c118e03b9111810714f1a1`；合并报告SHA：`6586dd3811fff32f32a577aec6814c0127ac047f0e511e494a2d557f95e46d7d`；执行审计SHA：`ed7b2322191d7221efa26d24b4345826deb3ad718a57693f6e99152842f01bc5`。原验证SHA：`23e23a137f4622b867e4841958c0b961425ce10fc739dbf7d3cacb4b1271417a`。

本地及AI Station当前报告完全一致，旧K-means的29个派生文件完整归档到`outputs/cifar_h84_formal150_v1/derived_report_archive/kmeans_20261029_before_paper/report`，早期Top20归档也保留。新report30个文件的SHA、归档和原始输入保护记录在`outputs/cifar_h84_formal_vert_paper150_v1/analysis/current_report_publication.json`；12图目视记录在`report/visual_qa.json`。原H84科学报告SHA仍`9234dd5e992382023a8c3cc61d5504d420f9b0c80717541723aa657384f40fc9`、导出仍`0ffacdf5ff65e2d09ebcc0aa62fef387df9be604ba476bb4d821414243fd7190`；旧K-means科学raw/analysis/export不改。正式入口23测试及独立报告器6测试均两端通过。

```bash
# 已完成；仅复现派生图表时运行，输出到尚不存在的新目录，不重训
python3 render_cifar_vert_paper.py --expected-sha256 2fd0d9cdad96b9cfacb6ed69da0b482dce4bfab537c118e03b9111810714f1a1 --output outputs/cifar_h84_formal_vert_paper150_v1/report_reproduction
```

以下保留历史实现、验证和启动记录；其中“当前”“运行中”等表述仅描述当时状态，最新结论与入口以上述最终结果为准。

---

## 历史：原文核心VERT正式报告工具准备（2026-10-29）

新增独立 `render_cifar_vert_paper.py` 适配105-source正式导出，不修改运行中的训练来源。要求完整30项新VERT、原其他150项值和轨迹不变、固定验证报告SHA及通过的预注册规则、4500实际选择/真实聚合审计、三seed全部排名和健康判定重算。拒绝旧Top20替补、正式后改参、缺失单元、改变其他方法或隐藏额外执行尝试。

全部训练及自动CPU审计结束、主进程退出后，先在AI Station对实际导出SHA运行 `python3 render_cifar_vert_paper.py --expected-sha256 <实际SHA> --audit-execution`。该操作只读本批controller与scheduler事件并保存派生execution_audit，核30单元逐项仅一次启动/成功退出、GPU身份/准入、无本批失败或恢复；若有新失败或恢复，拒绝无失败结论，必须先另行审查，不重训或自动恢复。下载并校验新export和execution_audit后，本地 `python3 render_cifar_vert_paper.py --expected-sha256 <同一SHA>` 生成每seed四组共12PNG/SVG及完整HTML；可显式指定 `--source --parent --validation --execution-audit --output`。输出目录必须尚不存在，不能直接覆盖旧报告。

新图只显示原文核心VERT及其余五法，MNIST配色、逐seed分开，含全部180原值/健康失败/六法原始排名/严格与Krum独立例外/跨seed均值样本SD/n/尾10、最终、攻击期和峰值/含扣crypto成本。旧科学原件保持，最终12图逐张目视之后才归档旧derived并切当前入口。此时训练仍进行，工具准备完成不代表正式结果或图表已经完成。

## VERT 原文核心：验证完成，固定参数补跑原三 seed（2026-10-29）

独立验证 `outputs/cifar_vert_paper_validation150_v1/analysis/paper_validation_results.json` 已完成30项150轮、3000轮实际VERT选择/权重、24组前缀及r24模型配对、8个FedAvg攻击门槛审计。报告SHA256：`23e23a137f4622b867e4841958c0b961425ce10fc739dbf7d3cacb4b1271417a`。只用训练衍生7500验证图，seed2026102901；此前已见官方test的后续开发如实披露。这是单seed描述性验证，不能直接当成原三seed正式test结果或统计显著性结论。

末10轮、八攻击场景等权均值：PaperVERT Acc48.967% / ASR48.9883%；CurrentVERT39.619% / 93.86%；FedAvg41.7512% / 93.4767%。预注册全部条件通过，固定原文核心Adam.01，不启动后备学习率搜索。改善主要来自IID：四攻击Acc57.4407% / ASR7.9967%；Dirichlet四攻击40.4933% / 89.98%，其中40/60/80%攻击ASR仍98.8533/95.4667/94.64%，不得概括为每场景都优于FedAvg。

用户已确认继续原seed2026102801/02/03×IID/Dirichlet(.5)×0/20/40/60/80%，30项VERT各150轮。新增独立入口/调度/审计/配置/测试，冻结原100来源，新105来源；只采用验证选定的dense256、raw A/B、每轮新投影器和预测器、history10、5epoch、Adam.01、Kmeans2及私有随机参与者顺序。公共模型、训练、攻击、数据划分、数值策略及原其他五法150项保持；这不是完整作者基准逐位复现。正式Acc为官方10000图，ASR为官方source5的1000图到target7，不再按正式成绩改参、换seed或控制名次。

```bash
python3 run_cifar_vert_paper_formal.py --ready
python3 run_cifar_vert_paper_formal.py --batch --max-gpus 6
# 只读状态；不得因历史PID或部分完成重复启动
python3 run_cifar_vert_paper_formal.py --summary
```

新输出 `outputs/cifar_h84_formal_vert_paper150_v1`。准入仍空闲8192MiB连续两次间隔5秒并锁内复核，利用率仅记录，一卡一项，隔离原GPU3和GPU6，最多六卡。TERM/INT停新派发并排空；任何新失败保留，不能自动重试partial或套用旧OOM恢复。每轮指标及审计完整，checkpoint仅r0/r24/末轮。

训练完自动核30新身份/初始化/数据/软件、4500实际选择与聚合权重、实际攻击target views/SGD及覆盖、24前缀配对，再将新30VERT与原150合并，重算所有健康/原始排名/严格标准及Krum独立例外。产物为 `analysis/supplement_results.json`、`analysis/combined_results.json`、`export/combined_bundle.json`。主进程活着且panel_terminal时可能CPU审计，禁止重启；只有全worker终态且主入口异常退出，才能`--finish`修复派生报告而不重训。新图每seed单独、MNIST配色、不展示旧Top20；缺项不可用旧VERT补齐。原科学raw/analysis/export保留，最终交付再归档旧derived并切当前报告入口。耗时是实测总墙钟减四个不重叠crypto区间，不能作为共享环境下的独占速度证据。

## 历史：VERT论文配置优先验证方案（2026-10-29，已完成）

用户要求优先复用原文配置；若验证集明显提升，固定该配置再补原三个seed正式对比，否则才另行有界调参。新入口 `run_cifar_vert_paper_validation.py` 仅使用训练拆分的7500验证图（source5共750），独立开发seed2026102901，训练40000/攻击辅助2500、公共CNN/E1/lr.05/.99/150轮/V2攻击全部保持。已见旧官方test成绩的后续开发如实披露；不以test成绩或第二/第三名选参，不覆盖当前已交付六法报告。

已核论文VI-A/VI-C3及作者 `mylab426/VERT@cc71a9dff95085219cdc2d6c033c10c13ed3e6e7` 的 defenses.py、utils/utils.py、models/ProjectHead.py、main-single.py、conf.json。论文主比较为Top-k；本研究沿用户要求采用作者未知比例K-means K=2分支。当前适配版与原核心的差异包括signed-hash投影、投影后系数、固定投影器、预测器lr.001和固定字典序处理。新独立 `cifar_vert_paper_core.py` 恢复稠密线性投影、原梯度空间A/B、每轮新投影器/预测器/全1系数、共享预测器按随机参与者顺序逐个训练即评分，以及history10/5epoch/Adam.01/256维/三Linear两ReLU。作者CIFAR模型62006参数，本项目公共CNN1687818参数；输入维度适配当前模型，不能称完整论文基准逐位复现。随机初始化/参与者顺序使用私有seed以保持共同训练RNG；历史转移批处理经作者逐条L2求和+三个Adam优化器数值对照测试。

先运行独立12轮资源探针（seed2026102990，仅资源/正确性，不作性能选参）；再运行三个固定臂 CurrentVERT/PaperVERT/FedAvg × IID/Dir(.5) × 0/20/40/60/80%，共30项/4500轮。所有项完整保留，不按早期成绩剪枝。预先锁定“明显提升”：全30完整有限、全部clean/attack前缀/r24配对和八FedAvg攻击门槛通过；PaperVERT两clean精度距FedAvg≤5pp；八攻击场景等权平均Acc≥FedAvg且较CurrentVERT下降≤1pp；平均ASR较FedAvg和CurrentVERT各降低≥5pp。该规则不是显著性检验，也不规定排名。失败时停止扩展并读诊断，不自动反复试至满意。

```bash
python3 run_cifar_vert_paper_validation.py --config configs/cifar10_vert_paper_resource_probe12_v1.json --ready
python3 run_cifar_vert_paper_validation.py --config configs/cifar10_vert_paper_resource_probe12_v1.json --batch --max-gpus 1
python3 run_cifar_vert_paper_validation.py --ready
python3 run_cifar_vert_paper_validation.py --batch --max-gpus 6
```

远端原生PyTorch Adam连带导入旧ONNX时触发Protobuf描述符兼容错误，发生在单元测试阶段，尚未训练。新入口进程内固定 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，并固定OMP/OpenBLAS/MKL线程2；不安装、卸载或降级共享依赖，不改变Adam公式。环境进入配置、启动和worker锁校验，三臂使用相同进程设置。兼容设置下两端27测试通过。

输出分别 `outputs/cifar_vert_paper_resource_probe12_v1`、`outputs/cifar_vert_paper_validation150_v1`；最终验证审计 `analysis/paper_validation_results.json`。每轮实际分数/聚类/选择和实际聚合权重独立记录，核7500验证身份、实际攻击target views/覆盖、初始化与配对；不读取官方test指标来选参。所有臂仅在r0/r24/末轮写完整checkpoint，逐轮分类/选择/状态全部保留；partial不自动恢复。共享GPU准入仍8GiB连续两次5秒+锁内复核、利用率仅记录、一卡一项，隔离既有GPU3/GPU6，最多六卡。TERM/INT停派发并排空，不杀其他任务；资源错误保留，不自动重试。原95冻结源码和旧科学输出不改。

---

## 历史结果：VERT K-means 30项补跑与逐seed图已完成（2026-10-29）

原seed2026102801/02/03 × IID/Dirichlet(.5) × 0/20/40/60/80%，30项VERT均完成150轮、数值有限并通过原单元健康和六组clean效用检查。北京时间11:07:39启动、13:07:46训练结束；自动审计完成，没有追加训练或根据成绩选版本。当前六法报告已统一替换全部30项VERT，其余五法150项成绩和轨迹逐值不变。

[当前完整六法报告](outputs/cifar_h84_formal150_v1/derived_report_archive/kmeans_20261029_before_paper/report/report.html)包含180项原值、60组跨seed均值/样本SD及n、尾10/最终轮/全攻击期/峰值、原始排名/健康资格、原严格与既有Krum IID效用例外并列。每seed四组图（尾10、IID全程、Dirichlet全程、含/扣crypto耗时），共12 PNG＋12 SVG，均已逐张目视核验，颜色严格沿用MNIST。尾10主图：[seed2026102801](outputs/cifar_h84_formal150_v1/derived_report_archive/kmeans_20261029_before_paper/report/tail10_s2026102801.png)、[seed2026102802](outputs/cifar_h84_formal150_v1/derived_report_archive/kmeans_20261029_before_paper/report/tail10_s2026102802.png)、[seed2026102803](outputs/cifar_h84_formal150_v1/derived_report_archive/kmeans_20261029_before_paper/report/tail10_s2026102803.png)。**新图和当前对比中不保留旧Top20曲线、数值或附图。**

六法攻击场景总览（先对每seed的八个攻击场景等权平均，再计算三seed均值±样本SD，n=3；单位%，尾10=141..150；所有失败结果均保留）：

| 方法 | Accuracy | ASR |
|---|---:|---:|
| Ours (H84) | 54.571±0.453 | 8.563±0.999 |
| TAD | 49.334±0.751 | 50.798±1.760 |
| VERT (K-means) | 41.049±0.299 | 90.834±0.412 |
| AlignIns | 43.619±0.639 | 78.788±0.657 |
| Krum | 37.125±0.839 | 12.298±4.637 |
| FedAvg | 42.188±0.853 | 91.604±1.619 |

VERT K-means逐场景结果（每场景三个seed，均值±样本SD，单位%）：

| 分布 | 恶意比例 | Accuracy | ASR |
|---|---:|---:|---:|
| IID | 0% | 56.000±0.560 | 7.560±1.370 |
| IID | 20% | 41.826±1.563 | 91.467±1.810 |
| IID | 40% | 42.134±2.854 | 90.690±2.423 |
| IID | 60% | 42.480±3.553 | 89.357±5.221 |
| IID | 80% | 41.194±2.265 | 89.017±4.830 |
| Dirichlet(.5) | 0% | 55.290±0.908 | 7.737±2.138 |
| Dirichlet(.5) | 20% | 38.690±0.911 | 94.573±2.169 |
| Dirichlet(.5) | 40% | 40.635±1.372 | 92.997±2.061 |
| Dirichlet(.5) | 60% | 40.187±1.978 | 90.927±2.579 |
| Dirichlet(.5) | 80% | 41.245±2.299 | 87.643±4.817 |

K-means消除了固定准入20个的限制：4500轮实际分数、聚类标签、选中客户端与权重已回放，除两轮全体预热外实际准入范围3–76。六组clean相对FedAvg精度损失3.623–4.422pp，满足原5pp健康效用门槛，因此新VERT可进入三seed两分布的健康参照。但八个攻击场景的跨seed平均ASR仍87.643%–94.573%，本项目适配实现未有效抵抗当前V2攻击；不能将健康通过写成防御有效，也不能据此断言作者原方案普遍无效。

H84原30项健康通过。重新纳入新VERT后，既有Krum IID例外口径三个seed仍8/10、8/10、9/10（25/30），原严格口径8/10、9/10、10/10（27/30）；五项当前比较失败及13项TAD单元健康失败完整保留。Krum较低ASR伴随精度损失，Non-IID clean效用不通过；原始总览不等于健康筛选后的名次或统计显著性检验。

科学提交`d795da7e131057c90f681b2e923a9a785e909141`的95项冻结源与Git树匹配，2148份合并输入SHA核验不变。新VERT151200次实际攻击调用、目标视图/SGD与恶意覆盖、24组攻击前前缀/r24模型配对全部通过；合并782593次攻击调用和原其余方法审计保留。所有新VERT初始化均与原正式同seed模型相同。完整导出SHA：`81852f8566d609180bb3b8c18c7241acdc348887e2b933d8f748d60a330ed9ea`；合并报告SHA：`8b990d75a5b67067bf53c7a673dcc1f40e23500faaeab4a45d265e99d6051e90`。

本次31执行尝试对应30科学单元，唯一已审查的初始化OOM成本2.178092秒、额外完成训练轮0，失败原件永久保留；失败未保存初始模型，不能声称与失败模型逐位比较。VERT累计任务墙钟36882.042秒（10.245小时），并行批次墙钟约2小时00分07秒。H84原累计总墙钟39024.610秒，扣除四个不重叠crypto墙钟区间后18362.239秒；这不是关闭密码学重跑，共享GPU/CPU耗时不能作为独占算法速度证据。

两端当前报告为同一套已核验图表，旧22份派生报告文件完整移入`outputs/cifar_h84_formal150_v1/derived_report_archive/top20_20261029_before_kmeans/report`，原科学analysis/export不改。`analysis/current_report_publication.json`记录归档/新文件SHA，`report/visual_qa.json`记录12图目视检查。原报告SHA仍`9234dd5e992382023a8c3cc61d5504d420f9b0c80717541723aa657384f40fc9`，原导出SHA仍`0ffacdf5ff65e2d09ebcc0aa62fef387df9be604ba476bb4d821414243fd7190`。

论文核验边界不变：VI-A主比较Top-κ；VI-C3未知恶意比例使用K-means(K=2)。本次仅修筛选，保留本项目预测器与投影适配；它是看过正式test后由用户要求的补充修订，不能称作者完整模型复现或未见测试的独立选择。补跑/恢复/绘图两端共28测试通过；本次完成后不再训练、调参或自动补跑。

```bash
# 已完成；仅在需要复现派生图表时运行，指定一个尚不存在的新目录
python3 render_cifar_vert_kmeans.py --expected-sha256 81852f8566d609180bb3b8c18c7241acdc348887e2b933d8f748d60a330ed9ea --output outputs/cifar_h84_formal_vert_kmeans150_v1/report_reproduction
```

以下为实现与启动记录；其中原H84正式结果节属于替换前历史，当前比较以上述新报告为准。

## VERT K-means：逐 seed 图表与当前六法报告

补跑完整审计后执行 `python3 render_cifar_vert_kmeans.py --expected-sha256 <export/combined_bundle.json的SHA>`。新入口核30项新VERT身份、其余150项值不变、180条完整曲线、三seed均值/样本SD、原严格及Krum例外排名重算。要求完整补跑及独立恢复审计，不用旧VERT补缺失项。

每个seed分别输出 `tail10_s<seed>`、`trajectories_iid_s<seed>`、`trajectories_dirichlet_s<seed>`、`timing_s<seed>` 四组PNG/SVG，三个seed共12组。颜色与MNIST相同：Ours浅蓝、VERT粉红、AlignIns青、Krum黄绿、TAD橙、FedAvg棕；线型辅助区分。新图只绘K-means版VERT，没有旧Top20曲线、原始数值或附图。HTML保留所有六法、失败seed、尾10/最终轮/全攻击期/峰值和含/扣密码学成本，并公开事后修订与剩余实现适配边界。原科学目录只归档，不从中择优展示。

## VERT K-means 补跑：一次已审查的初始化显存故障恢复（2026-10-29）

`VERT_dirichlet_r20_s2026102801` 首次在物理 GPU3 运行，派发锁内复查空闲16634MiB；约4秒后初始Accuracy前向OOM，另一进程占22.56GiB。没有完成round0、没有训练轮/模型检查点/成绩。first_failure SHA为`12198451a193669bf80e72d4449377dc2e48c1e8c7cb58779d3076026256b2d1`。原队列自动停止派发、自然排空其他6项，失败不计作算法健康失败。

独立 `recover_vert_kmeans_pre_eval_oom.py` 默认只读，只允许上述已审SHA、task和未完成初始评估的确切故障；拒绝partial/其他失败/重复恢复。`--apply --failure-sha256 <上述SHA>` 在旧worker全部结束后完整封存失败目录及日志，`--continue` 仅一次同seed/config fresh接续。95项冻结源/manifest不改；隔离GPU3及原GPU6，其余6卡动态，每卡一项、8GiB连续两次准入不变。没有自动重试循环，不杀其他任务、不降batch或改数值策略。

补跑最多31次尝试对应30科学单元。最终 `analysis/recovery_final_audit.json` 独立绑定合并报告SHA，核身份/数据/目标相同、新模型与原正式同seed初始化匹配，并披露失败额外成本2.178092秒。失败发生时没有保存初始模型，因此不声称失败attempt模型已逐位核验；失败证据永久保留。

## VERT K-means 同种子补跑（2026-10-29）

用户在原180项正式结果后明确要求把VERT改为K-means，其他条件保持一致。独立入口 `run_cifar_vert_kmeans_supplement.py` 使用 `configs/cifar10_vert_kmeans_supplement150_v1.json`，只补跑VERT：seed2026102801/02/03 × IID/Dirichlet(.5) × 0/20/40/60/80%，共30项×150轮。原其余五法150项完整复用；不依据新成绩选择是否替换，30项全部完成审计后统一替换当前对比里的VERT。

已核用户提供的正式论文 DOI `10.1109/TDSC.2025.3630175`：VI-A主实验实际采用统一Top-κ，VI-C3（2756–2757页）研究未知投毒率，以K-means K=2筛选高相似度簇。作者 `mylab426/VERT` 提交 `cc71a9dff95085219cdc2d6c033c10c13ed3e6e7` 的 `defenses.py` 233–247行包含该聚类分支（原仓库中被注释）。本补跑启用其分数升序/补零列/最大分数所属簇策略，`random_state=42`；明确固定Station现有scikit-learn1.2.0的有效默认值n_init10、k-means++、Lloyd、max_iter300、tol1e-4，防止库版本默认值漂移。每轮保存实际分数/簇标签/选中ID/权重，结束后回放核验4500轮决策。两轮预热保留；无真实比例、恶意身份或test信息输入聚类器。

唯一ExperimentConfig变化为 `vert_top_k:20→0`，`vert_use_ratio_prior=false`保持。预测器history10/epochs5/lr.001、signed-hash256投影及投影空间系数、数据划分、CNN、SGD、V2攻击、数值策略、150轮和测试分母全部保持原样；仅用独立作用域适配官方聚类，不修改原89项科学源。此处是选择策略修订，不声称已完整复现作者的模型/投影实现。原test已经查看，补跑作为明确披露的事后协议修订，不称未见测试，不新增搜索或结果择优。

```bash
python3 run_cifar_vert_kmeans_supplement.py             # 计划，只读
python3 run_cifar_vert_kmeans_supplement.py --ready     # 来源/环境/数据检查
python3 run_cifar_vert_kmeans_supplement.py --batch --max-gpus 8
python3 watch_cifar_experiment.py --help               # 只读进度显示用法
```

新输出 `outputs/cifar_h84_formal_vert_kmeans150_v1`；共享GPU沿用8GiB空闲、连续两次间隔5秒、UUID锁内复核和每卡一项。保留此前异常GPU6 UUID隔离，实际最多七卡；不杀其他任务，不自动重试partial/失败。结束自动生成 `analysis/supplement_results.json`、`analysis/combined_results.json` 和 `export/combined_bundle.json`。合并必须有30个完整finite新VERT，否则不能混入旧VERT补齐；旧原始记录独立归档，新的六法图表与数值只含新K-means VERT。

最终按每个seed单独出六法图，采用MNIST的固定配色：Ours #8DBAD9、VERT #E99092、AlignIns #8ADDE4、Krum #DCDD8F、TAD #FEBD85、FedAvg #C4A9A2。保留完整Acc/ASR轨迹、尾10/最终轮/全攻击期/峰值、健康与全部原始排名、原严格及既有Krum IID例外判定；总时间、crypto四段及扣除时间沿用原定义。共享负载时间不作独占速度结论。

# SM9-RRS-FL: 联邦学习中基于SM9环签名的鲁棒聚合方案

## 历史：替换VERT前的H84 CIFAR-10正式结果（2026-10-29）

固定H84、三个新seed2026102801/02/03、两分布、五比例、六方法，180项全部完成150轮且数值有限。北京时间10月28日11:47启动，29日05:39训练收尾、05:40完成自动审计；批次训练墙钟约17小时52分，包含共享资源等待。**H84全部30项原健康通过，但预先固定的比较标准未全面通过**：沿用此前Krum IID clean效用例外，三个seed分别8/10、8/10、9/10（25/30）；原严格口径分别8/10、9/10、10/10（27/30）。正式test没有回流调参、换seed、重选候选或按成绩剪枝。

H84尾10轮均值±样本标准差（独立seed数n=3，单位%；Non-IID为Dirichlet alpha=0.5）：

| 分布 | 恶意比例 | Accuracy | ASR | 最终诚实误撤销率 |
|---|---:|---:|---:|---:|
| IID | 0% | 59.170±0.248 | 7.240±0.372 | 3.667±4.726 |
| IID | 20% | 58.560±0.545 | 7.660±0.562 | 3.333±3.146 |
| IID | 40% | 57.586±0.352 | 7.980±0.202 | 1.667±0.000 |
| IID | 60% | 55.047±0.440 | 8.330±0.165 | 0±0 |
| IID | 80% | 52.073±0.396 | 10.167±1.556 | 0±0 |
| Non-IID | 0% | 57.683±0.692 | 7.250±0.918 | 4.000±2.646 |
| Non-IID | 20% | 56.459±0.588 | 6.817±0.047 | 3.750±2.165 |
| Non-IID | 40% | 54.942±0.926 | 7.577±1.735 | 3.889±0.962 |
| Non-IID | 60% | 52.539±0.905 | 11.050±2.022 | 3.333±2.887 |
| Non-IID | 80% | 49.360±2.067 | 8.927±5.785 | 1.667±2.887 |

最新口径的五个未达标场景均保留：seed2801的IID clean/Non-IID clean精度差距1.873/1.551pp，以及seed2802的Non-IID clean差距2.042pp，超过clean比较标准1pp；seed2802 IID20的ASR名次第三（差距1.21pp仍不能豁免名次）；seed2803 IID80的ASR为11.96%，比Krum高5.75pp，超过5pp。clean原健康效用与比较门槛不同，不能把这三项clean比较失败误称为H84健康失败。参照资格按同seed、同分布五比例及clean效用确定，不为跨seed均值另造资格。

六法原始结果/排名全部展示，13项TAD误撤销健康失败保留在原始均值中；VERT与Non-IID Krum的clean效用失败及各seed排除原因另列。Krum IID例外只豁免原clean精度效用失败，不代表该门槛原本通过。报告同时展示最终轮、0..150全程曲线、25..150全攻击期均值和逐seed峰值：例如H84 IID80第三seed峰ASR为24%，不能用尾10均值掩盖。

审计核对89项冻结来源与科学提交`409d1eb9fe52e8eb44523ab2093885e19464a8d7`、1764份输入SHA、数据与软件/native身份、782593次实际攻击调用和目标视图/SGD、H84逐客户端机制及TAD4500轮实际无先验决策。120组要求配对的前缀/r24模型检查通过，24个FedAvg攻击门槛通过。Krum另24组配对中23组有差异，原Byzantine数量配置差异单列披露，不要求其与clean一致；未声称TAD内部谱状态逐位回放。

H84三十项累计任务墙钟39024.610秒（10.840小时），其中四段实测密码学墙钟20662.371秒（5.740小时），扣除后18362.239秒（5.101小时）。全部180项累计任务墙钟44.065小时，与并行批次17.868小时不同。**扣除密码学是同次总墙钟减四个不重叠协议区间，非关闭密码学重跑**；并行sign/verify累计值不得重复扣除。基线不执行Ours密码协议，crypto为0；总墙钟含加载、评估、观测、审计和I/O，checkpoint写盘是结构化持久化时间的子集。共享GPU/CPU时间仅为本次运行成本，不能作为独占算法速度优势证据。

原VERT round0 OOM及日志永久封存，独立恢复审计确认同task identity/初始化：180科学单元、181执行尝试、额外完成训练轮数0，失败尝试成本5.421214秒单列。没有追加其他重试或修改冻结训练源。最终原报告`analysis/formal_results.json` SHA256为`9234dd5e992382023a8c3cc61d5504d420f9b0c80717541723aa657384f40fc9`，独立`analysis/recovery_final_audit.json`与其绑定。

独立只读绘图入口`render_cifar_h84_formal.py`验证固定导出SHA、180项轨迹、60组统计及三个seed的原严格/最新判定，生成[归档的替换前HTML报告](outputs/cifar_h84_formal150_v1/derived_report_archive/top20_20261029_before_kmeans/report/report.html)、六组PNG/SVG/PDF和核验manifest；六组图已逐张目视核对。原始报告保持不变，输出目录默认必须不存在。失败seed不删、缺失不填0，只有跨Python版本派生SD的浮点末位差允许极小容差，原样本、均值、计数和判定精确核对。

```bash
python3 -m unittest tests.test_h84_formal_render -v
# 仅重算派生统计和绘图，不训练/不重评官方test；若report已存在，用新的--output目录
python3 render_cifar_h84_formal.py --expected-sha256 0ffacdf5ff65e2d09ebcc0aa62fef387df9be604ba476bb4d821414243fd7190
```

以下保留正式实验启动、开发及恢复历史说明，不表示需要再次启动训练。

## 接受H84后的CIFAR-10正式实验（2026-10-28）

用户在看过完整排名后明确接受当前H84，授权安排正式实验。这是结果已知后的独立采纳决定，覆盖下文历史状态中“尚不能进入正式实验”的推进限制；原最新标准8/10、原严格标准9/10及两个失败场景仍原样保留，不改写成全部达标。H84、V2攻击及六法参数均冻结，正式测试结果不回流调参、不按成绩剪枝、不复用开发成绩填补正式单元。

独立入口`run_cifar_h84_formal.py`与`configs/cifar10_h84_formal150_v1.json`已获用户明确确认：seed2026102801/02/03 × IID与Dirichlet(alpha0.5) × 0/20/40/60/80% × FedAvg/H84/TAD-no-prior/VERT/AlignIns/Krum，共180项、每项150轮（27000轮）。用户于2026-10-28明确选择三个新种子、180项。输出`outputs/cifar_h84_formal150_v1`独立于开发目录。共享GPU动态准入沿用空闲显存≥8192MiB、连续2次5秒采样、利用率仅记录；每卡一项、最多8卡，同一UUID锁跨批次共用。partial、OOM或未知失败保留，禁止自动重试或改参数。

数据仍为40000张联邦训练图和2500张攻击辅助图，不把7500张开发验证图并回训练。正式Accuracy在官方10000张测试图计算，独立ASR为其中1000张source=5测试图被预测为target=7的比例；实际攻击优化仅使用原训练辅助池的128张图及固定增强。正式测试不作选择输入。TAD继续原无先验实现，Krum继续原Byzantine数量配置，两者边界不混淆。

逐轮0..150保存Accuracy、独立ASR、辅助目标指标、分类混淆矩阵、误撤销/恶意撤销和累计阶段耗时。主要汇总为141..150轮均值，同时保留最终轮、完整曲线、全攻击期和峰值；跨seed统计使用独立seed的均值和样本标准差、明确实际n，不把轮数当样本量。所有六法原始结果与失败保留；原严格健康口径和此前Krum IID clean效用例外分开报告，均只是固定标准下的结果描述，不启动新一轮测试集搜参。

`status.json`/`result.json`中的`summary`保存`runtime_seconds`（含密码学）和`runtime_without_crypto_seconds`（扣除密码学墙钟区间）；`timing.json`单独校验并保存四个不重叠区间：setup、packet、audit、finalize，以及训练、攻击、检测、聚合、评估、hash、包构造、签名、验签分项。**不含密码学耗时=同次运行总墙钟时间−四个实测密码协议墙钟区间，并非关闭密码学重跑。** 并行签名/验签操作耗时可重叠，不能把这些累计值再次从总耗时扣除。

当前worker总耗时含数据加载、评估、观测、审计和checkpoint I/O；新`timing.json.measured_persistence`另记录实际结构化写盘和checkpoint写入时间。原继承的`summary.checkpoint_io_seconds`没有被该worker填充，不应误当已剔除I/O。每轮`cumulative_stage_timings`保存阶段累计值。共享GPU/CPU争用影响墙钟时间，因此这些数值用于可核验运行开销记录，不能直接宣称独占硬件下的方法速度优势。

```bash
python3 -m unittest tests.test_h84_formal tests.test_cifar_run_observer -v
python3 run_cifar_h84_formal.py          # 仅列出180项计划和来源指纹，不训练
python3 run_cifar_h84_formal.py --ready  # 核原开发报告SHA、数据与CUDA/native环境，不训练
python3 -u run_cifar_h84_formal.py --batch --max-gpus 8
# 另一终端实时查看；Ctrl-C只关闭显示，不中断后台实验
python3 watch_cifar_experiment.py --output outputs/cifar_h84_formal150_v1
# 一次快照，包括逐任务Acc/ASR、GPU、轮数及已完成耗时
python3 watch_cifar_experiment.py --output outputs/cifar_h84_formal150_v1 --json
```

动态显示包含总进度条、完成/失败项数、各卡任务与轮数、当前Acc/ASR、用时及基于已观察速度的ETA。非TTY按间隔输出日志。训练与显示分离，断开显示不影响worker。完成后自动审计来源、每次攻击、Ours机制、TAD决策及攻击前配对，生成`analysis/formal_results.json`；若主入口在训练全部完成后异常退出，可用`--finish`仅恢复派生报告。`--summary`生成当前未审计汇总，不把完成项数当作健康通过。

2026-10-28 11:47北京时间，正式批次已在AI Station启动，冻结科学提交`409d1eb`、89项源码配置已与Git提交树逐项核对，两端25项测试和远端`--ready`通过。原数据/软件/native环境一致；初始实际三卡并发，空余卡动态加入，仍以实时状态为准。

第四项`VERT_iid_clean_s2026102801`在GPU6复查空闲17287MiB后派发，数秒后发生OOM：错误记录显示另一进程占21.89GiB，尚未完成第1训练轮。控制器停止新派发并排空已有任务，失败不是算法健康或性能结论。独立`recover_cifar_h84_initial_oom.py`仅允许这一个经审查的round0 VERT OOM恢复：默认只读核验，`--apply --failure-sha256 <已核错误SHA>`须原控制器完全收尾，完整封存失败目录和日志后才允许同配置同seed fresh尝试一次；partial、其他错误、训练后期失败和再次失败均不自动重试。`--continue`接续原180项manifest，对本批隔离实际发生显存突增的GPU UUID，其余七张卡仍动态调度、每卡一项，其他准入及科学参数不变。恢复控制器的SHA和设备隔离独立记录，不修改89项冻结执行来源。最终另生成`analysis/recovery_final_audit.json`保留原失败成本、180科学单元/181执行尝试、同初始化核验；不得只展示成功尝试而隐藏OOM。

## CP1的IID失效诊断与独立CUDA开发（2026-10-27）

固定CP1本地60项验证已完成，但尚未达标：IID60%诚实误撤销7/40=17.5%，IID60/80%末10轮ASR为93.37/92.88%。IID20/40%末10轮Accuracy为59.06/58.16%、ASR为9.96/10.36%；40%首轮攻击仍有短暂严重退化，不能只报告尾段。首轮漏入先于误撤销，恶意更新随后进入正常历史。原MPS结果和科学源码保持冻结。

新入口`run_cifar_cp1_cuda_development.py`使用`configs/cifar10_cp1_cuda_diagnosis150_v1.json`，在独立目录开展150轮开发。第一批为IID clean/60/80%各FedAvg、CP1、W85、W80，共12项1800轮；W85/W80仅改变Ours告警、历史准入及漂移容限三项阈值，其他参数（含累计撤销容忍60）不变。固定V2交替最小化攻击、100客户端、数据划分和已观察开发seed2026102201不变。此批用于检验早期漏入机制，不是最终资格或正式实验。

CUDA统一启用确定性算法，禁用TF32及cuDNN benchmark，设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`。平台/软件变化单独记录，不能将MPS与CUDA成绩混作配对基线。输入数据的全部数组与原MPS哈希核对；使用原攻击及观测代码、原健康函数。任务完整保留失败；partial和无身份残留禁止自动重试。原始串行入口保持冻结。2026-10-27按用户授权，新增独立调度适配器`run_cifar_cp1_dynamic_gpu.py`接续已有manifest：动态检查所有可见4090D，每卡一个独立实验，最多8卡同时运行，不固定GPU4或任何UUID。每张卡仍须连续3次（间隔10秒）满足空闲显存至少16GiB、利用率不高于5%，获取UUID锁后再次检查。无任务运行时等候上限24小时，超时仅表示资源阻塞，不代表算法失败。此抽样准入不等于集群独占预约，其他用户进程不会被终止。

```bash
# 在已配置PyTorch CUDA和SM9 native库的解释器下执行
python3 -m unittest tests.test_cp1_shared_gpu tests.test_cp1_cuda_development tests.test_cifar_tad_no_prior -v
python3 run_cifar_cp1_cuda_development.py          # 仅检查协议与来源
python3 run_cifar_cp1_shared_gpu.py --max-gpus 8   # 仅检查
# 只在旧控制器退出、已有任务均为终态时接续，切勿同时运行两个控制器
python3 -u run_cifar_cp1_shared_gpu.py --batch --max-gpus 8
python3 run_cifar_cp1_cuda_development.py --summary
```

用户随后确认AI Station是共享GPU，允许无视其他任务占用、以本实验正常运行为准。当前改用独立`run_cifar_cp1_shared_gpu.py`：利用率只记录、不阻止派发；每卡空闲显存至少8192MiB，连续2次5秒采样并在锁内复查，仍每卡一项、最多8卡。本批CNN估算worker上界3178MiB、含原75%安全系数需求4238MiB，8192MiB是额外保守准入，不代表已测量完整CUDA峰值或能够预留他人后续使用的显存。约3991/7648MiB空闲的卡暂不准入。新策略仅写`scheduler_shared_v1`，原dynamic_v1审计、71项科学来源、JSON及任务身份保持不变；工作进程仍加载原配置。旧控制器收到TERM后让活动实验自然完成，新入口只在旧控制器退出且无partial时接续。共享负载下耗时只作开发运行记录，不用于论文算法速度对比。

并行调度修订记录在`scheduler_dynamic_v1/policy.json`，每次接续保存原结果SHA和启动/退出事件；冻结manifest、原71项执行来源、攻击和筛选参数均不重写。已完成任务不重跑，partial或身份不明任务阻止派发；工作进程异常后停止派发新任务、等其他实验完成后保留失败证据。SIGTERM/SIGINT也只停止新派发并等待已运行任务收尾；不可把强杀控制器当作安全暂停。跨GPU实验必须实际核验攻击前轨迹和r24模型配对，不能仅凭确定性设置认定它们一致。

阶段完成后检查匹配clean前24轮、FedAvg攻击有效性、原健康、首轮冲击、恢复时间及尾10指标，再决定下一批。最终仍须两分布各0/20/40/60/80%及六法的完整同平台证据：方法需在同分布全部五比例健康后才进入参照集，Ours也必须健康；clean不限制名次，Accuracy/ASR距健康最优≤1/2pp，攻击场景双前二且差距≤2/5pp。六法原始结果和健康失败原因全部保留；不能以本12项开发面板宣称达标，也不自动启动正式测试集实验。

`diagnose_cifar_cp1_iid.py`仅供拥有原本地输出的环境执行，校验原64项科学来源及引用、逐客户端计数/撤销/聚合质量和时序，输出独立诊断JSON；远端不需要复制这些历史输出才能启动新CUDA面板。


### W85历史准入单变量验证（2026-10-27）

用户进一步授权以W85为基准，只调整历史准入。新入口`run_cifar_w85_history_admission.py`与协议`configs/cifar10_w85_history_admission150_v1.json`固定告警0.85、漂移容限0.85、历史确认1、累计撤销容忍60及全部攻击/训练参数，仅历史阈值从W85的0.75改为H80=0.80或H84=0.84（必须严格低于告警阈值0.85）。两个候选各IID clean/60/80%，6项×150轮，共新增900轮；同样使用共享GPU8GiB准入、每卡一项、最多8卡，不依据分数剪枝。H80表示历史阈值0.80，与上一轮告警0.80的W80不同。

先用`audit_cifar_cuda_panel.py`完整审计原12项，包括任务/数据/来源、真实逐次攻击SGD及增强种子与128辅助图身份、全部在线恶意client-round覆盖、配对0..24分类/r24模型、撤销计数/历史准入及聚合质量。新实验仅复用已审计W85和FedAvg的完成结果，不重跑对照；所有原失败照常保留。独立`reference_lock.json`在新训练前冻结原manifest与审计原始输入SHA，旧科学结果不改。

```bash
python3 -m unittest tests.test_w85_history_admission tests.test_cp1_shared_gpu tests.test_cp1_cuda_development tests.test_cifar_tad_no_prior -q
python3 audit_cifar_cuda_panel.py  # 原12项完整后执行一次；冻结引用后勿重复改写旧审计时间戳
python3 run_cifar_w85_history_admission.py  # 仅检查6项设计、来源与参数差异
python3 -u run_cifar_w85_history_admission.py --batch
# 已有批次的只读训练结果汇总及独立派生审计（不会重新训练）
python3 run_cifar_w85_history_admission.py --finish
```

此阶段推进条件独立预声明：三个场景原健康通过、clean效用门槛不变、clean误撤销严格少于W85的10人；相对W85，clean尾10 Accuracy最多降低1pp/ASR最多增加2pp，60/80尾10 Accuracy最多降低2pp/ASR最多增加5pp。符合者依次按clean误撤销数、最坏攻击ASR增量、攻击平均Accuracy、候选ID排序。完整保留首攻、全攻击期、峰值、恢复和机制记录，不把末段成绩当作全程无代价。以上仅用于本轮开发推进，不替代或放宽最终健康参照内双前二、差距1/2pp（clean无名次）及2/5pp（攻击）的标准。入选后仍须补齐IID/Dirichlet各0/20/40/60/80%和六法的同CUDA平台验证；现阶段不使用官方测试集。


### 固定H84完整CUDA验证（2026-10-27）

历史单变量6项已全部完成并审计。H80/H84的clean误撤销均为5/100，较W85的10/100减半；两候选三个场景均通过原健康和开发推进门槛。依预声明选择顺序选H84：clean FP同为5，最坏攻击尾10 ASR增量为0.04pp，小于H80的0.133333pp。H84尾10结果为clean Accuracy58.904%/ASR9.266667%，IID60为55.750667%/10.12%，IID80为51.836%/11.12%；误撤销分别5/100、1/40、0/20。H80在60%下ASR峰值33.866667%，高于W85/H84的20.933333%，保留该早期代价；H84在60%全攻击期ASR比W85高0.213757pp，80%低0.043386pp，不将尾段结果表述为全程无退化。

独立入口`run_cifar_h84_full_validation.py`、配置`configs/cifar10_h84_full_validation150_v1.json`和输出`outputs/cifar_h84_full_validation150_v1`覆盖IID与Dirichlet(alpha0.5)、0/20/40/60/80%、FedAvg/H84/TAD-no-prior/VERT/AlignIns/Krum，共60单元。同CUDA平台复用FedAvg和H84各IID clean/60/80六项，补跑54项×150轮=8100轮；此前12+6项开发预算与全部失败保留。候选、基线、V2攻击、数据、数值策略和最终健康参照筛选规则固定。仍为已观察seed2026102201的开发验证，不能称为正式或未用seed资格。

训练前核验父审计、选择规则、引用SHA以及Python/PyTorch/CUDA/cuDNN/native库一致性；GPU UUID可不同。每卡一项、最多8卡、共享GPU8GiB准入继续适用，不依据成绩剪枝。结束后逐单元核验实际攻击target views、完整结果与Ours机制、TAD1500轮实际无先验决策、攻击前分类/r24模型、FedAvg八项攻击门槛、原健康和六法原始排名。Krum原基线使用配置的Byzantine数量，故单独披露它的clean/attack前缀差异，不冒称配对一致；其参数未调整。TAD不接收该先验。任何缺失/未核验单元阻止完整资格判定，不作为健康失败排除。正式实验仅在完整达标后准备方案，不自动运行官方测试集。

```bash
python3 -m unittest tests.test_h84_full_validation tests.test_w85_history_admission tests.test_cp1_shared_gpu tests.test_cp1_cuda_development tests.test_cifar_tad_no_prior -q
python3 run_cifar_h84_full_validation.py          # 仅检查54新增+6复用的冻结设计
python3 run_cifar_h84_full_validation.py --ready  # 只读核对真实父引用、选择与运行环境
python3 -u run_cifar_h84_full_validation.py --batch
# 仅在全部训练结束、主入口已退出且需要恢复派生报告时执行
python3 run_cifar_h84_full_validation.py --finish
```

父审计和history_decision已作为新reference_lock冻结，勿重新运行父报告器改写时间戳。新任务失败/partial不自动重试；新入口结束后的`analysis/full_validation.json`保留完整原始结果、健康排除原因、健康排名与差距。绘图及正式方案是完成后的独立步骤。


### Krum IID参照的独立用户修订（2026-10-27，结果已知后）

Krum IID clean末10 Accuracy43.792%，比FedAvg59.198667%低15.406667pp，原5pp效用门槛判定仍保留。用户随后明确：若差表现有可分析原因，Krum仍可作参照。只读审计发现其150轮均仅采用1/100更新，19个客户端曾被选中，其中client-94获选113轮；该客户端400张训练图覆盖十类。保存模型的CPU复核：Krum在该客户端准确率100%、7500张独立验证图43.826667%；FedAvg同客户端64.5%、验证59.293333%。初始化、数据和训练配置一致，仅聚合方法不同。该现象支持重复选择集中和局部过拟合的机制解释，未作额外训练；实际CUDA每轮距离分数未保存，不能宣称已回放全部选择或量化全部差距的因果份额。原始单更新Krum选择规则见[原论文第4节](https://papers.nips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)，本项目固定客户端多步local SGD与其独立无偏梯度假设不同。

独立`configs/cifar10_h84_krum_iid_reference_exception_v1.json`记录人类指令、已知结果背景、证据SHA和范围：仅豁免Krum的IID clean Accuracy效用损失；该分布五比例其他原健康仍全部必须通过。Non-IID、其他方法、所有排名/差距门槛不变。这个Krum是按用户例外获得参照资格，不应改称其原clean效用检查通过。原82冻结来源、训练协议、`full_validation.json`严格报告均不修改。

完整矩阵结束后另运行`python3 assess_h84_krum_reference.py`，核对原报告、父引用、机制证据后生成`analysis/krum_reference_exception_assessment.json`；启动前可用`--verify-only`只读核验证据。最新达标结论依这个独立判定，原严格结论一并展示。若纳入Krum导致Ours ASR落到第三，即使差距小于5pp也仍判失败；其他健康失败不豁免，缺失方法不默认为失败而排除。机制证据为单独数据artifact（随协议记录SHA，非新训练），须位于配置指定路径。


### H84完整验证结论（2026-10-27）

60/60单元均完成150轮并通过执行审计；固定H84在最新用户标准下通过8/10场景，尚不能进入正式实验。Ours自身10/10原健康通过；两个未达标项为IID20的ASR第三，以及Non-IID clean Accuracy差1.734667pp超过1pp上限。原严格口径（未采用Krum IID例外）为9/10，仅Non-IID clean失败；两个报告均保留，没有放宽最终名次或差距。

| 分布 | 恶意比例 | Ours Accuracy / ASR (%) | 参照内Acc / ASR排名 | 距最优Acc / ASR (pp) | 结论 |
|---|---:|---:|---:|---:|---|
| IID | 0% | 58.904 / 9.267 | 4 / 3 | 0.295 / 0.707 | 通过（clean不限制名次） |
| IID | 20% | 58.640 / 10.040 | 2 / 3 | 0.433 / 3.107 | ASR名次未通过 |
| IID | 40% | 57.919 / 9.520 | 1 / 1 | 0 / 0 | 通过 |
| IID | 60% | 55.751 / 10.120 | 1 / 2 | 0 / 1.573 | 通过 |
| IID | 80% | 51.836 / 11.120 | 1 / 1 | 0 / 0 | 通过 |
| Non-IID | 0% | 56.847 / 6.480 | 3 / 1 | 1.735 / 0 | Accuracy差距未通过 |
| Non-IID | 20% | 55.963 / 6.680 | 1 / 1 | 0 / 0 | 通过 |
| Non-IID | 40% | 54.903 / 4.733 | 1 / 1 | 0 / 0 | 通过 |
| Non-IID | 60% | 51.453 / 2.853 | 1 / 1 | 0 / 0 | 通过 |
| Non-IID | 80% | 49.585 / 7.267 | 1 / 1 | 0 / 0 | 通过 |

参照资格按同分布五比例整体判断：IID六法参评，其中Krum仅依据已记录的用户例外保留，其原clean效用失败不改写。Non-IID参评FedAvg/Ours/VERT/AlignIns；TAD因clean误撤销21/100排除，Krum因clean Accuracy损失27.409333pp排除，IID例外不适用于Non-IID。所有六法原始结果、排名、排除原因及完整轨迹仍展示。IID20的ASR分别为Krum6.933333%、TAD9.44%、Ours10.04%；距第一虽小于5pp，但第三名仍失败。

两分布八项FedAvg攻击门槛全部通过，Accuracy下降14.148–19.337pp、ASR上升82.093–86.773pp。审计核82冻结来源、162父引用、638原始输入、261871次真实攻击调用；十个Ours机制覆盖和TAD1500轮实际无先验决策通过。48组攻击前配对中40组要求一致者全部通过，8组Krum因原配置Byzantine数量不同而前缀不同，按预声明披露。TAD内部谱状态未逐位回放；其论文歧义解释和默认参数仍完整保存。

尾10之外仍有代价：H84 Non-IID80在r29–57的ASR不低于20%，r53峰值30.4%，全攻击期ASR均值13.308%；该场景r26出现恶意历史准入，不能把末段7.267%解释为全程无退化。其诚实误撤销2/20=10%恰达健康上限。历史阈值优化的有限成功与本轮两处失败均保留，不推断整个参数空间无解。

独立报告入口`render_cifar_h84_validation.py`读取已核验的远端导出`outputs/cifar_h84_full_validation150_v1/export/final_audited_bundle.json`，复算原严格及最新用户判定，核对60条完整轨迹与尾10/健康/来源后，生成`report/report.html`及三组PNG/SVG/PDF（六法原始指标、最新差距/名次、Ours完整轨迹）。不改训练或审计输入；本地导出SHA256为`a3f066ed731e10f99c7e0dbf00379410d491e2ae1c63e840a659361e1b9213b0`。

```bash
python3 render_cifar_h84_validation.py --expected-sha256 a3f066ed731e10f99c7e0dbf00379410d491e2ae1c63e840a659361e1b9213b0
```

这是已观察seed2026102201的开发结果，累计12+6+54个独立训练任务、10800轮；没有使用官方测试集，未宣称正式或未用seed资格。当前冻结候选验证到此得出有边界的未达标结论，不自动启动正式实验或追加未冻结搜索。

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
