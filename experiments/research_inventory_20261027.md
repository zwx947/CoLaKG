# CoLaKG 已执行研究方案总览（2026-10-27）

依据各分支实验记录及 `D:\CoLaKGResearch` 下成功完成运行的 result.json 核对。这里的“成功完成”只表示程序执行完毕，不表示性能改进成功。不加载 checkpoint，不重新训练或评估。

## 口径

共20种方法实现（含A本身）：A 1种、强化A 2种、A+B 10种、原始模型上的其他独立方向7种。不把配对原始对照、种子补跑、仅对齐消融或启动错误各算成一种新方法。多数方向只有短预算证据，不能合称20个完整2000轮失败实验。

所有方案均以原CoLaKG为基础，但代码起点不同：A+B保留A的几何辅助项，独立B从原始基线出发。新候选从随机初始化训练，不是加载A训练完成的权重微调。固定A开发对照的1000→2000续跑是同一实验完整状态的延续，另行标注。

原 main、原始基线、固定A均保留；主要在 experiments/research_methods 中派生模型和目标，不覆盖原 rec_code/model.py。未发表研究分支仅本地保存。

## A与两次强化

| 方向 | 实际改动 | 已完成预算 | 结果 |
|---|---|---|---|
| A：pair_geometry | BPR基础上增加借鉴SGCL的几何辅助项，温度0.2、权重0.05 | 200轮三种子；2000轮及周期测试补跑 | 原测试400次评估最佳0.3487022586，相对原始0.3473341594约+0.3939%；包含2000轮终点共401次最佳0.3489572007，约+0.4673% |
| a_confidence_geometry | 50轮热身后，辅助对齐只保留batch内余弦最高80%的正例；BPR仍保留所有正例 | 2000轮 | 开发验证最佳0.2724738714，相对固定A的0.2734122028下降0.3432% |
| a_geometry_primary | 去掉BPR，保留0.05×几何目标和原正则，以几何项承担主监督 | 2000轮 | 开发验证最佳0.2406947570，相对固定A下降11.9663% |

强化A两次没有扩大收益；这不抹去原A小幅测试收益。A在相同开发划分下的2000轮最佳0.2734122028，低于原始开发对照0.2772647382；因此不能把A目前的收益称为跨划分/跨种子稳定结论。上述开发值与0.34左右原测试值不能混比。旧1.82%是最终轮对最终轮，不是最佳对最佳。

对应分支：codex/pair-geometry、codex/a-confidence-alignment、codex/a-geometry-primary。

## 在A上叠加的10种B

| 方法名 | 改动 | 已完成预算 | 证据边界 |
|---|---|---|---|
| degree_fusion | 度数等条件控制ID/语义融合 | 200轮 | 未通过短预算推进门槛 |
| cross_residual | ID×语义、ID−语义的低维交叉残差 | 200轮 | 短预算未改善 |
| layer_mixture | 学习0–3层逐维组合权重 | 200轮三种子；2000轮原测试 | 短预算积极，完整预算最佳测试NDCG@20为0.3402583489，未超过A |
| context_negative_mix | 历史条件化难负样本合成 | 200轮 | 短预算未改善 |
| collaborative_rank_residual | 共享ID参数的协同图排序残差 | 200轮 | 短预算未改善 |
| private_cf_rank | 独立ID参数的协同排序分支 | 200轮 | 短预算未改善 |
| gradient_priority | 协调BPR/正则与A辅助项的梯度 | 200轮 | 小幅变化未达到推进门槛 |
| gradient_id_routing | 在ID参数上进行梯度协调/路由 | 200轮 | 未通过短预算推进门槛 |
| propagation_rms | 分类型、分通道校正传播幅度 | 200轮三种子；2020延长1000轮 | 短预算曾积极，1000轮比较未超过对应A；未完成2000轮，不能断言完整预算一定失败 |
| cavity_propagation | 修正多跳传播的立即回返路径 | 200轮 | 未通过短预算推进门槛 |

分支分别为 codex/b-degree-fusion、b-cross-residual、b-layer-mixture、b-context-negative-mix、b-collaborative-rank-residual、b-private-cf-rank、b-gradient-priority、b-gradient-id-routing、b-propagation-rms、b-cavity-propagation（均使用codex/前缀）。

## 从原始CoLaKG独立改的7个方向

| 方法名 | 改动 | 已完成预算 | 结果 |
|---|---|---|---|
| catalog_contrastive | 原BPR外增加目录级辅助对比 | 200轮三种子 | 末三次验证均值相对对照约+0.39%～+0.58%，未达当时推进门槛；无完整测试结论 |
| consensus_residual | 共识邻居与协同残差 | 200轮 | 末三次验证均值约−0.51% |
| semantic_graph_decoder | 独立语义图评分框架 | 200轮 | 末三次验证均值约−1.66% |
| aligned_catalog | 余弦目录目标替换BPR，匹配打分目标 | 200轮 | 末三次验证均值约−6.94% |
| topk_boundary | 排名边界驱动UI消息传播 | 1000轮；同次状态延续到2000轮 | 完整最佳验证0.2769043911，对原始0.2772647382约−0.1300% |
| source_aware_fusion | 按节点度数、表示一致性及幅度比学习输入融合 | 2000轮 | 完整最佳验证0.2638499041，相对原始−4.8383% |
| decoupled_semantic_routing | CF单独传播，语义保留邻居增强，归一化拼接排序 | 2000轮 | 完整最佳验证0.2540454540，相对原始−8.3744% |

早期4个方向的比例为事先固定的末三次验证均值；后3项为最佳验证NDCG@20，不在两种评分规则之间直接排序或拼接。

分支：research/catalog-contrastive、research/consensus-residual、research/semantic-graph-decoder、research/aligned-catalog、codex/b-topk-boundary、codex/b-source-aware-fusion、codex/b-decoupled-semantic-routing。

## 不纳入方法数的事项

- pair_alignment_only是A的组件消融：200轮，仅保留对齐后明显变差，用于理解几何目标，不算独立B。
- 语义图扩散/新增连接、模块B若干文献审查：仅阅读与分析，没有真实训练结果，不能称为实验失败。
- 误启动的师姐参考组合已在56轮停止；停止原因是范围理解错误，不是性能淘汰，不纳入有效候选计数。
- 同一方案不同种子、1000→2000同次状态延续、周期测试补跑及启动器错误修复不重复计数。

所有完成运行的路径、执行提交、预算与方法名见配套 research_records/research_inventory_20261027.json。原始训练日志、曲线和权重保持各自D盘目录。
