# LastFM 中心—邻居逐维融合：完整训练集直接测试探索

用户2026-10-28明确授权变更评估规则。本次重新随机初始化训练，旧验证实验保留。
代码版本：`9b28b63c2383050e5a01d4b808d1b785b9628502`。完整2000轮、114000次更新；原始train57786、test13508。

| 口径 | 实际训练轮数 | P@10 | P@20 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始：400次最佳 | 1920 | 0.200161377 | 0.138945670 | 0.270616990 | 0.376867577 | 0.295404035 | 0.347334159 |
| 共享动态版：400次最佳 | 1645 | 0.200484131 | 0.140559441 | 0.270165945 | 0.379403716 | 0.293662249 | 0.347189833 |
| 非对称注意力对照：400次最佳 | 1650 | 0.201990317 | 0.141420118 | 0.273206921 | 0.381109275 | 0.294477115 | 0.347649019 |
| 本候选：400次最佳 | 1300 | 0.199193115 | 0.139026358 | 0.267328825 | 0.374897965 | 0.290005546 | 0.342513867 |
| 本候选：401次最佳 | 1300 | 0.199193115 | 0.139026358 | 0.267328825 | 0.374897965 | 0.290005546 | 0.342513867 |
| 本候选：最终2000轮 | 2000 | 0.195051103 | 0.135718128 | 0.261247940 | 0.366080928 | 0.282763043 | 0.333809382 |

原协议最佳相对变化：-1.388%；含终点最佳相对变化：-1.388%。
总耗时166.49分钟，峰值分配显存203.77MiB。
计时分项（秒）：`{"training_including_sampling_seconds": 9593.730636300024, "sampling_seconds": 3442.895152299982, "training_after_sampling_seconds": 6150.835484000042, "evaluation_seconds": 337.4628285000022, "checkpoint_seconds": 16.191615799998104}`。训练分项包含采样；评估不含checkpoint保存，保存单独计量，其余差额包括状态JSON、校验与其他开销。
配置、数据/源码哈希、环境及命令见manifest.json，完整曲线见history.json/evaluations.json。
最佳点注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.2559545040130615, "changed_top1_fraction_on_cyclic_query_replacement": 0.025595448911190033, "normalized_attention_entropy": 0.972615122795105, "mean_max_attention_weight": 0.16707190871238708, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 0.0, "full_catalog_logit_min": -9.360589027404785, "full_catalog_logit_max": 16.02808952331543, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder", "mean_absolute_forward_reverse_edge_logit_difference": 0.4394662082195282, "query_key_projection_cosine": 0.020242804661393166, "attention_bias_l2": 0.5270357131958008, "fusion_gate_mean": 0.02790842391550541, "fusion_gate_std": 0.15907897055149078, "fusion_gate_min": 0.0, "fusion_gate_max": 1.0}`。
最终注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.21862779557704926, "changed_top1_fraction_on_cyclic_query_replacement": 0.01635264791548252, "normalized_attention_entropy": 0.9137623310089111, "mean_max_attention_weight": 0.23824791610240936, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 5.354376543088293e-06, "full_catalog_logit_min": -8.746946334838867, "full_catalog_logit_max": 17.818490982055664, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder", "mean_absolute_forward_reverse_edge_logit_difference": 0.8756291270256042, "query_key_projection_cosine": 0.018210330978035927, "attention_bias_l2": 0.5738969445228577, "fusion_gate_mean": 0.026920679956674576, "fusion_gate_std": 0.1572810262441635, "fusion_gate_min": 0.0, "fusion_gate_max": 1.0}`。
在固定非对称动态注意力上，将中心—邻居固定均值替换为内容条件逐维融合；输入ID/语义融合、邻居集合、传播、BPR与原六向量正则保留，新增门控参数L2。方法及来源见content_neighbor_fusion_protocol.md。
反复查看原测试集得到的最大值属于探索最佳，不是独立验证选型后的泛化结论；单种子不证明稳定性。未按曲线调整本次参数或预算。
公开语义资产生成来源不完整这一既有限制保留；没有生成或覆盖原资产。

运行目录：`D:\CoLaKGResearch\content-neighbor-fusion-v1\direct-test-content_neighbor_fusion-s2020-20261030-025447-df2704`。best-protocol.pt、best.pt、final.pt、recovery.pt保存完整模型/Adam/RNG。

```powershell
& 'D:\CondaEnvs\CoLaKG\python.exe' -B experiments/content_neighbor_fusion_2000.py
```

本次进化：完整中心/邻居表示及差值、乘积经过256→128→64的MLP学习逐维融合，无度数输入。
最佳NDCG@20相对共享动态版本：-1.3468%。

Feature-wise center/neighbor fusion on asymmetric attention; see content_neighbor_fusion_protocol.md.
Parent best NDCG@20: 0.3476490190228204
Relative gain vs parent: -0.014771080618479981
