# LastFM 非对称语义注意力：完整训练集直接测试探索

用户2026-10-28明确授权变更评估规则。本次重新随机初始化训练，旧验证实验保留。
代码版本：`627f459677c396d9335dc2741be3a7995049b278`。完整2000轮、114000次更新；原始train57786、test13508。

| 口径 | 实际训练轮数 | P@10 | P@20 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始：400次最佳 | 1920 | 0.200161377 | 0.138945670 | 0.270616990 | 0.376867577 | 0.295404035 | 0.347334159 |
| 共享动态版：400次最佳 | 1645 | 0.200484131 | 0.140559441 | 0.270165945 | 0.379403716 | 0.293662249 | 0.347189833 |
| 本候选：400次最佳 | 1650 | 0.201990317 | 0.141420118 | 0.273206921 | 0.381109275 | 0.294477115 | 0.347649019 |
| 本候选：401次最佳 | 1650 | 0.201990317 | 0.141420118 | 0.273206921 | 0.381109275 | 0.294477115 | 0.347649019 |
| 本候选：最终2000轮 | 2000 | 0.199838623 | 0.138407746 | 0.270930779 | 0.373091865 | 0.291304108 | 0.341800437 |

原协议最佳相对变化：+0.091%；含终点最佳相对变化：+0.091%。
总耗时126.64分钟，峰值分配显存192.49MiB。
计时分项（秒）：`{"training_including_sampling_seconds": 7280.924716000012, "sampling_seconds": 2855.184408299995, "training_after_sampling_seconds": 4425.740307700017, "evaluation_seconds": 266.5468341999962, "checkpoint_seconds": 15.494445999997286}`。训练分项包含采样；评估不含checkpoint保存，保存单独计量，其余差额包括状态JSON、校验与其他开销。
配置、数据/源码哈希、环境及命令见manifest.json，完整曲线见history.json/evaluations.json。
最佳点注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.06647706776857376, "changed_top1_fraction_on_cyclic_query_replacement": 0.009598293341696262, "normalized_attention_entropy": 0.9807971119880676, "mean_max_attention_weight": 0.15498904883861542, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 0.0, "full_catalog_logit_min": -11.384486198425293, "full_catalog_logit_max": 16.48358154296875, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder", "mean_absolute_forward_reverse_edge_logit_difference": 0.33146408200263977, "query_key_projection_cosine": 0.028205739334225655, "attention_bias_l2": 0.462803453207016}`。
最终注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.03590472787618637, "changed_top1_fraction_on_cyclic_query_replacement": 0.00817632395774126, "normalized_attention_entropy": 0.9671120643615723, "mean_max_attention_weight": 0.17665249109268188, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 6.884198412542092e-06, "full_catalog_logit_min": -10.66634464263916, "full_catalog_logit_max": 18.741167068481445, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder", "mean_absolute_forward_reverse_edge_logit_difference": 0.45939376950263977, "query_key_projection_cosine": 0.022065848112106323, "attention_bias_l2": 0.48344001173973083}`。
仅替换原模块C的语义对注意力打分，保留用户/物品语义、原融合、邻居集合、传播、BPR与六向量正则。方法及来源见asymmetric_semantic_attention_protocol.md。
反复查看原测试集得到的最大值属于探索最佳，不是独立验证选型后的泛化结论；单种子不证明稳定性。未按曲线调整本次参数或预算。
公开语义资产生成来源不完整这一既有限制保留；没有生成或覆盖原资产。

运行目录：`D:\CoLaKGResearch\asymmetric-semantic-attention-v1\direct-test-asymmetric_semantic_attention-s2020-20261028-233455-8d6c5a`。best-protocol.pt、best.pt、final.pt、recovery.pt保存完整模型/Adam/RNG。

```powershell
& 'D:\CondaEnvs\CoLaKG\python.exe' -B experiments/asymmetric_semantic_attention_2000.py
```

本次进化：中心Q与邻居K独立投影，非线性前偏置可训练，其余编码器继承共享动态版。
最佳NDCG@20相对共享动态版本：+0.1323%。
