# LastFM 中心条件消息转换与尺度校准融合：完整训练集直接测试探索

用户2026-10-28明确授权变更评估规则。本次重新随机初始化训练，旧验证实验保留。
代码版本：`dddbc989829cb3970a8466a1241b097956d4e29f`。完整2000轮、114000次更新；原始train57786、test13508。

| 口径 | 实际训练轮数 | P@10 | P@20 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始：400次最佳 | 1920 | 0.200161377 | 0.138945670 | 0.270616990 | 0.376867577 | 0.295404035 | 0.347334159 |
| 共享动态版：400次最佳 | 1645 | 0.200484131 | 0.140559441 | 0.270165945 | 0.379403716 | 0.293662249 | 0.347189833 |
| 非对称注意力对照：400次最佳 | 1650 | 0.201990317 | 0.141420118 | 0.273206921 | 0.381109275 | 0.294477115 | 0.347649019 |
| 本候选：400次最佳 | 425 | 0.185529855 | 0.130231307 | 0.248043837 | 0.349478726 | 0.267533457 | 0.317108506 |
| 本候选：401次最佳 | 425 | 0.185529855 | 0.130231307 | 0.248043837 | 0.349478726 | 0.267533457 | 0.317108506 |
| 本候选：最终2000轮 | 2000 | 0.143033889 | 0.100215169 | 0.182419717 | 0.258895378 | 0.201173100 | 0.238679663 |

原协议最佳相对变化：-8.702%；含终点最佳相对变化：-8.702%。
总耗时156.87分钟，峰值分配显存207.95MiB。
计时分项（秒）：`{"training_including_sampling_seconds": 9073.53639389998, "sampling_seconds": 2883.6214947000126, "training_after_sampling_seconds": 6189.914899199967, "evaluation_seconds": 280.5221757000137, "checkpoint_seconds": 14.085415700002791}`。训练分项包含采样；评估不含checkpoint保存，保存单独计量，其余差额包括状态JSON、校验与其他开销。
配置、数据/源码哈希、环境及命令见manifest.json，完整曲线见history.json/evaluations.json。
最佳点注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.010309278033673763, "changed_top1_fraction_on_cyclic_query_replacement": 0.0021329540759325027, "normalized_attention_entropy": 0.9948007464408875, "mean_max_attention_weight": 0.12557505071163177, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 0.0, "full_catalog_logit_min": -9.729482650756836, "full_catalog_logit_max": 15.770469665527344, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder", "mean_absolute_forward_reverse_edge_logit_difference": 0.1816689670085907, "query_key_projection_cosine": 0.01759312115609646, "attention_bias_l2": 0.36983540654182434, "fusion_output_norm_mean": 4.6792192459106445, "fusion_reference_norm_mean": 4.6792192459106445, "fusion_norm_max_error": 4.76837158203125e-07, "fusion_mixed_near_zero_count": 0, "adapter_near_zero_count": 0, "fusion_gate_mean": 0.796257734298706, "fusion_gate_std": 0.11966519802808762, "fusion_gate_min": 0.09852010756731033, "fusion_gate_max": 0.9646452069282532, "fusion_gate_below_001": 0.0, "fusion_gate_above_099": 0.0}`。
最终注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.011375755071640015, "changed_top1_fraction_on_cyclic_query_replacement": 0.00746533926576376, "normalized_attention_entropy": 0.9991899728775024, "mean_max_attention_weight": 0.10984485596418381, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 4.01578240731622e-05, "full_catalog_logit_min": -3.0638718605041504, "full_catalog_logit_max": 20.637407302856445, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder", "mean_absolute_forward_reverse_edge_logit_difference": 0.07493892312049866, "query_key_projection_cosine": 0.016515042632818222, "attention_bias_l2": 0.5501391887664795, "fusion_output_norm_mean": 3.335745096206665, "fusion_reference_norm_mean": 3.335745096206665, "fusion_norm_max_error": 4.76837158203125e-07, "fusion_mixed_near_zero_count": 0, "adapter_near_zero_count": 0, "fusion_gate_mean": 0.9039306640625, "fusion_gate_std": 0.07373540103435516, "fusion_gate_min": 0.4630800485610962, "fusion_gate_max": 0.9820674657821655, "fusion_gate_below_001": 0.0, "fusion_gate_above_099": 0.0}`。
在非对称注意力上，中心条件消息转换、每物品标量门控、输出范数控制；训练支持计数仅来自原train；新增adapter与gate参数正则。方法及来源见calibrated_neighbor_fusion_protocol.md。
反复查看原测试集得到的最大值属于探索最佳，不是独立验证选型后的泛化结论；单种子不证明稳定性。未按曲线调整本次参数或预算。
公开语义资产生成来源不完整这一既有限制保留；没有生成或覆盖原资产。

运行目录：`D:\CoLaKGResearch\calibrated-neighbor-fusion-v1\direct-test-calibrated_neighbor_fusion-s2020-20261030-175809-4afb9d`。best-protocol.pt、best.pt、final.pt、recovery.pt保存完整模型/Adam/RNG。

```powershell
& 'D:\CondaEnvs\CoLaKG\python.exe' -B experiments/calibrated_neighbor_fusion_2000.py
```

本次候选：先转换邻居内容，按训练支持量和内容学习每物品比例，单独控制融合输出范数。
最佳NDCG@20相对共享动态版本：-8.6642%。

Center-conditioned message adaptation and norm-controlled scalar item fusion; see calibrated_neighbor_fusion_protocol.md.
Parent best NDCG@20: 0.3476490190228204
Relative gain vs parent: -0.08784869673419926
