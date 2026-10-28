# LastFM 中心条件化语义邻居注意力：完整训练集直接测试探索

用户2026-10-28明确授权变更评估规则。本次重新随机初始化训练，旧验证实验保留。
代码版本：`a1e2693b580c50bc16b8e092d91de5131b9dd083`。完整2000轮、114000次更新；原始train57786、test13508。

| 口径 | 实际训练轮数 | P@10 | P@20 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始：400次最佳 | 1920 | 0.200161377 | 0.138945670 | 0.270616990 | 0.376867577 | 0.295404035 | 0.347334159 |
| 本候选：400次最佳 | 1645 | 0.200484131 | 0.140559441 | 0.270165945 | 0.379403716 | 0.293662249 | 0.347189833 |
| 本候选：401次最佳 | 1645 | 0.200484131 | 0.140559441 | 0.270165945 | 0.379403716 | 0.293662249 | 0.347189833 |
| 本候选：最终2000轮 | 2000 | 0.199300699 | 0.138569123 | 0.269802206 | 0.374106876 | 0.291613278 | 0.342915431 |

原协议最佳相对变化：-0.042%；含终点最佳相对变化：-0.042%。
总耗时121.86分钟，峰值分配显存191.99MiB。
计时分项（秒）：`{"training_including_sampling_seconds": 7003.52328250001, "sampling_seconds": 2778.062457899995, "training_after_sampling_seconds": 4225.460824600014, "evaluation_seconds": 261.2858596000041, "checkpoint_seconds": 14.814286099998341}`。训练分项包含采样；评估不含checkpoint保存，保存单独计量，其余差额包括状态JSON、校验与其他开销。
配置、数据/源码哈希、环境及命令见manifest.json，完整曲线见history.json/evaluations.json。
最佳点注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.18236757814884186, "changed_top1_fraction_on_cyclic_query_replacement": 0.07643085718154907, "normalized_attention_entropy": 0.9717899560928345, "mean_max_attention_weight": 0.1676829308271408, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 1.9122773368172476e-07, "full_catalog_logit_min": -12.016319274902344, "full_catalog_logit_max": 16.67066192626953, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder"}`。
最终注意力诊断：`{"changed_neighbor_order_fraction_on_cyclic_query_replacement": 0.3270529508590698, "changed_top1_fraction_on_cyclic_query_replacement": 0.10238179564476013, "normalized_attention_entropy": 0.9635475277900696, "mean_max_attention_weight": 0.1787375956773758, "full_catalog_sigmoid_exact_one_fraction_including_known_positives": 7.075426146223816e-06, "full_catalog_logit_min": -10.769573211669922, "full_catalog_logit_max": 18.894235610961914, "interpretation": "query dependence and numerical diagnostics only; no method selection or test-label use inside the encoder"}`。
仅替换原模块C的语义对注意力打分，保留用户/物品语义、原融合、邻居集合、传播、BPR与六向量正则。方法及来源见dynamic_semantic_attention_protocol.md。
反复查看原测试集得到的最大值属于探索最佳，不是独立验证选型后的泛化结论；单种子不证明稳定性。未按曲线调整本次参数或预算。
公开语义资产生成来源不完整这一既有限制保留；没有生成或覆盖原资产。

运行目录：`D:\CoLaKGResearch\dynamic-semantic-attention-v1\direct-test-dynamic_semantic_attention-s2020-20261028-211059-bfa969`。best-protocol.pt、best.pt、final.pt、recovery.pt保存完整模型/Adam/RNG。

```powershell
& 'D:\CondaEnvs\CoLaKG\python.exe' -B experiments/dynamic_semantic_attention_2000.py
```
