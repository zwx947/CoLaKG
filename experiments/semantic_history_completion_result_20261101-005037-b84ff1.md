# 联合图上的行为监督语义历史补全：完整训练结果

代码版本：db35f6cfbcc02bfb9f55ffe3e8466ff2e448eec4
联合图SSC父模型保留，新增共享物品语义映射的训练历史补全，多正例CE权重0.01；随机初始化，无A/旧checkpoint。
配置、预先推演及论文来源见semantic_history_completion_protocol.md。

| 口径 | 完成轮数 | P10 | P20 | R10 | R20 | N10 | N20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始400次最佳 | 1920 | 0.200161377 | 0.138945670 | 0.270616990 | 0.376867577 | 0.295404035 | 0.347334159 |
| 联合图父模型400次最佳 | 1930 | 0.200268962 | 0.140882195 | 0.272189989 | 0.381990872 | 0.294414889 | 0.348326480 |
| 本候选400次最佳 | 1650 | 0.202097902 | 0.141554599 | 0.273736947 | 0.384140859 | 0.296347495 | 0.350219127 |
| 本候选401次最佳 | 1650 | 0.202097902 | 0.141554599 | 0.273736947 | 0.384140859 | 0.296347495 | 0.350219127 |
| 本候选最终2000轮 | 2000 | 0.197633136 | 0.140048413 | 0.267685520 | 0.379927338 | 0.290111293 | 0.345000427 |

原协议最佳相对变化：+0.8306%
相对联合图父模型变化：+0.5434%
总耗时：239.90分钟；峰值分配显存：82.40MiB。
计时分项（秒）：{"training_including_sampling_seconds": 14058.382840300013, "sampling_seconds": 2935.214023599984, "training_after_sampling_seconds": 11123.16881670003, "evaluation_seconds": 267.9599658999893, "checkpoint_seconds": 28.99569309999876, "auxiliary_sampling_seconds_within_training": 171.20609840008433, "auxiliary_forward_cuda_seconds_within_training": 2503.2076095008847, "timing_note": "aux forward CUDA events exclude shared backward; sampling/aux timings are subsets or overlaps, not additive extra wall time"}
完整2000轮/114000更新/401评估；其他指标取同一N20选中点。
主图和推理同父模型；仅训练新增历史补全，辅助RNG包含于model的extra_state；总loss包含辅助项，不与父模型BPR直接混比。
直接原test反复选型及单种子不证明稳定泛化；语义资产生成来源不完整限制保留。
运行目录：D:\CoLaKGResearch\semantic-history-completion-v1\direct-test-semantic_history_completion-s2020-20261101-005037-b84ff1
best-protocol.pt/best.pt/final.pt/recovery.pt保留模型/Adam/全部RNG状态。
复现：& D:/CondaEnvs/CoLaKG/python.exe -B experiments/semantic_history_completion_2000.py
