# LastFM 原始 CoLaKG 完整 epoch 与评估链路验证

日期：2026-10-22（北京时间）。本次只验证原始训练、排名评估和保存链路；不调参、不选模型，不宣称推荐性能提升。未运行正式 2000 轮。

## 输入、环境与配置

- 原始源码/数据版本：`be3aa19b59419d197dd4e20f44db737041821f58`；`rec_code/` 与原数据文件没有修改。
- Python：`D:\CondaEnvs\CoLaKG\python.exe`，3.8.19；PyTorch 1.13.0 / CUDA 11.7；MX450 2 GiB。
- NumPy 1.24.4、pandas 2.0.3、SciPy 1.10.1、scikit-learn 1.3.2、tensorboardX 2.6.2.2、tqdm 4.66.4。
- 配置直接读取 `train_lastfm.sh`：batch=1024，3 层，neighbor_k=10，dim=64，seed=2020，lr=0.001，BPR 正则系数=1e-4，item/user/neighbor dropout=0.6/0.2/0.6，use_drop_edge=0，keep_prob=1.0。
- testbatch=100，topks=[10,20]，multicore=0，A_split=False；load=0、pretrain=0，模型从随机参数开始。
- TensorBoard 按原始开关启用；使用原始 Python 负采样回退路径。没有安装编译扩展或更换采样算法。

数据 SHA-256：

| 文件 | SHA-256 |
|---|---|
| train.txt | `643619d1afd8d9bfbe536d0b9369b160bee932f69fdb59329132fcc07ade65a1` |
| test.txt | `0857015b628798e95e195daed6f9b289ab3bf888519e1c6445525de867f0008e` |
| lastfm_embeddings_simcse_kg.pt | `8d826d30a8bac4a25fcde9e98dc4f7116da6fda954f05dae764de86634e64fab` |
| lastfm_embeddings_simcse_kg_user.pt | `8ce9c4d833a9f17937478c76ccdff0ea1214b5b347e4ff2232562a63dc593215` |

## 实际结果

运行编号：`lastfm-one-epoch-20261022-231115-0f37bf66`；完整结果位于 `D:\CoLaKGExperiments\` 下该目录。

- 原样调用一次 `Procedure.BPR_train_original`：57786 个负采样三元组，56 个 1024 大小的 batch 和最后一个 442 大小的 batch，**57 次真实 Adam 更新**。
- 8 组活跃参数的 Adam step 均为 57；已有梯度、损失、最终参数均为有限值。原始 `W_u`、`a_u` 未参与当前前向，保持原样。
- 平均 minibatch 总损失：**0.3457275236907758**，采用原始函数的 batch 等权平均；原函数输出 `loss0.346`。按样本数加权的辅助统计为 0.34675076888575757。
- 训练结束后调用 **1 次**原始 `Procedure.Test`，覆盖 1859 个测试用户、19 个用户 batch，保留原始训练物品屏蔽、全量排名和指标计算。

| K | Precision | Recall | NDCG |
|---|---:|---:|---:|
| 10 | 0.1186659494 | 0.1498089517 | 0.1650242553 |
| 20 | 0.0824905863 | 0.2083734641 | 0.1945436220 |

| 阶段 | 实测秒数 | 口径 |
|---|---:|---|
| 准备 | 12.955448 | 导入、数据、语义邻居、建图、模型/优化器及 writer 初始化 |
| 完整训练 epoch | **14.799932** | 包含以下三项，不能重复相加 |
| ├ 负采样 | 1.352461 | 原始 Python sampler |
| ├ 57 次 BPR 训练调用 | 6.212989 | 含 Adam 0.223018 秒、梯度检查 0.356541 秒 |
| └ 其他训练开销 | 7.234482 | 传输、shuffle、TensorBoard、循环等未进一步归因开销 |
| 原始排名评估 | **0.862553** | 训练后一次完整 Test |
| 新 checkpoint 保存 | **0.013118** | 与 main.py 一样仅保存 model.state_dict；不含后续哈希校验 |
| 验证流程总墙钟时间 | **28.688256** | 含准备、训练、评估及保存，不含最终文件审计 |

训练峰值 allocated=**303.27 MiB**、reserved=**324.00 MiB**；评估峰值 allocated=281.93 MiB。训练/评估后驱动报告整卡空闲约 **582.37 MiB**。分配器统计不包含全部驱动/库工作区，整卡空闲也受其他进程影响。

没有 OOM、Windows 兼容性错误或非有限值。新 checkpoint 为本次独立目录下 `checkpoints/colakg-lastfm-epoch1.pth.tar`，1987219 字节；未加载任何旧 checkpoint，也未覆盖其他运行。

## 2000 轮成本与结论

- 原始协议对应 114000 次训练更新、400 次评估、2000 次当前 checkpoint 保存。
- 首轮直接线性外推：训练约 8.22 小时，评估约 5.75 分钟，保存约 26.24 秒，加准备后共 **8.33 小时**。采样约 45.08 分钟，已包含在训练内。
- 只扣除已测首 batch 的 3.925406 秒冷启动差值：暖 epoch 估计 10.874526 秒，训练约 6.04 小时，总计 **6.15 小时**。
- 以上是单次观测的两种外推场景，不是置信区间。首轮其他开销中可能还有初始化成本；散热、频率、后台 GPU 占用和后续迭代也会影响实际速度。建议预留约 **9 小时**连续运行窗口，不能承诺精确完成时间。
- 每次权重约 1.90 MiB；原作者不断覆盖同一个运行内的当前 checkpoint，最终保留一份，2000 次累计写入约 3.70 GiB，另需 TensorBoard/文本日志空间。
- 当前 MX450 的原始配置已经通过完整训练、评估和保存链路，显存可以继续尝试正式基线。建议安排连续运行窗口后启动；本轮不自动执行。
- 原始 checkpoint 不保存 Adam/RNG 状态，因此不能保证中断后精确续训；正式基线应从随机初始化开始并保持原协议。

## 复现命令及协议差异

```powershell
Set-Location 'D:\GitHub\colakg\CoLaKG'
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = 'D:\CoLaKGCache\tmp'
$env:TMP = 'D:\CoLaKGCache\tmp'
$env:TORCH_HOME = 'D:\CoLaKGCache\torch'
& 'D:\CondaEnvs\CoLaKG\python.exe' -B -u `
  'D:\GitHub\colakg\CoLaKG\experiments\lastfm_one_epoch.py'
```

每次产生新目录，不接受模型/轮数覆盖参数。脚本直接复用原始模型、图、损失、优化器、BPR_train_original 和 Test，仅添加观测与输出隔离。

与原作者 main.py 的差异：本次只训练一轮，并在训练后测试一次；原 main.py 在训练第 0/5/.../1995 轮之前测试。此次指标不能与完整论文结果等同，也没有用于选模型。正式协议应恢复原始评估时机。输出与图缓存重定向到 D 盘，增加了梯度有限值检查和同步计时；未改变损失公式、传播方式或超参数。原有 60 个跟踪文件哈希一致，main 引用未变。

正式 2000 轮的可执行计划见 [full_baseline_plan.md](full_baseline_plan.md)，当前尚未运行。
