# LastFM 原始 CoLaKG 单 minibatch 验证

- 日期：2026-10-22（北京时间）。目的：确认原始模型能在本机完成一次真实 GPU 训练更新，不评价推荐效果。
- 分支：`exp/colakg-improvement`；原始 `rec_code/`、数据和 main 分支均未修改。
- 环境：`D:\CondaEnvs\CoLaKG\python.exe`，Python 3.8.19，PyTorch 1.13.0 / CUDA 11.7，NVIDIA GeForce MX450（2 GiB）。NumPy 1.24.4、pandas 2.0.3、SciPy 1.10.1、scikit-learn 1.3.2、tensorboardX 2.6.2.2、tqdm 4.66.4。

## 原始配置与复现命令

脚本直接解析 `rec_code/train_lastfm.sh`，执行未经修改的 `world.py` 配置初始化，不接受模型超参数覆盖。

| 参数 | 实际值 |
|---|---:|
| 模型 / 数据 | CoLaKG / LastFM |
| 随机种子 | 2020 |
| minibatch / embedding 维度 | 1024 / 64 |
| 传播层数 / 语义邻居数 | 3 / 10 |
| 学习率 / BPR 正则系数 | 0.001 / 0.0001 |
| item / user / neighbor dropout | 0.6 / 0.2 / 0.6 |
| use_drop_edge / keep_prob | 0 / 1.0 |
| A_split / pretrain / load | False / 0 / 0 |
| 优化器 | 原始 `utils.BPRLoss` 内的 Adam，其他参数为原始默认值 |

```powershell
Set-Location 'D:\GitHub\colakg\CoLaKG'
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = 'D:\CoLaKGCache\tmp'
$env:TMP = 'D:\CoLaKGCache\tmp'
$env:TORCH_HOME = 'D:\CoLaKGCache\torch'
& 'D:\CondaEnvs\CoLaKG\python.exe' -B -u `
  'D:\GitHub\colakg\CoLaKG\experiments\lastfm_one_minibatch.py'
```

每次产生新的唯一结果目录，不覆盖旧结果。脚本执行原始 Python 负采样、shuffle，并取原始 epoch 中第一个完整 minibatch；只调用一次 `utils.BPRLoss.stageOne()`。没有 AMP、梯度裁剪、减层、减邻居、减 batch 或 CPU 回退。

## 本次实测结果

运行目录：`D:\CoLaKGExperiments\lastfm-one-minibatch-20261022-224930-f21b084e`。

| 检查 | 结果 |
|---|---|
| 初始化 / 前向 / 反向 / Adam 更新 | 全部成功 |
| 用户 / 物品 / 训练交互 | 1859 / 2813 / 57786 |
| 参数总数 | 495872 |
| BPR 损失 | 0.6225869655609131 |
| 加权正则项 | 0.0006974270820617676 |
| 总损失 | 0.6232843995094299 |
| 前向及损失计算 | 3.209553 秒 |
| 反向区间（含 zero_grad 和损失组合） | 0.067776 秒 |
| Adam 更新 | 0.015258 秒 |
| 单步墙钟时间（含梯度有限值检查） | 3.301943 秒 |
| 验证流程耗时（含导入、输入、建图及采样） | 18.229004 秒 |
| PyTorch 峰值 allocated / reserved | 297.97 / 318.00 MiB |
| 步后驱动报告的整卡空闲显存 | 约 588.37 MiB |
| 损失、已有梯度、更新后全部参数 | 均无 NaN/Inf |
| 优化器更新证据 | 8 组参数变化，对应 Adam step 均为 1 |
| checkpoint 读取 / 保存，Recall/NDCG 计算 | 均为 0 次 |

`W_u`、`a_u` 在原始 CoLaKG 中已定义，但当前原始前向没有使用，因此这两组参数没有梯度、没有更新；本次保持原样，没有修改数学逻辑。其余 8 组参数完成更新。

这是第一次冷启动训练步骤，不应直接乘以 minibatch 数推算 2000 轮总耗时。allocated/reserved 是 PyTorch 分配器统计，不包含全部 CUDA 驱动、库工作区及其他进程占用；整卡空闲是瞬时值，不是本进程显存峰值。

本次没有失败或 OOM。若以后失败，脚本保留完整 traceback、失败阶段和显存诊断，不自动改变配置或设备。

## 运行脚本的必要适配

1. 不导入或执行 `main.py` / `register.py`，绕开第 0 轮测试、2000 轮循环和每轮 checkpoint 写入。
2. 原样执行 `world.py` 的源码，仅为该运行时模块设置隔离的逻辑 `__file__`，将它自动创建的输出目录放到本次 D 盘运行目录。配置数值不变，traceback 仍指向真实源码。
3. Loader 读取原始 train/test 文件；test 只用于原加载器的维度与索引构建，没有评价或选参。仅重定向 `s_pre_adj_mat.npz` 的读写路径，仍执行原始图归一化和稀疏转换。
4. 完整复用 `model.CoLaKG`、原始 BPR 损失、Adam、Python 负采样及 minibatch 工具。只增加损失观察、更新前梯度检查、更新后参数对比，不重写训练公式。
5. `torch.load` 仅允许读取两份现有 LastFM 语义嵌入，`torch.save` 在本次进程中被禁止。模型可训练参数随机初始化。

运行目录的 `result.json` 包含完整配置、版本、数据与脚本哈希、每组梯度/参数更新、耗时和显存记录；`run.log` 为完整日志。运行前后全部 60 个原有跟踪文件的 SHA-256 一致，main 引用未变化，没有创建 checkpoint。

## 下一步

2 GiB MX450 已通过原配置单步 GPU 训练，可以继续验证完整 epoch 和原始评估路径，但单步不能证明长期运行及评估一定稳定。建议先原配置运行 1 个完整 epoch（57 次更新）并验证原始评估，再从 seed=2020 的全新随机初始化执行正式 2000 轮，不用测试集选择超参数或最优 checkpoint。

下一阶段的可执行步骤见 [完整基线计划](full_baseline_plan.md)。本轮未执行该计划，也没有提交或推送。
