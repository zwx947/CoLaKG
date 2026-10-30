# 中心条件消息转换与尺度校准融合：单候选完整训练

2026-10-30用户授权设计后训练，明确不先跑大量对照。本轮只训练一个完整候选，不重跑已有原始、固定A、非对称父模型，不自动排队消融。

## 动机与依据

上一版逐维融合2000轮最佳N20=0.3425138673，门控很早偏向中心。冻结训练目标敏感性分析发现直接减少邻居可降低训练BPR，但不能推出泛化改善。两路幅度不同，直接比例同时影响表示内容与幅度。详见item_fusion_sensitivity_20261030.md；这些证据没有证明唯一失败原因。

借鉴[COHESION SIGIR2025](https://arxiv.org/html/2504.04452v2)先处理内容再融合，以及[DCGL SIGIR2026](https://arxiv.org/html/2605.07314v1)基于内容与交互支持条件学习权重。不是原创门控，不照搬多模态双通道，不宣称转换后一定是纯互补信息。来源代码核对记录见fusion_message_literature_review_20261029.md。

## 固定实现

在已有非对称语义注意力父模型上，将中心/邻居均值处替换为以下机制；无A。ID/语义输入均值仍保留。

- x为原中心物品表示，m为ELU后的原邻居聚合消息。
- u(v)=v/max(norm(v),1e-6)，保持正负方向。特征f(a,b)=[a,b,b-a,a*b]。
- t=MLP(f(u(x),u(m)))，256→64→64，隐层ELU，PyTorch默认Linear初始化。
- 每物品标量g=sigmoid(MLP([f(u(x),u(t)),log(1+训练交互数)]))，257→64→1，隐层ELU。没有预设低频物品多用邻居；不读取测试交互数。
- r=norm((x+m)/2)；z=r*u((1-g)u(x)+g*u(t))。r不detach；门控仍可通过表示方向影响排序，但不直接改变本层输出范数（非退化情况下）。这不能排除全模型通过其他参数改变幅度。
- 两路原始dropout全部保留，转换和比例使用当前前向的表示；没有额外稳定性损失、强制平衡、对比损失或热身预算。
- BPR及原六向量正则保留；新增adapter+gate参数按0.5平方和/实际batch计入原decay=1e-4。支持计数为冻结buffer。

归一化可能放大弱消息，内容转换也可能丢失有用信息；比例不代表可靠性概率。记录范数误差、近零情况和门控分布，不因权重极端自动调整当前训练。

## 预算与比较

Python D:\CondaEnvs\CoLaKG\python.exe。LastFM原train57786/test13508、seed2020、完整2000轮、batch1024、dim64、UI传播3层、neighbor10、lr0.001、decay0.0001、dropout_i/u/n=0.6/0.2/0.6均沿用原配置。

推荐参数随机初始化，不加载任何训练权重。0/5/…/1995及最终2000评估；400次最佳、401次最佳、最终分别报告；其他指标取同一NDCG20选中点。参照已有原始最佳0.3473341594与非对称父模型0.3476490190，不将旧1.82%终点对比作基准。反复测试选择和单种子局限保留。

若整体有明确积极结果，再讨论必要消融及种子确认；不是本轮必须完成的附加训练，不按曲线改配置或提前淘汰。无效结果也保存。新目录唯一创建在D:\CoLaKGResearch\calibrated-neighbor-fusion-v1；完整模型/Adam/Python/NumPy/CPU/CUDA RNG状态随best、final及20轮recovery保存。

```powershell
& 'D:\CondaEnvs\CoLaKG\python.exe' -B experiments/calibrated_neighbor_fusion_2000.py
```

启动前CPU检查：check_calibrated_neighbor_fusion.py三项通过，覆盖绕过新融合时与父模型train/eval完整输出一致、正常及零输入的范数控制、比例改变能影响方向、BPR对转换和门控均有非零梯度、实际Adam更新、模型/Adam/RNG恢复一致、排名缓存一致。这些是代码正确性检查，不是额外模型性能实验。D盘约105.7GiB可用；实际启动前GPU守卫再次执行。
