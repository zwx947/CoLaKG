# 同类型表示几何辅助目标

分支 `codex/pair-geometry`，首轮执行提交 `8ba7eb3`。原CoLaKG编码器、原语义邻居、BPR、Adam和6项正则保留，辅助项改为SGCL（RecSys2025）式交互对齐及用户—用户、正例物品—物品的表示分散：归一化后拼接UU/II相似矩阵的logsumexp，减去对应UI余弦。使用数值稳定形式，温度0.2、辅助系数0.05沿用目录辅助对比，不做权重搜索。对角及重复ID按公开SGCL形式保留。本项目用辅助损失均值，官方原实现是独立求和损失；不宣称整体复现SGCL，也不把现有目标的迁移声称为论文创新。

相较目录负例，它不直接排斥所有未知物品；这是有依据的目标对照，而不是已经证实的有效原因。后续需消融才能归因。

首轮使用固定fit46953/validation10833，随机初始化seed2020，200轮9200次更新。末三次验证NDCG@20均值 **0.2646505291**，对照 **0.2596643656**，相对 **+1.9202%**；最终验证Recall@20 **0.3364481672**，NDCG@20 **0.2650995454**。总循环564.990秒，峰值allocated338578944字节（322.89MiB），无数值/OOM异常，源文件与输入哈希通过。

运行：`D:\CoLaKGResearch\development-pair_geometry-s2020-20261023-163010-ca0314`。原test未用于开发；尚未得到2000轮最终测试结论。按事前门槛，该初步信号超过1%，进入seed2021/2022固定配置确认，复用相同种子的已完成原始对照。

复现：`D:\CondaEnvs\CoLaKG\python.exe -B -u D:\GitHub\colakg\CoLaKG\experiments\research_runner.py --method pair_geometry --epochs 200 --seed 2020`。
