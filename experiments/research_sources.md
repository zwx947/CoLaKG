# 2025—2026论文与公开实现阅读记录

阅读日期：2026-10-23。仅阅读原论文/会议页面和作者公开源码；没有安装这些仓库依赖。所有本项目方法都是针对现有CoLaKG任务的迁移检验，不是下列完整算法的复现，也未据此确认新颖性。

| 工作 | 已核验来源 | 阅读的代码与借鉴点 |
|---|---|---|
| SGCL，RecSys2025 | https://doi.org/10.1145/3705328.3748036 ，https://github.com/DavidZWZ/SGCL | recbole_gnn/model/general_recommender/sgcl.py；交互对齐和同类型表示分散，原实现独立求和损失，本项目后续改为BPR辅助均值 |
| LightCCF，SIGIR2025 | https://arxiv.org/abs/2504.10113 ，https://github.com/ZzYUuuu/LightCCF | LightCCF.py、utility/losses.py；对比目标可承担邻域聚合，不能把任意InfoNCE直接当作完整LightCCF |
| CCLRec，ICML2026 | https://proceedings.mlr.press/v306/guo26p.html ，https://github.com/dongyu-pei/CCLRec | model.py、main.py；训练结构和语义的一致信号加权。本项目首轮改邻居图的方案与原方法的共识对比加权并不等价 |
| AlphaRec，ICLR2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/e4bab1843c8d5a69f5abfd0824593493-Abstract-Conference.html ，https://github.com/LehengTHU/AlphaRec | models/General/AlphaRec.py；共享语义投影与图传播。本项目额外保留ID评分路径，并用fit历史构造用户语义，属于不同框架 |

官方源码阅读快照仅保留在 `D:\CoLaKGResearch\references`，不是训练输入。具体SHA256见同目录来源记录及研究总表；这些源自公开分支的字节快照用于追溯阅读版本，不声称固定了完整官方仓库环境。

第一轮反馈促使最后补充同类型几何目标：目录辅助对比只有弱收益，而单纯对齐目录目标与余弦解码表现更差，因此检验SGCL式用户—用户、物品—物品分散与交互对齐是否比未知目录物品排斥更适合BPR。温度0.2、辅助权重0.05沿用第一方案，未进行网格搜索。此动机是待实验检验的假设，不是已证实的失败原因。

未来形成论文贡献还需同预算强对照、组件消融、长期训练、多种子和多数据集证据，并与上述原算法区分；某个迁移技巧有提升本身不足以证明贡献。

SGCL正文方法与公式5另核验于 https://arxiv.org/html/2507.13336v1 。本项目几何项与其公开目标直接相关，保留BPR的辅助式训练不属于原文的单目标框架，因此即使实验有效，也不能把此目标本身包装为新损失或声称已经解决多目标梯度冲突。
