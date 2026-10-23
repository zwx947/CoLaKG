# LastFM 研究开发协议 v1

原始公开基线保留在 exp/colakg-improvement，代码提交 b98629e；所有研究分支仅本地保存，不推送。环境继续使用 D:\CondaEnvs\CoLaKG，不安装依赖。

## 开发与测试隔离

只从原 train.txt 为每用户抽出约 20% 交互作开发验证，split_seed=314159；至少保留一个训练正例，并保证每件物品在训练图中仍有交互。派生文件仅写入 D:\CoLaKGResearch\protocol-v1，绝不覆盖原数据。开发加载器的 test.txt 实际是这个验证集；原项目 test.txt 仅记录哈希，不进入开发训练、图构造、选模型或方法比较。

固定资产为原始 item/user 语义张量。用户语义生成链路不完整，可能含原 train 中被抽为验证的历史，因此这是监督图/标签隔离的开发协议，不宣称语义资产层面完全无泄漏；所有方法使用同一资产以保证对照公平。原测试集不据此重新划分。

## 首轮有限实验

1. 原始 CoLaKG 对照与两个不同方向，各随机初始化训练 200 轮，seed=2020；每 20 轮验证。选择依据预先固定为 160/180/200 三点 NDCG@20 均值，不用原测试集，不以单个峰值挑选。
2. 方向 A：借鉴 RecSys2025 SGCL / SIGIR2025 LightCCF，对完整物品目录进行监督对比，排除已知其他训练正例，保留 BPR 与显式正则作为对照起点。
3. 方向 B：借鉴 ICML2026 CCLRec 的结构—语义共识，使用仅训练图计算的物品共现修正语义邻居；增加独立协同评分残差，避免两种表示直接相加带来的交叉项。使用原 BPR。
4. 根据验证反馈作有依据的结构消融或有限修订。积极候选须与同预算原始对照进行额外种子检验；失败方法和代码也保留。
5. 方法与配置冻结后，才在完整原 train 上从头训练并进行固定预算最终测试。相对于历史最佳测试值的比较仅作背景，主要比较固定预算、相同种子与资产的对照。原论文五次平均、历史测试曲线最大值和新固定预算测试分开报告。

除明确记录的新方法参数外，沿用 train_lastfm.sh：batch1024、dim64、3层、k10、lr0.001、decay0.0001、seed2020、dropout_i/u/n=0.6/0.2/0.6。原始 Python 负采样与 BPR_train_original 保留。开发阶段关闭 TensorBoard，仅改变记录开销；评估缓存一次 eval 表示，仍按原指标函数全排名、屏蔽训练正例。

每次运行保存配置、Git分支/commit、源文件与输入SHA256、环境版本、loss/验证曲线、时钟、显存及同次模型/Adam/RNG快照。异常停止，不自动降低模型或改CPU。目标4%–5%不是承诺；没有真实证据时不称提升或论文贡献。

参考：
- SGCL, RecSys2025: https://doi.org/10.1145/3705328.3748036 ，https://github.com/DavidZWZ/SGCL
- LightCCF, SIGIR2025: https://arxiv.org/abs/2504.10113 ，https://github.com/ZzYUuuu/LightCCF
- CCLRec, ICML2026: https://proceedings.mlr.press/v306/guo26p.html ，https://github.com/dongyu-pei/CCLRec
- Language Representations Can be What Recommenders Need, ICLR2025: https://openreview.net/forum?id=eIJfOIMN9z

上述工作提供迁移灵感，不等于本项目完整复现这些方法或已经具备新颖性。
