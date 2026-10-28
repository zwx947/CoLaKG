# 中心物品条件化语义注意力：固定实验协议

2026-10-28。用户批准执行候选方向后，首先检验方向1。独立分支 `codex/b-dynamic-semantic-attention`，从原始基线 `exp/colakg-improvement@b98629e` 创建，不叠加A。保护main、原始基线及固定A。

## 问题与方法依据

原CoLaKG注意力打分为 `LeakyReLU([s_neighbor W || s_center W] a64)`。对固定的共同候选集合，中心项在最终单调非线性前只提供一个常数偏置，因此中心物品不能改变这些候选的打分排序；它仍可影响权重的尖锐程度。该表达限制成立，不等于已经证实它限制了LastFM性能。

借鉴 How Attentive are Graph Attention Networks?（ICLR 2022）的动态打分：将非线性移到最终线性打分之前。论文 https://arxiv.org/html/2105.14491v3 ，作者实现 https://github.com/tech-srl/how_attentive_are_gats 。该来源是2022年工作，不能标成近期新发明；它对应本轮从原代码定位的具体问题。作者固定源码提交 `6f6d29089603c2f51651bd083dcd44f206b1100e` 的 gatv2_conv_PyG.py 已完整阅读并另存D盘，SHA256见来源记录。

## 本轮准确改动

使用作者支持的 `share_weights=True` 变体，将中心和邻居投影到同一个32维语义空间：

`new_logits(i,j) = a32^T LeakyReLU(s_j W + s_i W)`。

采用单头、无bias、LeakyReLU负斜率0.2，保留原W的Xavier初始化，新的a32仍按原Xavier gain=1.414随机初始化。作者默认share_weights=False；此处共享投影是为保持CoLaKG原有同源语义投影设置。只借鉴打分机制，不复制作者层的值投影、自环添加或整套节点分类模型。不安装PyG、DGL或扩展。

- 原固定Top-10候选完全保留，仍用 `sklearn cosine_similarity` 及 `np.argsort(... )[:,1:11]`，不扩候选、不加自环。
- 原用户和物品语义向量、Linear/ELU、输入及输出dropout、1:1 ID融合保留；原语义邻居的value、dropout与ELU聚合保留。
- UI图、3层传播与含0层均值、点积/Sigmoid、原BPR、六向量L2、负采样及Adam保留。
- batch=1024、dim=64、K=10、seed=2020、lr=0.001、decay=1e-4、dropout_i/u/n=0.6/0.2/0.6、无边丢弃。
- 参数总数相对原模型少32（a64变为a32），没有新加查询投影。原本未使用的W_u/a_u不被借用，继续保留。参数量不是本研究候选的筛选标准；本次设计用于检验打分机制。
- 由原computer复制形成实验派生类，替换打分块。检查脚本用原公式替回后，比较完整前向、BPR/六项正则及其余梯度，以验证替换边界；原rec_code源码不改。

## 运行与观察

完整原train57786、原test13508，随机初始化训练2000轮，无旧推荐checkpoint加载。按已完成0/5/…/1995轮测试（400次），另测2000轮终点（401次）。不提前淘汰、不按测试曲线改当前配置或延长预算。选中最大NDCG@20对应的一组指标，分别报告400次最佳、401次最佳和最终值。

复用原始2000轮同种子基线，最佳NDCG@20=0.3473341593547369；不重跑原始或A。完整测试反复用于探索的选择局限保留；单种子不能证明稳定收益。后续有效时再安排跨种子、与A组合和必要消融；当前不会自动排队其他候选。

额外记录不参与训练的诊断：保持邻居固定，将中心语义按固定循环移位后，邻居顺序/首位改变比例；注意力熵与最大权重；全目录logit范围与Sigmoid恰好为1的比例（包含已知正例，明确不是只统计候选负例）。这些只帮助解释动态性和数值尺度，不能当作性能或正确邻居标签。计算不消耗RNG，也不读取测试标签。

保存完整loss/指标/诊断曲线、实际更新次数、采样/训练/评估/保存耗时、PyTorch峰值分配/保留显存；每20轮保存本次recovery，另保留best-protocol/best/final，包含模型、Adam、Python/NumPy/CPU/CUDA RNG和输入/代码身份。日志和权重位于D盘唯一目录。

训练前检查其他GPU任务；OOM、非有限梯度/参数或严重错误时停止并记录，不改结构或降到CPU。输入及源码结束时核对哈希，不覆盖数据、语义或旧实验。公开语义生成来源不完整的限制继续保留。

```powershell
cd D:\GitHub\colakg\CoLaKG
& 'D:\CondaEnvs\CoLaKG\python.exe' -B experiments/check_dynamic_semantic_attention.py -v
& 'D:\CondaEnvs\CoLaKG\python.exe' -B -u experiments/dynamic_semantic_attention_2000.py
```

代码和最终中文记录仅本地中文commit，不推送。是否值得继续由完整结果和机制证据共同判断；失败不自动证明整个动态注意力方向无效，也不自动启动无依据的参数扫描。
