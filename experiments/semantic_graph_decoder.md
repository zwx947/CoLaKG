# 仅训练历史语义与双路径图评分：首轮结果

分支 `research/semantic-graph-decoder`，执行提交 `2b05db3`。借鉴AlphaRec（ICLR2025）的共享语义映射思路，建立不同于原CoLaKG的框架：去掉物品邻居注意力和公开用户语义输入，以居中物品语义的fit正例均值构造用户特征，共享1024→64投影，分别传播ID图与语义图，使用独立评分相加。原BPR、Adam、3层、64维/路径、学习率和正则保留；公开用户语义不参与此模型。不是AlphaRec复现，不声称新颖性。

协议：固定fit46953/validation10833，随机初始化seed2020，200轮9200更新。末三次（160/180/200）验证NDCG@20均值 **0.2553591885**，原始对照 **0.2596643656**，相对 **-1.66%**。最终验证Recall@20 **0.3268936192**，NDCG@20 **0.2547889154**。原test未用于开发。

循环总耗时466.086秒，对照532.446秒；峰值allocated58749440字节（56.03MiB），对照302.61MiB。整体耗时仍有原Python负采样开销，不能把显存下降比例当作速度提升比例。资源效率有价值，准确率未改善，暂不进入原测试集或做参数搜索。可能失去公开用户语义及物品注意力提供的信号，但当前同时变化多个组件，不能单独归因。

运行目录：`D:\CoLaKGResearch\development-semantic_graph_decoder-s2020-20261023-152430-9076ed`。源文件、输入哈希一致，无NaN/OOM。

首次启动 `...-152234-dce238` 在训练前的源归档环节因Python3.8相对`__file__`报错，零训练更新；修复为绝对路径后从头新运行，失败目录保留。该修复不涉及数学实现。

复现：`D:\CondaEnvs\CoLaKG\python.exe -B -u D:\GitHub\colakg\CoLaKG\experiments\research_runner.py --method semantic_graph_decoder --epochs 200 --seed 2020`。
