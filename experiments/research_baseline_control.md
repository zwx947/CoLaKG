# LastFM 开发验证对照

原始 CoLaKG、seed2020、200轮、train_lastfm.sh参数，在 protocol-v1（fit46953/validation10833）从头训练。原 test.txt 未用于方法开发。运行目录：`D:\CoLaKGResearch\development-baseline-s2020-20261023-143751-b77fd2`；执行提交 `e0e2a84`。

原 BPR_train_original 共9200次更新，最后160/180/200轮验证NDCG@20均值 **0.2596643656479679**，最终轮 **0.2589287191162085**；实际训练约532.446秒，峰值allocated317312512字节。原始模型源码、输入及运行入口哈希验证通过。该分数属于开发验证，不与原测试曲线直接比较。

启动：`D:\CondaEnvs\CoLaKG\python.exe -B -u experiments\research_runner.py --method baseline --epochs 200 --seed 2020`。
