from pathlib import Path
import json
import statistics

root=Path(r'D:\GitHub\colakg\CoLaKG')
study=Path(r'D:\CoLaKGResearch\study-v1')
aggregate=json.loads((study/'aggregate.json').read_text(encoding='utf-8'))
rows=aggregate['runs']
names={'baseline':'原始CoLaKG','catalog_contrastive':'目录辅助对比','consensus_residual':'共识邻居+协同残差',
       'semantic_graph_decoder':'独立语义图评分框架','aligned_catalog':'余弦目录目标替换BPR',
       'pair_geometry':'同类型表示几何辅助','pair_alignment_only':'仅对齐组件消融'}
branches={'catalog_contrastive':'research/catalog-contrastive','consensus_residual':'research/consensus-residual',
          'semantic_graph_decoder':'research/semantic-graph-decoder','aligned_catalog':'research/aligned-catalog',
          'pair_geometry':'codex/pair-geometry'}
lines=['# CoLaKG LastFM 首轮方法研究报告','',
       '开发研究已完成，冻结的2000轮最终训练进行中；本报告的最终测试段在运行结束后补齐。',
       '', '## 比较协议','',
       '保持已复现基线 exp/colakg-improvement@b98629e 与 main 不变，所有研究分支只在本地保存。未安装/升级依赖，未修改原始数据、语义资产或 rec_code 模型源码。新模型/目标位于 experiments/research_methods。所有训练从随机初始化开始；自己的训练快照只用于同次状态保存与明确标注的事后表示分析。',
       '',
       '只从原train建立固定fit46953/validation10833，split_seed314159，用户1859、物品2813均保留。原test标签没有用于开发图、损失、方法比较、调参或选种子。每个开发实验200轮9200次原BPR_train_original更新，每20轮验证，事前固定160/180/200轮NDCG@20均值作为评分。以下0.26左右的数值全部是验证集，不能与历史0.3473测试数值直接作差。',
       '',
       '共同参数：batch1024、dim64、3层、邻居10、lr0.001、decay0.0001、dropout_i/u/n=0.6/0.2/0.6，原Python负采样、Adam。模型框架改动和新增目标参数另记于各方法记录。评估缓存一次eval表示，仍使用原Procedure.Test、全排名和已知训练正例屏蔽。TensorBoard关闭，只改变记录开销。',
       '',
       '环境：Windows，D:\\CondaEnvs\\CoLaKG\\python.exe，Python3.8.19，PyTorch1.13.0/CUDA11.7，MX450 2GiB；完整pip清单在本轮study及最终运行目录。输入/源SHA、运行命令、每轮损失、指标和完整同次模型/Adam/RNG状态均在独立D盘目录。所有完成的运行源与输入哈希检查通过，没有NaN/OOM。',
       '', '## 首轮方法与结果','',
       '| 方法（seed2020） | 末三次验证NDCG@20 | 配对变化 | 最终验证Recall@20 | 峰值allocated MiB | 循环秒 | 执行代码 |',
       '|---|---:|---:|---:|---:|---:|---|']
for row in rows:
    if row['seed']==2020:
        lines.append('| {} | {:.8f} | {:+.2%} | {:.8f} | {:.2f} | {:.2f} | {} |'.format(names[row['method']],row['score'],row['relative_gain'],row['final_recall20'],row['peak_allocated_mib'],row['seconds'],row['git_commit'][:7]))
lines += ['', '显存是PyTorch allocated峰值，不含驱动/其他程序；循环时间包括采样、训练、验证与状态保存，不含启动。单机有短时耗时波动，这些数值不是严格受控速度基准。损失包含不同辅助项，不能横向比较绝对值；完整损失见research_records/development_results_v1.json及各运行history.json。',
          '', '![开发验证曲线](research_figures/screening-ndcg20.png)','',
          '路线覆盖监督目标、训练图邻居、独立协同评分与整体语义图框架，而非只修原注意力。目录辅助对比只有微小收益；共识重排/残差及独立语义图未改善验证；余弦目录目标在原参数下明显更差。以上仅是200轮筛选，不宣称这些框架在所有预算/数据上均无效。具体假设与限制分别见各方法中文记录。',
          '', '## 初始化稳定性','',
          '| seed | 原始对照 | 目录辅助 | 目录变化 | 几何辅助 | 几何变化 |','|---|---:|---:|---:|---:|---:|']
lookup={(r['method'],r['seed']):r for r in rows}
for seed in [2020,2021,2022]:
    b,a,e=[lookup[(method,seed)] for method in ['baseline','catalog_contrastive','pair_geometry']]
    lines.append('| {} | {:.8f} | {:.8f} | {:+.2%} | {:.8f} | {:+.2%} |'.format(seed,b['score'],a['score'],a['relative_gain'],e['score'],e['relative_gain']))
lines += ['',
          '目录辅助三种子平均配对提升约0.50%，未达新增种子训练前写下的1%推进门槛；没有对其执行原测试。几何辅助平均配对提升约1.80%，两新增种子均正，达到推进门槛。此为固定验证划分、短预算的初始化稳定性，不能声称三种子完整测试提升或论文级显著性。目标4%–5%尚未由开发结果证明。',
          '', '## 机制证据及边界','',
          '仅对齐消融移除同类型logsumexp分散项，其验证分数0.23248324，相对原始对照-10.47%；完整几何项为0.26465053。对本轮自身200轮快照的CPU事后分析发现：仅对齐使fit正例余弦由0.68383升至0.79653，却使物品平均异节点余弦从0.07382升到0.24667，并伴随有效秩下降；完整几何项物品异节点余弦降至0.04072。',
          '',
          '结果支持需要平衡交互对齐与表示分散的解释。完整几何项中心化有效秩并没有超过原始对照，不能包装成普遍的“提高秩”机制。分析不训练、不做原test排名，详见pair_geometry.md、research_records/geometry_analysis_v1.json；单种子关联和组件消融仍不足以证明普遍因果。',
          '', '## 冻结后的完整训练','',
          '配置frozen_pair_geometry_v1.json在原测试前生成，完整校验源文件、原始资产、ID映射、种子、模型参数与共同配置。seed2020、2000轮、全部原train57786交互，从新随机初始化，禁止开发checkpoint作为起点。每20轮覆盖本次独立目录内的最新完整状态，只在完成2000轮后做一次原test。',
          '',
          '运行目录：`D:\\CoLaKGResearch\\final-pair_geometry-s2020-20261023-171221-e3dfce`；启动提交 `06c2035bd5261fba7a0442ab6110727ce6d1f605`。此处待最终结果完成后补充。',
          '', '复现命令（在codex/pair-geometry分支，创建全新唯一输出目录）：','',
          '```powershell',
          "& 'D:\\CondaEnvs\\CoLaKG\\python.exe' -B -u D:\\GitHub\\colakg\\CoLaKG\\experiments\\research_runner.py --method pair_geometry --phase final --epochs 2000 --seed 2020 --frozen-config D:\\GitHub\\colakg\\CoLaKG\\experiments\\frozen_pair_geometry_v1.json",'```','',
          '与历史原基线协议差异：原基线每5轮训练前评估并保留测试峰值，本次只在固定2000轮后评估；保存频率和日志方式不同，原训练超参数及负采样一致。主要比较历史最终2000轮NDCG@20=0.34284222、Recall@20=0.37498634；历史测试峰值0.34733416另列，不能把反复测试取最大、单次固定预算及论文五次平均混为一谈。',
          '', '## 可追溯性与可信度限制','',
          '原始资产数据版本见lastfm_data_audit.md：仓库公开train/test共71294交互，与论文86608的统计差异仍未解释，但文件与原公开版本相同。公开用户语义的历史生成链不完整；开发划分只隔离监督图/标签，不证明语义资产层面完全无泄漏。所有对照沿用同一公开资产版本；独立语义图候选主动舍弃公开用户语义，属于其明确框架差异。',
          '',
          'SGCL、LightCCF、CCLRec和AlphaRec论文/官方代码阅读及迁移差异见research_sources.md；几何辅助直接借鉴既有SGCL目标，保留BPR的做法也不是原SGCL单目标框架。它目前不是已证实的新损失或论文贡献。后续需要同预算强基线、更多完整种子、多数据集和针对性新机制，而不是改名包装。',
          '',
          '本轮完成开发运行{}次，合计{}轮、{}次更新、循环时间约{:.1f}分钟；这是多个独立200轮实验之和，不能当作某个新模型已经跑完2000轮。原始基线分支和main不变；没有执行任何远程push。'.format(aggregate['completed_runs'],aggregate['completed_epochs'],aggregate['updates'],aggregate['training_loop_seconds']/60),
          '', '| 方法分支 | 开发代码起点 | 本地结果记录提交 |','|---|---|---|',
          '| research/catalog-contrastive | 83b8fe7 | d0068c8 |',
          '| research/consensus-residual | f83ce1a | b8e1e77 |',
          '| research/semantic-graph-decoder | 2b05db3 | d97d41e |',
          '| research/aligned-catalog | 88dac9f | f3c98bd |',
          '| codex/pair-geometry | 8ba7eb3 | 06c2035（完整运行后另存报告提交） |','',
          '复现各开发方法时，先切换表中相应分支，再用绝对Python/脚本路径执行：`D:\\CondaEnvs\\CoLaKG\\python.exe -B -u D:\\GitHub\\colakg\\CoLaKG\\experiments\\research_runner.py --method 方法名 --epochs 200 --seed 2020`。方法名按各中文记录；新输出全部在D盘且唯一命名。三种子原始对照、每个运行ID、原始数值与结果SHA保存在research_records/development_results_v1.json。',
          '',
          '唯一启动失败为语义图候选的Python3.8相对源路径归档错误，发生在零更新时；修复启动路径后从头新运行，失败目录保留。无数据/语义资产/原模型公式被改写，无已发表或未发表研究分支被推送。']
content='\n'.join(lines)+'\n'
final_dir=Path(r'D:\CoLaKGResearch\final-pair_geometry-s2020-20261023-171221-e3dfce')
if (final_dir/'result.json').exists():
    final=json.loads((final_dir/'result.json').read_text(encoding='utf-8'))
    assert final['success'] and final['source_unchanged'] and final['input_unchanged']
    assert final['evaluation_count']==1 and final['completed_epochs']==2000 and final['updates']==114000
    metrics=final['final_metrics'];ndcg=metrics['ndcg'][1];recall=metrics['recall'][1]
    content=content.replace('开发研究已完成，冻结的2000轮最终训练进行中；本报告的最终测试段在运行结束后补齐。',
                            '已完成五个方向、三种子开发确认、组件消融及冻结配置后的2000轮一次最终测试。以下保留所有负面结果，并区分开发、固定预算测试与历史测试峰值。')
    summary=('最终完成2000轮、114000次更新、一次原测试；源/输入哈希通过。\n\n'
             '| 指标 | 当前几何辅助最终模型 | 原始固定2000轮模型 |\n|---|---:|---:|\n'
             '| Recall@20 | %.10f | 0.3749863384 |\n| NDCG@20 | %.10f | 0.3428422200 |\n\n'
             '相对原始固定预算最终NDCG@20变化 **%+.4f%%**；相对历史测试峰值0.3473341594变化 **%+.4f%%**。'
             '本轮最终测试只有seed2020，不能声称多种子最终测试稳定提升或论文五次平均结论。\n\n'
             '完整循环耗时%.3f秒（%.2f小时），allocated峰值%.2fMiB，reserved峰值%.2fMiB，最终联合损失%.8f。'
             '计时包括采样、训练、最终评估和状态保存；每轮training_seconds包含原Python采样。\n\n'
             '其他最终指标：Precision@10/20=%s，Recall@10/20=%s，NDCG@10/20=%s。模型与优化器/RNG状态为该运行目录checkpoint.pt，'
             '完整日志console.log、history.json、evaluations.json、manifest.json、result.json、pip-freeze.txt。'
             % (recall,ndcg,(ndcg/.34284221999548964-1)*100,(ndcg/.3473341593547369-1)*100,
                final['elapsed_seconds'],final['elapsed_seconds']/3600,final['peak_allocated_bytes']/2**20,
                final['peak_reserved_bytes']/2**20,final['latest']['loss'],str(metrics['precision']),str(metrics['recall']),str(metrics['ndcg'])))
    gain_vs_peak=ndcg/.3473341593547369-1
    timing_history=json.loads((final_dir/'history.json').read_text(encoding='utf-8'))
    timing_evals=json.loads((final_dir/'evaluations.json').read_text(encoding='utf-8'))
    sampling_and_training=sum(row['training_seconds'] for row in timing_history)
    evaluation_seconds=sum(row['seconds'] for row in timing_evals)
    summary+='\n\n耗时拆分：采样+训练合计%.3f秒，原评估函数计时%.3f秒，其余状态保存、数值检查、日志及缓存表示等约%.3f秒。入口未分别计采样和纯GPU训练，不能进一步拆分；原基线评估/保存更频繁，因此总时间差不能直接归因为算法加速。' % (sampling_and_training,evaluation_seconds,final['elapsed_seconds']-sampling_and_training-evaluation_seconds)
    if ndcg>.34284221999548964:
        verdict='该固定预算单种子测试相对原始最终模型有正收益，但完整测试尚未进行多种子配对确认。'
    else:
        verdict='短预算验证优势未在本次固定2000轮测试中保持；不得据此改温度/权重、挑中间轮次或反复测试挽救结果。'
    if gain_vs_peak<.04:
        verdict+='相对于已知原始最佳观测值的4%–5%研究目标尚未达到。'
    else:
        verdict+='该单种子数值达到或超过相对于原始最佳观测值的4%目标，但不能声称已经获得稳定平均提升。'
    content=content.replace('此处待最终结果完成后补充。',summary+'\n\n'+verdict)
    content=content.replace('06c2035（完整运行后另存报告提交）','06c2035（完整训练执行版本；报告提交见本分支HEAD）')
    content=content.replace('这是多个独立200轮实验之和，不能当作某个新模型已经跑完2000轮。','这部分是多个独立200轮开发实验之和；另有一个冻结后的2000轮完整候选，分开记录。')
content+='\n## Windows换行与严格哈希复现\n\n历史冻结v1文件保持原样。Git的LF/CRLF转换可能改变字节哈希；`research_freeze_replay.py`逐文件核验当前代码及文本数据与训练提交06c2035的Git规范化内容完全相同，二进制语义资产哈希必须相同，然后只创建一个新冻结JSON，绝不改写代码/数据/历史记录，也不允许任意源变更通过。源与数据实际字节哈希记录在新清单里。\n\n复现时先运行 `D:\\CondaEnvs\\CoLaKG\\python.exe -B -u D:\\GitHub\\colakg\\CoLaKG\\experiments\\research_freeze_replay.py`，再用输出的新路径替换完整训练命令的 `--frozen-config`。其余方法、seed2020与2000轮不变。如果源内容超出换行变化，脚本拒绝刷新，不能绕过。\n'
if (final_dir/'result.json').exists():
    content+='\n## 本轮结论与后续优先级\n\n得到小幅正收益候选，尚未达到相对历史最佳值4%–5%的目标。当前没有把它整合到稳定分支：完整预算测试仍只有一个种子。优先补充固定协议的完整预算种子2021/2022配对对照，检验效应是否持续；随后才考虑用仅训练图和语义共识为几何分散项提供置信权重，并与已有CCLRec/SGCL作强对照。这是待验证方向，不是已实现或已证实的创新。\n\n![完整训练联合损失](research_figures/final-training-loss.png)\n\n绘图使用已有Codex附属Python中的ReportLab4.4.9和Sharp生成及核验SVG/PNG，没有给CoLaKG训练环境添加绘图依赖。研究的完整训练和开发阶段共14次运行；13次为独立200轮开发实验，另1次为冻结的2000轮最终实验，不能混成单次训练预算。\n\n最终快照完整性见research_records/final_artifact_integrity.json；机器可读最终结果见research_records/final_pair_geometry_s2020.json。没有为本次快照完整性检查再次评估原test。\n'
(root/'experiments/research_study_v1.md').write_text(content,encoding='utf-8')
print(root/'experiments/research_study_v1.md')
