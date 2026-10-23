# LastFM 原始 CoLaKG：正式运行与恢复协议

本轮 2000 轮已完成，实测指标、曲线、中断及恢复经过见 [lastfm_baseline.md](lastfm_baseline.md)。下面的命令用于用户主动复现，不会在本轮自动再启动训练。历史单步、单轮验收记录继续保留。

## 配置与执行

- 项目：`D:\GitHub\colakg\CoLaKG`，分支 `exp/colakg-improvement`。
- 解释器：`D:\CondaEnvs\CoLaKG\python.exe`，独立环境不重建、不安装新包。
- 原始 `train_lastfm.sh` 配置：batch=1024、3 层、neighbor_k=10、dim=64、seed=2020、lr=0.001、decay=0.0001、dropout_i/u/n=0.6/0.2/0.6、use_drop_edge=0、keep_prob=1.0；实际解析结果保存在 manifest。
- `lastfm_baseline.py` 不改原数学实现，导入原 CoLaKG、Loader、BPRLoss 和 Adam，每轮调用 `Procedure.BPR_train_original`。使用已验证的原始 Python 负采样回退，不编译扩展。
- 正式预算固定 2000 轮，不接受超参数覆盖。新运行只加载项目已有语义张量，可训练参数随机初始化，不加载其他实验权重。
- 原始训练前测试在 epoch=0,5,...,1995 执行，表示实际已训练 0,5,...,1995 轮；原 main 显示标签却为 1,6,...,1996。完成 2000 轮后另行评估最终模型。
- 最大 NDCG@20 只是原协议测试曲线最大观测值，存在反复观察测试集的选择偏差，不用于调参或选取后续创新模型。最终 2000 轮结果也来自同一测试集，并非新的独立留出集。

## 全新复现命令

以下 PowerShell 环境变量只作用于当前启动进程及子进程，不修改系统配置：

```powershell
Set-Location 'D:\GitHub\colakg\CoLaKG'
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = 'D:\CoLaKGCache\tmp'
$env:TMP = 'D:\CoLaKGCache\tmp'
$env:TORCH_HOME = 'D:\CoLaKGCache\torch'
$env:PIP_CACHE_DIR = 'D:\CoLaKGCache\pip'
$env:CONDA_PKGS_DIRS = 'D:\CoLaKGCache\conda-pkgs'
& 'D:\CondaEnvs\CoLaKG\python.exe' -B -u 'D:\GitHub\colakg\CoLaKG\experiments\lastfm_windows.py'
```

该命令新建 `D:\CoLaKGExperiments\lastfm-baseline-日期时间-唯一编号`。已有目录不会复用或清空。训练、评估及保存都使用本地文件，不需要持续联网；GitHub 推送才需要网络。

Windows 入口 `lastfm_windows.py` 导入未修改的正式训练脚本，仅替换原子输出函数；对 Windows WinError 5/32/33 最多尝试 10 次，等待总上限 5.55 秒，不关闭安全检查，不消耗随机数，不无限重试。其他错误立即停止。训练、采样、优化器及评估代码不受适配影响。

## 状态与监视

运行目录保存 `manifest.json`、依赖导出、源码副本、数据 SHA-256、逐事件 JSONL、TensorBoard、每轮状态及完整指标。每轮实际更新数、采样数、loss、耗时和 CUDA allocator 峰值均单独记录。附加有限值检查不修改梯度或参数，不使用 AMP、裁剪或调度器。

需要独立监视时，在另一 PowerShell 进程指定实际运行目录：

```powershell
$run = 'D:\CoLaKGExperiments\本次实际lastfm-baseline目录'
$state = Get-Content (Join-Path $run 'status.json') -Raw | ConvertFrom-Json
& 'D:\GitHub\colakg\CoLaKG\experiments\monitor_lastfm.ps1' -RunDirectory $run -TrainingProcessId $state.pid
```

监视器以允许共享删除/替换的句柄读取状态文件，每 30 秒记录整卡显存、温度和 D 盘余量。整卡数据包含桌面及其他进程，不能当作模型独占显存。它只在进程运行期间请求防止空闲休眠，不修改电源计划；合盖、关机或强制休眠仍可能中断运行。

## 有限保存与同次恢复

- `checkpoints/colakg-lastfm-latest.pth.tar`：本次目录内每轮原子更新的模型 state_dict，最终对应完整 2000 轮，不按测试指标选择。
- `recovery-0.pt` / `recovery-1.pt`：初始和每 10 轮保存模型、Adam、Python/NumPy/PyTorch CPU/CUDA RNG、历史和已完成轮数；两个槽轮换，避免保留 2000 份权重。
- `recovery.json`：最近完整快照文件名和 SHA-256。中断时最多回滚不足 10 轮；本次恰好恢复到完整保存的 1870 轮，未丢失或重计已完成更新。
- 任何 OOM、NaN/Inf 或无法保证一致性的错误都停止报告，不调整配置、不改用 CPU、不自动重启。

已退出的同一次运行，经核对后恢复：

```powershell
& 'D:\CondaEnvs\CoLaKG\python.exe' -B -u 'D:\GitHub\colakg\CoLaKG\experiments\lastfm_windows.py' --resume 'D:\CoLaKGExperiments\本次实际lastfm-baseline目录'
```

恢复前严格比较原入口源码、原模型源码、Git HEAD、全部输入哈希、实际配置及环境。Windows 适配不豁免这些检查。恢复所用快照在覆盖轮换槽前另行归档到本次 `io-adaptations/`，保留旧状态、适配器、适配哈希与命令；只允许该运行自身快照。恢复全部 RNG 后才进入下一次原始评估和训练。

恢复段单独记日志，汇总以快照中已提交历史加后续事件为准，不把回滚事件重复计入结果。保存完整 RNG 不等于已经证明默认 CUDA 稀疏计算的跨运行逐位确定性；本次中断与恢复在报告中明确披露。

## 完成后汇总

`summarize_lastfm_baseline.py` 只读 JSON 与文件哈希，不导入 torch、不加载模型，验证 2000 轮、114000 次更新和 400+1 次评估，输出 CSV 与中文报告。指定可选 `--plots` 时，用独立绘图解释器中现有的 ReportLab 生成 SVG；不向训练环境安装绘图库。

原始 `result.json` 保留入口生成时的原始内容，其中 status/phase 是完成前复制的字段；最终是否完成以 `status.json` 的 completed、result.success、completed_epochs=2000 和完整性记录联合核验。跨会话总实际耗时见 `resource-audit.json`，明确区分原始最终会话时钟、总活动时间和含中断等待的日历跨度。

Git 仅提交安全的入口、适配器、监视器、汇总脚本、说明及聚合曲线；权重、缓存、运行日志和原始数据不加入提交。只推送 origin 的实验分支，不修改 main、不推送 upstream。