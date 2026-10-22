# LastFM 正式 2000 轮基线计划（尚未启动）

保持 `train_lastfm.sh` 的全部模型/优化器参数和现有数据划分，使用同一 D 盘环境。继续使用原始 Python 负采样回退路径并记录该选择，不安装 C++ 扩展。

1. 单 minibatch、完整 epoch、训练后一次原始排名评估及一次隔离 checkpoint 保存均已通过；实测与协议差异见 [lastfm_one_epoch.md](lastfm_one_epoch.md)。复现该短跑使用 `lastfm_one_epoch.py`，不要使用下方正式入口。
2. 正式运行需另行审核，使用下方 **2000** 轮入口并全新随机初始化；不续用单步/单轮模型，不加载旧 checkpoint。正式保留原始 epoch=0,5,...,1995 的训练前评估节奏和每轮保存策略。
3. 保存运行配置、Git HEAD 与未提交补丁、输入哈希、环境导出、日志及耗时。checkpoint/TensorBoard/图缓存全部在新的唯一运行目录；checkpoint 沿用原作者的“每轮更新当前文件”策略，不覆盖其他运行。
4. 单轮外推总耗时约 6.15–8.33 小时（两种假设，非置信区间），建议预留约 9 小时；若长时运行或评估 OOM，停止并报告，不自动改 batch、模型、图传播或设备。

以下 PowerShell 命令是正式阶段的可执行入口，**当前没有运行**。它复用单步脚本的配置解析和 I/O 隔离工具，再执行原始 `main.py`；没有修改任何原始源码，正式阶段固定为 2000 轮。

```powershell
$Epochs = 2000  # 仅在正式运行经审核后执行；全新随机初始化
$env:COLAKG_BASELINE_EPOCHS = "$Epochs"
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = 'D:\CoLaKGCache\tmp'
$env:TMP = 'D:\CoLaKGCache\tmp'
$env:TORCH_HOME = 'D:\CoLaKGCache\torch'
Set-Location 'D:\GitHub\colakg\CoLaKG\rec_code'
$Launch = @'
import builtins, contextlib, json, os, runpy, sys, uuid
from datetime import datetime
from pathlib import Path
root = Path(r"D:\GitHub\colakg\CoLaKG")
sys.path.insert(0, str(root / "experiments"))
from lastfm_one_minibatch import (
    ROOT, REC, OUTPUT, CHINA, Tee, git, digest, tracked_hashes,
    original_arguments, bootstrap_world, isolated_graph_cache, write_json,
)
assert git("branch", "--show-current") == "exp/colakg-improvement"
assert Path(sys.executable).resolve() == Path(r"D:\CondaEnvs\CoLaKG\python.exe").resolve()
epochs = int(os.environ["COLAKG_BASELINE_EPOCHS"])
assert epochs == 2000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
run = OUTPUT / ("lastfm-baseline-" + datetime.now(CHINA).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
run.mkdir(parents=True, exist_ok=False)
(run / "logs").mkdir()
assignments, arguments = original_arguments()
sys.path.insert(0, str(REC))
world = bootstrap_world(arguments, run)
assert world.LOAD == 0 and world.config["pretrain"] == 0
world.TRAIN_epochs = epochs
import torch
import scipy.sparse as sparse
assert torch.cuda.is_available() and world.device.type == "cuda"
write_json(run / "configuration.json", {
    "epochs": epochs, "seed": world.seed, "config": world.config,
    "arguments": arguments, "assignments": assignments,
    "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short", "--branch"),
    "tracked_input_hashes": tracked_hashes(), "main_source_sha256": digest(REC / "main.py"),
    "checkpoint_loading": False,
})
real_open = builtins.open
def isolated_open(file, *args, **kwargs):
    if isinstance(file, (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(file)).resolve()
        if path.parent == (ROOT / "logs").resolve():
            file = run / "logs" / path.name
    return real_open(file, *args, **kwargs)
builtins.open = isolated_open
print("RUN_DIRECTORY=" + str(run), flush=True)
try:
    with (run / "console.log").open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            with isolated_graph_cache(sparse, ROOT / "data" / "lastfm", run):
                runpy.run_path(str(REC / "main.py"), run_name="__main__")
finally:
    builtins.open = real_open
'@
& 'D:\CondaEnvs\CoLaKG\python.exe' -B -u -c $Launch
```

注意：正式入口会进行 2000 轮训练、400 次原始评估和每轮 checkpoint 保存，必须另行审核后运行。本轮只执行了独立的一轮训练、训练后一次 Test 和一次新 checkpoint 保存，没有执行此正式入口。
