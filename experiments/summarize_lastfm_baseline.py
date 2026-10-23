"""Audit a completed baseline's JSON records; export CSV and a Chinese report.

This never imports torch or reads model weights. Optional SVG plots use
ReportLab in a separate plotting interpreter; no training package is installed.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def export_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(destination, history, evaluations, reference):
    from reportlab.graphics.shapes import Drawing, String, Line, Rect
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.graphics import renderSVG
    from reportlab.lib import colors
    destination.mkdir(parents=True, exist_ok=True)
    for stem, title, ylabel, series, names in [
        ("lastfm-baseline-loss", "LastFM CoLaKG: training loss (seed 2020)", "Mean minibatch BPR + regularization",
         [[(row["completed_epochs"], row["mean_minibatch_loss"]) for row in history]], ["Original training, 2000 epochs"]),
        ("lastfm-baseline-ndcg20", "LastFM CoLaKG: test NDCG@20 (seed 2020)", "NDCG@20",
         [[(row["completed_epochs"], row["metrics"]["ndcg"][1]) for row in evaluations if row["kind"] == "original_protocol"],
          [(0, reference), (2000, reference)],
          [(2000, evaluations[-1]["metrics"]["ndcg"][1])]],
         ["400 original pre-training evaluations", "Reference 0.34498495", "Final fixed-budget evaluation"]),
    ]:
        drawing = Drawing(960, 470)
        drawing.add(Rect(0, 0, 960, 470, fillColor=colors.white, strokeColor=None))
        drawing.add(String(72, 438, title, fontName="Helvetica-Bold", fontSize=18, fillColor=colors.HexColor("#172A3A")))
        drawing.add(String(72, 413, ylabel, fontName="Helvetica", fontSize=11, fillColor=colors.HexColor("#475569")))
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 72, 76, 844, 312
        chart.data = series
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 0, 2000
        chart.xValueAxis.valueSteps = [0, 500, 1000, 1500, 2000]
        chart.yValueAxis.valueMin = 0
        chart.yValueAxis.valueMax = math.ceil(max(y for group in series for _, y in group) * 1.05 * 20) / 20
        chart.yValueAxis.labelTextFormat = "%.2f"
        chart.xValueAxis.labels.fontSize = chart.yValueAxis.labels.fontSize = 11
        chart.yValueAxis.visibleGrid = True
        chart.yValueAxis.gridStrokeColor = colors.HexColor("#E2E8F0")
        palette = ["#1769AA", "#64748B", "#E87500"]
        for index in range(len(series)):
            chart.lines[index].strokeColor = colors.HexColor(palette[index])
            chart.lines[index].strokeWidth = 1.0
        if len(series) > 1:
            chart.lines[1].strokeDashArray = [5, 4]
            chart.lines[2].symbol = makeMarker("FilledCircle")
            chart.lines[2].symbol.size = 7
            chart.lines[2].symbol.fillColor = colors.HexColor(palette[2])
            chart.lines[2].symbol.strokeColor = colors.HexColor(palette[2])
        drawing.add(chart)
        drawing.add(String(494, 36, "Completed training epochs", textAnchor="middle", fontName="Helvetica", fontSize=12))
        for index, name in enumerate(names):
            x = [72, 415, 665][index]
            drawing.add(Line(x, 13, x + 18, 13, strokeColor=colors.HexColor(palette[index]), strokeWidth=2))
            drawing.add(String(x + 23, 9, name, fontName="Helvetica", fontSize=9))
        renderSVG.drawToFile(drawing, str(destination / (stem + ".svg")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "lastfm_baseline.md")
    parser.add_argument("--plots", type=Path, help="Optional destination for ReportLab SVG curves")
    args = parser.parse_args()
    run = args.run.resolve()
    result, status, manifest = [read(run / name) for name in ["result.json", "status.json", "manifest.json"]]
    history, evaluations = read(run / "training-history.json"), read(run / "evaluations.json")
    assert result["success"] and status["status"] == "completed"
    assert result["completed_epochs"] == 2000 and result["updates"] == 114000
    assert [row["completed_epochs"] for row in history] == list(range(1, 2001))
    assert all(row["updates_this_epoch"] == 57 and row["sampled_triplets"] == 57786 for row in history)
    assert all(math.isfinite(row["mean_minibatch_loss"]) for row in history)
    protocol = [row for row in evaluations if row["kind"] == "original_protocol"]
    final = [row for row in evaluations if row["kind"] == "final_2000_epochs"]
    assert [row["completed_epochs"] for row in protocol] == list(range(0, 2000, 5))
    assert len(final) == 1 and final[0]["completed_epochs"] == 2000
    integrity_paths = sorted(run.glob("integrity-*.json"))
    assert integrity_paths
    for path in integrity_paths:
        integrity = read(path)
        assert all(integrity[key] for key in ["tracked_files_unchanged", "main_ref_unchanged", "branch_unchanged"])
    assert sha256(Path(result["final_model"])) == result["final_model_sha256"]
    best = max(protocol, key=lambda row: row["metrics"]["ndcg"][1])
    recall_best = max(protocol, key=lambda row: row["metrics"]["recall"][1])
    assert best == result["best_observed_original_protocol"]
    assert final[0] == result["final_2000_epochs"]
    export_csv(run / "loss-curve.csv", [{key: row[key] for key in [
        "completed_epochs", "cumulative_updates", "mean_minibatch_loss", "sample_weighted_loss",
        "training_seconds", "sampling_seconds", "stage_seconds", "numeric_checks_seconds", "model_save_seconds",
        "peak_allocated_bytes", "peak_reserved_bytes"]} for row in history])
    export_csv(run / "ranking-curves.csv", [{
        "kind": row["kind"], "completed_epochs": row["completed_epochs"],
        "original_epoch_argument": row["original_epoch_argument"], "original_display_epoch": row["original_display_epoch"],
        **{name + "@" + str(k): row["metrics"][name][i] for name in ["precision", "recall", "ndcg"] for i, k in enumerate([10, 20])},
        "evaluation_seconds": row["seconds"]} for row in evaluations])
    if args.plots:
        plots(args.plots, history, evaluations, result["reference_ndcg20"])
    totals, identity = result["totals"], manifest["identity"]
    failures = [read(path) for path in sorted(run.glob("failure-*.json"))]
    # The original runner records wall time per session. Include failed sessions,
    # but exclude the overnight pause from active runtime; preserve raw result.json.
    unique_failures = {row["session_id"]: row for row in failures}
    previous_wall = sum(row["session_elapsed_seconds"] for row in unique_failures.values())
    active_wall = previous_wall + totals["session_wall_seconds"]
    started = min([row["started_at"] for row in failures] + [result["started_at"]])
    calendar_wall = (datetime.fromisoformat(result["finished_at"]) - datetime.fromisoformat(started)).total_seconds()
    peak_allocated = max([result["peak_allocated_bytes"]] + [row["peak_allocated_bytes"] for row in history])
    peak_reserved = max([result["peak_reserved_bytes"]] + [row["peak_reserved_bytes"] for row in history])
    recovery = result.get("recovered_from")
    resource_audit = {"active_wall_seconds_all_sessions": active_wall,
                      "calendar_wall_seconds_including_pause": calendar_wall,
                      "pause_seconds": calendar_wall - active_wall,
                      "prior_failed_session_seconds": previous_wall,
                      "final_session_seconds": totals["session_wall_seconds"],
                      "peak_allocated_bytes_all_sessions": peak_allocated,
                      "peak_reserved_bytes_all_sessions": peak_reserved,
                      "session_ids": list(unique_failures) + [result["session_id"]],
                      "recovery": recovery}
    (run / "resource-audit.json").write_text(json.dumps(resource_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for label, row in [("原协议 NDCG@20 最大观测点", best), ("固定 2000 轮最终模型", final[0])]:
        values = [row["metrics"][name][i] for name in ["precision", "recall", "ndcg"] for i in [0, 1]]
        rows.append("| {} | {} | {} |".format(label, row["completed_epochs"], " | ".join("{:.10f}".format(value) for value in values)))
    input_rows = "\n".join("| `{}` | {} | `{}` |".format(name, value["bytes"], value["sha256"]) for name, value in identity["inputs"].items())
    version_rows = "、".join("{} {}".format(name, value) for name, value in identity["versions"].items())
    difference = result["best_minus_reference"]
    final_difference = result["final_minus_reference"]
    recoveries = sorted(run.glob("console-*.log"))
    report = f'''# LastFM 原始 CoLaKG：2000 轮基线复现

## 结论

已完成 **2000/2000 轮、114000 次 Adam 更新**，400 次原协议训练前测试及 1 次固定预算最终测试。全程从本次随机初始化开始，没有加载其他实验 checkpoint，没有修改模型结构、图传播、损失或超参数，没有按测试结果提前停止或调参。

| 指标口径 | 实际已训练轮数 | Precision@10 | Precision@20 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

原协议 Recall@20 单独最大值为 **{recall_best['metrics']['recall'][1]:.10f}**，发生在已完成 **{recall_best['completed_epochs']}** 轮；它不一定与最大 NDCG 点相同。表中最佳行的所有指标来自同一个最大 NDCG@20 观测点。

参考 NDCG@20 为 **{result['reference_ndcg20']:.8f}**。原协议最大观测值与参考的有符号差为 **{difference:+.10f}**（{100*difference:+.6f} 个百分点，参考相对差 {100*difference/result['reference_ndcg20']:+.4f}%）；固定 2000 轮最终值差为 **{final_difference:+.10f}**。此处只比较给定参考数值，不能据此宣称模型创新或性能提升。

**评估局限：**“最佳”是在同一测试集上反复观察 400 次后的最大值，存在测试集选择偏差，属于原作者代码报告口径；不是通过验证集选择得到的独立结论。最终模型严格取固定 2000 轮预算，另行调用原始 Test，但仍使用同一测试集，不能视为新增独立留出集。单种子复现不提供多种子均值或显著性结论。

## 配置、环境与版本

- 项目：`D:\\GitHub\\colakg\\CoLaKG`；分支：`exp/colakg-improvement`。
- 启动时 Git HEAD：`{identity['git_head']}`；正式入口当时是待提交的新文件，实际执行 SHA-256：`{identity['source_sha256']['experiments' + chr(92) + 'lastfm_baseline.py']}`。运行目录保留相同源码副本，提交后可按此哈希追溯；脚本与本报告的保存提交可用 `git log -1 -- experiments/lastfm_baseline.md` 查询。
- 解释器：`D:\\CondaEnvs\\CoLaKG\\python.exe`，{identity['python']}。
- 依赖：{version_rows}；CUDA runtime {identity['cuda_runtime']}；设备 {identity['gpu']}，2 GiB。未安装新依赖，未修改其他环境。
- 完整 world 配置：`{json.dumps(identity['config'], ensure_ascii=False, sort_keys=True)}`。
- seed={identity['seed']}，epochs={identity['epochs']}，topks={identity['topks']}；Adam 参数见运行目录 `status.json`/`result.json` 的 optimizer 字段。BPR 显式正则 0.0001，Adam 自身 weight_decay=0，未加调度器、AMP 或梯度裁剪。
- {result['data']['users']} 用户、{result['data']['items']} 物品、{result['data']['train_interactions']} 训练交互、{result['data']['test_interactions']} 测试交互。输入来自当前仓库原始 LastFM 划分与已提供语义张量，不重新生成嵌入，不改数据划分。

| 输入 | 字节数 | SHA-256 |
| --- | ---: | --- |
{input_rows}

## 实际执行协议

读取原始 `train_lastfm.sh`，通过未修改的 `world.py` 解析。新脚本只隔离路径、记录日志、检查有限值并保存同次恢复状态；每轮调用原始 `Procedure.BPR_train_original`，每次评估调用原始 `Procedure.Test`。模型、Adam 和 BPRLoss 复用原类。

每轮使用原始 Python 回退负采样（`utils.UniformSample_original_python`）：抽取训练交互数规模的用户样本，为用户随机抽一个训练正例，再随机拒绝采样一个不在训练正例中的负例；neg_k=1。每轮实际采样 57786 个三元组，57 个 minibatch（56×1024+442），实际检查 Adam 步数与总更新数一致。

在 original epoch=0,5,...,1995 **训练开始前**评估，分别对应已完成训练轮数 0,5,...,1995。原 main 日志会显示 EPOCH[1]、[6]、...、[1996]，不能误读为已训练这些轮数。本次事件和 CSV 同时保留两种标签。第 0 次是随机模型评估。训练 2000 轮后额外执行一次最终测试，与原协议最大值分开保存。

原 TensorBoard BPR step 使用 `epoch * int(57786/1024) + batch_i`，相邻 epoch 的步号有一个重叠；本次保留该原始行为，曲线使用独立记录的实际 epoch 和真实更新数，避免误对齐。

## 耗时与显存

| 项目 | 实测 |
| --- | ---: |
| 训练循环合计（含采样、前反向、Adam、数值检查和原日志） | {totals['training_seconds']:.3f} 秒 |
| 其中原始负采样 | {totals['sampling_seconds']:.3f} 秒 |
| 其中 BPR stageOne 包裹耗时 | {totals['stage_seconds']:.3f} 秒 |
| 有限值检查读数（与 stageOne 时间部分重叠，不可相加） | {totals['numeric_checks_seconds']:.3f} 秒 |
| 400+1 次评估合计 | {totals['evaluation_seconds']:.3f} 秒 |
| 每轮模型保存合计 | {totals['model_save_seconds']:.3f} 秒 |
| 完整恢复快照保存合计 | {totals['snapshot_save_seconds']:.3f} 秒 |
| 两段实际运行合计（含准备及文件 I/O，不含停机等待） | {active_wall:.3f} 秒（{active_wall/3600:.3f} 小时） |
| 从首次启动到最终完成的日历跨度（含中断等待） | {calendar_wall:.3f} 秒（{calendar_wall/3600:.3f} 小时） |
| 每轮训练耗时中位数 / 最小 / 最大 | {statistics.median(row['training_seconds'] for row in history):.3f} / {min(row['training_seconds'] for row in history):.3f} / {max(row['training_seconds'] for row in history):.3f} 秒 |
| 两段 PyTorch 进程峰值 allocated / reserved | {peak_allocated/2**20:.2f} / {peak_reserved/2**20:.2f} MiB |
| 首轮 / 最终轮平均 minibatch loss | {history[0]['mean_minibatch_loss']:.10f} / {history[-1]['mean_minibatch_loss']:.10f} |

训练循环时间已经包含采样，不能再次相加。整卡监视日志包含桌面及其他程序占用，不能当作模型专属显存。恢复阶段峰值还包括反序列化快照暂存，不等于模型新增参数。未改系统电源计划，监视进程只在运行期间防止空闲休眠。原始 result 的 session_wall_seconds 仅是最终会话耗时；resource-audit.json 加上失败会话日志中的时钟值得到总实际运行时间，明确排除中断后的等待。完整快照耗时沿用原始计时，恢复点最近一次快照写入的极小尾部开销可能未包含在累计快照计时中。总耗时不包括此后作图和 Git 操作。

## 曲线与运行产物

![完整 2000 轮 Loss 曲线](figures/lastfm-baseline-loss.svg)

![完整原协议 NDCG@20 曲线及最终点评估](figures/lastfm-baseline-ndcg20.svg)

原始值不平滑、不丢弃下降点：运行目录 `loss-curve.csv` 有 2000 行，`ranking-curves.csv` 有 401 行，保留 Precision/Recall/NDCG@10/20。绘图仅读取 JSON，不加载权重，使用独立附带绘图运行时的 ReportLab，不向 CoLaKG 环境安装绘图库。

本次独立运行目录：`{run}`。

- 完整日志：`console-*.log`、`segments/*.jsonl`、`monitor-*.jsonl`、`tensorboard/`。
- 配置与依赖：`manifest.json`、`pip-freeze.txt`、`conda-explicit.txt`、`conda-environment.yml`、`source/`、`uncommitted.patch`。
- 指标与状态：`training-history.json`、`evaluations.json`、`result.json`、`status.json`、完整曲线 CSV。
- 最终模型：`{result['final_model']}`，SHA-256 `{result['final_model_sha256']}`。
- 同次恢复：`checkpoints/recovery-0.pt`、`recovery-1.pt` 和带 SHA-256 的 `recovery.json`。初始/每 10 轮保存模型、Adam、Python/NumPy/PyTorch CPU/CUDA RNG；两个槽轮换，不增长为 2000 个文件，不覆盖其他实验。实际会话日志数 {len(recoveries)}。

## 复现命令与协议差异

在 PowerShell 执行下列命令会**新建唯一目录并重新开始完整 2000 轮**，不会加载本次或其他旧模型：

```powershell
Set-Location 'D:\\GitHub\\colakg\\CoLaKG'
$env:PYTHONNOUSERSITE = '1'
$env:TEMP = 'D:\\CoLaKGCache\\tmp'
$env:TMP = 'D:\\CoLaKGCache\\tmp'
$env:TORCH_HOME = 'D:\\CoLaKGCache\\torch'
$env:PIP_CACHE_DIR = 'D:\\CoLaKGCache\\pip'
$env:CONDA_PKGS_DIRS = 'D:\\CoLaKGCache\\conda-pkgs'
& 'D:\\CondaEnvs\\CoLaKG\\python.exe' -B -u 'D:\\GitHub\\colakg\\CoLaKG\\experiments\\lastfm_windows.py'
```

汇总而不训练、不加载权重：

```powershell
& 'D:\\CondaEnvs\\CoLaKG\\python.exe' -B 'D:\\GitHub\\colakg\\CoLaKG\\experiments\\summarize_lastfm_baseline.py' '{run}'
```

恢复入口与监视器用法见 `full_baseline_plan.md`。Windows 适配器只替换原子输出函数，对 WinError 5/32/33 作有限重试，不消耗随机数、不跳过原始源码哈希检查；监视器通过 FileShare.ReadWrite|Delete 读取，避免阻止原子替换。恢复仅限同次运行、同样代码/数据/依赖，校验 manifest 与快照哈希后恢复全部 RNG，不从别的实验取权重。未宣称默认 CUDA 稀疏运算具备跨运行逐位确定性；后端设置在 manifest 中完整记录。

与原作者数学/训练协议一致；执行环境为 Windows/Python3.8/PyTorch1.13 CUDA11.7，使用 Python 负采样回退，I/O 路径改为 D 盘独立目录，新增数值检查、完整恢复快照与最终固定预算评估。这些差异不修改模型公式，但限制对参考值逐位一致的预期。未验证参考值所用硬件、编译采样后端与完整依赖快照，因此只报告实际数值差，不编造差异成因。

## 异常与完整性

所有已完成会话的完整性记录确认运行期间已跟踪项目文件内容、main 引用及实验分支保持不变。原始 `rec_code` 没有修改；所有训练输出位于本次 D 盘目录。没有加载其他实验 checkpoint，没有根据测试曲线调整配置。会话日志数为 {len(recoveries)}，失败记录数为 {len(failures)}。原始 result.json 中 status/phase 是完成前复制的字段，保留不改；最终完成状态联合核验 status.json=completed、result.success=true、completed_epochs=2000 及完整性记录，不依赖旧 status 字段。

首次运行在 2026-10-23 01:36:56 完成 1870 轮后，因 status.json 原子替换返回 PermissionError/WinError 5 停止。监视器读取与替换存在 Windows 文件共享竞争风险；实际原因可确定到文件替换失败，不能仅凭日志排除杀软等短暂占用。故障发生在第 1870 轮快照保存后、该轮次评估开始前，没有损失已完成训练更新。旧 status.json 没能写成 failed，failure-*.json 与进程退出才是此故障的最终状态依据。

经用户确认后，仅新增 Windows I/O 适配器并修正监视器共享读取。用真实 Windows 不允许删除共享的只读句柄复现占用，重试 4 次后替换成功，随机数状态不变；测试记录保存在 io-tests/。恢复输入快照已独立归档于 io-adaptations/，SHA-256 为 `{recovery['sha256'] if recovery else '未使用恢复'}`。适配器、原入口及同次快照校验均保留记录，原训练入口本身未改。恢复从已完成 {recovery['completed_epochs'] if recovery else 0} 轮继续，实际训练历史保持 1..2000 唯一连续，没有重计回滚事件。

中断与恢复是本次执行协议的一项差异，已如实披露。完成了全部 2000 轮与原协议评估频率；未额外证明恢复路径与不中断 CUDA 运行逐位等价，不将此单次结果描述为独立无偏性能结论。最终汇总通过 epoch/评估频率/更新数/数值有限性/最终模型哈希及完整性检查。
'''
    args.report.write_text(report, encoding="utf-8")
    summary = {"success": True, "run": str(run), "epochs": 2000, "updates": 114000,
               "best_ndcg20_epoch": best["completed_epochs"], "best_ndcg20": best["metrics"]["ndcg"][1],
               "recall20_at_best_ndcg": best["metrics"]["recall"][1], "best_recall20": recall_best["metrics"]["recall"][1],
               "best_recall20_epoch": recall_best["completed_epochs"], "final_recall20": final[0]["metrics"]["recall"][1],
               "final_ndcg20": final[0]["metrics"]["ndcg"][1], "wall_seconds": active_wall,
               "calendar_wall_seconds": calendar_wall,
               "peak_allocated_MiB": peak_allocated/2**20, "peak_reserved_MiB": peak_reserved/2**20}
    (run / "audited-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
