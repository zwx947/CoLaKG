"""Original LastFM CoLaKG: 2000 epochs, original pre-training Test schedule.

Model, loss, sampler, Adam and ranking evaluation are imported unchanged.
Only output isolation, measurement and same-run recovery are added here.
No command-line hyperparameter overrides are accepted.
"""

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime

from lastfm_one_minibatch import (
    CHINA, OUTPUT, PYTHON, REC, ROOT, Tee, bootstrap_world, digest, git,
    isolated_graph_cache, original_arguments, tracked_hashes, write_json,
)

EPOCHS = 2000
REFERENCE_NDCG20 = 0.34498495
PACKAGES = ["torch", "numpy", "pandas", "scipy", "scikit-learn", "tensorboardX", "tqdm"]


def now():
    return datetime.now(CHINA).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(str(temporary), str(path))


def atomic_torch_save(torch, value, path):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def capture_command(command, destination):
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    destination.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("Provenance command failed: " + str(command))


def all_finite(torch, tensors):
    flags = [torch.isfinite(tensor).all() for tensor in tensors]
    return not flags or bool(torch.stack(flags).all())


def execute(run, state, resume):
    session_start = time.perf_counter()
    assignments, arguments = original_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REC))
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from tensorboardX import SummaryWriter

    world = bootstrap_world(arguments, run)
    if importlib.util.find_spec("cppimport") is not None:
        raise RuntimeError("Refusing implicit compilation; verified Python sampler required")
    utils = importlib.import_module("utils")
    dataloader = importlib.import_module("dataloader")
    model = importlib.import_module("model")
    procedure = importlib.import_module("Procedure")
    assert world.TRAIN_epochs == EPOCHS and world.seed == 2020
    assert world.LOAD == 0 and world.config["pretrain"] == 0
    assert world.dataset == "lastfm" and world.model_name == "colakg"
    assert not utils.sample_ext and world.tensorboard == 1
    assert world.topks == [10, 20] and world.config["multicore"] == 0
    assert torch.cuda.is_available() and world.device.type == "cuda"
    assert "MX450" in torch.cuda.get_device_name(0)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    utils.set_seed(world.seed)
    random.seed(world.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    data = ROOT / "data" / "lastfm"
    semantic = [(REC / world.item_semantic_emb_file).resolve(),
                (REC / world.user_semantic_emb_file).resolve()]
    inputs = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)}
              for path in [data / "train.txt", data / "test.txt", *semantic]}
    source_paths = [*sorted(REC.glob("*.py")), REC / "train_lastfm.sh",
                    Path(__file__), Path(__file__).with_name("lastfm_one_minibatch.py")]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    identity = {"git_head": git("rev-parse", "HEAD"), "source_sha256": source_hashes,
                "inputs": inputs, "versions": versions, "python": sys.version,
                "config": dict(world.config), "seed": world.seed, "epochs": EPOCHS,
                "topks": world.topks, "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "backend": {"cudnn_version": torch.backends.cudnn.version(),
                            "cudnn_benchmark": torch.backends.cudnn.benchmark,
                            "cudnn_deterministic": torch.backends.cudnn.deterministic,
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "torch_num_threads": torch.get_num_threads()}}
    if resume:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise RuntimeError("Same-run recovery rejected: code/data/config/environment identity changed")
    else:
        manifest = {"run_id": run.name, "created_at": now(), "identity": identity,
                    "command": subprocess.list2cmdline([str(PYTHON), "-B", "-u", str(Path(__file__).resolve())]),
                    "original_cli": arguments, "original_shell_assignments": assignments,
                    "git_status": git("status", "--short", "--branch"),
                    "initialization": "Random trainable parameters; only supplied semantic tensors loaded",
                    "negative_sampler": "Unmodified utils.UniformSample_original_python; neg_k=1",
                    "evaluation_protocol": "Test BEFORE original epochs 0,5,...,1995 = completed epochs 0,5,...,1995; separate final Test at completed_epochs=2000",
                    "saving_protocol": "Original model state_dict every epoch (one current file); full same-run model/Adam/Python/NumPy/CPU+CUDA RNG snapshot initially and every 10 completed epochs; two alternating atomic slots",
                    "numerics": "No AMP, clipping, scheduler, deterministic-kernel override, early stopping or tuning"}
        write_json(run / "manifest.json", manifest)
        capture_command([str(PYTHON), "-m", "pip", "freeze"], run / "pip-freeze.txt")
        conda = r"D:\miniconda3\Scripts\conda.exe"
        capture_command([conda, "list", "--prefix", str(PYTHON.parent), "--explicit"], run / "conda-explicit.txt")
        capture_command([conda, "env", "export", "--prefix", str(PYTHON.parent)], run / "conda-environment.yml")
        (run / "source").mkdir()
        for path in source_paths:
            destination = run / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(path), str(destination))
        (run / "uncommitted.patch").write_text(git("diff", "--binary", "HEAD"), encoding="utf-8")
        (run / "checkpoints").mkdir()
        (run / "segments").mkdir()

    print("IDENTITY=" + json.dumps(identity, ensure_ascii=False), flush=True)
    original_load = torch.load

    def semantic_only(path, *args, **kwargs):
        if Path(path).resolve() not in semantic:
            raise RuntimeError("Loading external model weights forbidden")
        return original_load(path, *args, **kwargs)

    torch.load = semantic_only
    try:
        dataset = dataloader.Loader(config=world.config, path=str(data))
        item_emb = torch.load(semantic[0], map_location="cpu", weights_only=True)
        user_emb = torch.load(semantic[1], map_location="cpu", weights_only=True)
        assert list(item_emb.shape) == [dataset.m_items, 1024]
        assert list(user_emb.shape) == [dataset.n_users, 1024]
        similarity = cosine_similarity(item_emb.numpy())
        neighbours = torch.tensor(np.argsort(-similarity, axis=1)[:, 1:world.config["neighbor_k"] + 1]).long()
        del similarity
        # Recreate the same normalized graph in a new session cache on recovery.
        cache_root = run if not resume else run / ("session-" + state["session_id"])
        if resume:
            cache_root.mkdir(exist_ok=False)
        with isolated_graph_cache(dataloader.sp, data, cache_root):
            recmodel = model.CoLaKG(world.config, dataset, neighbours, item_emb, user_emb).to(world.device)
        bpr = utils.BPRLoss(recmodel, world.config)
    finally:
        torch.load = original_load
    assert (dataset.n_users, dataset.m_items, dataset.trainDataSize, dataset.testDataSize) == (1859, 2813, 57786, 13508)
    state.update({"data": {"users": dataset.n_users, "items": dataset.m_items,
                           "train_interactions": dataset.trainDataSize, "test_interactions": dataset.testDataSize},
                  "optimizer": {"class": type(bpr.opt).__name__, "defaults": dict(bpr.opt.defaults), "regularization": bpr.weight_decay}})
    history, evaluations = [], []
    totals = {name: 0.0 for name in ["training_seconds", "sampling_seconds", "stage_seconds", "numeric_checks_seconds",
                                    "evaluation_seconds", "model_save_seconds", "snapshot_save_seconds", "session_wall_seconds"]}
    completed, updates = 0, 0
    if resume:
        pointer = json.loads((run / "recovery.json").read_text(encoding="utf-8"))
        checkpoint = (run / "checkpoints" / pointer["file"]).resolve()
        if checkpoint.parent != (run / "checkpoints").resolve() or digest(checkpoint) != pointer["sha256"]:
            raise RuntimeError("Same-run snapshot integrity mismatch")
        payload = torch.load(checkpoint, map_location=world.device)
        if payload["run_id"] != run.name or payload["manifest_sha256"] != digest(run / "manifest.json"):
            raise RuntimeError("Snapshot belongs to a different run/manifest")
        recmodel.load_state_dict(payload["model"])
        bpr.opt.load_state_dict(payload["optimizer"])
        completed, updates = payload["completed_epochs"], payload["updates"]
        history, evaluations, totals = payload["history"], payload["evaluations"], payload["totals"]
        state["recovered_from"] = {"file": str(checkpoint), "completed_epochs": completed, "sha256": pointer["sha256"]}
    state.update({"completed_epochs": completed, "updates": updates, "status": "running", "phase": "training",
                  "preparation_seconds": time.perf_counter() - session_start})
    writer = SummaryWriter(str(run / "tensorboard" / state["session_id"]))
    segment = (run / "segments" / (state["session_id"] + ".jsonl")).open("x", encoding="utf-8", buffering=1)
    last_snapshot = completed
    manifest_sha = digest(run / "manifest.json")

    def status():
        state.update({"updated_at": now(), "completed_epochs": completed, "updates": updates,
                      "session_elapsed_seconds": time.perf_counter() - session_start,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "totals": totals, "last_snapshot_completed_epochs": last_snapshot})
        atomic_json(run / "status.json", state)

    def event(kind, value):
        segment.write(json.dumps({"kind": kind, **value}, ensure_ascii=False) + "\n")

    def snapshot():
        nonlocal last_snapshot
        begin = time.perf_counter()
        rng = {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}
        target = run / "checkpoints" / ("recovery-{}.pt".format((completed // 10) % 2))
        payload = {"run_id": run.name, "manifest_sha256": manifest_sha,
                   "completed_epochs": completed, "updates": updates, "model": recmodel.state_dict(),
                   "optimizer": bpr.opt.state_dict(), "rng": rng, "history": history,
                   "evaluations": evaluations, "totals": totals}
        atomic_torch_save(torch, payload, target)
        atomic_json(run / "recovery.json", {"file": target.name, "sha256": digest(target),
                                           "completed_epochs": completed, "run_id": run.name})
        totals["snapshot_save_seconds"] += time.perf_counter() - begin
        last_snapshot = completed

    def evaluate(kind, epoch):
        state["phase"] = "evaluation_" + kind
        status()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        metrics = procedure.Test(dataset, recmodel, epoch, w=writer, multicore=world.config["multicore"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        values = {name: array.tolist() for name, array in metrics.items()}
        if not all(math.isfinite(value) and 0 <= value <= 1 for array in values.values() for value in array):
            raise FloatingPointError("Invalid evaluation metric")
        row = {"kind": kind, "completed_epochs": completed, "original_epoch_argument": epoch,
               "original_display_epoch": epoch + 1 if kind == "original_protocol" else None,
               "metrics": values, "seconds": seconds, "at": now()}
        evaluations.append(row)
        event("evaluation", row)
        totals["evaluation_seconds"] += seconds
        atomic_json(run / "evaluations.json", evaluations)
        print("EVAL=" + json.dumps(row), flush=True)
        state["phase"] = "training"

    stats = {}
    original_sampler, original_stage, original_step = utils.UniformSample_original, bpr.stageOne, bpr.opt.step

    def measured_sampler(*args, **kwargs):
        begin = time.perf_counter()
        samples = original_sampler(*args, **kwargs)
        stats["sampling_seconds"] = time.perf_counter() - begin
        stats["sampled_triplets"] = len(samples)
        return samples

    def checked_step(*args, **kwargs):
        nonlocal updates
        begin = time.perf_counter()
        if not all_finite(torch, [p.grad for p in recmodel.parameters() if p.grad is not None]):
            raise FloatingPointError("Non-finite gradient before Adam update")
        stats["numeric_checks_seconds"] += time.perf_counter() - begin
        value = original_step(*args, **kwargs)
        updates += 1
        return value

    def measured_stage(users, positives, negatives):
        begin = time.perf_counter()
        value = original_stage(users, positives, negatives)
        stats["stage_seconds"] += time.perf_counter() - begin
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite BPR loss")
        checking = time.perf_counter()
        if not all_finite(torch, list(recmodel.parameters())):
            raise FloatingPointError("Non-finite parameter after Adam update")
        stats["numeric_checks_seconds"] += time.perf_counter() - checking
        stats["losses"].append(value)
        stats["batch_sizes"].append(len(users))
        return value

    utils.UniformSample_original, bpr.stageOne, bpr.opt.step = measured_sampler, measured_stage, checked_step
    try:
        if resume:
            # Restore last, after all reconstruction, writer and log setup.
            random.setstate(payload["rng"]["python"])
            np.random.set_state(payload["rng"]["numpy"])
            torch.set_rng_state(payload["rng"]["torch_cpu"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["rng"]["torch_cuda"]])
        if not resume:
            snapshot()  # This run's initial random model, optimizer and RNG, not loaded.
        for epoch in range(completed, EPOCHS):
            if epoch % 5 == 0:
                evaluate("original_protocol", epoch)
            stats = {"losses": [], "batch_sizes": [], "numeric_checks_seconds": 0.0, "stage_seconds": 0.0}
            previous_updates = updates
            torch.cuda.synchronize()
            begin = time.perf_counter()
            original_report = procedure.BPR_train_original(dataset, recmodel, bpr, epoch, neg_k=1, w=writer)
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - begin
            completed = epoch + 1
            if updates - previous_updates != 57 or len(stats["losses"]) != 57:
                raise RuntimeError("Unexpected number of original BPR updates")
            assert stats["sampled_triplets"] == sum(stats["batch_sizes"]) == 57786
            assert stats["batch_sizes"][-1] == 442
            assert updates == completed * 57
            adam_steps = [int(value["step"].item()) for value in bpr.opt.state.values() if "step" in value]
            assert adam_steps and all(step == updates for step in adam_steps)
            saving = time.perf_counter()
            atomic_torch_save(torch, recmodel.state_dict(), run / "checkpoints" / "colakg-lastfm-latest.pth.tar")
            save_seconds = time.perf_counter() - saving
            row = {"completed_epochs": completed, "updates_this_epoch": updates - previous_updates,
                   "cumulative_updates": updates, "mean_minibatch_loss": statistics.mean(stats["losses"]),
                   "sample_weighted_loss": sum(n * loss for n, loss in zip(stats["batch_sizes"], stats["losses"])) / 57786,
                   "sampled_triplets": stats["sampled_triplets"], "negative_sampler": "original_python",
                   "training_seconds": training_seconds, "sampling_seconds": stats["sampling_seconds"],
                   "stage_seconds": stats["stage_seconds"], "numeric_checks_seconds": stats["numeric_checks_seconds"],
                   "model_save_seconds": save_seconds, "original_report": original_report,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "at": now()}
            history.append(row)
            for name in totals:
                if name in row:
                    totals[name] += row[name]
            event("training", row)
            if completed % 10 == 0:
                writer.flush()
                snapshot()
                atomic_json(run / "training-history.json", history)
            state["latest_epoch"] = row
            if evaluations:
                state["latest_evaluation"] = evaluations[-1]
            status()
            print("PROGRESS epoch={}/{} updates={} loss={:.9f} train_s={:.3f} sample_s={:.3f} peak_MiB={:.2f}".format(
                completed, EPOCHS, updates, row["mean_minibatch_loss"], training_seconds,
                row["sampling_seconds"], row["peak_allocated_bytes"] / 2**20), flush=True)
        if not any(row["kind"] == "final_2000_epochs" for row in evaluations):
            evaluate("final_2000_epochs", EPOCHS)
        atomic_json(run / "training-history.json", history)
        snapshot()
        protocol = [row for row in evaluations if row["kind"] == "original_protocol"]
        assert len(protocol) == 400 and [row["completed_epochs"] for row in protocol] == list(range(0, 2000, 5))
        assert len(history) == 2000 and updates == 114000
        best = max(protocol, key=lambda row: row["metrics"]["ndcg"][1])
        totals["session_wall_seconds"] += time.perf_counter() - session_start
        result = {**state, "success": True, "completed_epochs": completed, "updates": updates,
                  "protocol_evaluation_count": len(protocol), "additional_final_evaluations": 1,
                  "best_observed_original_protocol": best, "final_2000_epochs": evaluations[-1],
                  "reference_ndcg20": REFERENCE_NDCG20,
                  "best_minus_reference": best["metrics"]["ndcg"][1] - REFERENCE_NDCG20,
                  "final_minus_reference": evaluations[-1]["metrics"]["ndcg"][1] - REFERENCE_NDCG20,
                  "test_selection_limitation": "Maximum over 400 repeatedly observed test evaluations; not validation selection or independent generalization evidence. Final2000 is fixed-budget evaluation on the same test set, not a new held-out dataset.",
                  "totals": totals, "finished_at": now(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "final_model": str(run / "checkpoints" / "colakg-lastfm-latest.pth.tar"),
                  "final_model_sha256": digest(run / "checkpoints" / "colakg-lastfm-latest.pth.tar"),
                  "model_source_modified": False}
        write_json(run / "result.json", result)
        state.update({"status": "completed", "phase": "completed", "success": True})
        status()
        print("RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        utils.UniformSample_original, bpr.stageOne, bpr.opt.step = original_sampler, original_stage, original_step
        writer.close()
        segment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Only a snapshot from this SAME run; identity must match")
    args = parser.parse_args()
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(PYTHON.resolve())):
        raise SystemExit("Use " + str(PYTHON))
    if git("branch", "--show-current") != "exp/colakg-improvement":
        raise SystemExit("Refusing another branch")
    before = tracked_hashes()
    main_before = git("rev-parse", "main")
    if args.resume:
        run = args.resume.resolve()
        if run.parent != OUTPUT.resolve() or not run.name.startswith("lastfm-baseline-"):
            raise SystemExit("Recovery requires a formal run directly under " + str(OUTPUT))
        previous = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            raise SystemExit("Completed run cannot be resumed")
        if previous.get("pid"):
            alive = subprocess.check_output(["tasklist", "/FI", "PID eq " + str(previous["pid"]), "/FO", "CSV", "/NH"], text=True)
            if '"' + str(previous["pid"]) + '"' in alive:
                raise SystemExit("Prior run process still exists; refusing concurrent recovery")
    else:
        run = OUTPUT / ("lastfm-baseline-" + datetime.now(CHINA).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "tracked-hashes-before.json", before)
    session_id = datetime.now(CHINA).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    state = {"run_directory": str(run), "session_id": session_id, "pid": os.getpid(),
             "started_at": now(), "status": "initializing", "completed_epochs": 0,
             "updates": 0, "success": False, "main_ref": main_before}
    atomic_json(run / "status.json", state)
    print("RUN_DIRECTORY=" + str(run), flush=True)
    failed = False
    with (run / ("console-" + session_id + ".log")).open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(run, state, args.resume is not None)
            except BaseException as error:
                failed = True
                state.update({"success": False, "status": "failed", "error_type": type(error).__name__,
                              "error": str(error), "traceback": traceback.format_exc(), "failed_at": now()})
                write_json(run / ("failure-" + session_id + ".json"), state)
                atomic_json(run / "status.json", state)
                traceback.print_exc()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    try:
                        (run / ("memory-failure-" + session_id + ".txt")).write_text(torch.cuda.memory_summary(), encoding="utf-8")
                    except Exception:
                        pass
            finally:
                after = tracked_hashes()
                integrity = {"session_id": session_id, "tracked_files_unchanged": before == after,
                             "main_ref_unchanged": main_before == git("rev-parse", "main"),
                             "branch_unchanged": git("branch", "--show-current") == "exp/colakg-improvement",
                             "git_head": git("rev-parse", "HEAD"), "checked_at": now()}
                write_json(run / ("integrity-" + session_id + ".json"), integrity)
                if not all(integrity[key] for key in ["tracked_files_unchanged", "main_ref_unchanged", "branch_unchanged"]):
                    failed = True
                    state.update({"success": False, "status": "failed", "error": "Project integrity changed during training"})
                    atomic_json(run / "status.json", state)
                    if (run / "result.json").exists():
                        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                        result.update({"success": False, "integrity_error": integrity})
                        write_json(run / "result.json", result)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
