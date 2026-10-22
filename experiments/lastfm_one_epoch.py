"""One original LastFM BPR_train_original epoch, one post-training Test, one save.

Outputs are isolated on D:. No checkpoint is loaded, no model code is changed,
and this entry point has no option for launching a 2000-epoch run.
"""

import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
import uuid
from datetime import datetime

from lastfm_one_minibatch import (
    CHINA, OUTPUT, PYTHON, REC, ROOT, Tee, bootstrap_world, digest, git,
    isolated_graph_cache, original_arguments, tracked_hashes, write_json,
)


def memory(torch):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "device_free_bytes": free, "device_total_bytes": total,
    }


def execute(run, result):
    started = time.perf_counter()
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
        raise RuntimeError("Refusing implicit C++ compilation; Python fallback required")
    utils = importlib.import_module("utils")
    dataloader = importlib.import_module("dataloader")
    model = importlib.import_module("model")
    procedure = importlib.import_module("Procedure")
    assert world.dataset == "lastfm" and world.model_name == "colakg"
    assert world.LOAD == 0 and world.config["pretrain"] == 0
    assert not utils.sample_ext and world.tensorboard == 1
    assert world.config["bpr_batch_size"] == 1024
    assert world.config["lightGCN_n_layers"] == 3 and world.config["neighbor_k"] == 10
    assert torch.cuda.is_available() and world.device.type == "cuda"
    utils.set_seed(world.seed)
    random.seed(world.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result.update({
        "config": dict(world.config), "seed": world.seed, "topks": list(world.topks),
        "original_shell_assignments": assignments, "original_cli_arguments": arguments,
        "original_effective_epochs": world.TRAIN_epochs, "executed_epochs": 1,
        "python": sys.version, "executable": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in [
            "torch", "numpy", "pandas", "scipy", "scikit-learn", "tensorboardX", "tqdm"]},
        "gpu": {"name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda},
        "negative_sampler": "original utils.UniformSample_original_python",
        "protocol_difference": "One evaluation AFTER one training epoch; original main.py evaluates BEFORE epochs 0,5,...,1995. No tuning or model selection.",
        "training_updates": 0, "test_calls": 0, "checkpoint_saves": 0,
    })
    assert "MX450" in result["gpu"]["name"]
    print("CONFIG=" + json.dumps(result["config"]), flush=True)
    data_path = ROOT / "data" / "lastfm"
    semantic_paths = [(REC / world.item_semantic_emb_file).resolve(),
                      (REC / world.user_semantic_emb_file).resolve()]
    real_load, real_save = torch.load, torch.save
    loaded = []
    writer = None

    def semantic_only(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved not in semantic_paths:
            raise RuntimeError("Checkpoint loading forbidden: " + str(resolved))
        loaded.append(str(resolved))
        return real_load(path, *args, **kwargs)

    def exclusive_checkpoint(state, path, *args, **kwargs):
        destination = Path(path).resolve()
        if run.resolve() not in destination.parents or destination.exists():
            raise RuntimeError("Refusing non-isolated checkpoint or overwrite")
        result["checkpoint_saves"] += 1
        assert result["checkpoint_saves"] == 1
        return real_save(state, path, *args, **kwargs)

    torch.load, torch.save = semantic_only, exclusive_checkpoint
    try:
        result["phase"] = "initialization"
        dataset = dataloader.Loader(config=world.config, path=str(data_path))
        item_emb = torch.load(semantic_paths[0], map_location="cpu", weights_only=True)
        user_emb = torch.load(semantic_paths[1], map_location="cpu", weights_only=True)
        assert list(item_emb.shape) == [dataset.m_items, 1024]
        assert list(user_emb.shape) == [dataset.n_users, 1024]
        similarity = cosine_similarity(item_emb.numpy())
        indices = np.argsort(-similarity, axis=1)[:, 1:world.config["neighbor_k"] + 1]
        neighbours = torch.tensor(indices).long()
        del similarity, indices
        with isolated_graph_cache(dataloader.sp, data_path, run) as cache:
            recmodel = model.CoLaKG(world.config, dataset, neighbours, item_emb, user_emb).to(world.device)
        bpr = utils.BPRLoss(recmodel, world.config)
        writer = SummaryWriter(str(run / "tensorboard"))
        result["data"] = {
            "users": dataset.n_users, "items": dataset.m_items,
            "train_interactions": dataset.trainDataSize, "test_interactions": dataset.testDataSize,
            "graph_cache": str(cache), "parameter_count": sum(p.numel() for p in recmodel.parameters()),
        }
        result["optimizer"] = {"class": type(bpr.opt).__name__, "defaults": dict(bpr.opt.defaults),
                               "loss_regularization": bpr.weight_decay}
        result["memory_initialization"] = memory(torch)
        result["preparation_seconds"] = time.perf_counter() - started
        losses, batch_sizes, batch_seconds, optimizer_seconds, check_seconds = [], [], [], [], []
        original_sampler, original_stage, original_step = utils.UniformSample_original, bpr.stageOne, bpr.opt.step

        def measured_sampler(*args, **kwargs):
            t0 = time.perf_counter()
            samples = original_sampler(*args, **kwargs)
            result["sampling_seconds"] = time.perf_counter() - t0
            result["sampled_triplets"] = len(samples)
            return samples

        def measured_optimizer_step(*args, **kwargs):
            checking = time.perf_counter()
            for name, parameter in recmodel.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Non-finite gradient: " + name)
            check_seconds.append(time.perf_counter() - checking)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            returned = original_step(*args, **kwargs)
            torch.cuda.synchronize()
            optimizer_seconds.append(time.perf_counter() - t0)
            result["training_updates"] += 1
            return returned

        def measured_stage(users, positives, negatives):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = original_stage(users, positives, negatives)
            torch.cuda.synchronize()
            batch_seconds.append(time.perf_counter() - t0)
            assert math.isfinite(loss), "Non-finite training loss"
            losses.append(loss)
            batch_sizes.append(len(users))
            if len(losses) == 1 or len(losses) % 10 == 0:
                print("UPDATE={} batch={} loss={:.9f}".format(len(losses), len(users), loss), flush=True)
            return loss

        utils.UniformSample_original, bpr.stageOne, bpr.opt.step = measured_sampler, measured_stage, measured_optimizer_step
        result["phase"] = "original_full_epoch_training"
        print("PHASE original Procedure.BPR_train_original: one full epoch", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        training_start = time.perf_counter()
        try:
            result["original_training_report"] = procedure.BPR_train_original(dataset, recmodel, bpr, 0, neg_k=1, w=writer)
            torch.cuda.synchronize()
            result["epoch_training_seconds"] = time.perf_counter() - training_start
        finally:
            utils.UniformSample_original, bpr.stageOne, bpr.opt.step = original_sampler, original_stage, original_step
        result["memory_training"] = memory(torch)
        expected = math.ceil(result["sampled_triplets"] / world.config["bpr_batch_size"])
        assert result["training_updates"] == expected == len(losses) == 57
        assert sum(batch_sizes) == result["sampled_triplets"] == dataset.trainDataSize
        assert batch_sizes[-1] == 442
        assert all(bool(torch.isfinite(p).all()) for p in recmodel.parameters())
        steps = [float(state["step"].item()) for state in bpr.opt.state.values() if "step" in state]
        assert steps and all(step == 57 for step in steps)
        result.update({
            "batch_sizes": batch_sizes, "batch_losses": losses, "batch_seconds": batch_seconds,
            "average_minibatch_loss": statistics.mean(losses),
            "sample_weighted_loss": sum(n * loss for n, loss in zip(batch_sizes, losses)) / sum(batch_sizes),
            "optimizer_step_values": steps, "optimizer_seconds": sum(optimizer_seconds),
            "batch_training_seconds": sum(batch_seconds), "gradient_checks_seconds": sum(check_seconds),
            "transfer_shuffle_logging_and_loop_seconds": max(0, result["epoch_training_seconds"] - result["sampling_seconds"] - sum(batch_seconds)),
        })

        result["phase"] = "one_post_training_original_test"
        print("PHASE original Procedure.Test: once after training", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        evaluation_start = time.perf_counter()
        metrics = procedure.Test(dataset, recmodel, 0, w=writer, multicore=world.config["multicore"])
        torch.cuda.synchronize()
        result["evaluation_seconds"] = time.perf_counter() - evaluation_start
        result["test_calls"] = 1
        result["memory_evaluation"] = memory(torch)
        result["evaluation_user_batches"] = math.ceil(len(dataset.testDict) / world.config["test_u_batch_size"])
        result["metrics"] = {name: values.tolist() for name, values in metrics.items()}
        assert all(math.isfinite(value) and 0 <= value <= 1 for values in result["metrics"].values() for value in values)
        writer.flush()
        writer.close()
        writer = None

        result["phase"] = "one_isolated_checkpoint_save_benchmark"
        destination = run / "checkpoints" / "colakg-lastfm-epoch1.pth.tar"
        destination.parent.mkdir()
        torch.cuda.synchronize()
        save_start = time.perf_counter()
        torch.save(recmodel.state_dict(), destination)
        result["checkpoint_save_seconds"] = time.perf_counter() - save_start
        result["checkpoint"] = {"path": str(destination), "bytes": destination.stat().st_size,
                                "sha256": digest(destination), "contents": "model state_dict only, same as original main.py"}
        result["semantic_tensor_reads"] = loaded
        assert len(loaded) == 2
        result["overall_peak_allocated_bytes"] = max(result[phase]["peak_allocated_bytes"] for phase in ["memory_initialization", "memory_training", "memory_evaluation"])
        result["overall_peak_reserved_bytes"] = max(result[phase]["peak_reserved_bytes"] for phase in ["memory_initialization", "memory_training", "memory_evaluation"])
        median_warm = statistics.median(batch_seconds[1:])
        cold_extra = max(0, batch_seconds[0] - median_warm)
        warm_epoch = result["epoch_training_seconds"] - cold_extra
        train_estimate = result["epoch_training_seconds"] + 1999 * warm_epoch
        eval_estimate, save_estimate = 400 * result["evaluation_seconds"], 2000 * result["checkpoint_save_seconds"]
        result["cost_estimate_2000_epochs"] = {
            "basis": "Single observed epoch; first-batch cold overhead removed once. Linear scenarios, not a confidence interval.",
            "original_evaluation_count": 400, "updates": 2000 * 57,
            "median_warm_batch_seconds": median_warm, "estimated_first_batch_cold_extra_seconds": cold_extra,
            "estimated_warm_epoch_seconds": warm_epoch,
            "training_seconds": train_estimate, "sampling_seconds_within_training": 2000 * result["sampling_seconds"],
            "evaluation_seconds": eval_estimate, "checkpoint_saving_seconds": save_estimate,
            "total_seconds": result["preparation_seconds"] + train_estimate + eval_estimate + save_estimate,
            "cold_epoch_repeated_total_seconds": result["preparation_seconds"] + 2000 * result["epoch_training_seconds"] + eval_estimate + save_estimate,
            "checkpoint_cumulative_write_bytes": 2000 * result["checkpoint"]["bytes"],
            "checkpoint_retained_bytes_original_overwrite_policy": result["checkpoint"]["bytes"],
            "limitations": "Windows desktop/background GPU load, temperature and frequency, later-epoch variability, and filesystem cache affect actual duration. Gradient checking overhead remains in this estimate."
        }
        result["phase"], result["success"] = "completed", True
    finally:
        torch.load, torch.save = real_load, real_save
        if writer is not None:
            writer.close()
        result["experiment_wall_seconds"] = time.perf_counter() - started


def main():
    if len(sys.argv) != 1:
        raise SystemExit("No epoch/hyperparameter overrides accepted; this script runs exactly one epoch")
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(PYTHON.resolve())):
        raise SystemExit("Use " + str(PYTHON))
    if git("branch", "--show-current") != "exp/colakg-improvement":
        raise SystemExit("Refusing to run on another branch")
    run = OUTPUT / ("lastfm-one-epoch-" + datetime.now(CHINA).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_DIRECTORY=" + str(run), flush=True)
    before = tracked_hashes()
    result = {"success": False, "run_directory": str(run), "git_head": git("rev-parse", "HEAD"),
              "main_ref_before": git("rev-parse", "main"), "started_at": datetime.now(CHINA).isoformat(),
              "script_sha256": digest(Path(__file__)), "helper_sha256": digest(Path(__file__).with_name("lastfm_one_minibatch.py")),
              "checkpoint_loading": False, "checkpoint_saving_purpose": "one new isolated state_dict save for timing only"}
    write_json(run / "tracked-hashes-before.json", before)
    with (run / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(run, result)
            except Exception as error:
                result.update({"success": False, "error_type": type(error).__name__, "error": str(error),
                               "traceback": traceback.format_exc(), "cuda_oom": "out of memory" in str(error).lower()})
                (run / "error.log").write_text(result["traceback"], encoding="utf-8")
                traceback.print_exc()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    try:
                        result["failure_memory"] = memory(torch)
                        (run / "cuda-memory-summary.txt").write_text(torch.cuda.memory_summary(), encoding="utf-8")
                    except Exception:
                        pass
            finally:
                after = tracked_hashes()
                result["original_tracked_files_unchanged"] = before == after
                result["main_ref_after"] = git("rev-parse", "main")
                result["main_ref_unchanged"] = result["main_ref_before"] == result["main_ref_after"]
                if not result["original_tracked_files_unchanged"] or not result["main_ref_unchanged"]:
                    result["success"] = False
                    result["integrity_error"] = "Original tracked files or main ref changed"
                result["finished_at"] = datetime.now(CHINA).isoformat()
                write_json(run / "tracked-hashes-after.json", after)
                write_json(run / "result.json", result)
                print("RESULT=" + json.dumps({key: result.get(key) for key in [
                    "success", "training_updates", "average_minibatch_loss", "metrics", "epoch_training_seconds",
                    "sampling_seconds", "evaluation_seconds", "checkpoint_save_seconds", "overall_peak_allocated_bytes",
                    "overall_peak_reserved_bytes", "experiment_wall_seconds", "error"]}), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
