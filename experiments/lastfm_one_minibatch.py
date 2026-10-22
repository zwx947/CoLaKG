"""Run exactly one original CoLaKG/LastFM BPR minibatch on CUDA.

No main.py, evaluation, checkpoint loading/saving, or model-source changes.
All generated outputs go into a new directory under D:/CoLaKGExperiments.
"""

import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import time
import traceback
import types
import uuid
from datetime import datetime, timedelta, timezone


ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "rec_code"
PYTHON = Path(r"D:\CondaEnvs\CoLaKG\python.exe")
OUTPUT = Path(r"D:\CoLaKGExperiments")
CHINA = timezone(timedelta(hours=8))


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tracked_hashes():
    names = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"]
    ).decode("utf-8").split("\0")
    return {name: digest(ROOT / name) for name in names if name}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def original_arguments():
    """Read shell assignments/CLI without executing the shell script."""
    assignments, command = {}, None
    source = REC / "train_lastfm.sh"
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "python main.py " in line:
            command = shlex.split(line)
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)=(.*)", line)
        if not match:
            raise ValueError("Unrecognized original shell configuration: " + line)
        tokens = shlex.split(match.group(2))
        if len(tokens) != 1:
            raise ValueError("Expected one literal shell assignment")
        assignments[match.group(1)] = tokens[0]
    if command is None or command[:3] != ["CUDA_VISIBLE_DEVICES=0", "python", "main.py"]:
        raise ValueError("Original LastFM launch command changed; review before running")

    def substitute(match):
        return assignments[match.group(1) or match.group(2)]

    arguments = [
        re.sub(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)", substitute, token)
        for token in command[3:]
    ]
    return assignments, arguments


def bootstrap_world(arguments, run):
    """Execute unchanged world.py; only relocate its __file__-derived outputs.

    world.py otherwise creates ROOT/code/checkpoints during import. Its logical
    __file__ lives under this run, while compilation/tracebacks reference the
    real source. The real Loader input path is passed explicitly later.
    """
    module = types.ModuleType("world")
    module.__file__ = str(run / "runtime" / "rec_code" / "world.py")
    module.__package__ = ""
    sys.modules["world"] = module
    previous = sys.argv
    try:
        sys.argv = [str(REC / "main.py"), *arguments]
        exec(compile((REC / "world.py").read_text(encoding="utf-8"),
                     str(REC / "world.py"), "exec"), module.__dict__)
    finally:
        sys.argv = previous
    return module


@contextlib.contextmanager
def isolated_graph_cache(sparse, input_directory, run):
    """Redirect only Loader's graph-cache I/O; retain all original graph math."""
    original_load, original_save = sparse.load_npz, sparse.save_npz
    original_path = (input_directory / "s_pre_adj_mat.npz").resolve()
    destination = run / "cache" / "s_pre_adj_mat.npz"
    destination.parent.mkdir()

    def mapped(path):
        return destination if Path(path).resolve() == original_path else path

    def load(path, *args, **kwargs):
        return original_load(mapped(path), *args, **kwargs)

    def save(path, *args, **kwargs):
        return original_save(mapped(path), *args, **kwargs)

    sparse.load_npz, sparse.save_npz = load, save
    try:
        yield destination
    finally:
        sparse.load_npz, sparse.save_npz = original_load, original_save


def experiment(run, result):
    started = time.perf_counter()
    assignments, arguments = original_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # same as train_lastfm.sh
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REC))
    result["phase"] = "imports_and_configuration"
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    world = bootstrap_world(arguments, run)
    # Import order matches main.py and avoids the project's circular import trap.
    if importlib.util.find_spec("cppimport") is not None:
        raise RuntimeError("Optional cppimport is present; refusing implicit compilation")
    utils = importlib.import_module("utils")
    dataloader = importlib.import_module("dataloader")
    model = importlib.import_module("model")
    result.update({
        "original_shell_assignments": assignments, "original_cli_arguments": arguments,
        "world_config": dict(world.config), "seed": world.seed,
        "original_effective_epochs_not_executed": world.TRAIN_epochs,
        "original_tensorboard_flag_writer_not_created": world.tensorboard,
        "load_checkpoint_flag": world.LOAD,
        "sampler": "original Python fallback" if not utils.sample_ext else "C++",
        "runtime_world_logical_file": world.__file__,
        "versions": {name: importlib.metadata.version(name) for name in [
            "torch", "numpy", "pandas", "scipy", "scikit-learn", "tensorboardX", "tqdm"]},
        "python": sys.version, "executable": sys.executable,
    })
    assert world.dataset == "lastfm" and world.model_name == "colakg"
    assert world.LOAD == 0 and world.config["pretrain"] == 0
    assert not utils.sample_ext
    assert torch.cuda.is_available() and world.device.type == "cuda"
    assert world.config["bpr_batch_size"] == 1024
    assert world.config["lightGCN_n_layers"] == 3 and world.config["neighbor_k"] == 10
    utils.set_seed(world.seed)
    random.seed(world.seed)
    result["seed_policy"] = "Original utils.set_seed plus Python random.seed; no deterministic-kernel overrides"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    result["gpu"] = {"name": props.name, "runtime": torch.version.cuda,
                     "total_bytes": total, "free_bytes_before_model": free}
    assert "MX450" in props.name

    # Only the two semantic input tensors may be read through torch.load.
    item_path = (REC / world.item_semantic_emb_file).resolve()
    user_path = (REC / world.user_semantic_emb_file).resolve()
    allowed = {item_path, user_path}
    original_load, original_save = torch.load, torch.save
    semantic_reads = []

    def load_semantic(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved not in allowed:
            raise RuntimeError("Forbidden checkpoint/non-semantic torch.load: " + str(resolved))
        semantic_reads.append(str(resolved))
        return original_load(path, *args, **kwargs)

    def forbid_save(*args, **kwargs):
        raise RuntimeError("Checkpoint saving is forbidden for this experiment")

    torch.load, torch.save = load_semantic, forbid_save
    try:
        result["phase"] = "data_and_semantic_inputs"
        print("PHASE data_and_semantic_inputs", flush=True)
        data_path = ROOT / "data" / "lastfm"
        dataset = dataloader.Loader(config=world.config, path=str(data_path))
        item_semantic = torch.load(item_path, map_location="cpu", weights_only=True)
        user_semantic = torch.load(user_path, map_location="cpu", weights_only=True)
        assert list(item_semantic.shape) == [dataset.m_items, 1024]
        assert list(user_semantic.shape) == [dataset.n_users, 1024]
        assert bool(torch.isfinite(item_semantic).all() and torch.isfinite(user_semantic).all())
        result["data"] = {"users": dataset.n_users, "items": dataset.m_items,
                          "training_interactions": dataset.trainDataSize,
                          "item_embedding_shape": list(item_semantic.shape),
                          "user_embedding_shape": list(user_semantic.shape),
                          "test_usage": "Loader reads test.txt for dimensions/indexing only; no evaluation/selection"}
        # Exactly the same cosine/sort/neighbour construction as main.py.
        cosine = cosine_similarity(item_semantic.numpy())
        indices = np.argsort(-cosine, axis=1)[:, 1:world.config["neighbor_k"] + 1]
        neighbours = torch.tensor(indices).long()
        del cosine, indices
        result["phase"] = "model_initialization"
        print("PHASE model_initialization", flush=True)
        with isolated_graph_cache(dataloader.sp, data_path, run) as cache:
            recmodel = model.CoLaKG(world.config, dataset, neighbours,
                                    item_semantic, user_semantic).to(world.device)
        bpr = utils.BPRLoss(recmodel, world.config)
        assert isinstance(bpr.opt, torch.optim.Adam)
        recmodel.train()
        torch.cuda.synchronize()
        result["model_initialized"] = True
        result["graph_cache"] = str(cache)
        result["parameter_count"] = sum(p.numel() for p in recmodel.parameters())
        result["optimizer"] = {"class": type(bpr.opt).__name__,
                               "lr": bpr.lr, "loss_regularization": bpr.weight_decay,
                               "defaults": dict(bpr.opt.defaults)}
        result["initialization_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["initialization_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        assert all(bool(torch.isfinite(p).all()) for p in recmodel.parameters())

        result["phase"] = "original_negative_sampling"
        sample_start = time.perf_counter()
        samples = utils.UniformSample_original(dataset)
        users = torch.Tensor(samples[:, 0]).long().to(world.device)
        positives = torch.Tensor(samples[:, 1]).long().to(world.device)
        negatives = torch.Tensor(samples[:, 2]).long().to(world.device)
        users, positives, negatives = utils.shuffle(users, positives, negatives)
        batch = next(utils.minibatch(users, positives, negatives,
                                    batch_size=world.config["bpr_batch_size"]))
        assert len(batch[0]) == 1024
        result["sampling_seconds"] = time.perf_counter() - sample_start
        result["sampled_triplets"] = len(samples)
        result["batch_size"] = len(batch[0])
        result["batch_sha256"] = hashlib.sha256(
            torch.stack(batch, dim=1).cpu().numpy().tobytes()).hexdigest()
        before = {name: p.detach().cpu().clone() for name, p in recmodel.named_parameters()}
        timings, captured, gradients = {}, {}, {}
        real_loss, real_step = recmodel.bpr_loss, bpr.opt.step
        forward_return_time = [None]

        def observed_loss(*args):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, regularizer = real_loss(*args)
            torch.cuda.synchronize()
            timings["forward_loss_seconds"] = time.perf_counter() - t0
            captured["bpr"] = loss.detach()
            captured["regularizer"] = regularizer.detach()
            assert bool(torch.isfinite(loss) and torch.isfinite(regularizer))
            forward_return_time[0] = time.perf_counter()
            return loss, regularizer

        def observed_step(*args, **kwargs):
            torch.cuda.synchronize()
            timings["backward_interval_seconds"] = time.perf_counter() - forward_return_time[0]
            checking = time.perf_counter()
            for name, parameter in recmodel.named_parameters():
                if parameter.grad is None:
                    gradients[name] = {"present": False}
                    continue
                values = parameter.grad.detach()
                finite = bool(torch.isfinite(values).all())
                gradients[name] = {"present": True, "finite": finite,
                                   "l2_norm": float(values.norm().item()),
                                   "max_abs": float(values.abs().max().item())}
                assert finite, "Non-finite gradient: " + name
            timings["gradient_checks_seconds"] = time.perf_counter() - checking
            update_start = time.perf_counter()
            returned = real_step(*args, **kwargs)
            torch.cuda.synchronize()
            timings["adam_update_seconds"] = time.perf_counter() - update_start
            captured["optimizer_calls"] = captured.get("optimizer_calls", 0) + 1
            return returned

        recmodel.bpr_loss, bpr.opt.step = observed_loss, observed_step
        result["phase"] = "one_original_bpr_minibatch"
        print("PHASE one_original_bpr_minibatch batch=1024 device=cuda:0", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        try:
            # Original forward, regularization, zero_grad, backward, and Adam step.
            total_loss = bpr.stageOne(*batch)
            torch.cuda.synchronize()
            timings["step_wall_seconds_including_gradient_checks"] = time.perf_counter() - step_start
        finally:
            recmodel.bpr_loss, bpr.opt.step = real_loss, real_step
        result["training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["training_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["overall_peak_allocated_bytes"] = max(result["initialization_peak_allocated_bytes"], result["training_peak_allocated_bytes"])
        result["overall_peak_reserved_bytes"] = max(result["initialization_peak_reserved_bytes"], result["training_peak_reserved_bytes"])
        free, total = torch.cuda.mem_get_info()
        result["gpu"]["free_bytes_after_step"] = free
        parameter_checks = {}
        for name, parameter in recmodel.named_parameters():
            current = parameter.detach().cpu()
            delta = current - before[name]
            parameter_checks[name] = {
                "finite": bool(torch.isfinite(current).all()),
                "changed": not torch.equal(current, before[name]),
                "update_l2_norm": float(delta.norm().item()),
                "update_max_abs": float(delta.abs().max().item()),
            }
        optimizer_steps = {
            name: float(bpr.opt.state[p]["step"].item())
            for name, p in recmodel.named_parameters() if "step" in bpr.opt.state.get(p, {})
        }
        result["loss"] = {"bpr": float(captured["bpr"].item()),
                          "regularizer_raw": float(captured["regularizer"].item()),
                          "regularizer_weighted": float(captured["regularizer"].item()) * bpr.weight_decay,
                          "total": total_loss}
        result["timings"] = timings
        result["gradients"] = gradients
        result["parameters_after_step"] = parameter_checks
        result["optimizer_steps"] = optimizer_steps
        result["optimizer_update_calls"] = captured["optimizer_calls"]
        result["parameters_updated"] = [name for name, check in parameter_checks.items() if check["changed"]]
        result["parameters_without_gradient"] = [name for name, check in gradients.items() if not check["present"]]
        assert math.isfinite(total_loss)
        assert all(check["finite"] for check in parameter_checks.values())
        assert result["parameters_updated"] and captured["optimizer_calls"] == 1
        assert optimizer_steps and all(step == 1 for step in optimizer_steps.values())
        result["semantic_tensor_reads"] = semantic_reads
        assert len(semantic_reads) == 2
        result["phase"] = "completed"
        result["success"] = True
    finally:
        torch.load, torch.save = original_load, original_save
        result["elapsed_experiment_seconds"] = time.perf_counter() - started


def main():
    if len(sys.argv) != 1:
        raise SystemExit("No hyperparameter overrides are accepted; read train_lastfm.sh")
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(PYTHON.resolve())):
        raise SystemExit("Use the dedicated absolute Python path: " + str(PYTHON))
    if git("branch", "--show-current") != "exp/colakg-improvement":
        raise SystemExit("Refusing to run outside exp/colakg-improvement")
    run = OUTPUT / ("lastfm-one-minibatch-" + datetime.now(CHINA).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_DIRECTORY=" + str(run), flush=True)
    result = {"success": False, "run_directory": str(run), "started_at": datetime.now(CHINA).isoformat(),
              "script_sha256": digest(Path(__file__)), "git_head": git("rev-parse", "HEAD"),
              "main_ref_before": git("rev-parse", "main"), "git_status_before": git("status", "--short", "--branch"),
              "checkpoint_loading": False, "checkpoint_saving": False,
              "evaluation_executed": False, "real_minibatches_requested": 1}
    before = tracked_hashes()
    write_json(run / "tracked-hashes-before.json", before)
    with (run / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                experiment(run, result)
            except Exception as error:
                result["success"] = False
                result["error_type"] = type(error).__name__
                result["error"] = str(error)
                result["traceback"] = traceback.format_exc()
                result["cuda_oom"] = "out of memory" in str(error).lower()
                (run / "error.log").write_text(result["traceback"], encoding="utf-8")
                traceback.print_exc()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    try:
                        result["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                        result["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                        (run / "cuda-memory-summary.txt").write_text(torch.cuda.memory_summary(), encoding="utf-8")
                    except Exception:
                        pass
            finally:
                after = tracked_hashes()
                result["original_tracked_files_unchanged"] = before == after
                result["main_ref_after"] = git("rev-parse", "main")
                result["main_ref_unchanged"] = result["main_ref_before"] == result["main_ref_after"]
                result["checkpoint_files_created"] = [str(p.relative_to(run)) for p in run.rglob("*")
                                                       if p.is_file() and p.name.endswith((".pt", ".pth", ".pth.tar", ".ckpt"))]
                if not result["original_tracked_files_unchanged"] or not result["main_ref_unchanged"] or result["checkpoint_files_created"]:
                    result["success"] = False
                    result["integrity_error"] = "Protected inputs/refs changed or a checkpoint was created"
                result["finished_at"] = datetime.now(CHINA).isoformat()
                write_json(run / "tracked-hashes-after.json", after)
                write_json(run / "result.json", result)
                print("RESULT=" + json.dumps({key: result.get(key) for key in [
                    "success", "phase", "loss", "timings", "overall_peak_allocated_bytes",
                    "overall_peak_reserved_bytes", "parameters_updated", "error"]}, ensure_ascii=False), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
