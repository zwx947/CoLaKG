"""Validation-first LastFM research; fresh parameters, no model warm starts."""
import argparse
import contextlib
import importlib
import importlib.metadata
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

from lastfm_one_minibatch import ROOT, REC, PYTHON, CHINA, Tee, git, digest, original_arguments, bootstrap_world
from lastfm_windows import replace_with_retry

RESEARCH = Path(r"D:\CoLaKGResearch")
PROTOCOL = RESEARCH / "protocol-v1"


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def prepare_validation(np):
    source = ROOT / "data" / "lastfm" / "train.txt"
    if PROTOCOL.exists():
        record = json.loads((PROTOCOL / "manifest.json").read_text(encoding="utf-8"))
        assert record["original_train_sha256"] == digest(source)
        for name, value in record["derived_sha256"].items():
            assert digest(PROTOCOL / name) == value
        return record
    PROTOCOL.mkdir(parents=True, exist_ok=False)
    rows = [list(map(int, line.split())) for line in source.read_text().splitlines() if line.strip()]
    degree = np.zeros(2813, dtype=np.int64)
    for row in rows:
        degree[row[1:]] += 1
    rng = np.random.RandomState(314159)
    fits, vals = [], []
    for row in rows:
        user, items = row[0], row[1:]
        target = min(len(items) - 1, max(1, len(items) // 5))
        chosen = set()
        for index in rng.permutation(len(items)):
            if len(chosen) >= target:
                break
            if degree[items[index]] > 1:
                chosen.add(int(index))
                degree[items[index]] -= 1
        fit = [value for index, value in enumerate(items) if index not in chosen]
        val = [value for index, value in enumerate(items) if index in chosen]
        assert fit and val
        fits.append([user, *fit]); vals.append([user, *val])
    for name, records in [("train.txt", fits), ("test.txt", vals)]:
        (PROTOCOL / name).write_text("\n".join(" ".join(map(str, row)) for row in records) + "\n", encoding="utf-8")
    record = {"split_seed": 314159, "original_train_sha256": digest(source),
              "derived_sha256": {name: digest(PROTOCOL / name) for name in ["train.txt", "test.txt"]},
              "fit_pairs": sum(len(row)-1 for row in fits), "validation_pairs": sum(len(row)-1 for row in vals),
              "item_coverage": bool((degree > 0).all()),
              "test_txt_meaning": "Validation derived only from original train; NOT original test",
              "semantic_asset_limitation": "Fixed released user semantics may include held-out development history; same assets for all methods"}
    assert record["fit_pairs"] + record["validation_pairs"] == 57786
    atomic_json(PROTOCOL / "manifest.json", record)
    return record


def run_experiment(args, run, state):
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    assignment, cli = original_arguments()
    sys.path.insert(0, str(REC))
    world = bootstrap_world(cli, run)
    world.tensorboard = 0
    utils = importlib.import_module("utils")
    dataloader = importlib.import_module("dataloader")
    original_model = importlib.import_module("model")
    procedure = importlib.import_module("Procedure")
    assert torch.cuda.is_available() and not utils.sample_ext
    protocol = prepare_validation(np)
    if args.phase == "development":
        data_path = PROTOCOL
    else:
        if not args.frozen_config:
            raise RuntimeError("Final testing requires an independently frozen config file")
        frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
        assert frozen["method"] == args.method and frozen["epochs"] == args.epochs
        assert frozen["selection_used_original_test"] is False
        data_path = ROOT / "data" / "lastfm"
    utils.set_seed(args.seed); random.seed(args.seed)
    # Loader writes graph caches only in this run, never into input data.
    dataset = dataloader.Loader(config=world.config, path=str(data_path))
    original_npz_load, original_npz_save = dataloader.sp.load_npz, dataloader.sp.save_npz
    def graph_path(path):
        return run / "graph-cache.npz" if Path(path).name == "s_pre_adj_mat.npz" else path
    dataloader.sp.load_npz = lambda path,*a,**kw: original_npz_load(graph_path(path),*a,**kw)
    dataloader.sp.save_npz = lambda path,*a,**kw: original_npz_save(graph_path(path),*a,**kw)
    semantic_paths = [ROOT/"data/lastfm/lastfm_embeddings_simcse_kg.pt", ROOT/"data/lastfm/lastfm_embeddings_simcse_kg_user.pt"]
    item, user = [torch.load(path, map_location="cpu", weights_only=True) for path in semantic_paths]
    similarity = cosine_similarity(item.numpy())
    neighbours = torch.tensor(np.argsort(-similarity,axis=1)[:, 1:11]).long()
    if args.method == "baseline":
        constructor = original_model.CoLaKG
    else:
        module = importlib.import_module("research_methods." + args.method)
        neighbours = module.prepare_neighbours(dataset, item, neighbours)
        constructor = module.ResearchModel
    model = constructor(world.config, dataset, neighbours, item, user).to(world.device)
    dataloader.sp.load_npz, dataloader.sp.save_npz = original_npz_load, original_npz_save
    bpr = utils.BPRLoss(model, world.config)
    source_paths = [*REC.glob("*.py"), REC/"train_lastfm.sh", Path(__file__), Path(__file__).with_name("lastfm_one_minibatch.py")]
    if args.method != "baseline":
        source_paths.append(Path(module.__file__))
    # Python 3.8 can retain a relative __file__ when launched by relative path.
    source_paths = [path.resolve() for path in source_paths]
    state.update({"config": dict(world.config), "seed":args.seed,"epochs":args.epochs,
                  "method":args.method,"phase":args.phase,"protocol":protocol,
                  "git_commit":git("rev-parse","HEAD"),"git_branch":git("branch","--show-current"),
                  "source_sha256":{str(p.relative_to(ROOT)):digest(p) for p in source_paths},
                  "input_sha256":{str(p):digest(p) for p in [data_path/"train.txt",data_path/"test.txt",*semantic_paths]},
                  "versions":{p:importlib.metadata.version(p) for p in ["torch","numpy","scipy","scikit-learn"]},
                  "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
                  "fit_pairs":dataset.trainDataSize,"evaluation_pairs":dataset.testDataSize,
                  "method_config":getattr(model,"method_config",{}),
                  "parameters":sum(p.numel() for p in model.parameters()),"checkpoint_loading":False,
                  "evaluation_label":"validation" if args.phase=="development" else "original_test_once_after_freeze"})
    atomic_json(run/"manifest.json",state)
    source_copy=run/"source";source_copy.mkdir()
    for path in source_paths:
        dest=source_copy/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    history, evaluations = [], []
    losses=[];actual_updates=0
    original_stage=bpr.stageOne
    def measured_stage(users, positives, negatives):
        nonlocal actual_updates
        value=original_stage(users,positives,negatives)
        if not math.isfinite(value): raise FloatingPointError("Non-finite training loss")
        actual_updates+=1;losses.append(value)
        return value
    bpr.stageOne=measured_stage
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for epoch in range(args.epochs):
        losses.clear();torch.cuda.synchronize();tick=time.perf_counter()
        procedure.BPR_train_original(dataset,model,bpr,epoch,neg_k=1,w=None)
        torch.cuda.synchronize()
        train_seconds=time.perf_counter()-tick
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Non-finite parameter")
        row={"completed_epochs":epoch+1,"updates":actual_updates,"loss":statistics.mean(losses),"training_seconds":train_seconds}
        history.append(row)
        evaluate = (args.phase=="development" and ((epoch+1)%20==0 or epoch+1==args.epochs)) or (args.phase=="final" and epoch+1==args.epochs)
        if evaluate:
            model.eval()
            original_rating=model.getUsersRating
            with torch.no_grad():
                # Same eval computation and original metrics, without 19 redundant graph passes.
                all_users,all_items=model.computer()
                model.getUsersRating=lambda users: model.f(all_users[users.long()]@all_items.t())
                tick=time.perf_counter()
                try:metrics=procedure.Test(dataset,model,epoch+1,w=None,multicore=0)
                finally:model.getUsersRating=original_rating
                evaluation={"completed_epochs":epoch+1,"metrics":{k:v.tolist() for k,v in metrics.items()},"seconds":time.perf_counter()-tick}
                evaluations.append(evaluation)
                del all_users,all_items
            print("EVALUATION="+json.dumps(evaluation),flush=True)
            atomic_json(run/"evaluations.json",evaluations)
        if (epoch+1)%20==0 or epoch+1==args.epochs:
            state_dict={"model":model.state_dict(),"optimizer":bpr.opt.state_dict(),"epoch":epoch+1,
                        "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()},
                        "run_id":run.name,"git_commit":state["git_commit"]}
            temp=run/"checkpoint.pt.tmp";torch.save(state_dict,temp);replace_with_retry(temp,run/"checkpoint.pt")
            atomic_json(run/"history.json",history)
        status={"status":"running","completed_epochs":epoch+1,"updates":actual_updates,"latest":row,
                "elapsed_seconds":time.perf_counter()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        atomic_json(run/"status.json",status)
        print("PROGRESS epoch={}/{} method={} loss={:.8f} seconds={:.2f}".format(epoch+1,args.epochs,args.method,row["loss"],train_seconds),flush=True)
    expected=math.ceil(dataset.trainDataSize/world.config["bpr_batch_size"])*args.epochs
    assert expected==actual_updates
    score=statistics.mean(row["metrics"]["ndcg"][1] for row in evaluations[-3:]) if args.phase=="development" else None
    result={**state,**status,"status":"completed","success":True,"evaluation_count":len(evaluations),
            "selection_score_last_three_ndcg20":score,"final_metrics":evaluations[-1]["metrics"],
            "source_unchanged":all(digest(p)==state["source_sha256"][str(p.relative_to(ROOT))] for p in source_paths),
            "input_unchanged":all(digest(Path(p))==v for p,v in state["input_sha256"].items())}
    assert result["source_unchanged"] and result["input_unchanged"]
    atomic_json(run/"result.json",result);atomic_json(run/"status.json",result)
    print("RESULT="+json.dumps({k:result[k] for k in ["success","method","selection_score_last_three_ndcg20","final_metrics","elapsed_seconds","peak_allocated_bytes"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",default="baseline")
    parser.add_argument("--phase",choices=["development","final"],default="development")
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--seed",type=int,default=2020)
    parser.add_argument("--frozen-config",type=Path)
    args=parser.parse_args()
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    assert git("branch","--show-current").startswith("research/")
    os.environ["CUDA_VISIBLE_DEVICES"]="0";sys.dont_write_bytecode=True
    run=RESEARCH/(args.phase+"-"+args.method+"-s"+str(args.seed)+"-"+datetime.now(CHINA).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    state={"run_directory":str(run),"pid":os.getpid(),"success":False,"args":vars(args).copy()}
    state["args"]["frozen_config"]=str(args.frozen_config) if args.frozen_config else None
    print("RUN_DIRECTORY="+str(run),flush=True)
    with (run/"console.log").open("x",encoding="utf-8",buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:run_experiment(args,run,state)
            except BaseException as error:
                state.update({"status":"failed","error":str(error),"traceback":traceback.format_exc()})
                atomic_json(run/"failure.json",state);traceback.print_exc();return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
