"""Explicitly authorized full-train/direct-test exploration, fresh 2000 epochs."""
import argparse
import contextlib
import ctypes
from datetime import datetime
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from lastfm_one_minibatch import ROOT,REC,PYTHON,CHINA,Tee,git,digest,original_arguments,bootstrap_world
from experiment_io import atomic_json,atomic_save
from research_gpu_guard import check_gpu_processes


OUTPUT=Path(r'D:\CoLaKGResearch\content-neighbor-fusion-v1')
BASE=Path(r'D:\CoLaKGExperiments\lastfm-baseline-20261022-233837-33d715d8')
PACKAGES=['torch','numpy','pandas','scipy','scikit-learn','tensorboardX','tqdm']
LIMITATION='User-authorized direct test feedback for exploration. Repeated test selection is not an independent validation-selected generalization conclusion.'


BRANCH='codex/b-content-neighbor-fusion'
PROTECTED={'main':'be3aa19b59419d197dd4e20f44db737041821f58',
           'exp/colakg-improvement':'b98629e3e32e3917d8dc1f0013cfc2e33e0eb8e9',
           'codex/pair-geometry':'2a5c8b1df0fbee46ff669d972249129ac060a2e4',
           'codex/b-dynamic-semantic-attention':'5c5ee015a98c5f89b1e00995f2e24a80e2b341c5',
           'codex/b-asymmetric-semantic-attention':'b7ecd77'}


def assert_checkout(commit):
    assert git('branch','--show-current')==BRANCH and git('rev-parse','HEAD')==commit
    assert not git('status','--porcelain'),'Concurrent changes; preserve them and stop automatic commit'
    for ref,sha in PROTECTED.items():assert git('rev-parse',ref)==git('rev-parse',sha),ref


def audit_reference():
    parent_path=Path(r'D:\CoLaKGResearch\dynamic-semantic-attention-v1\direct-test-dynamic_semantic_attention-s2020-20261028-211059-bfa969')
    parent=json.loads((parent_path/'result.json').read_text(encoding='utf-8-sig'))
    assert parent['success'] and parent['source_unchanged'] and parent['input_unchanged']
    assert parent['completed_epochs']==2000 and parent['updates']==114000 and parent['evaluation_count']==401
    rel='experiments/research_methods/dynamic_semantic_attention.py'
    archived=parent_path/'source'/rel
    assert archived.read_bytes().replace(b'\r\n',b'\n')==(ROOT/rel).read_bytes().replace(b'\r\n',b'\n')
    baseline=json.loads((BASE/'result.json').read_text(encoding='utf-8-sig'))
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8-sig'))
    evaluations=json.loads((BASE/'evaluations.json').read_text(encoding='utf-8-sig'))
    assert baseline['success'] and baseline['completed_epochs']==2000 and baseline['updates']==114000
    assert [r['completed_epochs'] for r in evaluations]==list(range(0,2000,5))+[2000]
    identity=manifest['identity']
    for name,item in identity['inputs'].items():assert digest(ROOT/name)==item['sha256'],name
    for name,sha in identity['source_sha256'].items():
        if name.replace('\\','/').startswith('rec_code/'):
            archived=BASE/'source'/name
            assert digest(archived)==sha,name
            assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n')==archived.read_bytes().replace(b'\r\n',b'\n'),name
    return baseline,identity,{'baseline_result_sha256':digest(BASE/'result.json'),
        'baseline_evaluations_sha256':digest(BASE/'evaluations.json'),'baseline_directory':str(BASE),
        'original_protocol_best':baseline['best_observed_original_protocol'],
        'shared_dynamic_best':parent['best_observed_original_protocol'],'shared_dynamic_final':parent['final_metrics'],
        'shared_dynamic_directory':str(parent_path),'shared_dynamic_result_sha256':digest(parent_path/'result.json'),
        'best_including_final':max(evaluations,key=lambda r:r['metrics']['ndcg'][1]),
        'final':evaluations[-1],'original_rec_sources_unchanged':True,'checkpoint_loaded':False}


def train(args,run,state):
    import numpy as np
    import torch
    baseline,identity,audit=audit_reference()
    _,cli=original_arguments();sys.path.insert(0,str(REC))
    world=bootstrap_world(cli,run);world.tensorboard=0
    utils=importlib.import_module('utils');dataloader=importlib.import_module('dataloader')
    procedure=importlib.import_module('Procedure')
    method=importlib.import_module('research_methods.content_neighbor_fusion')
    assert torch.cuda.is_available() and not utils.sample_ext
    versions={name:importlib.metadata.version(name) for name in PACKAGES}
    assert dict(world.config)==identity['config'] and versions==identity['versions']
    assert args.seed==identity['seed']==2020
    utils.set_seed(args.seed);random.seed(args.seed)
    data=ROOT/'data/lastfm'
    dataset=dataloader.Loader(config=world.config,path=str(data))
    old_load,old_save=dataloader.sp.load_npz,dataloader.sp.save_npz
    def graph_path(path):return run/'graph-cache.npz' if Path(path).name=='s_pre_adj_mat.npz' else path
    dataloader.sp.load_npz=lambda path,*a,**kw:old_load(graph_path(path),*a,**kw)
    dataloader.sp.save_npz=lambda path,*a,**kw:old_save(graph_path(path),*a,**kw)
    assets=[data/'lastfm_embeddings_simcse_kg.pt',data/'lastfm_embeddings_simcse_kg_user.pt']
    item,user=[torch.load(p,map_location='cpu',weights_only=True) for p in assets]
    from sklearn.metrics.pairwise import cosine_similarity
    neighbors=torch.tensor(np.argsort(-cosine_similarity(item.numpy()),axis=1)[:,1:11]).long()
    model=method.ResearchModel(world.config,dataset,neighbors,item,user).to(world.device)
    dataloader.sp.load_npz,dataloader.sp.save_npz=old_load,old_save
    assert dataset.trainDataSize==57786 and dataset.testDataSize==13508
    optimizer=utils.BPRLoss(model,world.config)
    (run/'checkpoints').mkdir()
    sources=[*REC.glob('*.py'),REC/'train_lastfm.sh',Path(__file__).resolve(),Path(method.__file__).resolve(),
             ROOT/'AGENTS.md',ROOT/'experiments/content_neighbor_fusion_protocol.md']
    sources += [ROOT/'experiments/check_dynamic_semantic_attention.py']
    sources += [ROOT/'experiments'/name for name in ['lastfm_one_minibatch.py','lastfm_windows.py','experiment_io.py','research_gpu_guard.py','check_content_neighbor_fusion.py']]
    sources += [(Path(method.__file__).parent/name).resolve() for name in method.SOURCE_FILES]
    sources=list(dict.fromkeys(p.resolve() for p in sources))
    inputs=[data/'train.txt',data/'test.txt',*assets,data/'user_map.txt',data/'item_map.txt']
    state.update(method='content_neighbor_fusion',phase='direct_test_exploration',seed=args.seed,epochs=2000,
        config=dict(world.config),method_config=model.method_config,versions=versions,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),parameters=sum(p.numel() for p in model.parameters()),
        training_interactions=dataset.trainDataSize,test_interactions=dataset.testDataSize,
        checkpoint_loading=False,initialization='fresh random recommendation parameters, fixed released semantic assets',
        evaluation_protocol='completed epochs 0,5,...,1995 before training, plus 2000 final',
        evaluation_data='original LastFM test, explicitly used for research feedback',
        test_selection_limitation=LIMITATION,reference=audit,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256={str(p):digest(p) for p in inputs})
    atomic_json(run/'manifest.json',state);atomic_json(run/'baseline-reuse-audit.json',audit)
    with (run/'pip-freeze.txt').open('x',encoding='utf-8') as out:
        subprocess.run([str(PYTHON),'-B','-m','pip','freeze'],stdout=out,check=True)
    for p in sources:
        target=run/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    history=[];evaluations=[];losses=[];components={};updates=0;best=None;primary=None
    sampling_seconds=0.;checkpoint_seconds=0.
    original_sample=utils.UniformSample_original
    def timed_sample(*a,**kw):
        nonlocal sampling_seconds
        tick=time.perf_counter()
        value=original_sample(*a,**kw)
        sampling_seconds+=time.perf_counter()-tick
        return value
    utils.UniformSample_original=timed_sample
    original_step=optimizer.opt.step;original_stage=optimizer.stageOne
    def checked_step(*a,**kw):
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),'Non-finite gradient'
        return original_step(*a,**kw)
    optimizer.opt.step=checked_step
    def stage(*a,**kw):
        nonlocal updates
        loss=original_stage(*a,**kw);assert math.isfinite(loss)
        losses.append(loss);updates+=1
        for name,value in model.latest_components.items():
            assert math.isfinite(value);components[name]=components.get(name,0.)+value
        return loss
    optimizer.stageOne=stage
    def checkpoint(name,completed):
        nonlocal checkpoint_seconds
        tick=time.perf_counter()
        value={'run_id':run.name,'git_commit':state['git_commit'],'epoch':completed,'updates':updates,
            'model':model.state_dict(),'optimizer':optimizer.opt.state_dict(),
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
            'history':history,'evaluations':evaluations,'best':best,'best_protocol':primary,
            'source_sha256':state['source_sha256'],'input_sha256':state['input_sha256']}
        atomic_save(torch,run/'checkpoints'/name,value)
        sha=digest(run/'checkpoints'/name);checkpoint_seconds+=time.perf_counter()-tick
        return sha
    def evaluate(completed,kind):
        nonlocal best,primary
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        mode=model.training;model.eval();tick=time.perf_counter()
        try:
            with torch.no_grad():
                model.begin_evaluation()
                metrics=procedure.Test(dataset,model,completed,w=None,multicore=0)
                diagnostics=model.evaluation_diagnostics()
                assert all(math.isfinite(v) for v in diagnostics.values() if isinstance(v,(int,float)))
                torch.cuda.synchronize()
        finally:
            model.end_evaluation();model.train(mode)
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        values={k:v.tolist() for k,v in metrics.items()}
        assert all(math.isfinite(x) and 0<=x<=1 for vals in values.values() for x in vals)
        row={'completed_epochs':completed,'kind':kind,'metrics':values,'seconds':time.perf_counter()-tick,'diagnostics':diagnostics}
        evaluations.append(row)
        if kind=='original_protocol' and (primary is None or values['ndcg'][1]>primary['metrics']['ndcg'][1]):
            primary=dict(row);primary['checkpoint_sha256']=checkpoint('best-protocol.pt',completed)
        if best is None or values['ndcg'][1]>best['metrics']['ndcg'][1]:
            best=dict(row);best['checkpoint_sha256']=checkpoint('best.pt',completed)
        atomic_json(run/'evaluations.json',evaluations);atomic_json(run/'best.json',best)
        print('EVALUATION='+json.dumps(row),flush=True)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for epoch in range(2000):
        if epoch%5==0:evaluate(epoch,'original_protocol')
        losses.clear();components.clear();sampling_seconds=0.;torch.cuda.synchronize();tick=time.perf_counter()
        procedure.BPR_train_original(dataset,model,optimizer,epoch,neg_k=1,w=None)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Non-finite parameter'
        row={'completed_epochs':epoch+1,'updates':updates,'loss':statistics.mean(losses),'training_seconds':seconds,'sampling_seconds':sampling_seconds,'training_after_sampling_seconds':seconds-sampling_seconds,
             'loss_components':{name:value/len(losses) for name,value in components.items()}}
        history.append(row)
        if (epoch+1)%20==0:
            checkpoint('recovery.pt',epoch+1);atomic_json(run/'history.json',history)
        status={'status':'running','completed_epochs':epoch+1,'updates':updates,'latest':row,
            'elapsed_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        atomic_json(run/'status.json',status)
        print('PROGRESS epoch={}/2000 loss={:.8f} seconds={:.2f}'.format(epoch+1,row['loss'],seconds),flush=True)
    evaluate(2000,'final_2000_epochs');final_sha=checkpoint('final.pt',2000)
    assert updates==114000 and len(evaluations)==401
    assert [e['completed_epochs'] for e in evaluations]==list(range(0,2000,5))+[2000]
    result={**state,**status,'success':True,'status':'completed','evaluation_count':401,
        'best_observed_original_protocol':primary,'best_including_final':best,'final_metrics':evaluations[-1]['metrics'],
        'final_checkpoint_sha256':final_sha,'final_diagnostics':evaluations[-1]['diagnostics'],'elapsed_seconds':time.perf_counter()-started,
        'timing_breakdown':{'training_including_sampling_seconds':sum(r['training_seconds'] for r in history),
            'sampling_seconds':sum(r['sampling_seconds'] for r in history),
            'training_after_sampling_seconds':sum(r['training_after_sampling_seconds'] for r in history),
            'evaluation_seconds':sum(r['seconds'] for r in evaluations),'checkpoint_seconds':checkpoint_seconds},
        'relative_gain_best_protocol':primary['metrics']['ndcg'][1]/audit['original_protocol_best']['metrics']['ndcg'][1]-1,
        'relative_gain_vs_shared_dynamic':primary['metrics']['ndcg'][1]/audit['shared_dynamic_best']['metrics']['ndcg'][1]-1,
        'relative_gain_best_including_final':best['metrics']['ndcg'][1]/audit['best_including_final']['metrics']['ndcg'][1]-1,
        'source_unchanged':all(digest(p)==state['source_sha256'][str(p.relative_to(ROOT))] for p in sources),
        'input_unchanged':all(digest(Path(p))==sha for p,sha in state['input_sha256'].items())}
    assert result['source_unchanged'] and result['input_unchanged']
    atomic_json(run/'result.json',result);atomic_json(run/'status.json',result)
    return result


def summarize(run,result):
    result['fusion_protocol'] = 'experiments/content_neighbor_fusion_protocol.md'

    parent=json.loads(Path(r'D:\CoLaKGResearch\asymmetric-semantic-attention-v1\direct-test-asymmetric_semantic_attention-s2020-20261028-233455-8d6c5a\result.json').read_text(encoding='utf-8-sig'))
    result['asymmetric_parent_best']=parent['best_observed_original_protocol']
    result['gain_vs_asymmetric_parent']=result['best_observed_original_protocol']['metrics']['ndcg'][1]/parent['best_observed_original_protocol']['metrics']['ndcg'][1]-1
    rows=[('原始：400次最佳',result['reference']['original_protocol_best']),
          ('共享动态版：400次最佳',result['reference']['shared_dynamic_best']),
          ('本候选：400次最佳',result['best_observed_original_protocol']),
          ('本候选：401次最佳',result['best_including_final']),
          ('本候选：最终2000轮',{'completed_epochs':2000,'metrics':result['final_metrics']})]
    rows.insert(2,('非对称注意力对照：400次最佳',result['asymmetric_parent_best']))
    atomic_json(run/'result.json',result)
    lines=['# LastFM 中心—邻居逐维融合：完整训练集直接测试探索','',
        '用户2026-10-28明确授权变更评估规则。本次重新随机初始化训练，旧验证实验保留。',
        '代码版本：`'+result['git_commit']+'`。完整2000轮、114000次更新；原始train57786、test13508。',
        '', '| 口径 | 实际训练轮数 | P@10 | P@20 | R@10 | R@20 | NDCG@10 | NDCG@20 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,row in rows:
        values=row['metrics']['precision']+row['metrics']['recall']+row['metrics']['ndcg']
        lines.append('| '+label+' | '+str(row['completed_epochs'])+' | '+' | '.join('{:.9f}'.format(v) for v in values)+' |')
    lines+=['','原协议最佳相对变化：{:+.3f}%；含终点最佳相对变化：{:+.3f}%。'.format(result['relative_gain_best_protocol']*100,result['relative_gain_best_including_final']*100),
        '总耗时{:.2f}分钟，峰值分配显存{:.2f}MiB。'.format(result['elapsed_seconds']/60,result['peak_allocated_bytes']/1024**2),
        '计时分项（秒）：`'+json.dumps(result['timing_breakdown'])+'`。训练分项包含采样；评估不含checkpoint保存，保存单独计量，其余差额包括状态JSON、校验与其他开销。',
        '配置、数据/源码哈希、环境及命令见manifest.json，完整曲线见history.json/evaluations.json。',
        '最佳点注意力诊断：`'+json.dumps(result['best_observed_original_protocol']['diagnostics'],ensure_ascii=False)+'`。',
        '最终注意力诊断：`'+json.dumps(result['final_diagnostics'],ensure_ascii=False)+'`。',
        '在固定非对称动态注意力上，将中心—邻居固定均值替换为内容条件逐维融合；输入ID/语义融合、邻居集合、传播、BPR与原六向量正则保留，新增门控参数L2。方法及来源见content_neighbor_fusion_protocol.md。',
        '反复查看原测试集得到的最大值属于探索最佳，不是独立验证选型后的泛化结论；单种子不证明稳定性。未按曲线调整本次参数或预算。',
        '公开语义资产生成来源不完整这一既有限制保留；没有生成或覆盖原资产。',
        '', '运行目录：`'+str(run)+'`。best-protocol.pt、best.pt、final.pt、recovery.pt保存完整模型/Adam/RNG。',
        '', '```powershell',"& 'D:\\CondaEnvs\\CoLaKG\\python.exe' -B experiments/content_neighbor_fusion_2000.py",'```']
    lines += ['', '本次进化：完整中心/邻居表示及差值、乘积经过256→128→64的MLP学习逐维融合，无度数输入。',
              '最佳NDCG@20相对共享动态版本：{:+.4f}%。'.format(result['relative_gain_vs_shared_dynamic']*100)]
    lines += ['', 'Feature-wise center/neighbor fusion on asymmetric attention; see content_neighbor_fusion_protocol.md.', 'Parent best NDCG@20: '+str(result['asymmetric_parent_best']['metrics']['ndcg'][1]), 'Relative gain vs parent: '+str(result['gain_vs_asymmetric_parent'])]
    report='\n'.join(lines)+'\n';(run/'report.md').write_text(report,encoding='utf-8')
    assert_checkout(result['git_commit'])
    stamp=run.name.rsplit('-s2020-',1)[-1]
    md=Path('experiments')/('content_neighbor_fusion_result_'+stamp+'.md')
    js=Path('experiments/research_records')/('content_neighbor_fusion_result_'+stamp+'.json')
    assert not (ROOT/md).exists() and not (ROOT/js).exists()
    (ROOT/md).write_text(report,encoding='utf-8');atomic_json(ROOT/js,result)
    subprocess.run(['git','add','--',str(md),str(js)],cwd=str(ROOT),check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=str(ROOT),check=True)
    subprocess.run(['git','commit','-m','记录中心邻居逐维融合完整训练探索结果'],cwd=str(ROOT),check=True)
    atomic_json(run/'record-commit.json',{'result_commit':git('rev-parse','HEAD')})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=[2020],default=2020)
    args=parser.parse_args();assert Path(sys.executable).resolve()==PYTHON.resolve()
    commit=git('rev-parse','HEAD');assert_checkout(commit);guard=check_gpu_processes()
    run=OUTPUT/('direct-test-content_neighbor_fusion-s2020-'+datetime.now(CHINA).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    state={'run_directory':str(run),'pid':os.getpid(),'success':False,'git_commit':commit,'git_branch':git('branch','--show-current'),'gpu_guard':guard}
    print('RUN_DIRECTORY='+str(run),flush=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001))
    try:
        with (run/'console.log').open('x',encoding='utf-8',buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                try:
                    result=train(args,run,state);summarize(run,result)
                    print('COMPLETED='+str(run),flush=True);return 0
                except BaseException:
                    state.update(status='failed_or_requires_review',traceback=traceback.format_exc())
                    atomic_json(run/'failure.json',state);traceback.print_exc();return 1
    finally:ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000))


if __name__=='__main__':raise SystemExit(main())
