from pathlib import Path
import hashlib
import json
import statistics

root=Path(r'D:\CoLaKGResearch')
methods=['baseline','catalog_contrastive','consensus_residual','semantic_graph_decoder','aligned_catalog','pair_geometry','pair_alignment_only']
rows=[]
for method in methods:
    for seed in [2020,2021,2022]:
        runs=[]
        for path in root.glob('development-'+method+'-s'+str(seed)+'-*'):
            if not (path/'result.json').exists():
                continue
            result=json.loads((path/'result.json').read_text(encoding='utf-8'))
            if result['epochs']==200:
                runs.append((path,result))
        assert len(runs)<=1, 'Duplicate completed run: select by a documented rule, never best score'
        for path,result in runs:
            assert result['success'] and result['source_unchanged'] and result['input_unchanged']
            assert result['updates']==9200 and result['phase']=='development'
            assert result['evaluation_label']=='validation'
            evaluations=json.loads((path/'evaluations.json').read_text(encoding='utf-8'))
            score=statistics.mean(e['metrics']['ndcg'][1] for e in evaluations[-3:])
            assert [e['completed_epochs'] for e in evaluations[-3:]]==[160,180,200]
            assert abs(score-result['selection_score_last_three_ndcg20'])<1e-12
            rows.append({'method':method,'seed':seed,'score':score,
                         'final_recall20':result['final_metrics']['recall'][1],
                         'final_ndcg20':result['final_metrics']['ndcg'][1],
                         'loss':result['latest']['loss'],'seconds':result['elapsed_seconds'],
                         'peak_allocated_mib':result['peak_allocated_bytes']/2**20,
                         'git_commit':result['git_commit'],'run':str(path),
                         'result_sha256':hashlib.sha256((path/'result.json').read_bytes()).hexdigest()})
controls={row['seed']:row for row in rows if row['method']=='baseline'}
for row in rows:
    row['relative_gain']=row['score']/controls[row['seed']]['score']-1
aggregate={'completed_runs':len(rows),'completed_epochs':200*len(rows),'updates':9200*len(rows),
           'training_loop_seconds':sum(row['seconds'] for row in rows),'runs':rows,
           'selection_used_original_test':False}
for method in methods[1:]:
    selected=[r for r in rows if r['method']==method]
    if selected:
        aggregate[method]={'seeds':[r['seed'] for r in selected],
                           'mean_score':statistics.mean(r['score'] for r in selected),
                           'mean_paired_relative_gain':statistics.mean(r['relative_gain'] for r in selected),
                           'all_paired_gains_positive':all(r['relative_gain']>0 for r in selected)}
(root/'study-v1'/'aggregate.json').write_text(json.dumps(aggregate,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in aggregate.items() if k!='runs'},indent=2))
