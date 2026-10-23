"""Post-hoc CPU geometry statistics of this study's OWN 200-epoch snapshots.

Not training, not a warm start, not test ranking. Original encoder is identical
for baseline, pair_geometry and its alignment-only ablation.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import uuid
from lastfm_one_minibatch import ROOT, REC, PYTHON, digest, original_arguments, bootstrap_world


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    args=parser.parse_args()
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    root=Path(r'D:\CoLaKGResearch')
    output=root/'study-v1'/('geometry-analysis-'+uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    sys.path.insert(0,str(REC))
    _,cli=original_arguments()
    world=bootstrap_world(cli,output);world.device=torch.device('cpu');world.tensorboard=0
    importlib.import_module('utils')
    dataloader=importlib.import_module('dataloader')
    model_module=importlib.import_module('model')
    protocol=root/'protocol-v1'
    split=json.loads((protocol/'manifest.json').read_text())
    for name,value in split['derived_sha256'].items():assert digest(protocol/name)==value
    dataset=dataloader.Loader(config=world.config,path=str(protocol))
    original_load,original_save=dataloader.sp.load_npz,dataloader.sp.save_npz
    def redirect(p):return output/'graph-cache.npz' if Path(p).name=='s_pre_adj_mat.npz' else p
    dataloader.sp.load_npz=lambda p,*a,**kw:original_load(redirect(p),*a,**kw)
    dataloader.sp.save_npz=lambda p,*a,**kw:original_save(redirect(p),*a,**kw)
    item=torch.load(ROOT/'data/lastfm/lastfm_embeddings_simcse_kg.pt',map_location='cpu',weights_only=True)
    user=torch.load(ROOT/'data/lastfm/lastfm_embeddings_simcse_kg_user.pt',map_location='cpu',weights_only=True)
    neighbours=torch.from_numpy(np.argsort(-cosine_similarity(item.numpy()),axis=1)[:,1:11]).long()
    model=model_module.CoLaKG(world.config,dataset,neighbours,item,user)
    dataloader.sp.load_npz,dataloader.sp.save_npz=original_load,original_save
    fit_u,fit_i=dataset.UserItemNet.nonzero()
    def statistics(embedding):
        normalized=torch.nn.functional.normalize(embedding,dim=1).double()
        count=len(normalized)
        cosine=normalized@normalized.t()
        uniformity=torch.log((torch.exp(-4+4*cosine).sum()-count)/(count*(count-1))).item()
        mean_cosine=(normalized.sum(0).square().sum()-count).item()/(count*(count-1))
        centered=normalized-normalized.mean(0,keepdim=True)
        eigen=torch.linalg.eigvalsh(centered.t()@centered/count).clamp(min=0)
        probabilities=eigen/eigen.sum()
        effective_rank=torch.exp(-(probabilities*torch.log(probabilities.clamp(min=1e-30))).sum()).item()
        return {'mean_off_diagonal_cosine':mean_cosine,'uniformity_log_mean_exp_negative_two_distance_squared':uniformity,
                'centered_covariance_effective_rank':effective_rank,'mean_raw_norm':embedding.norm(dim=1).mean().item()}
    rows=[]
    for run in args.runs:
        run=run.resolve();assert root in run.parents
        record=json.loads((run/'result.json').read_text(encoding='utf-8'))
        assert record['success'] and record['phase']=='development' and record['epochs']==200 and record['seed']==2020
        assert record['method'] in ['baseline','pair_geometry','pair_alignment_only']
        assert record['source_sha256'][str(Path('rec_code/model.py'))]==digest(REC/'model.py')
        checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu')
        assert checkpoint['run_id']==run.name and checkpoint['epoch']==200
        model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        with torch.no_grad():
            u,i=model.computer()
            normalized_u=torch.nn.functional.normalize(u,dim=1)
            normalized_i=torch.nn.functional.normalize(i,dim=1)
            fit_alignment=(normalized_u[fit_u]*normalized_i[fit_i]).sum(1).mean().item()
            rows.append({'method':record['method'],'run':str(run),'checkpoint_sha256':digest(run/'checkpoint.pt'),
                         'users':statistics(u),'items':statistics(i),'fit_positive_mean_cosine':fit_alignment})
    result={'purpose':'post-hoc representation analysis, no training or original-test ranking',
            'device':'cpu','seed':2020,'trained_epochs':200,'fit_pairs':len(fit_u),'rows':rows,
            'interpretation_limit':'one seed; association and controlled component ablation, not universal causality'}
    (output/'geometry.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GEOMETRY_DIRECTORY='+str(output));print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
