from pathlib import Path
import json, hashlib, shutil, re, argparse
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description='Summarize historical data only; no training or solving.')
parser.add_argument('--project-root',type=Path,default=Path(r'D:\project\Resource_Allocation'))
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'outputs'/'stage_report_20260908_2245')
args=parser.parse_args()
ROOT=args.project_root
OUT=args.output_dir
for name in ['tables','source_data','figures','review','reproducibility']:(OUT/name).mkdir(parents=True,exist_ok=True)
TZ=timezone(timedelta(hours=8)); CUTOFF=datetime(2026,9,8,22,45,tzinfo=TZ)
sources=[]; checks=[]
def source(rel):
 p=ROOT/rel
 sources.append({'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'modified_at':datetime.fromtimestamp(p.stat().st_mtime,TZ).isoformat(),'before_cutoff':p.stat().st_mtime<=CUTOFF.timestamp()})
 return p
def read(rel): return pd.read_csv(source(rel))
def save(d,name): d.to_csv(OUT/'tables'/name,index=False,encoding='utf-8-sig')
def check(name,condition,detail=''):
 checks.append({'check':name,'passed':bool(condition),'detail':str(detail)})
 if not condition: raise AssertionError((name,detail))
m=read('data_processed/base_instances/manifest.csv');m['V']=m.instance_name.str.extract(r'V(\d+)').astype(int)
check('4500 unique instances',len(m)==4500 and m.instance_uid.nunique()==4500)
splits={s:read(f'data_processed/splits/{s}.csv') for s in ['train','validation','test']}
for a,b in [('train','validation'),('train','test'),('validation','test')]:check(f'{a}/{b} source disjoint',set(splits[a].source_identity).isdisjoint(splits[b].source_identity))
save(m.groupby(['group','V'],sort=True).agg(instances=('n','size'),N_min=('n','min'),N_median=('n','median'),N_max=('n','max')).reset_index(),'dataset_scale.csv')
save(pd.concat([d.groupby('topology').size().rename(s) for s,d in splits.items()],axis=1).reset_index(),'dataset_split.csv')
save(pd.concat([d.assign(split=s) for s,d in splits.items()]).groupby(['split','group','topology']).agg(instances=('n','size'),N_min=('n','min'),N_max=('n','max')).reset_index(),'dataset_cells.csv')
selection={}; cats=[]; masks=[]
for p in m.source_csv.unique():
 d=pd.read_csv(ROOT/p);d=d[d.row_type=='instance']; selection[p]={Path(r.instance_name).stem:np.array([int(x) for x in re.findall(r'-?\d+',r.x)]) for r in d.itertuples()}
for r in m.itertuples():
 p=ROOT/r.output_json;d=json.loads(p.read_text(encoding='utf-8'))
 x=selection[r.source_csv][r.instance_name]
 assert set(np.unique(x))<={0,1} and len(x)==r.V
 assert np.array_equal(np.flatnonzero(x)+1,np.array(d['selected_ids']))
 assert d['N']==r.n and d['M']==11 and d['K']==6 and r.n==r.selected_count
 for field in ['student_pref','teacher_pref','effect_matrix','feasible_mask']: assert np.array(d[field]).shape==(r.n,11)
 assert len(set(d['selected_ids']))==r.n
 cats.append({'topology':r.topology,'instance_uid':r.instance_uid,**{name:float(np.mean(np.array(d['category_id'])==i)) for i,name in enumerate(d['category_names'])}})
 masks.append(bool(np.all(d['feasible_mask'])))
check('all JSON shapes, selections, IDs and sizes',True,'4500 JSON checked against source selection CSV')
check('all static masks allow every method',all(masks),'dynamic mask still removes current assignment')
save(pd.DataFrame(cats).groupby('topology').mean(numeric_only=True).reset_index(),'category_composition.csv')
sample=json.loads((ROOT/m.iloc[0].output_json).read_text(encoding='utf-8'))
(OUT/'source_data'/'example_assignment_instance.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2),encoding='utf-8')
scenario_stats=[]
for name in ['validation_interpolation','test_interpolation','test_heldout']:
 bank=json.loads(source(f'data_processed/splits/scenarios/{name}.json').read_text(encoding='utf-8'))
 scenario_stats.append({'bank':name,'scenarios':len(bank),'template_pairs':len({(x['student_template'],x['teacher_template']) for x in bank})})
save(pd.DataFrame(scenario_stats),'scenario_banks.csv')
train_summaries=[]; train_frames=[]; val_frames=[]
for seed in range(3):
 base=f'results/formal_joint_seed{seed}_run01'
 tr=read(base+'/metrics/train_updates.csv');va=read(base+'/metrics/validation_updates.csv');vi=read(base+'/metrics/validation_instances.csv');ti=read(base+'/metrics/train_instances.csv')
 overall=va[va.scope=='overall'];best=overall.loc[overall.total_score_mean.idxmax()]
 check(f'seed{seed} updates',tr['update'].tolist()==list(range(1,1501)))
 check(f'seed{seed} train records',len(ti)==6000 and ti.groupby('update').size().eq(4).all())
 check(f'seed{seed} validation records',len(vi)==5400 and overall['update'].tolist()==list(range(25,1501,25)))
 check(f'seed{seed} no invalid actions',tr.illegal_action_count.sum()==0 and vi.mask_violation_count.sum()==0 and vi.hard_feasible_rate.eq(1).all())
 meta=json.loads(source(base+'/run_metadata.json').read_text(encoding='utf-8'))
 for kind,v in meta['inputs'].items():check(f'seed{seed} input hash {kind}',hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'])
 train_summaries.append({'seed':seed,'updates':len(tr),'training_instance_visits':len(ti),'transitions':int(tr.transitions.sum()),'validation_points':len(overall),'best_update':int(best['update']),'best_validation_J':best.total_score_mean,'last_validation_J':overall.iloc[-1].total_score_mean,'last_four_mean_J':overall.tail(4).total_score_mean.mean(),'entropy_min':tr.normalized_entropy.min(),'KL_p90':tr.approx_kl.quantile(.9),'clip_mean':tr.clip_fraction.mean(),'update_minutes':tr.update_elapsed_sec.sum()/60,'validation_minutes':vi.runtime.sum()/60})
 train_frames.append(tr.assign(seed=seed));val_frames.append(va.assign(seed=seed))
 for filename in ['run_metadata.json','run_summary.json','checkpoint_index.csv']:
  src=source(base+'/'+filename);shutil.copy2(src,OUT/'source_data'/f'seed{seed}_{filename}')
save(pd.DataFrame(train_summaries),'training_summary_recomputed.csv')
pd.concat(train_frames).to_csv(OUT/'source_data'/'training_updates.csv',index=False,encoding='utf-8-sig')
pd.concat(val_frames).to_csv(OUT/'source_data'/'validation_updates.csv',index=False,encoding='utf-8-sig')
test_frames=[];summary=[];seed_results=[]
for dataset,suffix in [('interpolation','interpolation'),('heldout','heldout')]:
 f=read(f'results/final_evaluation/seed1_best_test_{suffix}_with_baselines.csv')
 check(dataset+' 3375 rows',len(f)==3375)
 check(dataset+' weights',all(json.loads(x)==sample['weights'] for x in f.weights_json.unique()))
 reconstructed=.3*f.effect_score+.25*f.student_pref_score+.25*f.teacher_pref_score+.1*f.global_score-.1*f.soft_penalty
 check(dataset+' objective reconstruction',np.max(np.abs(reconstructed-f.total_score))<1e-12)
 check(dataset+' all feasible',f.hard_feasible_rate.eq(1).all() and f.mask_violation_count.sum()==0)
 keys=['instance_path','scenario_id'];ref=set(map(tuple,f[f.solver_name=='ppo_shared_policy'][keys].to_numpy()))
 for solver,g in f.groupby('solver_name'):
  check(dataset+' paired '+solver,len(g)==675 and set(map(tuple,g[keys].to_numpy()))==ref)
  summary.append({'dataset':dataset,'solver':solver,'n':len(g),'J_mean':g.total_score.mean(),'J_sd_instances':g.total_score.std(ddof=1),'runtime_mean_s':g.runtime.mean(),'runtime_median_s':g.runtime.median(),'runtime_p90_s':g.runtime.quantile(.9),'F_E':g.effect_score.mean(),'F_S':g.student_pref_score.mean(),'F_T':g.teacher_pref_score.mean(),'F_G':g.global_score.mean(),'V_soft':g.soft_penalty.mean()})
 test_frames.append(f.assign(dataset=dataset))
 for seed in range(3):
  q=f[f.solver_name=='ppo_shared_policy'] if seed==1 else read(f'results/final_evaluation/seed{seed}_best_test_{suffix}_ppo.csv')
  check(f'{dataset} seed{seed} pairs',len(q)==675 and set(map(tuple,q[keys].to_numpy()))==ref)
  for top,g in [('all',q),*list(q.groupby('topology'))]:seed_results.append({'dataset':dataset,'seed':seed,'topology':top,'n':len(g),'J_mean':g.total_score.mean()})
save(pd.DataFrame(summary),'test_algorithm_summary.csv');save(pd.DataFrame(seed_results),'test_seed_topology_summary.csv')
pd.concat(test_frames).to_csv(OUT/'source_data'/'test_deployment_paired.csv',index=False,encoding='utf-8-sig')
f=read('results/exact_hybrid_validation_pilot/fair_runtime/paired_results_fair_runtime.csv')
key=['instance_uid','scenario_id'];opt=f[f.solver_name=='gurobi_exact_or_bounded'].set_index(key)
check('45 exact optima',len(opt)==45 and opt.proven_optimal.eq(True).all() and opt.mip_gap.eq(0).all())
check('pilot 315 paired rows',len(f)==315 and f.groupby(key).solver_name.nunique().eq(7).all())
check('pilot objective recomputation error',opt.objective_recompute_error.abs().max()<1e-8)
f=f.join(opt.total_score.rename('J_star'),on=key);f['gap_absolute_recomputed']=f.J_star-f.total_score;f['gap_relative_percent_recomputed']=100*f.gap_absolute_recomputed/f.J_star
save(f.groupby('solver_name').agg(n=('n','size'),J_mean=('total_score','mean'),gap_absolute_mean=('gap_absolute_recomputed','mean'),gap_relative_percent_mean=('gap_relative_percent_recomputed','mean'),runtime_mean_s=('runtime','mean'),runtime_median_s=('runtime','median'),runtime_p90_s=('runtime',lambda x:x.quantile(.9)),runtime_max_s=('runtime','max')).reset_index(),'pilot_algorithm_summary.csv')
save(f.groupby(['group','solver_name']).agg(N_min=('n','min'),N_max=('n','max'),J_mean=('total_score','mean'),runtime_mean_s=('runtime','mean'),gap_relative_percent_mean=('gap_relative_percent_recomputed','mean')).reset_index(),'pilot_by_scale.csv')
save(f.groupby(['topology','solver_name']).agg(J_mean=('total_score','mean'),gap_absolute_mean=('gap_absolute_recomputed','mean'),gap_relative_percent_mean=('gap_relative_percent_recomputed','mean')).reset_index(),'pilot_by_topology.csv')
f.to_csv(OUT/'source_data'/'pilot_paired.csv',index=False,encoding='utf-8-sig')
for folder in ['tables','source_data']:
 for p in (ROOT/'results/publication_ready_20260908'/folder).glob('*.csv'):
  shutil.copy2(source(str(p.relative_to(ROOT))),OUT/folder/('historical_'+p.name))
shutil.copy2(source('results/ppo_parameter_search/round_300_final/candidate_summary.csv'),OUT/'tables'/'parameter_candidates.csv')
shutil.copy2(source('results/exact_hybrid_validation_pilot/summary_fair_runtime/hybrid_budget_decision.json'),OUT/'source_data'/'hybrid_budget_decision.json')
for p in (ROOT/'results/publication_ready_20260908/figures').glob('*'):shutil.copy2(source(str(p.relative_to(ROOT))),OUT/'figures'/p.name)
jsonl=source('results/exact_hybrid_validation_pilot/paired_results.jsonl')
records=[json.loads(line) for line in jsonl.read_text(encoding='utf-8').splitlines()]
# Representative case: nearest to median selected N among uniform pilot cases, fixed before looking at quality.
pool=f[(f.solver_name=='ppo')&(f.topology=='uniform')];row=pool.iloc[(pool.n-pool.n.median()).abs().argmin()]
case=[r for r in records if r['instance_uid']==row.instance_uid]
inst=json.loads(Path(row.instance_path).read_text(encoding='utf-8'))
dist=[]
for r in case:
 for i,name in enumerate(inst['method_names']):dist.append({'instance_uid':row.instance_uid,'scenario_id':r['scenario_id'],'solver_name':r['solver_name'],'method':name,'method_index':i,'proportion':float(np.mean(np.array(r['assignment'])==i)),'target':sample['target_distribution'][i]})
save(pd.DataFrame(dist),'case_method_distribution.csv')
(OUT/'source_data'/'representative_case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
save(pd.DataFrame(checks),'audit_checks.csv')
pd.DataFrame(sources).drop_duplicates('path').to_csv(OUT/'reproducibility'/'source_manifest.csv',index=False,encoding='utf-8-sig')
assert all(s['before_cutoff'] for s in sources),'A historical source was modified after cutoff; inspect before inclusion.'
print(json.dumps({'output':str(OUT),'checks_passed':len(checks),'sources':len(set(s['path'] for s in sources)),'case':row.instance_uid,'training':train_summaries},ensure_ascii=False,indent=2))
