from pathlib import Path
import json,math,hashlib,re
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Recompute historical saved solutions; does not run a solver.')
parser.add_argument('--project',type=Path,default=Path(r'D:\project\Resource_Allocation'))
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
R=args.project;O=args.output;O.mkdir(parents=True,exist_ok=True)
bank=json.loads((R/'data_processed/splits/scenarios/validation_interpolation.json').read_text(encoding='utf-8'))
rows=[json.loads(s) for s in (R/'results/exact_hybrid_validation_pilot/paired_results.jsonl').read_text(encoding='utf-8').splitlines()]
errors=[];entropy_errors=[];cache={};logcheck=[];proofbound=11e-8/math.log(11)
for row in rows:
 path=row['instance_path']
 if path not in cache:cache[path]=json.loads(Path(path).read_text(encoding='utf-8'))
 inst=cache[path];N=inst['N'];M=inst['M'];a=np.array(row['assignment']);E=np.array(inst['effect_matrix']);U=np.array(inst['concept_need']);Q=np.array(inst['cognitive_load']);Rattr=np.array(inst['method_attr'])
 scenario=bank[row['manifest_index']%len(bank)];assert scenario['scenario_id']==row['scenario_id']
 z=U[:,None,:5]*Rattr[None,:,:5];load=.6*Q[:,None]+.4*Rattr[None,:,5]
 P=[]
 for field in ['student_profile','teacher_profile']:
  theta=np.array(scenario[field]);phi=(1-np.abs(z-theta[None,None,:5])).mean(axis=2);psi=1-np.maximum(0,load-theta[5]);P.append(np.clip(.75*phi+.25*psi,0,1))
 p=np.bincount(a,minlength=M)/N;raw=-np.sum(p*np.log(p+1e-8))/math.log(M);H=float(np.clip(raw,0,1));hp=float(np.maximum(0,-p*np.log(p+1e-8)/math.log(M)).sum());entropy_errors.append(abs(H-hp))
 FE=E[np.arange(N),a].mean();FS=P[0][np.arange(N),a].mean();FT=P[1][np.arange(N),a].mean();FG=1-.5*np.abs(p-np.array(scenario['target_distribution'])).sum();V=max(0,.55-H)**2+np.maximum(0,p-.35).dot(np.maximum(0,p-.35));J=.3*FE+.25*FS+.25*FT+.1*FG-.1*V
 errors.append(abs(J-row['total_score']));assert len(a)==N and np.array(inst['feasible_mask'])[np.arange(N),a].all()
 assert abs(FS-row['student_pref_score'])<1e-12 and abs(FT-row['teacher_pref_score'])<1e-12
for path in (R/'results/exact_hybrid_validation_pilot/gurobi_logs').glob('*.log'):
 s=path.read_text(encoding='utf-8');pat=re.search(r'Optimize a model with (\d+) rows, (\d+) columns',s);types=re.search(r'Variable types: (\d+) continuous, (\d+) integer \((\d+) binary\)',s)
 if pat and types:
  nbin=int(types[3]);N=nbin//11;assert int(pat[1])==N+46 and int(pat[2])==nbin+46 and int(types[1])==35 and int(types[2])==nbin+11
  assert 'Model has 12 quadratic objective terms' in s and '11 PWL' in s
  logcheck.append(path.name)
assert max(errors)<1e-12 and max(entropy_errors)<1e-12
result={'audit_date':'2026-10-02','historical_cutoff':'2026-09-08T22:45:00+08:00','saved_assignments_recomputed':len(rows),'instances':len(cache),'max_J_error':max(errors),'max_entropy_form_equivalence_error':max(entropy_errors),'logs_structure_verified':len(logcheck),'entropy_smoothing_uniform_bound':proofbound,'J_difference_bound_vs_standard_Shannon':.11*proofbound,'note':'Read-only independent formula reconstruction, no solver rerun.'}
(O/'mathematical_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
sources=['pa_moap_rl/utils/scoring.py','pa_moap_rl/data/build_assignment_instance.py','pa_moap_rl/solvers/gurobi_exact_solver.py','pa_moap_rl/experiments/compare_exact_hybrid.py','pa_moap_rl/experiments/remeasure_gurobi_runtime.py','results/exact_hybrid_validation_pilot/paired_results.jsonl','data_processed/splits/scenarios/validation_interpolation.json','docs/exact_hybrid_experiment_guide.md']
(O/'source_hashes.json').write_text(json.dumps([{'path':x,'sha256':hashlib.sha256((R/x).read_bytes()).hexdigest()} for x in sources],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
