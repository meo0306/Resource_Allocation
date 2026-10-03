import sys
from pathlib import Path
import argparse
BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Plot saved historical summary tables, without new experiments.')
parser.add_argument('--output-dir',type=Path,default=BASE/'outputs/stage_report_20260908_2245')
parser.add_argument('--qa-path',type=Path,default=BASE/'work/figure_contact_sheet.png')
args=parser.parse_args()
sys.path.insert(0,str(BASE/'work/python_deps'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
OUT=args.output_dir
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':11,'axes.titlesize':14,'axes.labelsize':12,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'legend.frameon':False,'savefig.facecolor':'white'})
COL=['#2876A8','#D58E3D','#4F9384']; top=['uniform','random','hourglass','bottleneck','staircase']
def read(name,folder='tables'):return pd.read_csv(OUT/folder/name)
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
sc=read('dataset_scale.csv');cat=read('category_composition.csv').set_index('topology').loc[top]
fig,axs=plt.subplots(1,2,figsize=(12.8,5),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
ax=axs[0];ax.errorbar(sc.V,sc.N_median,yerr=[sc.N_median-sc.N_min,sc.N_max-sc.N_median],fmt='o-',color=COL[0],capsize=4);ax.plot([0,1000],[0,1000],':',color='#AAB1B8');ax.set(xlabel='原始网络节点数 V',ylabel='已选知识点数 N',title='a  内容选择后进入分配的实际规模',xlim=(0,1040),ylim=(0,1040));ax.text(70,900,'点：中位数；线：最小–最大\n每档 500 个实例',fontsize=10)
ax=axs[1];im=ax.imshow(cat.to_numpy()*100,cmap='Blues',vmin=0,vmax=70,aspect='auto');ax.set_xticks(range(6),cat.columns);ax.set_yticks(range(5),top);ax.set_title('b  五类拓扑的知识点类别组成 (%)')
for i in range(5):
 for j in range(6):ax.text(j,i,f'{cat.iloc[i,j]*100:.1f}',ha='center',va='center',color='white' if cat.iloc[i,j]>.45 else '#243746')
fig.colorbar(im,ax=ax,shrink=.75,label='实例内占比的拓扑均值 (%)');save(fig,'figure_4_dataset')

tr=read('training_updates.csv','source_data');fig,axs=plt.subplots(2,2,figsize=(12.8,7.5),layout='constrained')
for ax,(field,title,ylabel) in zip(axs.flat,[('episode_return','a  训练回合回报','记录的平均 episode return'),('normalized_entropy','b  策略标准化熵','策略熵 / 合法动作数对数'),('approx_kl','c  策略更新幅度','approx KL'),('clip_fraction','d  PPO 裁剪比例','clip fraction')]):
 for seed,g in tr.groupby('seed'):
  ax.plot(g['update'],g[field],color=COL[seed],lw=.5,alpha=.12)
  ax.plot(g['update'],g[field].rolling(50,min_periods=1).mean(),color=COL[seed],lw=1.7,label=f'seed {seed}')
 ax.set(title=title,xlabel='PPO update',ylabel=ylabel,xlim=(0,1500));ax.grid(axis='y',alpha=.15)
axs[0,0].legend(ncol=3);fig.suptitle('原始轨迹 + 50-update 移动平均；训练分布随采样变化',fontsize=13);save(fig,'figure_5_training_diagnostics')

names=['ppo','ppo_plus_short_local_search_4','ppo_plus_short_local_search_8','ppo_plus_short_local_search_16','ppo_plus_short_local_search_32','one_point_best_improvement_local_search','gurobi_exact_or_bounded']
labels=['PPO','PPO + 4','PPO + 8','PPO + 16','PPO + 32','单点局部搜索','Gurobi']
colors=[COL[0],'#95B8D0','#7CA9C7','#5C93B8','#407CA5',COL[1],COL[2]]
pi=read('pilot_algorithm_summary.csv').set_index('solver_name').loc[names]
fig,axs=plt.subplots(1,2,figsize=(12.8,5.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.1]})
ax=axs[0];ax.barh(labels,pi.gap_relative_percent_mean,color=colors);ax.invert_yaxis();ax.set(xlabel='逐实例相对最优差距的均值 (%)',title='a  解质量相对精确最优的差距',xlim=(0,3.1))
for i,x in enumerate(pi.gap_relative_percent_mean):ax.text(x+.04,i,f'{x:.3f}%',va='center',fontsize=10)
ax=axs[1]
offsets=[(-8,-18),(-8,10),(-8,10),(-8,10),(-8,10),(-105,-10),(8,0)]
for i,(name,r) in enumerate(pi.iterrows()):
 ax.scatter(r.runtime_mean_s,r.J_mean,s=65,c=colors[i],marker='D' if i==6 else 'o')
 ax.annotate(labels[i],(r.runtime_mean_s,r.J_mean),xytext=offsets[i],textcoords='offset points',fontsize=10)
ax.set(xscale='log',xlabel='修正口径平均求解耗时 (s，对数轴)',ylabel='平均综合目标 J',title='b  同一批 45 个验证算例',xlim=(.85,31),ylim=(.669,.695));ax.grid(alpha=.15);save(fig,'figure_6_exact_hybrid')

sc=read('pilot_by_scale.csv');fig,axs=plt.subplots(1,2,figsize=(12.8,5),layout='constrained')
for name,label,color in [(names[0],labels[0],COL[0]),(names[-2],labels[-2],COL[1]),(names[-1],labels[-1],COL[2])]:
 g=sc[sc.solver_name==name].sort_values('group');x=np.arange(1,10)
 axs[0].plot(x,g.runtime_mean_s,'o-',label=label,color=color)
 axs[1].plot(x,g.gap_relative_percent_mean,'o-',label=label,color=color)
axs[0].set(yscale='log',title='a  各规模组平均求解时间',ylabel='平均时间 (s，对数轴)');axs[1].set(title='b  各规模组平均相对最优差距',ylabel='相对最优差距 (%)')
for ax in axs:ax.set_xticks(range(1,10),[60,80,100,150,200,300,500,800,1000],rotation=35);ax.set_xlabel('规模组对应的原始 V（每组 5 个实例）');ax.grid(alpha=.15)
axs[0].legend();save(fig,'figure_7_scale_diagnostic')

df=read('case_method_distribution.csv');fig,ax=plt.subplots(figsize=(12.8,5.5),layout='constrained');methods=df.sort_values('method_index').method.unique();x=np.arange(11)
for offset,(name,label,color) in zip([-.24,0,.24],[(names[0],labels[0],COL[0]),(names[-2],labels[-2],COL[1]),(names[-1],labels[-1],COL[2])]):
 g=df[df.solver_name==name].sort_values('method_index');ax.bar(x+offset,g.proportion*100,width=.23,label=label,color=color,alpha=.9)
g=df[df.solver_name==names[0]].sort_values('method_index');ax.scatter(x,g.target*100,marker='_',s=250,c='black',label='参考分布 ρ',zorder=5);ax.set_xticks(x,methods,rotation=30,ha='right');ax.set(ylabel='方法使用比例 (%)',title='代表算例：uniform / group5，按中位规模规则选取');ax.legend(ncol=4);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15);save(fig,'figure_8_case_distribution')

# QA contact sheet, produced with the same Python backend.
from PIL import Image, ImageOps, ImageDraw
images=[]
for p in sorted((OUT/'figures').glob('*.png')):
 im=Image.open(p).convert('RGB');im.thumbnail((960,570));tile=Image.new('RGB',(980,610),'white');tile.paste(im,((980-im.width)//2,30));ImageDraw.Draw(tile).text((10,8),p.stem,fill='black');images.append(tile)
sheet=Image.new('RGB',(1960,610*((len(images)+1)//2)), '#E8EDF1')
for i,im in enumerate(images):sheet.paste(im,((i%2)*980,(i//2)*610))
args.qa_path.parent.mkdir(parents=True,exist_ok=True)
sheet.save(args.qa_path)
print('Exported',len(images),'figure sets')
