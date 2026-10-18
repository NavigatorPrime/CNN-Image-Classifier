from pathlib import Path
import json, zipfile, os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/'figures'
TAB = ROOT/'tables'
for p in [FIG/'task1', FIG/'task3', FIG/'task4', FIG/'task5', FIG/'task6', TAB]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Load verified data ----------
opt = pd.read_csv(ROOT/'results/optimizer_comparison/optimizer_comparison.csv')
profiles = pd.read_csv(ROOT/'results/computational_cost/model_profiles.csv')
opt_json = json.load(open(ROOT/'results/optimizer_comparison/optimizer_comparison.json'))
sota = json.load(open(ROOT/'results/sota_benchmark_summary.json'))
ha = json.load(open(ROOT/'results/model_a/training_history.json'))
hb = json.load(open(ROOT/'results/model_b/training_history.json'))
manifest = pd.read_csv(ROOT/'split_manifest.csv')

# Parse classification reports compactly
def parse_report(path):
    txt = Path(path).read_text(encoding='utf-8')
    acc = None; test_loss=None; macro_p=macro_r=macro_f1=None
    m=re.search(r'Test Accuracy:\s*([0-9.]+)%', txt)
    if m: acc=float(m.group(1))
    m=re.search(r'Test Loss:\s*([0-9.]+)', txt)
    if m: test_loss=float(m.group(1))
    m=re.search(r'macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', txt)
    if m: macro_p,macro_r,macro_f1=map(float,m.groups())
    return dict(test_accuracy=acc,test_loss=test_loss,macro_precision=macro_p,macro_recall=macro_r,macro_f1=macro_f1)

metrics = {
    'Model A': parse_report(ROOT/'results/model_a/classification_report.txt'),
    'Model B': parse_report(ROOT/'results/model_b/classification_report.txt'),
    'MobileNetV2': parse_report(ROOT/'results/mobilenet_v2/classification_report.txt'),
    'ShuffleNetV2 1.0x': parse_report(ROOT/'results/shufflenet_v2/classification_report.txt'),
}
# Exact summary values where classification report rounds to 2 decimals
metrics['MobileNetV2'].update(test_accuracy=sota['mobilenet_v2']['test_accuracy'], test_loss=sota['mobilenet_v2']['test_loss'], macro_f1=sota['mobilenet_v2']['macro_f1'])
metrics['ShuffleNetV2 1.0x'].update(test_accuracy=sota['shufflenet_v2']['test_accuracy'], test_loss=sota['shufflenet_v2']['test_loss'], macro_f1=sota['shufflenet_v2']['macro_f1'])
metrics['Model A']['test_accuracy']=68.9002
metrics['Model B']['test_accuracy']=70.4297

# Actual checkpoint file sizes
ckpts={
 'Model A': ROOT/'results/model_a/model_a_best.pth',
 'Model B': ROOT/'results/model_b/model_b_best.pth',
 'MobileNetV2': ROOT/'results/mobilenet_v2/mobilenet_v2_best.pth',
 'ShuffleNetV2 1.0x': ROOT/'results/shufflenet_v2/shufflenet_v2_best.pth',
}
size_bytes={k:v.stat().st_size for k,v in ckpts.items()}
size_mib={k:b/(1024**2) for k,b in size_bytes.items()}
size_kib={k:b/1024 for k,b in size_bytes.items()}

# ---------- Helpers ----------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def tex_escape(s):
    return str(s).replace('&','\\&').replace('%','\\%').replace('_','\\_')

def df_to_tex(df, path, float_format=None):
    text=df.to_latex(index=False, escape=True, float_format=float_format, caption=None, label=None)
    Path(path).write_text(text, encoding='utf-8')

# ---------- Task 1: samples + distribution ----------
# Class distribution
counts=manifest.groupby(['class','split']).size().unstack(fill_value=0).reindex(columns=['train','val','test'])
ax=counts.plot(kind='bar', figsize=(9,5))
ax.set_title('DeFungi Class Distribution by Split')
ax.set_xlabel('Class')
ax.set_ylabel('Number of images')
ax.grid(axis='y', alpha=0.25)
savefig(FIG/'task1/class_distribution.png')

# Representative samples from packaged dataset
zip_path=ROOT/'dataset_64x64.zip'
with zipfile.ZipFile(zip_path) as zf:
    names=zf.namelist()
    samples=[]
    for cls in ['H1','H2','H3','H5','H6']:
        cand=sorted([n for n in names if f'/train/{cls}/' in n and n.lower().endswith(('.jpg','.jpeg','.png'))])
        if not cand:
            cand=sorted([n for n in names if f'/{cls}/' in n and n.lower().endswith(('.jpg','.jpeg','.png'))])
        samples.append((cls,cand[0]))
    fig,axs=plt.subplots(1,5,figsize=(12,2.8))
    for ax,(cls,name) in zip(axs,samples):
        with zf.open(name) as f:
            im=Image.open(f).convert('RGB')
            ax.imshow(im)
        ax.set_title(cls)
        ax.axis('off')
    fig.suptitle('Representative 64x64 DeFungi Images', y=1.03)
    plt.tight_layout()
    plt.savefig(FIG/'task1/representative_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

# ---------- Task 3: optimizer curves ----------
res={r['optimizer']:r for r in opt_json['results']}
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for name,r in res.items():
    e=np.arange(1,len(r['history']['train_loss'])+1)
    axs[0].plot(e,r['history']['train_loss'],marker='o',markersize=2.5,label=name)
    axs[1].plot(e,r['history']['validation_loss'],marker='o',markersize=2.5,label=name)
    axs[2].plot(e,np.array(r['history']['validation_accuracy'])*100,marker='o',markersize=2.5,label=name)
axs[0].set_title('Training Loss'); axs[1].set_title('Validation Loss'); axs[2].set_title('Validation Accuracy')
for ax in axs:
    ax.set_xlabel('Epoch'); ax.grid(alpha=0.25)
axs[0].set_ylabel('Cross-entropy loss'); axs[1].set_ylabel('Cross-entropy loss'); axs[2].set_ylabel('Accuracy (%)')
axs[2].legend(fontsize=8)
fig.suptitle('Task 3: Controlled Optimizer Comparison on Model B')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(FIG/'task3/optimizer_comparison_curves.png',dpi=300,bbox_inches='tight')
plt.close()

# ---------- Task 4: custom model curves ----------
e=np.arange(1,21)
fig,axs=plt.subplots(2,2,figsize=(12,8))
axs[0,0].plot(e,ha['train_loss'],label='Model A train'); axs[0,0].plot(e,ha['val_loss'],label='Model A validation')
axs[0,1].plot(e,hb['train_loss'],label='Model B train'); axs[0,1].plot(e,hb['val_loss'],label='Model B validation')
axs[1,0].plot(e,np.array(ha['train_acc'])*100,label='Model A train'); axs[1,0].plot(e,np.array(ha['val_acc'])*100,label='Model A validation')
axs[1,1].plot(e,np.array(hb['train_acc'])*100,label='Model B train'); axs[1,1].plot(e,np.array(hb['val_acc'])*100,label='Model B validation')
for j,title in enumerate(['Model A Loss','Model B Loss']): axs[0,j].set_title(title)
for j,title in enumerate(['Model A Accuracy','Model B Accuracy']): axs[1,j].set_title(title)
for ax in axs.flat:
    ax.set_xlabel('Epoch'); ax.grid(alpha=0.25); ax.legend(fontsize=8)
axs[0,0].set_ylabel('Loss'); axs[0,1].set_ylabel('Loss'); axs[1,0].set_ylabel('Accuracy (%)'); axs[1,1].set_ylabel('Accuracy (%)')
fig.suptitle('Task 4: Custom Model Training and Validation Curves')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(FIG/'task4/custom_model_training_curves.png',dpi=300,bbox_inches='tight')
plt.close()
# Copy confusion matrices with stable final names
for src,dst in [(ROOT/'results/model_a/confusion_matrix.png',FIG/'task4/model_a_confusion_matrix.png'),(ROOT/'results/model_b/confusion_matrix.png',FIG/'task4/model_b_confusion_matrix.png')]:
    dst.write_bytes(src.read_bytes())

# ---------- Task 5: SOTA curves ----------
sm=sota['mobilenet_v2']['history']; ss=sota['shufflenet_v2']['history']; e=np.arange(1,21)
fig,axs=plt.subplots(2,2,figsize=(12,8))
axs[0,0].plot(e,sm['train_loss'],label='Train'); axs[0,0].plot(e,sm['val_loss'],label='Validation'); axs[0,0].set_title('MobileNetV2 Loss')
axs[0,1].plot(e,ss['train_loss'],label='Train'); axs[0,1].plot(e,ss['val_loss'],label='Validation'); axs[0,1].set_title('ShuffleNetV2 Loss')
axs[1,0].plot(e,np.array(sm['train_acc'])*100,label='Train'); axs[1,0].plot(e,np.array(sm['val_acc'])*100,label='Validation'); axs[1,0].set_title('MobileNetV2 Accuracy')
axs[1,1].plot(e,np.array(ss['train_acc'])*100,label='Train'); axs[1,1].plot(e,np.array(ss['val_acc'])*100,label='Validation'); axs[1,1].set_title('ShuffleNetV2 Accuracy')
for ax in axs.flat:
    ax.set_xlabel('Epoch'); ax.grid(alpha=0.25); ax.legend(fontsize=8)
axs[0,0].set_ylabel('Loss'); axs[0,1].set_ylabel('Loss'); axs[1,0].set_ylabel('Accuracy (%)'); axs[1,1].set_ylabel('Accuracy (%)')
fig.suptitle('Task 5: Fine-Tuning Curves for Lightweight Pre-Trained Models')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(FIG/'task5/sota_training_curves.png',dpi=300,bbox_inches='tight')
plt.close()
for src,dst in [(ROOT/'results/mobilenet_v2/confusion_matrix.png',FIG/'task5/mobilenet_v2_confusion_matrix.png'),(ROOT/'results/shufflenet_v2/confusion_matrix.png',FIG/'task5/shufflenet_v2_confusion_matrix.png')]:
    dst.write_bytes(src.read_bytes())

# ---------- Tables ----------
# Task 3 table
opt_table=opt.copy()
for c in ['best_validation_accuracy','final_validation_accuracy','test_accuracy','macro_precision','macro_recall','macro_f1']:
    opt_table[c]=opt_table[c]*100 if 'accuracy' in c else opt_table[c]
opt_table=opt_table.rename(columns={
 'optimizer':'Optimizer','learning_rate':'Learning Rate','momentum':'Momentum','best_epoch':'Best Epoch',
 'best_validation_accuracy':'Best Val Accuracy (%)','test_accuracy':'Test Accuracy (%)',
 'macro_precision':'Macro Precision','macro_recall':'Macro Recall','macro_f1':'Macro F1',
 'average_epoch_seconds':'Avg Epoch Time (s)'})
opt_cols=['Optimizer','Learning Rate','Momentum','Best Epoch','Best Val Accuracy (%)','Test Accuracy (%)','Macro Precision','Macro Recall','Macro F1','Avg Epoch Time (s)']
opt_table=opt_table[opt_cols]
opt_table.to_csv(TAB/'optimizer_comparison.csv',index=False,float_format='%.4f')
df_to_tex(opt_table,TAB/'optimizer_comparison.tex',float_format='%.4f')

# Task 4 custom table
pa=int(profiles.loc[profiles.model=='Model A','total_parameters'].iloc[0]); pb=int(profiles.loc[profiles.model=='Model B','total_parameters'].iloc[0])
custom=pd.DataFrame([
 {'Metric':'Total parameters','Model A':f'{pa:,}','Model B':f'{pb:,}'},
 {'Metric':'Actual checkpoint size','Model A':f'{size_kib["Model A"]:.1f} KiB','Model B':f'{size_kib["Model B"]:.1f} KiB'},
 {'Metric':'FP32 parameter storage','Model A':f'{pa*4/1024:.1f} KiB','Model B':f'{pb*4/1024:.1f} KiB'},
 {'Metric':'Average training time / epoch','Model A':'29.44 s','Model B':'21.71 s'},
 {'Metric':'Test accuracy','Model A':f'{metrics["Model A"]["test_accuracy"]:.2f}%','Model B':f'{metrics["Model B"]["test_accuracy"]:.2f}%'},
 {'Metric':'Macro precision','Model A':f'{metrics["Model A"]["macro_precision"]:.4f}','Model B':f'{metrics["Model B"]["macro_precision"]:.4f}'},
 {'Metric':'Macro recall','Model A':f'{metrics["Model A"]["macro_recall"]:.4f}','Model B':f'{metrics["Model B"]["macro_recall"]:.4f}'},
 {'Metric':'Macro F1','Model A':f'{metrics["Model A"]["macro_f1"]:.4f}','Model B':f'{metrics["Model B"]["macro_f1"]:.4f}'},
 {'Metric':'Calculated MACs (Conv/Linear only)','Model A':f'{profiles.loc[profiles.model=="Model A","million_macs"].iloc[0]:.2f} M','Model B':f'{profiles.loc[profiles.model=="Model B","million_macs"].iloc[0]:.2f} M'},
])
custom.to_csv(TAB/'model_a_vs_b.csv',index=False)
df_to_tex(custom,TAB/'model_a_vs_b.tex')

# Task5 SOTA table
sota_table=pd.DataFrame([
 {'Model':'MobileNetV2','Parameters':2230277,'Checkpoint Size (MiB)':size_mib['MobileNetV2'],'Test Accuracy (%)':86.31,'Macro F1':0.8805,'Avg Epoch Time (s)':40.90,'MACs (M)':24.454912},
 {'Model':'ShuffleNetV2 1.0x','Parameters':1258729,'Checkpoint Size (MiB)':size_mib['ShuffleNetV2 1.0x'],'Test Accuracy (%)':83.76,'Macro F1':0.8590,'Avg Epoch Time (s)':20.59,'MACs (M)':11.750752},
])
sota_table.to_csv(TAB/'sota_comparison.csv',index=False,float_format='%.4f')
df_to_tex(sota_table,TAB/'sota_comparison.tex',float_format='%.4f')

# Final comparison all four
train_times={'Model A':29.44,'Model B':21.71,'MobileNetV2':40.90,'ShuffleNetV2 1.0x':20.59}
final_rows=[]
for model in ['Model A','Model B','ShuffleNetV2 1.0x','MobileNetV2']:
    pr=profiles[profiles.model==model].iloc[0]
    met=metrics[model]
    final_rows.append({
      'Model':model,'Test Accuracy (%)':met['test_accuracy'],'Macro Precision':met['macro_precision'],'Macro Recall':met['macro_recall'],'Macro F1':met['macro_f1'],
      'Total Parameters':int(pr.total_parameters),'Checkpoint Size (MiB)':size_mib[model],'FP32 Weights (MiB)':int(pr.total_parameters)*4/(1024**2),
      'MACs (M)':pr.million_macs,'FLOPs (M; 2/MAC)':pr.million_flops,'Avg Epoch Time (s)':train_times[model]
    })
final=pd.DataFrame(final_rows)
final.to_csv(TAB/'final_model_comparison.csv',index=False,float_format='%.4f')
df_to_tex(final,TAB/'final_model_comparison.tex',float_format='%.4f')

# ---------- Task 6 figures ----------
main=['Model B','ShuffleNetV2 1.0x','MobileNetV2']
# accuracy
vals=[metrics[m]['test_accuracy'] for m in main]
plt.figure(figsize=(7.5,4.8)); plt.bar(main,vals); plt.ylabel('Test accuracy (%)'); plt.title('Final Test Accuracy Comparison'); plt.ylim(0,100); plt.grid(axis='y',alpha=.2)
for i,v in enumerate(vals): plt.text(i,v+1,f'{v:.2f}%',ha='center',fontsize=9)
savefig(FIG/'task6/test_accuracy_comparison.png')
# params log
vals=[int(profiles.loc[profiles.model==m,'total_parameters'].iloc[0]) for m in main]
plt.figure(figsize=(7.5,4.8)); plt.bar(main,vals); plt.yscale('log'); plt.ylabel('Trainable parameters (log scale)'); plt.title('Parameter Count Comparison'); plt.grid(axis='y',alpha=.2)
for i,v in enumerate(vals): plt.text(i,v*1.15,f'{v:,}',ha='center',fontsize=9)
savefig(FIG/'task6/parameter_count_comparison.png')
# size log
vals=[size_mib[m] for m in main]
plt.figure(figsize=(7.5,4.8)); plt.bar(main,vals); plt.yscale('log'); plt.ylabel('Checkpoint size (MiB, log scale)'); plt.title('Checkpoint Size Comparison'); plt.grid(axis='y',alpha=.2)
for i,v in enumerate(vals): plt.text(i,v*1.15,f'{v:.3f}',ha='center',fontsize=9)
savefig(FIG/'task6/model_size_comparison.png')
# MACs
vals=[float(profiles.loc[profiles.model==m,'million_macs'].iloc[0]) for m in main]
plt.figure(figsize=(7.5,4.8)); plt.bar(main,vals); plt.ylabel('MACs per inference (millions)'); plt.title('Calculated Computational Cost'); plt.grid(axis='y',alpha=.2)
for i,v in enumerate(vals): plt.text(i,v+.5,f'{v:.2f}M',ha='center',fontsize=9)
savefig(FIG/'task6/mac_comparison.png')
# Scatter size vs acc all 4 + Pareto
allm=['Model A','Model B','ShuffleNetV2 1.0x','MobileNetV2']
x=[size_mib[m] for m in allm]; y=[metrics[m]['test_accuracy'] for m in allm]
plt.figure(figsize=(8,5.3)); plt.scatter(x,y,s=70)
for xx,yy,m in zip(x,y,allm): plt.annotate(m,(xx,yy),xytext=(6,6),textcoords='offset points',fontsize=9)
# Pareto frontier: increasing resource x, retain increasing max acc
pts=sorted(zip(x,y,allm)); frontier=[]; best=-1
for xx,yy,m in pts:
    if yy>best:
        frontier.append((xx,yy,m)); best=yy
plt.plot([p[0] for p in frontier],[p[1] for p in frontier],linestyle='--',linewidth=1.5,label='Pareto frontier')
plt.xscale('log'); plt.xlabel('Checkpoint size (MiB, log scale)'); plt.ylabel('Test accuracy (%)'); plt.title('Accuracy vs Checkpoint Size'); plt.grid(alpha=.2); plt.legend()
savefig(FIG/'task6/accuracy_vs_model_size.png')
# Scatter MAC vs acc
x=[float(profiles.loc[profiles.model==m,'million_macs'].iloc[0]) for m in allm]; y=[metrics[m]['test_accuracy'] for m in allm]
plt.figure(figsize=(8,5.3)); plt.scatter(x,y,s=70)
for xx,yy,m in zip(x,y,allm): plt.annotate(m,(xx,yy),xytext=(6,6),textcoords='offset points',fontsize=9)
pts=sorted(zip(x,y,allm)); frontier=[]; best=-1
for xx,yy,m in pts:
    if yy>best:
        frontier.append((xx,yy,m)); best=yy
plt.plot([p[0] for p in frontier],[p[1] for p in frontier],linestyle='--',linewidth=1.5,label='Pareto frontier')
plt.xlabel('MACs per inference (millions)'); plt.ylabel('Test accuracy (%)'); plt.title('Accuracy vs Calculated MACs'); plt.grid(alpha=.2); plt.legend()
savefig(FIG/'task6/accuracy_vs_macs.png')

# Three-metric normalized radar intentionally omitted: not requested, and accuracy-memory-compute is clearer as tables/scatters.

# Comparison calculations for report
b=final[final.Model=='Model B'].iloc[0]; sh=final[final.Model=='ShuffleNetV2 1.0x'].iloc[0]; mb=final[final.Model=='MobileNetV2'].iloc[0]
calc={
 'model_b_param_reduction_vs_a_pct':100*(1-pb/pa),
 'model_b_param_factor_vs_a':pa/pb,
 'mobilenet_accuracy_gain_pp':mb['Test Accuracy (%)']-b['Test Accuracy (%)'],
 'shufflenet_accuracy_gain_pp':sh['Test Accuracy (%)']-b['Test Accuracy (%)'],
 'mobilenet_param_factor_vs_b':mb['Total Parameters']/b['Total Parameters'],
 'shufflenet_param_factor_vs_b':sh['Total Parameters']/b['Total Parameters'],
 'mobilenet_mac_factor_vs_b':mb['MACs (M)']/b['MACs (M)'],
 'shufflenet_mac_factor_vs_b':sh['MACs (M)']/b['MACs (M)'],
 'mobilenet_size_factor_vs_b':mb['Checkpoint Size (MiB)']/b['Checkpoint Size (MiB)'],
 'shufflenet_size_factor_vs_b':sh['Checkpoint Size (MiB)']/b['Checkpoint Size (MiB)'],
}
(ROOT/'tables/derived_tradeoffs.json').write_text(json.dumps(calc,indent=2),encoding='utf-8')
print('Generated figures and tables.')
