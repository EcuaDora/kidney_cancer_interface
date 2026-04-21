"""
Survival plots: forest, Kaplan-Meier по риск-группам, td-AUC over time.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sksurv.metrics import cumulative_dynamic_auc

with open('survival_models.pkl', 'rb') as f:
    st = pickle.load(f)

models = st['models']
scaler = st['scaler']
feature_cols = st['feature_cols']
df_tr = st['df_tr']; df_te = st['df_te']
y_tr = st['y_tr']; y_te = st['y_te']
X_tr = df_tr[feature_cols].values.astype(float)
X_te = df_te[feature_cols].values.astype(float)
X_tr_s = scaler.transform(X_tr); X_te_s = scaler.transform(X_te)

ci = pd.read_csv('survival_cindex_ci.csv')
delta = pd.read_csv('survival_delta_cindex.csv')

# ============================================================================
# 1. FOREST PLOT: C-index с CI + ΔC-index vs IMDC
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Левый: C-index с CI
ax = axes[0]
order = ['IMDC-Cox','MSKCC-Cox','Cox (Ridge)','CoxNet (ElasticNet)',
         'Random Survival Forest','Gradient Boosted Survival']
ci['ord'] = ci['model'].map({m:i for i,m in enumerate(order)})
ci_sorted = ci.sort_values('ord', ascending=False).reset_index(drop=True)

y_pos = np.arange(len(ci_sorted))
colors = ['#888780' if m in ['IMDC-Cox','MSKCC-Cox'] else '#534AB7' for m in ci_sorted['model']]

ax.errorbar(ci_sorted['c_mean'], y_pos,
            xerr=[ci_sorted['c_mean']-ci_sorted['c_lo'], ci_sorted['c_hi']-ci_sorted['c_mean']],
            fmt='o', markersize=0, ecolor='gray', elinewidth=1.5, capsize=4)
for i, c in enumerate(colors):
    ax.scatter([ci_sorted['c_mean'].iloc[i]], [i], color=c, s=110, zorder=5)

ax.set_yticks(y_pos); ax.set_yticklabels(ci_sorted['model'], fontsize=10)
ax.set_xlabel("Harrell's C-index", fontsize=11)
ax.set_title('Survival models — C-index (test, 95% CI)', fontsize=12)
ax.grid(alpha=0.25, axis='x')
ax.set_xlim(0.62, 0.84)
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

# Правый: ΔC-index vs IMDC-Cox
ax = axes[1]
delta['ord'] = delta['model'].map({m:i for i,m in enumerate(order)})
delta_sorted = delta.sort_values('delta', ascending=True).reset_index(drop=True)
y_pos = np.arange(len(delta_sorted))

ax.errorbar(delta_sorted['delta'], y_pos,
            xerr=[delta_sorted['delta']-delta_sorted['lo'], delta_sorted['hi']-delta_sorted['delta']],
            fmt='s', markersize=9, color='#534AB7', ecolor='#534AB7',
            elinewidth=1.8, capsize=4)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_yticks(y_pos); ax.set_yticklabels(delta_sorted['model'], fontsize=10)
ax.set_xlabel('ΔC-index (model − IMDC-Cox)', fontsize=11)
ax.set_title('Paired bootstrap ΔC-index vs IMDC', fontsize=12)
ax.grid(alpha=0.25, axis='x')
ax.set_xlim(-0.01, 0.14)

for i, (_, row) in enumerate(delta_sorted.iterrows()):
    p = row['p']
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    ax.text(0.13, i, sig, fontsize=10, va='center', ha='right')

plt.tight_layout()
plt.savefig('survival_forest.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. KAPLAN-MEIER по risk-группам от лучшей модели (Cox Ridge)
# ============================================================================
cox = models['Cox (Ridge)']
risk_tr = cox.predict(X_tr_s)
risk_te = cox.predict(X_te_s)

# Разделение по терцилям риска (из train)
t1, t2 = np.percentile(risk_tr, [33.33, 66.67])
def to_group(r):
    if r <= t1: return 'Low risk'
    if r <= t2: return 'Intermediate risk'
    return 'High risk'

group_te = np.array([to_group(r) for r in risk_te])
print(f"\nRisk groups on test: Low={np.sum(group_te=='Low risk')}, "
      f"Int={np.sum(group_te=='Intermediate risk')}, High={np.sum(group_te=='High risk')}")

# Для сравнения: IMDC-группы на test (1/2/3)
group_imdc = df_te['imdc'].map({1:'Low risk', 2:'Intermediate risk', 3:'High risk'}).values

# Два KM-plots рядом
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
colors_km = {'Low risk':'#1D9E75', 'Intermediate risk':'#EF9F27', 'High risk':'#E24B4A'}

# KM по Cox-risk группам
ax = axes[0]
for g in ['Low risk','Intermediate risk','High risk']:
    mask = group_te == g
    if mask.sum() == 0: continue
    kmf = KaplanMeierFitter()
    kmf.fit(df_te.loc[mask, 'time_months'].values,
            event_observed=df_te.loc[mask, 'event'].values, label=f'{g} (n={mask.sum()})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors_km[g], linewidth=2)

ax.set_title('Kaplan-Meier by Cox-predicted risk tertiles', fontsize=12)
ax.set_xlabel('Time since metastatic progression (months)', fontsize=11)
ax.set_ylabel('Survival probability', fontsize=11)
ax.set_xlim(0, 120); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)

# Логранк-тест
lr_cox = multivariate_logrank_test(
    df_te['time_months'].values, group_te, df_te['event'].values
)
ax.text(0.05, 0.05,
        f'Log-rank p = {lr_cox.p_value:.1e}\nχ² = {lr_cox.test_statistic:.1f}',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
ax.legend(loc='upper right', fontsize=9)

# KM по IMDC-группам
ax = axes[1]
for g in ['Low risk','Intermediate risk','High risk']:
    mask = group_imdc == g
    if mask.sum() == 0: continue
    kmf = KaplanMeierFitter()
    kmf.fit(df_te.loc[mask, 'time_months'].values,
            event_observed=df_te.loc[mask, 'event'].values, label=f'{g} (n={mask.sum()})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors_km[g], linewidth=2)

ax.set_title('Kaplan-Meier by IMDC-score groups', fontsize=12)
ax.set_xlabel('Time since metastatic progression (months)', fontsize=11)
ax.set_xlim(0, 120); ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)

lr_imdc = multivariate_logrank_test(
    df_te['time_months'].values, group_imdc, df_te['event'].values
)
ax.text(0.05, 0.05,
        f'Log-rank p = {lr_imdc.p_value:.1e}\nχ² = {lr_imdc.test_statistic:.1f}',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('survival_km_groups.png', dpi=150, bbox_inches='tight')
plt.close()

# Статистика: медианы выживания по группам
print("\n=== Median survival by risk group (test) ===")
for name, groups in [('Cox (Ridge)', group_te), ('IMDC', group_imdc)]:
    print(f"\n{name}:")
    for g in ['Low risk','Intermediate risk','High risk']:
        mask = groups == g
        if mask.sum() == 0: continue
        kmf = KaplanMeierFitter()
        kmf.fit(df_te.loc[mask, 'time_months'].values,
                df_te.loc[mask, 'event'].values)
        med = kmf.median_survival_time_
        print(f"  {g:<20} n={mask.sum():<4} median = {med:.1f} mo")

print(f"\nCox log-rank χ² = {lr_cox.test_statistic:.1f}, p = {lr_cox.p_value:.1e}")
print(f"IMDC log-rank χ² = {lr_imdc.test_statistic:.1f}, p = {lr_imdc.p_value:.1e}")
