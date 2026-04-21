"""
Этап 3.4: Survival analysis на всей выборке (977 пациентов).

Что это добавляет:
  - Использует ВСЕХ 977 пациентов (включая 178 цензурированных),
    а не только 909 complete-case из классификации
  - Корректно обрабатывает цензурирование (right-censoring, non-informative)
  - Даёт предсказания времени выживания, а не только бинарный исход
  - Закрывает вторую половину критики рецензентов о правильном survival framework

Модели:
  - IMDC-Cox / MSKCC-Cox: Cox на одном предикторе (baseline)
  - Cox: линейный Cox со всеми признаками
  - CoxNet (Lasso): регуляризованный Cox — feature selection
  - Random Survival Forest (RSF): нелинейный ensemble
  - Gradient Boosted Survival (GBS): бустинг с Cox partial likelihood

Метрики:
  - Harrell's C-index
  - Uno's C-index (робастнее к цензурированию)
  - Time-dependent AUC в t={12, 24, 60} месяцев
  - Integrated Brier Score в [0, 60] мес
"""
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                             integrated_brier_score, cumulative_dynamic_auc)

RANDOM_STATE = 42

# ============================================================================
# 1. ЗАГРУЗКА — ВСЕ 977 пациентов
# ============================================================================
df = pd.read_pickle('step1_clean.pkl')
print(f"Загружено: {df.shape}")

feature_cols = [
    'age', 'sex', 'localization', 'variant', 'differentiation',
    'T_stage', 'N_stage', 'M_stage', 'is_synchronous',
    'n_mets_category', 'num_organs_with_mets',
    'mets_bone', 'mets_lung', 'mets_liver', 'mets_brain', 'mets_ln',
    'mets_adrenal', 'mets_visceral', 'mets_cns',
    'ecog',
    'hem', 'creatinine', 'ldh', 'platelets', 'esr',
    'operation_code', 'nexavar_first_line', 'intermittent',
    'mets_surgery', 'radiotherapy',
]
print(f"Признаков: {len(feature_cols)}")

X = df[feature_cols].values.astype(float)
y = Surv.from_arrays(event=df['event'].astype(bool).values,
                      time=df['time_months'].values)

# Stratified по event (чтобы сохранить долю цензурирования)
idx_tr, idx_te = train_test_split(
    np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE,
    stratify=df['event'].values
)
X_tr, X_te = X[idx_tr], X[idx_te]
y_tr, y_te = y[idx_tr], y[idx_te]
df_tr, df_te = df.iloc[idx_tr], df.iloc[idx_te]

print(f"Train: {len(idx_tr)} (event rate: {df_tr['event'].mean()*100:.1f}%)")
print(f"Test:  {len(idx_te)} (event rate: {df_te['event'].mean()*100:.1f}%)")

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

# ============================================================================
# 2. МЕТРИКИ
# ============================================================================
# Harrell C-index: простой, но смещён при высоком цензурировании
# Uno C-index: робастнее, требует IPCW
# Для Uno и td-AUC нужны risk scores и временные точки

def harrell_cindex(y_true, risk_score):
    return concordance_index_censored(
        y_true['event'], y_true['time'], risk_score
    )[0]

def uno_cindex(y_train, y_test, risk_score, tau=60):
    return concordance_index_ipcw(y_train, y_test, risk_score, tau=tau)[0]

# Временные точки для td-AUC и IBS
eval_times = np.array([12, 24, 60])
# IBS требует сетку времён в пределах наблюдаемого диапазона
t_max_test  = df_te['time_months'].max()
t_min_test  = df_te['time_months'].min()
ibs_grid = np.linspace(max(1.0, t_min_test + 0.1), min(60.0, t_max_test - 0.1), 50)

# ============================================================================
# 3. МОДЕЛИ
# ============================================================================
print("\n" + "="*70)
print("TRAINING SURVIVAL MODELS")
print("="*70)

models = {}

# --- IMDC-Cox: Cox на одном признаке ---
imdc_cox = CoxPHSurvivalAnalysis(alpha=0.01)
imdc_cox.fit(df_tr[['imdc']].values.astype(float), y_tr)
models['IMDC-Cox'] = (imdc_cox, df_tr[['imdc']].values.astype(float),
                                  df_te[['imdc']].values.astype(float))

# --- MSKCC-Cox ---
mskcc_cox = CoxPHSurvivalAnalysis(alpha=0.01)
mskcc_cox.fit(df_tr[['mskcc']].values.astype(float), y_tr)
models['MSKCC-Cox'] = (mskcc_cox, df_tr[['mskcc']].values.astype(float),
                                    df_te[['mskcc']].values.astype(float))

# --- Cox (Ridge-регуляризация для стабильности) ---
cox = CoxPHSurvivalAnalysis(alpha=0.1, ties='efron')
cox.fit(X_tr_s, y_tr)
models['Cox (Ridge)'] = (cox, X_tr_s, X_te_s)

# --- CoxNet (Elastic Net, для feature selection) ---
try:
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=10000)
    coxnet.fit(X_tr_s, y_tr)
    # Выбираем оптимальную alpha через концу пути
    best_alpha = coxnet.alphas_[len(coxnet.alphas_) // 2]
    coxnet_final = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[best_alpha], max_iter=10000)
    coxnet_final.fit(X_tr_s, y_tr)
    models['CoxNet (ElasticNet)'] = (coxnet_final, X_tr_s, X_te_s)
    print(f"CoxNet: выбрано alpha={best_alpha:.4f}")
except Exception as e:
    print(f"CoxNet не сошёлся: {e}")

# --- Random Survival Forest ---
rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                            max_features='sqrt', n_jobs=-1,
                            random_state=RANDOM_STATE)
rsf.fit(X_tr, y_tr)
models['Random Survival Forest'] = (rsf, X_tr, X_te)

# --- Gradient Boosted Survival ---
gbs = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                         max_depth=3, random_state=RANDOM_STATE)
gbs.fit(X_tr, y_tr)
models['Gradient Boosted Survival'] = (gbs, X_tr, X_te)

# ============================================================================
# 4. ОЦЕНКА
# ============================================================================
print("\n" + "="*70)
print("EVALUATION (test set, n={})".format(len(idx_te)))
print("="*70)

results = []
for name, (model, Xtr_m, Xte_m) in models.items():
    # Risk score (для Cox моделей — это predict; для RSF/GBS — predict тоже дает risk)
    try:
        risk_te = model.predict(Xte_m)
    except Exception:
        # Для некоторых моделей predict может не работать одинаково
        risk_te = model.predict(Xte_m).ravel()

    # 1. Harrell C-index
    c_harrell = harrell_cindex(y_te, risk_te)

    # 2. Uno C-index (tau=60 мес)
    try:
        c_uno = uno_cindex(y_tr, y_te, risk_te, tau=60)
    except Exception as e:
        c_uno = np.nan

    # 3. Time-dependent AUC и IBS
    try:
        # Нужны survival-функции
        surv_fn_te = model.predict_survival_function(Xte_m)
        # risks на каждый eval_time: 1 - S(t|x)
        risk_at_t = np.array([[1 - fn(t) for t in eval_times] for fn in surv_fn_te])
        td_auc, td_mean = cumulative_dynamic_auc(y_tr, y_te, risk_at_t, eval_times)

        # IBS
        surv_at_grid = np.array([[fn(t) for t in ibs_grid] for fn in surv_fn_te])
        ibs = integrated_brier_score(y_tr, y_te, surv_at_grid, ibs_grid)
    except Exception as e:
        td_auc = [np.nan, np.nan, np.nan]
        td_mean = np.nan
        ibs = np.nan

    row = {
        'model': name,
        'c_harrell': c_harrell,
        'c_uno': c_uno,
        'td_auc_12mo': td_auc[0] if len(td_auc)>0 else np.nan,
        'td_auc_24mo': td_auc[1] if len(td_auc)>1 else np.nan,
        'td_auc_60mo': td_auc[2] if len(td_auc)>2 else np.nan,
        'ibs': ibs,
    }
    results.append(row)

res_df = pd.DataFrame(results)
order = ['IMDC-Cox','MSKCC-Cox','Cox (Ridge)','CoxNet (ElasticNet)',
         'Random Survival Forest','Gradient Boosted Survival']
res_df['ord'] = res_df['model'].map({m:i for i,m in enumerate(order)})
res_df = res_df.sort_values('ord').drop(columns=['ord']).reset_index(drop=True)
res_df.to_csv('survival_results.csv', index=False)

print(f"\n{'model':<28}{'Harrell C':>12}{'Uno C':>12}{'AUC@12mo':>12}{'AUC@24mo':>12}{'AUC@60mo':>12}{'IBS':>12}")
for _, r in res_df.iterrows():
    print(f"{r['model']:<28}{r['c_harrell']:>12.3f}{r['c_uno']:>12.3f}"
          f"{r['td_auc_12mo']:>12.3f}{r['td_auc_24mo']:>12.3f}{r['td_auc_60mo']:>12.3f}{r['ibs']:>12.3f}")

# ============================================================================
# 5. BOOTSTRAP CI для Harrell C-index
# ============================================================================
print("\n" + "="*70)
print("BOOTSTRAP 1000 iter — Harrell C-index CI")
print("="*70)

N_BOOT = 1000
rng = np.random.default_rng(RANDOM_STATE)
n_te = len(idx_te)

ci_rows = []
for name, (model, _, Xte_m) in models.items():
    risk_te = model.predict(Xte_m)
    c_values = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_te, n_te)
        if y_te['event'][idx].sum() < 2:
            continue
        c = concordance_index_censored(
            y_te['event'][idx], y_te['time'][idx], risk_te[idx]
        )[0]
        c_values.append(c)
    c_arr = np.array(c_values)
    ci_rows.append({
        'model': name,
        'c_mean': c_arr.mean(),
        'c_lo': np.percentile(c_arr, 2.5),
        'c_hi': np.percentile(c_arr, 97.5),
    })

ci_df = pd.DataFrame(ci_rows)
ci_df['ord'] = ci_df['model'].map({m:i for i,m in enumerate(order)})
ci_df = ci_df.sort_values('ord').drop(columns=['ord']).reset_index(drop=True)
ci_df.to_csv('survival_cindex_ci.csv', index=False)

print(f"\n{'model':<28}{'C-index [95% CI]':>36}")
for _, r in ci_df.iterrows():
    print(f"{r['model']:<28}{r['c_mean']:.3f} [{r['c_lo']:.3f}, {r['c_hi']:.3f}]".rjust(36+28))

# ============================================================================
# 6. PAIRED BOOTSTRAP: ΔC-index vs IMDC-Cox
# ============================================================================
print("\n" + "="*70)
print("PAIRED BOOTSTRAP: ΔC-index (модель − IMDC-Cox)")
print("="*70)

baseline_risk = models['IMDC-Cox'][0].predict(df_te[['imdc']].values.astype(float))
delta_rows = []
for name, (model, _, Xte_m) in models.items():
    if name == 'IMDC-Cox':
        continue
    risk = model.predict(Xte_m)
    deltas = []
    rng2 = np.random.default_rng(RANDOM_STATE)
    for _ in range(N_BOOT):
        idx = rng2.integers(0, n_te, n_te)
        if y_te['event'][idx].sum() < 2:
            continue
        c_m = concordance_index_censored(y_te['event'][idx], y_te['time'][idx], risk[idx])[0]
        c_b = concordance_index_censored(y_te['event'][idx], y_te['time'][idx], baseline_risk[idx])[0]
        deltas.append(c_m - c_b)
    d_arr = np.array(deltas)
    p = 2 * min((d_arr <= 0).mean(), (d_arr >= 0).mean())
    delta_rows.append({
        'model': name,
        'delta': d_arr.mean(),
        'lo': np.percentile(d_arr, 2.5),
        'hi': np.percentile(d_arr, 97.5),
        'p': p,
    })

delta_df = pd.DataFrame(delta_rows)
delta_df['ord'] = delta_df['model'].map({m:i for i,m in enumerate(order)})
delta_df = delta_df.sort_values('ord').drop(columns=['ord']).reset_index(drop=True)
delta_df.to_csv('survival_delta_cindex.csv', index=False)

print(f"\n{'model':<28}{'ΔC-index [95% CI]':>34}{'p':>10}")
for _, r in delta_df.iterrows():
    s = f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
    sig = '***' if r['p']<0.001 else ('**' if r['p']<0.01 else ('*' if r['p']<0.05 else ''))
    print(f"{r['model']:<28}{s:>34}{r['p']:>8.4f} {sig}")

# Сохранение моделей
with open('survival_models.pkl','wb') as f:
    pickle.dump({
        'models': {name: m[0] for name, m in models.items()},
        'scaler': scaler,
        'feature_cols': feature_cols,
        'idx_tr': idx_tr, 'idx_te': idx_te,
        'y_tr': y_tr, 'y_te': y_te,
        'df_tr': df_tr, 'df_te': df_te,
        'eval_times': eval_times, 'ibs_grid': ibs_grid,
    }, f)

