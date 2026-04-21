"""
Этап 3.3: Устойчивость оценок через CV и bootstrap.

Что делается:
  1. 5-fold stratified CV на train (727) → mean±SD для F1_macro, AUC, Brier
  2. Bootstrap 2000 итераций на test (182) → 95% CI для тех же метрик
  3. Два новых честных бенчмарка:
       - IMDC-LR: логистика, обученная ТОЛЬКО на imdc-score как ordinal
       - MSKCC-LR: то же самое для mskcc
     Исправляем замечание, что линейная проекция (3-score)/2
     несправедлива к IMDC. Теперь IMDC/MSKCC на равных с ML-моделями.
  4. Bootstrap-test на разность AUC: ΔAUC(модель − IMDC) с 95% CI и p-value
"""
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, accuracy_score
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
try:
    from lightgbm import LGBMClassifier
    has_lgbm = True
except ImportError:
    has_lgbm = False

RANDOM_STATE = 42
N_BOOT = 2000

# ============================================================================
# 1. ЗАГРУЗКА
# ============================================================================
with open('step3_models.pkl','rb') as f:
    st = pickle.load(f)

feature_cols = st['feature_cols']
train_df = st['train_df']
test_df  = st['test_df']
X_train, y_train = st['X_train'], st['y_train']
X_test,  y_test  = st['X_test'],  st['y_test']

# ============================================================================
# 2. ФАБРИКА МОДЕЛЕЙ
# ============================================================================
def make_models():
    m = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE), True),
        'Random Forest': (RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                  random_state=RANDOM_STATE, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                          random_state=RANDOM_STATE), False),
        'SVM (RBF)': (SVC(C=1.0, probability=True, random_state=RANDOM_STATE), True),
    }
    if has_xgb:
        m['XGBoost'] = (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        eval_metric='logloss', random_state=RANDOM_STATE,
                                        n_jobs=-1), False)
    if has_lgbm:
        m['LightGBM'] = (LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), False)
    return m

# ============================================================================
# 3. 5-FOLD STRATIFIED CV на train
# ============================================================================
print("="*70)
print("5-FOLD CV (train, n=727)")
print("="*70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []

# ML-модели
for name, (_, needs_s) in make_models().items():
    fold_metrics = {'f1_macro':[], 'f1_pos':[], 'auc':[], 'brier':[], 'acc':[]}
    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if needs_s:
            sc = StandardScaler().fit(X_tr)
            X_tr = sc.transform(X_tr); X_va = sc.transform(X_va)
        model, _ = make_models()[name]
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        yhat = (p >= 0.5).astype(int)
        fold_metrics['f1_macro'].append(f1_score(y_va, yhat, average='macro'))
        fold_metrics['f1_pos'].append(f1_score(y_va, yhat, pos_label=1))
        fold_metrics['auc'].append(roc_auc_score(y_va, p))
        fold_metrics['brier'].append(brier_score_loss(y_va, p))
        fold_metrics['acc'].append(accuracy_score(y_va, yhat))
    row = {'model': name}
    for k, v in fold_metrics.items():
        row[f'{k}_mean'] = np.mean(v)
        row[f'{k}_std']  = np.std(v)
    cv_rows.append(row)

# Честные IMDC/MSKCC: обучаем LogReg на одном признаке
for score_name, score_col in [('IMDC-LR', 'imdc'), ('MSKCC-LR', 'mskcc')]:
    s = train_df[score_col].values.reshape(-1, 1).astype(float)
    fold_metrics = {'f1_macro':[], 'f1_pos':[], 'auc':[], 'brier':[], 'acc':[]}
    for tr_idx, va_idx in skf.split(s, y_train):
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(s[tr_idx], y_train[tr_idx])
        p = lr.predict_proba(s[va_idx])[:, 1]
        yhat = (p >= 0.5).astype(int)
        fold_metrics['f1_macro'].append(f1_score(y_train[va_idx], yhat, average='macro'))
        fold_metrics['f1_pos'].append(f1_score(y_train[va_idx], yhat, pos_label=1))
        fold_metrics['auc'].append(roc_auc_score(y_train[va_idx], p))
        fold_metrics['brier'].append(brier_score_loss(y_train[va_idx], p))
        fold_metrics['acc'].append(accuracy_score(y_train[va_idx], yhat))
    row = {'model': score_name}
    for k, v in fold_metrics.items():
        row[f'{k}_mean'] = np.mean(v); row[f'{k}_std'] = np.std(v)
    cv_rows.append(row)

cv_df = pd.DataFrame(cv_rows)
# Упорядочим
order = ['IMDC-LR','MSKCC-LR','Logistic Regression','SVM (RBF)','Random Forest',
         'Gradient Boosting','XGBoost','LightGBM']
cv_df['ord'] = cv_df['model'].map({m:i for i,m in enumerate(order)})
cv_df = cv_df.sort_values('ord').drop(columns=['ord']).reset_index(drop=True)

print("\nCV результаты (mean ± SD по 5 фолдам):")
print(f"{'model':<22}{'F1 macro':>18}{'AUC':>18}{'Brier':>18}")
for _, r in cv_df.iterrows():
    f1  = f"{r['f1_macro_mean']:.3f} ± {r['f1_macro_std']:.3f}"
    auc = f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}"
    br  = f"{r['brier_mean']:.3f} ± {r['brier_std']:.3f}"
    print(f"{r['model']:<22}{f1:>18}{auc:>18}{br:>18}")
cv_df.to_csv('work/cv_results.csv', index=False)

# ============================================================================
# 4. BOOTSTRAP 95% CI на test
# ============================================================================
print("\n" + "="*70)
print(f"BOOTSTRAP {N_BOOT} iter (test, n=182)")
print("="*70)

# Обучаем финальные модели на полном train
trained_final = {}
scaler_final = StandardScaler().fit(X_train)
X_train_s = scaler_final.transform(X_train)
X_test_s  = scaler_final.transform(X_test)

for name, (model, needs_s) in make_models().items():
    Xtr = X_train_s if needs_s else X_train
    model.fit(Xtr, y_train)
    trained_final[name] = (model, needs_s)

# IMDC-LR и MSKCC-LR
imdc_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
imdc_lr.fit(train_df[['imdc']].values.astype(float), y_train)
mskcc_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
mskcc_lr.fit(train_df[['mskcc']].values.astype(float), y_train)

# Собираем все предсказания на test
probas_test = {
    'IMDC-LR':  imdc_lr.predict_proba(test_df[['imdc']].values.astype(float))[:, 1],
    'MSKCC-LR': mskcc_lr.predict_proba(test_df[['mskcc']].values.astype(float))[:, 1],
}
for name, (model, needs_s) in trained_final.items():
    X = X_test_s if needs_s else X_test
    probas_test[name] = model.predict_proba(X)[:, 1]

# Bootstrap
rng = np.random.default_rng(RANDOM_STATE)
n = len(y_test)

def boot_metric(y_true, y_prob, metric_fn, n_boot=N_BOOT):
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = metric_fn(y_true[idx], y_prob[idx])
            values.append(v)
        except Exception:
            pass
    arr = np.array(values)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

def f1m(y, p): return f1_score(y, (p>=0.5).astype(int), average='macro')
def auc_(y, p): return roc_auc_score(y, p)
def brier(y, p): return brier_score_loss(y, p)

boot_rows = []
for name in order:
    if name not in probas_test:
        continue
    p = probas_test[name]
    row = {'model': name}
    m, lo, hi = boot_metric(y_test, p, f1m); row['f1_macro'] = m
    row['f1_macro_lo'], row['f1_macro_hi'] = lo, hi
    m, lo, hi = boot_metric(y_test, p, auc_); row['auc'] = m
    row['auc_lo'], row['auc_hi'] = lo, hi
    m, lo, hi = boot_metric(y_test, p, brier); row['brier'] = m
    row['brier_lo'], row['brier_hi'] = lo, hi
    boot_rows.append(row)

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv('bootstrap_ci.csv', index=False)

print("\nTest-set метрики с 95% CI (bootstrap):")
print(f"{'model':<22}{'F1 macro [95% CI]':>28}{'AUC [95% CI]':>28}{'Brier [95% CI]':>28}")
for _, r in boot_df.iterrows():
    f1s = f"{r['f1_macro']:.3f} [{r['f1_macro_lo']:.3f}, {r['f1_macro_hi']:.3f}]"
    aus = f"{r['auc']:.3f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]"
    brs = f"{r['brier']:.3f} [{r['brier_lo']:.3f}, {r['brier_hi']:.3f}]"
    print(f"{r['model']:<22}{f1s:>28}{aus:>28}{brs:>28}")

# ============================================================================
# 5. ПАРНОЕ СРАВНЕНИЕ AUC: ΔAUC(модель − IMDC-LR)
# ============================================================================
print("\n" + "="*70)
print("PAIRED BOOTSTRAP: ΔAUC (модель − IMDC-LR)")
print("="*70)

# Paired bootstrap: один и тот же resample для обеих моделей
baseline_name = 'IMDC-LR'
p_base = probas_test[baseline_name]

delta_rows = []
for name in order:
    if name == baseline_name or name not in probas_test:
        continue
    p_model = probas_test[name]
    deltas = []
    rng2 = np.random.default_rng(RANDOM_STATE)
    for _ in range(N_BOOT):
        idx = rng2.integers(0, n, n)
        try:
            d = roc_auc_score(y_test[idx], p_model[idx]) - roc_auc_score(y_test[idx], p_base[idx])
            deltas.append(d)
        except Exception:
            pass
    deltas = np.array(deltas)
    delta_rows.append({
        'model': name,
        'delta_auc': deltas.mean(),
        'delta_lo': np.percentile(deltas, 2.5),
        'delta_hi': np.percentile(deltas, 97.5),
        'p_two_sided': 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()),
    })

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv('delta_auc_vs_imdc.csv', index=False)

print(f"\n{'model':<22}{'ΔAUC [95% CI]':>32}{'p-value':>12}{'signif.':>10}")
for _, r in delta_df.iterrows():
    ds = f"{r['delta_auc']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
    sig = '***' if r['p_two_sided']<0.001 else ('**' if r['p_two_sided']<0.01 else ('*' if r['p_two_sided']<0.05 else ''))
    print(f"{r['model']:<22}{ds:>32}{r['p_two_sided']:>12.4f}{sig:>10}")

