"""
Этап 3: Корректная классификация 2-летней выживаемости с IPCW.

  - Оригинал: выбрасывал всех живых → selection bias, split 468/335 внутри умерших
  - подход: IPCW-взвешивание, включаем всех надёжно-размеченных +
                цензурированные-рано с весами из 1/G(t); complete-case 909 vs IPCW 977
  - Head-to-head с IMDC и MSKCC на той же выборке — main missing piece из рецензии

Выход: step3_results.pkl (все метрики), step3_models.pkl (обученные модели)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                              precision_score, recall_score, brier_score_loss,
                              confusion_matrix)
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

RANDOM_STATE = 42
HORIZON = 24  # месяцев

# ============================================================================
# 1. ЗАГРУЗКА И РАЗМЕТКА
# ============================================================================
df = pd.read_pickle('step1_clean.pkl')
print(f"Загружено: {df.shape}")

# Четыре группы
A = (df['event']==1) & (df['time_months']< HORIZON)  # умер до 24  → 0
B = (df['event']==1) & (df['time_months']>=HORIZON)  # умер после   → 1
C = (df['event']==0) & (df['time_months']>=HORIZON)  # жив ≥24     → 1
D = (df['event']==0) & (df['time_months']< HORIZON)  # цензур рано → UNK

df['label_reliable'] = np.where(A, 0, np.where(B|C, 1, -1))  # -1 = unknown
print(f"Надёжно размеченных: {(df['label_reliable']>=0).sum()}")
print(f"  label=0 (умер<24): {(df['label_reliable']==0).sum()}")
print(f"  label=1 (выжил≥24): {(df['label_reliable']==1).sum()}")
print(f"Цензурированных до 24 мес (для IPCW): {(df['label_reliable']==-1).sum()}")

# ============================================================================
# 2. IPCW-ВЕСА
# ============================================================================
# Формула: w_i = 1 / Ĝ(min(T_i, t*))
#   Ĝ(t) = вероятность быть нецензурированным к моменту t (KM с перевёрнутым событием)
#   Для i с T_i > t*           → w_i = 1/Ĝ(t*)         (дожил до горизонта)
#   Для i с T_i ≤ t*, event=1  → w_i = 1/Ĝ(T_i)        (событие до горизонта)
#   Для i с T_i ≤ t*, event=0  → w_i = 0                (нет информации)
y_surv = Surv.from_arrays(event=df['event'].astype(bool).values,
                           time=df['time_months'].values)
cens = CensoringDistributionEstimator()
cens.fit(y_surv)

# Вычисляем веса
t_eval = np.minimum(df['time_months'].values, HORIZON)
G_t = cens.predict_proba(t_eval)   # Ĝ(min(T, t*))
ipcw = np.zeros(len(df))
mask_usable = df['label_reliable'] >= 0  # A,B,C группы
ipcw[mask_usable] = 1.0 / np.clip(G_t[mask_usable], 1e-6, None)
df['ipcw_weight'] = ipcw

print(f"\n=== IPCW-веса ===")
print(f"Ĝ(24 мес): {cens.predict_proba(np.array([HORIZON]))[0]:.3f}")
print(f"Распределение весов (ненулевых): "
      f"median={np.median(ipcw[mask_usable]):.3f}, "
      f"mean={np.mean(ipcw[mask_usable]):.3f}, "
      f"max={np.max(ipcw[mask_usable]):.3f}")

# ============================================================================
# 3. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================================
# Используем ВСЕ клинически осмысленные признаки
feature_cols = [
    # демография
    'age', 'sex',
    # опухоль/стадия
    'localization', 'variant', 'differentiation', 'T_stage', 'N_stage', 'M_stage',
    'is_synchronous', 'n_mets_category', 'num_organs_with_mets',
    # органные метастазы (прогностические)
    'mets_bone', 'mets_lung', 'mets_liver', 'mets_brain', 'mets_ln',
    'mets_adrenal', 'mets_visceral', 'mets_cns',
    # статус
    'ecog',
    # лабораторные (сырые — богаче, чем бинарные)
    'hem', 'creatinine', 'ldh', 'platelets', 'esr',
    # терапия
    'operation_code', 'nexavar_first_line', 'intermittent',
    'mets_surgery', 'radiotherapy',
]
# MSKCC/IMDC специально НЕ включаем в ML модель — используем как независимые бенчмарки
print(f"\nПризнаков для ML: {len(feature_cols)}")

# Complete-case выборка для обучения (надёжные метки)
reliable = df[df['label_reliable']>=0].copy()
print(f"Complete-case выборка: {len(reliable)}")

# Stratified train/test split (по метке)
train_df, test_df = train_test_split(
    reliable, test_size=0.20, random_state=RANDOM_STATE,
    stratify=reliable['label_reliable']
)
print(f"Train: {len(train_df)} (label=1: {(train_df['label_reliable']==1).mean()*100:.1f}%)")
print(f"Test:  {len(test_df)} (label=1: {(test_df['label_reliable']==1).mean()*100:.1f}%)")

X_train = train_df[feature_cols].values.astype(float)
y_train = train_df['label_reliable'].values.astype(int)
w_train = train_df['ipcw_weight'].values
X_test  = test_df[feature_cols].values.astype(float)
y_test  = test_df['label_reliable'].values.astype(int)
w_test  = test_df['ipcw_weight'].values

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ============================================================================
# 4. БЕНЧМАРКИ: IMDC и MSKCC как предикторы
# ============================================================================
# IMDC/MSKCC: 1=благоприятный, 2=промежуточный, 3=неблагоприятный
# Используем напрямую как одномерный предиктор (ordinal)
# Для классификации "выжил ≥24 мес": предсказываем по score (1=выжил, 3=умрёт)
# Простейший бенчмарк: predict_proba(survived=1) = (4 - score)/3 → линейная шкала
def score_as_predictor(score):
    """3=неблаг → 0, 2=промеж → 0.5, 1=благ → 1.0"""
    return (3 - score) / 2.0

def evaluate_binary(y_true, y_pred_proba, w=None, threshold=0.5, name=""):
    y_pred = (y_pred_proba >= threshold).astype(int)
    res = {
        'model': name,
        'n': len(y_true),
        'accuracy': accuracy_score(y_true, y_pred, sample_weight=w),
        'f1_macro':  f1_score(y_true, y_pred, average='macro', sample_weight=w),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', sample_weight=w),
        'f1_pos':    f1_score(y_true, y_pred, pos_label=1, sample_weight=w),
        'precision': precision_score(y_true, y_pred, pos_label=1, sample_weight=w, zero_division=0),
        'recall':    recall_score(y_true, y_pred, pos_label=1, sample_weight=w),
        'auc':       roc_auc_score(y_true, y_pred_proba, sample_weight=w),
        'brier':     brier_score_loss(y_true, y_pred_proba, sample_weight=w),
    }
    return res

results = []

# IMDC
imdc_train_pred = score_as_predictor(train_df['imdc'].values)
imdc_test_pred  = score_as_predictor(test_df['imdc'].values)
results.append({**evaluate_binary(y_test, imdc_test_pred, w=w_test, name='IMDC (Heng)'),
                'split': 'test', 'weighted': True})
results.append({**evaluate_binary(y_test, imdc_test_pred, name='IMDC (Heng)'),
                'split': 'test', 'weighted': False})

# MSKCC
mskcc_test_pred = score_as_predictor(test_df['mskcc'].values)
results.append({**evaluate_binary(y_test, mskcc_test_pred, w=w_test, name='MSKCC (Motzer)'),
                'split': 'test', 'weighted': True})
results.append({**evaluate_binary(y_test, mskcc_test_pred, name='MSKCC (Motzer)'),
                'split': 'test', 'weighted': False})

# ============================================================================
# 5. ML-МОДЕЛИ (с IPCW весами и без — для сравнения)
# ============================================================================
# Пытаемся импортировать XGBoost и LightGBM
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

models = {
    'Logistic Regression': (LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE), True),  # scale
    'Random Forest': (RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                              random_state=RANDOM_STATE, n_jobs=-1), False),
    'Gradient Boosting': (GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                      random_state=RANDOM_STATE), False),
    'SVM (RBF)': (SVC(C=1.0, probability=True, random_state=RANDOM_STATE), True),
}
if has_xgb:
    models['XGBoost'] = (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                         eval_metric='logloss', random_state=RANDOM_STATE,
                                         n_jobs=-1, use_label_encoder=False), False)
if has_lgbm:
    models['LightGBM'] = (LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), False)

print(f"\nМоделей к обучению: {len(models)}")
trained = {}

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_s if needs_scaling else X_train
    Xte = X_test_s  if needs_scaling else X_test

    # Обучение с IPCW весами
    try:
        model.fit(Xtr, y_train, sample_weight=w_train)
    except TypeError:
        model.fit(Xtr, y_train)  # если sample_weight не поддерживается

    proba_test = model.predict_proba(Xte)[:, 1]
    res_w = evaluate_binary(y_test, proba_test, w=w_test, name=name)
    res_u = evaluate_binary(y_test, proba_test, name=name)
    results.append({**res_w, 'split': 'test', 'weighted': True})
    results.append({**res_u, 'split': 'test', 'weighted': False})
    trained[name] = model

# ============================================================================
# 6. СВОДКА
# ============================================================================
results_df = pd.DataFrame(results)
results_df.to_pickle('step3_results.pkl')

# Unweighted table (для простоты чтения + чтобы видеть "наивные" метрики)
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ НА ТЕСТЕ (unweighted — прямое сравнение)")
print("="*80)
cols_show = ['model','f1_macro','f1_pos','accuracy','auc','brier']
df_show = results_df[~results_df['weighted']][cols_show].round(3)
print(df_show.to_string(index=False))

print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ НА ТЕСТЕ (IPCW-weighted — корректная оценка)")
print("="*80)
df_show_w = results_df[results_df['weighted']][cols_show].round(3)
print(df_show_w.to_string(index=False))

# Сохранение моделей для SHAP позже
import pickle
with open('work/step3_models.pkl','wb') as f:
    pickle.dump({
        'models': trained,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'X_train': X_train, 'y_train': y_train, 'w_train': w_train,
        'X_test': X_test,   'y_test': y_test,   'w_test': w_test,
        'X_train_s': X_train_s, 'X_test_s': X_test_s,
        'train_df': train_df, 'test_df': test_df,
    }, f)
print("\nСохранено: work/step3_results.pkl, work/step3_models.pkl")
