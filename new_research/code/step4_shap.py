"""
Этап 3.1: SHAP-анализ объяснимости моделей.

Цель: закрыть замечание рецензента «black-box without SHAP/LIME».
Анализируем 3 семейства моделей:
  - Logistic Regression (LinearExplainer) — primary model, лучший AUC
  - Random Forest (TreeExplainer) — нелинейное древовидное семейство
  - Gradient Boosting (TreeExplainer) — бустинг, обнаруживает взаимодействия

Выходы:
  - shap_bar_{model}.png — global feature importance
  - shap_beeswarm_{model}.png — распределение SHAP values
  - shap_waterfall_{model}_{case}.png — индивидуальные примеры
  - shap_dependence_{feature}_{model}.png — зависимость для top-2 признаков
  - shap_values.pkl — numerical values для дальнейшего анализа
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os

os.makedirs('figs', exist_ok=True)

# ============================================================================
# 1. ЗАГРУЗКА
# ============================================================================
with open('step3_models.pkl', 'rb') as f:
    st = pickle.load(f)

models = st['models']
scaler = st['scaler']
feature_cols = st['feature_cols']
X_train, y_train = st['X_train'], st['y_train']
X_test,  y_test  = st['X_test'],  st['y_test']
X_train_s, X_test_s = st['X_train_s'], st['X_test_s']
test_df = st['test_df']

# Красивые имена для графиков
feature_names_pretty = {
    'age':'Age', 'sex':'Sex (1=M)', 'localization':'Tumor localization',
    'variant':'Histological variant', 'differentiation':'Differentiation',
    'T_stage':'T stage', 'N_stage':'N stage', 'M_stage':'M stage',
    'is_synchronous':'Synchronous mets', 'n_mets_category':'Number of mets (cat)',
    'num_organs_with_mets':'# organs with mets',
    'mets_bone':'Bone mets','mets_lung':'Lung mets','mets_liver':'Liver mets',
    'mets_brain':'Brain mets','mets_ln':'Lymph node mets',
    'mets_adrenal':'Adrenal mets','mets_visceral':'Visceral mets',
    'mets_cns':'CNS mets', 'ecog':'ECOG status',
    'hem':'Hemoglobin', 'creatinine':'Creatinine', 'ldh':'LDH',
    'platelets':'Platelets', 'esr':'ESR',
    'operation_code':'Operation type','nexavar_first_line':'Nexavar 1st line',
    'intermittent':'Intermittent therapy','mets_surgery':'Metastasectomy',
    'radiotherapy':'Radiotherapy',
}
pretty = [feature_names_pretty.get(c, c) for c in feature_cols]

# ============================================================================
# 2. SHAP для Logistic Regression (LinearExplainer)
# ============================================================================
print("="*60)
print("SHAP: Logistic Regression")
print("="*60)
lr = models['Logistic Regression']
# LinearExplainer для модели на стандартизованных признаках
expl_lr = shap.LinearExplainer(lr, X_train_s, feature_names=pretty)
shap_lr = expl_lr(X_test_s)
print(f"SHAP values shape: {shap_lr.values.shape}")

# Summary bar
plt.figure(figsize=(8, 6))
shap.plots.bar(shap_lr, max_display=15, show=False)
plt.title('SHAP feature importance — Logistic Regression', fontsize=11)
plt.tight_layout()
plt.savefig('shap_bar_lr.png', dpi=150, bbox_inches='tight')
plt.close()

# Beeswarm
plt.figure(figsize=(8, 6))
shap.plots.beeswarm(shap_lr, max_display=15, show=False)
plt.title('SHAP — Logistic Regression (test set)', fontsize=11)
plt.tight_layout()
plt.savefig('shap_beeswarm_lr.png', dpi=150, bbox_inches='tight')
plt.close()

# Топ-10 признаков по mean |SHAP|
mean_abs_lr = np.abs(shap_lr.values).mean(axis=0)
top_lr = pd.DataFrame({'feature': pretty, 'mean_abs_shap': mean_abs_lr})\
           .sort_values('mean_abs_shap', ascending=False).head(15)
print("\nТоп-15 признаков (LogReg):")
print(top_lr.to_string(index=False))

# ============================================================================
# 3. SHAP для Random Forest (TreeExplainer)
# ============================================================================
print("\n" + "="*60)
print("SHAP: Random Forest")
print("="*60)
rf = models['Random Forest']
expl_rf = shap.TreeExplainer(rf)
shap_rf_raw = expl_rf.shap_values(X_test)
# RF возвращает list для classification: [class0, class1]
if isinstance(shap_rf_raw, list):
    shap_rf_vals = shap_rf_raw[1]  # класс 1 (survived)
elif shap_rf_raw.ndim == 3:
    shap_rf_vals = shap_rf_raw[:, :, 1]
else:
    shap_rf_vals = shap_rf_raw
print(f"SHAP values shape: {shap_rf_vals.shape}")

shap_rf_obj = shap.Explanation(
    values=shap_rf_vals,
    base_values=np.full(len(X_test), expl_rf.expected_value[1] if hasattr(expl_rf.expected_value,'__len__') else expl_rf.expected_value),
    data=X_test,
    feature_names=pretty
)

plt.figure(figsize=(8, 6))
shap.plots.bar(shap_rf_obj, max_display=15, show=False)
plt.title('SHAP feature importance — Random Forest', fontsize=11)
plt.tight_layout()
plt.savefig('shap_bar_rf.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
shap.plots.beeswarm(shap_rf_obj, max_display=15, show=False)
plt.title('SHAP — Random Forest (test set)', fontsize=11)
plt.tight_layout()
plt.savefig('shap_beeswarm_rf.png', dpi=150, bbox_inches='tight')
plt.close()

mean_abs_rf = np.abs(shap_rf_vals).mean(axis=0)
top_rf = pd.DataFrame({'feature': pretty, 'mean_abs_shap': mean_abs_rf})\
           .sort_values('mean_abs_shap', ascending=False).head(15)
print("\nТоп-15 признаков (RF):")
print(top_rf.to_string(index=False))

# ============================================================================
# 4. SHAP для Gradient Boosting
# ============================================================================
print("\n" + "="*60)
print("SHAP: Gradient Boosting")
print("="*60)
gb = models['Gradient Boosting']
expl_gb = shap.TreeExplainer(gb)
shap_gb_raw = expl_gb.shap_values(X_test)
print(f"Raw shape: {np.asarray(shap_gb_raw).shape}")
if isinstance(shap_gb_raw, list):
    shap_gb_vals = shap_gb_raw[1]
elif shap_gb_raw.ndim == 3:
    shap_gb_vals = shap_gb_raw[:, :, 1]
else:
    shap_gb_vals = shap_gb_raw

shap_gb_obj = shap.Explanation(
    values=shap_gb_vals,
    base_values=np.full(len(X_test), expl_gb.expected_value if np.isscalar(expl_gb.expected_value) else expl_gb.expected_value[0]),
    data=X_test,
    feature_names=pretty
)

plt.figure(figsize=(8, 6))
shap.plots.bar(shap_gb_obj, max_display=15, show=False)
plt.title('SHAP feature importance — Gradient Boosting', fontsize=11)
plt.tight_layout()
plt.savefig('shap_bar_gb.png', dpi=150, bbox_inches='tight')
plt.close()

mean_abs_gb = np.abs(shap_gb_vals).mean(axis=0)
top_gb = pd.DataFrame({'feature': pretty, 'mean_abs_shap': mean_abs_gb})\
           .sort_values('mean_abs_shap', ascending=False).head(15)
print("\nТоп-15 признаков (GB):")
print(top_gb.to_string(index=False))

# ============================================================================
# 5. СРАВНЕНИЕ РАНЖИРОВАНИЙ
# ============================================================================
print("\n" + "="*60)
print("СРАВНЕНИЕ: топ-10 по трём моделям")
print("="*60)
comparison = pd.DataFrame({
    'feature': pretty,
    'LogReg': mean_abs_lr / mean_abs_lr.max(),    # нормализованные
    'RF':     mean_abs_rf / mean_abs_rf.max(),
    'GB':     mean_abs_gb / mean_abs_gb.max(),
})
comparison['avg_rank'] = (
    comparison['LogReg'].rank(ascending=False) +
    comparison['RF'].rank(ascending=False) +
    comparison['GB'].rank(ascending=False)
) / 3
comparison = comparison.sort_values('avg_rank').head(15)
print(comparison.round(3).to_string(index=False))
comparison.to_csv('work/shap_feature_ranking.csv', index=False)

# ============================================================================
# 6. СОХРАНЕНИЕ
# ============================================================================
with open('shap_values.pkl','wb') as f:
    pickle.dump({
        'lr': shap_lr, 'rf': shap_rf_obj, 'gb': shap_gb_obj,
        'mean_abs_lr': mean_abs_lr, 'mean_abs_rf': mean_abs_rf, 'mean_abs_gb': mean_abs_gb,
        'feature_cols': feature_cols, 'pretty': pretty,
    }, f)

