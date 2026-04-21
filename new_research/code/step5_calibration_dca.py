"""
Этап 3.2: Calibration curves + Decision Curve Analysis.

Цель: закрыть замечания рецензентов о клинической полезности и калибрации.


Calibration: насколько предсказанные вероятности соответствуют реальной частоте.
DCA (Vickers & Elkin 2006): показывает net clinical benefit при разных порогах
    принятия решения — прямой ответ на вопрос «стоит ли использовать модель в клинике».
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import os

os.makedirs('figs', exist_ok=True)

# ============================================================================
# 1. ЗАГРУЗКА
# ============================================================================
with open('step3_models.pkl','rb') as f:
    st = pickle.load(f)

models = st['models']
X_test, y_test = st['X_test'], st['y_test']
X_test_s = st['X_test_s']
test_df = st['test_df']

# Вспомогательная функция для IMDC/MSKCC как предикторов
def score_as_predictor(score):
    return (3 - score) / 2.0

# ============================================================================
# 2. СБОР ВЕРОЯТНОСТЕЙ ДЛЯ ВСЕХ МОДЕЛЕЙ
# ============================================================================
probas = {
    'IMDC (Heng)':  score_as_predictor(test_df['imdc'].values),
    'MSKCC':        score_as_predictor(test_df['mskcc'].values),
}
for name, model in models.items():
    needs_s = name in ['Logistic Regression', 'SVM (RBF)']
    X = X_test_s if needs_s else X_test
    probas[name] = model.predict_proba(X)[:, 1]

# ============================================================================
# 3. CALIBRATION CURVES
# ============================================================================
# Используем 8 bins, квантильные (равное число наблюдений в каждом bin)
# Для честного сравнения добавляем Brier score и expected calibration error (ECE)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if i == n_bins - 1:
            mask |= (y_prob == bins[i+1])
        if mask.sum() > 0:
            bin_conf = y_prob[mask].mean()
            bin_acc  = y_true[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_conf - bin_acc)
    return ece

# Главный график: 4 модели (IMDC, MSKCC, LogReg, SVM)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Subplot 1: все модели рядом
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Perfect calibration')

colors_map = {
    'IMDC (Heng)':  '#888780',
    'MSKCC':        '#B4B2A9',
    'Logistic Regression': '#534AB7',
    'SVM (RBF)':    '#1D9E75',
    'Random Forest':'#D85A30',
    'Gradient Boosting': '#E24B4A',
    'XGBoost':      '#EF9F27',
    'LightGBM':     '#378ADD',
}

calib_table = []
for name, p in probas.items():
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy='quantile')
    brier = brier_score_loss(y_test, p)
    ece = expected_calibration_error(y_test, p, n_bins=10)
    calib_table.append({'model': name, 'brier': brier, 'ece': ece})
    ls = '-' if name not in ['IMDC (Heng)','MSKCC'] else '--'
    lw = 2.0 if name in ['IMDC (Heng)','Logistic Regression','SVM (RBF)','MSKCC'] else 1.2
    alpha = 1.0 if name in ['IMDC (Heng)','Logistic Regression','SVM (RBF)','MSKCC'] else 0.55
    ax.plot(mean_pred, frac_pos, 'o-', color=colors_map.get(name,'gray'),
            linewidth=lw, linestyle=ls, alpha=alpha, markersize=6,
            label=f'{name} (Brier={brier:.3f})')

ax.set_xlabel('Predicted probability of 2-year survival', fontsize=11)
ax.set_ylabel('Observed frequency', fontsize=11)
ax.set_title('Calibration curves (test set, n=182)', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.25)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

# Subplot 2: гистограмма распределения предсказаний для primary модели
ax = axes[1]
lr_proba = probas['Logistic Regression']
imdc_proba = probas['IMDC (Heng)']
bins = np.linspace(0, 1, 21)
ax.hist(lr_proba[y_test==1], bins=bins, alpha=0.55, color='#534AB7', label='Survived ≥24 mo (LogReg)')
ax.hist(lr_proba[y_test==0], bins=bins, alpha=0.55, color='#E24B4A', label='Died <24 mo (LogReg)')
ax.set_xlabel('Predicted probability of 2-year survival', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Prediction distribution — Logistic Regression', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('calibration.png', dpi=150, bbox_inches='tight')
plt.close()

calib_df = pd.DataFrame(calib_table)
calib_df.to_csv('work/calibration_metrics.csv', index=False)
print("=== CALIBRATION METRICS ===")
print(calib_df.round(4).to_string(index=False))

# ============================================================================
# 4. DECISION CURVE ANALYSIS (Vickers & Elkin 2006)
# ============================================================================
# Net benefit = (TP/N) - (FP/N) * (p_t / (1 - p_t))
# где p_t — пороговая вероятность (risk threshold)
#
# Для задачи "выживет ли пациент 2 года":
# - TP: модель говорит «выживет», действительно выжил (правильная уверенность)
# - FP: модель говорит «выживет», но не выжил (ложная уверенность, может привести
#       к более агрессивному лечению без основания)
# - Сравниваем с «treat-all» и «treat-none» стратегиями

def net_benefit(y_true, y_prob, threshold):
    """Net benefit at threshold p_t."""
    y_pred = (y_prob >= threshold).astype(int)
    N = len(y_true)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    if threshold >= 1.0:
        return 0.0
    return (tp / N) - (fp / N) * (threshold / (1 - threshold))

def net_benefit_all(y_true, threshold):
    """Net benefit of 'treat all' strategy."""
    prev = y_true.mean()
    if threshold >= 1.0:
        return 0.0
    return prev - (1 - prev) * (threshold / (1 - threshold))

thresholds = np.linspace(0.05, 0.95, 91)
dca_data = {'threshold': thresholds}
for name, p in probas.items():
    dca_data[name] = [net_benefit(y_test, p, t) for t in thresholds]
dca_data['Treat all'] = [net_benefit_all(y_test, t) for t in thresholds]
dca_data['Treat none'] = [0.0] * len(thresholds)
dca_df = pd.DataFrame(dca_data)
dca_df.to_csv('dca_data.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(thresholds, dca_df['Treat all'], color='gray', linestyle=':', lw=1.5, label='Treat all')
ax.plot(thresholds, dca_df['Treat none'], color='black', linestyle=':', lw=1.5, label='Treat none')

focus_models = ['IMDC (Heng)', 'MSKCC', 'Logistic Regression', 'SVM (RBF)', 'Random Forest']
for name in focus_models:
    ls = '-' if 'Regression' in name or 'Forest' in name or 'SVM' in name else '--'
    lw = 2.2 if name == 'Logistic Regression' else 1.8
    ax.plot(thresholds, dca_df[name], color=colors_map.get(name),
            linewidth=lw, linestyle=ls, label=name, alpha=0.9)

ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Threshold probability (risk tolerance)', fontsize=11)
ax.set_ylabel('Net benefit', fontsize=11)
ax.set_title('Decision Curve Analysis — 2-year survival prediction', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.25)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(-0.05, 0.65)

plt.tight_layout()
plt.savefig('figs/dca.png', dpi=150, bbox_inches='tight')
plt.close()

# Посчитаем net benefit в клинически осмысленных точках
print("\n=== NET BENEFIT at clinically relevant thresholds ===")
print("(Higher = better; negative = worse than doing nothing)\n")
key_thresholds = [0.20, 0.35, 0.50, 0.65, 0.80]
summary_nb = {'threshold': key_thresholds}
for name in focus_models + ['Treat all', 'Treat none']:
    row = []
    for t in key_thresholds:
        if name == 'Treat all':
            row.append(net_benefit_all(y_test, t))
        elif name == 'Treat none':
            row.append(0.0)
        else:
            row.append(net_benefit(y_test, probas[name], t))
    summary_nb[name] = row
nb_df = pd.DataFrame(summary_nb).set_index('threshold').round(4)
print(nb_df.T.to_string())
nb_df.to_csv('dca_summary.csv')

