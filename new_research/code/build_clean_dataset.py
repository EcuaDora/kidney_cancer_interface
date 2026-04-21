"""
Этап 2: Сборка чистого survival-датасета

Вход:  step0_raw981.pkl (981 x 176, очищено от служебных строк)
Выход: step1_clean.csv

Ключевые решения:
  - Таргет времени: ВПП (время после прогрессирования), подтверждено min/max=0.75/183.57
  - Событие: 1=умер, 0=цензурирован; пациенты с 'неизвестно' (4 шт) удаляются
  - Нет выбрасывания живых! Все цензурированные остаются для корректного survival-анализа
  - 47 пациентов с пропусками в гистологии (Variant/Differentiation/T) сохраняются
    с категорией 'unknown' — терять ~5% выборки нецелесообразно
  - Feature engineering: num_organs_with_mets, num_therapy_lines, органо-специфичные флаги
"""
import pandas as pd
import numpy as np

df = pd.read_pickle('work/step0_raw981.pkl')
print(f"Вход: {df.shape}")

# ============================================================================
# 1. ОТБОР И ПЕРЕИМЕНОВАНИЕ КОЛОНОК
# ============================================================================
col_map = {
    'Номер': 'patient_id',
    'возраст (К 60)': 'age',
    'Пол К': 'sex',                            # 1=муж, 2=жен
    'Локализация': 'localization',             # 1=лев, 2=прав, 3=двуст
    'дата опер': 'operation_date',
    'Кодировка операций': 'operation_code',    # 1-5
    'Вар ПКР 2': 'variant',                    # 1/2, NaN→unknown
    'Дифф 2': 'differentiation',               # 1/2, NaN→unknown
    'Т 2': 'T_stage',                          # 1/2, NaN→unknown
    'N': 'N_stage',                            # 0/1/2
    'M': 'M_stage',                            # 0/1
    'Синх Метах К': 'sync_metach',             # 1=sync, 2=metach
    'Кол-во мтс 1-солитарные\n2- единичные\n3- множественные': 'n_mets_category',
    'Кодировка статуса ECOG': 'ecog',          # 0-3
    'MSKCC Код (благоприятный 1\n2- промежуточный\n3 -Неблагоприятный)': 'mskcc',
    'Heng Код (благоприятный 1\n2- промежуточный\n3 -Неблагоприятный)': 'imdc',
    'Нексавар': 'nexavar_first_line',          # 0/1
    'Прерывисто': 'intermittent_raw',          # текст, нужна нормализация
    'Хирургия метастазов': 'mets_surgery',     # 0/1
    'Кодировка лучей': 'radiotherapy',         # 0/1
    'Кодировка прогресс': 'progress_code',     # 1-4
    # Лабораторные (сырые значения)
    'Hem': 'hem', 'CF': 'creatinine', 'ЛДГ': 'ldh', 'Tr': 'platelets', 'SOE': 'esr',
    'Neu': 'neutrophils', 'Ca O': 'calcium',
    # Лабораторные (бинаризованные врачами)
    'L-Hem': 'hem_flag', 'L-CF': 'creatinine_flag', 'L-LDG': 'ldh_flag',
    'L-Tr': 'platelets_flag', 'L-SOE': 'esr_flag',
    # Метастазы по органам
    'Кости': 'mets_bone', 'Легкие': 'mets_lung', 'Почка': 'mets_kidney',
    'Надпочечник': 'mets_adrenal', 'Печень': 'mets_liver', 'Л/у': 'mets_ln',
    'Гол мозг': 'mets_brain', 'Яичник': 'mets_ovary', 'Другие': 'mets_other',
    # Outcome
    'ОВ': 'os_months',          # общая выживаемость от операции
    'ВБП': 'pfs_months',        # время до прогрессирования
    'ВПП': 'time_months',       # *** ОСНОВНОЙ ТАРГЕТ: от прогрессирования до исхода ***
    'Жив_ умер К': 'status_raw',
}

# Колонки линий терапии
for i, rus in enumerate(['мес 2-й линии','мес 3-й линии','мес 4-й линии','мес 5-й линии','мес 6-й линии'], start=2):
    col_map[rus] = f'line{i}_months'
col_map['Месяцы таргет суммарно'] = 'total_targeted_months'

# Берём только нужные колонки
missing = [c for c in col_map if c not in df.columns]
assert not missing, f"Отсутствуют: {missing}"
clean = df[list(col_map.keys())].rename(columns=col_map).copy()
print(f"После отбора колонок: {clean.shape}")

# ============================================================================
# 2. OUTCOME: event, фильтрация неизвестных
# ============================================================================
clean['event'] = clean['status_raw'].map({1.0: 0, 2.0: 1})
n_before = len(clean)
clean = clean[clean['event'].notna()].copy()
clean['event'] = clean['event'].astype(int)
print(f"Удалено пациентов с неизвестным статусом: {n_before - len(clean)}")
print(f"После фильтра исхода: {len(clean)} (ожидалось 977)")

# Проверка таргета времени
assert clean['time_months'].notna().all(), "time_months содержит NaN"
assert clean['time_months'].min() > 0, "time_months содержит неположительные значения"

# ============================================================================
# 3. НОРМАЛИЗАЦИЯ КАТЕГОРИАЛЬНЫХ
# ============================================================================
# Пол: 1=муж, 2=жен → male=1, female=0 (оставляем понятное кодирование)
clean['sex'] = clean['sex'].map({1.0: 1, 2.0: 0}).astype('Int8')  # 1=M, 0=F

# Synchronous/Metachronous: 1=sync, 2=metach → is_synchronous
clean['is_synchronous'] = (clean['sync_metach'] == 1.0).astype('Int8')
clean = clean.drop(columns=['sync_metach'])

# Intermittent: 'да'/'нет'/'Нет'/'нет ' → 0/1
clean['intermittent'] = (
    clean['intermittent_raw']
    .astype(str).str.strip().str.lower()
    .map({'да': 1, 'нет': 0})
    .astype('Int8')
)
clean = clean.drop(columns=['intermittent_raw'])
assert clean['intermittent'].notna().all(), "intermittent содержит NaN"

# Гистологические признаки: 47 NaN → категория 'unknown' (=0)
for col in ['variant', 'differentiation', 'T_stage']:
    clean[col] = clean[col].fillna(0).astype(int)  # 0=unknown, 1, 2

# Приведение остальных int-колонок
int_cols = ['localization', 'operation_code', 'N_stage', 'M_stage',
            'n_mets_category', 'ecog', 'mskcc', 'imdc', 'nexavar_first_line',
            'mets_surgery', 'radiotherapy', 'progress_code']
for c in int_cols:
    clean[c] = clean[c].astype(int)

# Метастазы по органам: 0/1, уже int
mets_cols = ['mets_bone','mets_lung','mets_kidney','mets_adrenal','mets_liver',
             'mets_ln','mets_brain','mets_ovary','mets_other']
for c in mets_cols:
    clean[c] = clean[c].fillna(0).astype('Int8')

# Лабораторные флаги
for c in ['hem_flag','creatinine_flag','ldh_flag','platelets_flag','esr_flag']:
    clean[c] = clean[c].astype(int)

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
# Количество органов с метастазами
clean['num_organs_with_mets'] = clean[mets_cols].sum(axis=1).astype(int)

# Количество линий терапии (по наличию длительности > 0)
line_cols = [f'line{i}_months' for i in range(2, 7)]
clean['num_therapy_lines'] = 1 + clean[line_cols].notna().sum(axis=1).astype(int)
# (1-я линия есть у всех, считаем дополнительные)

# Has any line >= 2, 3
clean['has_second_line'] = (clean['num_therapy_lines'] >= 2).astype('Int8')
clean['has_third_plus_line'] = (clean['num_therapy_lines'] >= 3).astype('Int8')

# Ключевые органные метастазы (прогностически важные)
clean['mets_visceral'] = ((clean['mets_liver']==1) | (clean['mets_lung']==1)).astype('Int8')
clean['mets_cns'] = (clean['mets_brain']==1).astype('Int8')
clean['mets_bone_present'] = (clean['mets_bone']==1).astype('Int8')

# ============================================================================
# 5. СБОРКА ФИНАЛЬНОЙ СХЕМЫ
# ============================================================================
final_cols = [
    # ID и outcome
    'patient_id', 'time_months', 'event',
    # Вспомогательные outcome (для survival analysis)
    'os_months', 'pfs_months',
    # Демография
    'age', 'sex',
    # Опухоль
    'localization', 'variant', 'differentiation', 'T_stage', 'N_stage', 'M_stage',
    'is_synchronous', 'n_mets_category', 'num_organs_with_mets',
    # Органные метастазы
    'mets_bone','mets_lung','mets_kidney','mets_adrenal','mets_liver',
    'mets_ln','mets_brain','mets_ovary','mets_other',
    'mets_visceral','mets_cns','mets_bone_present',
    # Статус пациента
    'ecog', 'mskcc', 'imdc',
    # Лабы
    'hem','creatinine','ldh','platelets','esr',
    'hem_flag','creatinine_flag','ldh_flag','platelets_flag','esr_flag',
    # Терапия
    'operation_code','nephrectomy_done' if False else 'operation_code',  # placeholder
    'nexavar_first_line','intermittent','mets_surgery','radiotherapy',
    'progress_code','num_therapy_lines','has_second_line','has_third_plus_line',
    'total_targeted_months',
]
# убираем дубликаты с сохранением порядка
seen = set(); final_cols = [c for c in final_cols if not (c in seen or seen.add(c))]

clean_final = clean[final_cols].copy()
print(f"\nФинальный датасет: {clean_final.shape}")
print(f"Колонки: {list(clean_final.columns)}")

# Сохранение
clean_final.to_pickle('work/step1_clean.pkl')
clean_final.to_csv('work/step1_clean.csv', index=False)
print("\nСохранено: work/step1_clean.pkl, work/step1_clean.csv")

# ============================================================================
# 6. САНИТИ-ЧЕК
# ============================================================================
print("\n" + "="*60)
print("САНИТИ-ЧЕК")
print("="*60)
print(f"N пациентов:              {len(clean_final)}")
print(f"События (умер):           {clean_final['event'].sum()} ({clean_final['event'].mean()*100:.1f}%)")
print(f"Цензурировано (жив):      {(clean_final['event']==0).sum()} ({(1-clean_final['event'].mean())*100:.1f}%)")
print(f"Медиана time_months:      {clean_final['time_months'].median():.1f}")
print(f"Медиана time_months умерших:  {clean_final[clean_final['event']==1]['time_months'].median():.1f}")
print(f"Медиана time_months цензурир: {clean_final[clean_final['event']==0]['time_months'].median():.1f}")
print(f"\nПропусков по колонкам (только те, где есть):")
miss = clean_final.isna().sum()
print(miss[miss > 0].to_string() if (miss > 0).any() else "  (нет пропусков)")

print(f"\num_organs_with_mets: min={clean_final['num_organs_with_mets'].min()}, "
      f"median={clean_final['num_organs_with_mets'].median()}, max={clean_final['num_organs_with_mets'].max()}")
print(f"num_therapy_lines: {clean_final['num_therapy_lines'].value_counts().sort_index().to_dict()}")
