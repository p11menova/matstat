import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, chi2_contingency

# Функция для расчёта Cramér's V
def cramers_v(conf_matrix):
    chi2_val, _, _, _ = chi2_contingency(conf_matrix)
    n = conf_matrix.sum().sum()
    r, k = conf_matrix.shape
    return np.sqrt(chi2_val / (n * (min(r, k) - 1)))

# 1. Загрузка и подготовка данных
df = pd.read_csv("fifa_players_stats.csv", sep=";")
df = df[['Nationality', 'Overall']].dropna()

# Чтобы таблица была «тёплой» и не слишком разреженной, оставим
# только топ-5 стран по числу игроков, остальные — в группу "Other"
top_countries = df['Nationality'].value_counts().nlargest(5).index
df['Nat_group'] = df['Nationality'].where(df['Nationality'].isin(top_countries))

# Рейтинг тоже сделаем категориальным — разобьём на три уровня
bins = [0, 60, 75, 100]
labels = ['Low (≤60)', 'Medium (61–75)', 'High (>75)']
df['Rating_cat'] = pd.cut(df['Overall'], bins=bins, labels=labels, right=True)

# 2. Собираем контингентную таблицу
cont_table = pd.crosstab(df['Nat_group'], df['Rating_cat'])
print("Наблюдаемая таблица (counts):\n", cont_table)

# 3. «Ручной» χ²-тест
observed = cont_table.values
row_sums = observed.sum(axis=1, keepdims=True)
col_sums = observed.sum(axis=0, keepdims=True)
total = observed.sum()
expected = row_sums @ col_sums / total

chi2_stat = ((observed - expected) ** 2 / expected).sum()
dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
p_manual = 1 - chi2.cdf(chi2_stat, dof)

print(f"\n=== Ручной χ²-тест ===")
print(f"χ² = {chi2_stat:.2f}, dof = {dof}, p-value = {p_manual:.4f}")
if p_manual < 0.05:
    print("Отвергаем H0: есть зависимость между рейтингом и национальностью")
else:
    print("Нет оснований отвергать H0: рейтинг и национальность, видимо, независимы")

# 4. Готовая реализация из scipy
chi2_stat2, p_scipy, dof2, expected2 = chi2_contingency(cont_table)
print(f"\n=== chi2_contingency из SciPy ===")
print(f"χ² = {chi2_stat2:.2f}, dof = {dof2}, p-value = {p_scipy:.4f}")
if p_scipy < 0.05:
    print("Отвергаем H0")
else:
    print("Нет оснований отвергать H0")

# 5. Расчёт Cramér's V
cramer_v = cramers_v(cont_table.values)
print(f"\n=== Cramér's V ===")
print(f"Cramér's V: {cramer_v:.4f}  (0 — нет связи, 1 — идеально связаны)")
print("Умеренная ассоциация между рейтингом и национальностью")

# 6. Визуализация — «тепловая карта» с тёмными цветами для больших значений
# 6. Визуализация — «тепловая карта» с подписями
plt.figure(figsize=(8, 5))
plt.imshow(cont_table, aspect='auto', cmap='PuBu')  # Обратная палитра
plt.colorbar(label='Count')

# Подписи осей
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(cont_table.index)), cont_table.index)
plt.title("Таблица сопряжённости\nРейтинг и Национальность")
plt.xlabel("Рейтинг")
plt.ylabel("Национальность")

# Добавляем числа в ячейки
for i in range(cont_table.shape[0]):
    for j in range(cont_table.shape[1]):
        count = cont_table.iat[i, j]
        plt.text(j, i, str(count), ha='center', va='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()
