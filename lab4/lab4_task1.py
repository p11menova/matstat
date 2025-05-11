import pandas as pd
import numpy as np
from scipy.stats import t, f

# Загрузка данных
df = pd.read_csv("kc_house_data.csv")

# Выбор переменных
X_raw = df[['sqft_living', 'sqft_lot', 'sqft_above']]
y = df['price'].to_numpy().reshape(-1, 1)

# Формирование матрицы X с единицами
X = np.hstack((np.ones((X_raw.shape[0], 1)), X_raw.to_numpy()))

# Оценка коэффициентов: b = (X^T X)^-1 X^T y
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
beta_hat = XtX_inv @ Xty

# Предсказания и остатки
y_pred = X @ beta_hat
residuals = y - y_pred

# Остаточная дисперсия
n, k = X.shape
rss = np.sum(residuals ** 2)
sigma2 = rss / (n - k)

# Стандартные ошибки и доверительные интервалы
var_b = sigma2 * XtX_inv
se_b = np.sqrt(np.diag(var_b))
t_crit = t.ppf(0.975, df=n - k)
conf_ints = np.column_stack((beta_hat.flatten() - t_crit * se_b,
                             beta_hat.flatten() + t_crit * se_b))

# R^2
tss = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - rss / tss

# Проверка гипотез
# H0: b1 = 0, b2 = 0 одновременно (sqft_living и sqft_above)
R = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
q = np.array([[0], [0]])
Rb = R @ beta_hat
middle = np.linalg.inv(R @ XtX_inv @ R.T)
F_stat = ((Rb - q).T @ middle @ (Rb - q)) / R.shape[0] / sigma2
p_value = 1 - f.cdf(F_stat, dfn=R.shape[0], dfd=n - k)

# Вывод
print("коэффициенты линейной модели:")
for name, coef, interval in zip(['intercept', 'sqft_living', 'sqft_lot', 'sqft_above'], beta_hat.flatten(), conf_ints):
    print(f"{name:15}: {coef:10.2f}  95%-довер. интерв.: [{interval[0]:.2f}, {interval[1]:.2f}]")

# Автоматический вывод по модели
names = ['intercept', 'sqft_living', 'sqft_lot', 'sqft_above']
coef_rounded = np.round(beta_hat.flatten(), 2)
print()
print(f"полученная линейная модель: y^ = {coef_rounded[0]} + ({coef_rounded[1]})*sqft_living + ({coef_rounded[2]})*sqft_lot + ({coef_rounded[3]})*sqft_above\n")

print(f"• при увеличении жилой площади на 1 кв. фут, цена в среднем увеличивается на {coef_rounded[1]:.2f} у.е.")
print(f"• при увеличении площади участка на 1 кв. фут, цена в среднем изменяется на {coef_rounded[2]:.2f} у.е.")
print(f"• при увеличении площади над землёй на 1 кв. фут, цена в среднем изменяется на {coef_rounded[3]:.2f} у.е.")
print("\nИТОГО:")
print(f"- гипотеза 'Чем больше жилая площадь, тем больше цена' {'подтверждается' if beta_hat[1][0] > 0 and (2 * (1 - t.cdf(abs(beta_hat[1][0]/se_b[1]), df=n-k))) < 0.05 else 'Не подтверждается'}")
print(f"- гипотеза 'Цена зависит от площади участка' {'подтверждается' if (2 * (1 - t.cdf(abs(beta_hat[2][0]/se_b[2]), df=n-k)) < 0.05) else 'Не подтверждается'}")

# TSS: Total Sum of Squares — общая сумма квадратов отклонений от среднего - насколько сильно разбросаны реальные значения
tss = np.sum((y - np.mean(y)) ** 2)

# RSS: Residual Sum of Squares — сумма квадратов остатков - то, что модель не объяснила (ошибки)
rss = np.sum((y - y_pred) ** 2)

# R²: какая доля дисперсии объясняется моделью
r_squared = 1 - (rss / tss)

# Альтернативно: можно распечатать части по отдельности
print("\nрасчет R²:")
print(f"TSS (общая дисперсия): {tss:.2e}")
print(f"RSS (необъяснённая дисперсия): {rss:.2e}")
print(f"R² = 1 - RSS / TSS = 1 - {rss:.2e} / {tss:.2e} = {r_squared:.4f}")

print(f"коэффициент детерминации R² = {r_squared:.4f}")
print("модель объясняет часть разброса, однако возможны улучшения точности предсказания")
# 📊 Вывод по F-статистике (гипотеза H0: b1 = b3 = 0)
print("\nпроверка гипотезы: H₀: коэффициенты при sqft_living и sqft_above равны нулю")

print(f"F-статистика = {F_stat[0][0]:.2f}")
print(f"p-value = {p_value[0][0]:.4f}")

if p_value[0][0] < 0.05:
    print("отвергаем H₀ на уровне значимости 5%: хотя бы один из коэффициентов значим.")
else:
    print("недостаточно оснований отвергнуть H₀: переменные могут быть незначимы.")


import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'DejaVu Sans'  # чтобы кириллица отобразилась

# 1. Реальные значения vs Предсказания
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.5, label='наблюдения', color="pink")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', color="lime",label='идеальное совпадение')
plt.xlabel("фактическая цена")
plt.ylabel("предсказанная цена")
plt.title("сравнение фактической и предсказанной цены")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("task1_graphic.png")
plt.show()




