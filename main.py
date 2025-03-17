"""
1. В файле iris.csv' (здесь и далее ссылкы кликабельны) представлены данные о параметрах различных
экземплярах цветка ириса.
Какой вид в датасете представлен больше всего, какой - меньше?
Рассчитайте выборочное среднее,
выборочную дисперсию,
 выборочную медиану и
 выборочную квантиль порядка 2/5 для суммарной площади (более точно - оценки площади) чашелистика и лепестка
 всей совокупности и отдельно для каждого вида.
Построить график эмпирической функции распределения, гистограмму и
 box-plot суммарной площади чашелистика и лепестка для всей совокупности и каждого вида.
"""
import math
from random import sample

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm, gamma, expon, uniform

irises_species = dict()  # мапа для разделения по видам
irises = []  # просто в целом список всех ирисов


def read_data(filename):
	global irises_species, irises
	with open(filename) as f:
		data = f.readlines()
		for line in data[1:]:
			sl, sw, pl, pw, s = line.strip().split(",")
			iris = {"sepal_length": float(sl),
			        "sepal_width": float(sw),
			        "petal_length": float(pl),
			        "petal_width": float(pw),
			        "specie": str(s).replace('"', '')
			        }
			irises.append(iris)
			if str(s).replace('"', '') not in irises_species.keys():
				irises_species[str(s).replace('"', '')] = [iris]
			else:
				irises_species[str(s).replace('"', '')].append(iris)


def print_max_min_specie():
	global irises_species
	r = dict(sorted(irises_species.items(), key=lambda x: len(x[1])))

	max_amount_specie = next(iter(r))
	min_amount_specie = next(iter(reversed(r)))

	print(f"наиболее частый вид, встречаемый в датасете: {max_amount_specie}, "
	      f"\n\tколичество цветков такого вида: {len(irises_species[max_amount_specie])}\n")
	print(f"наименее частый вид, встречаемый в датасете: {min_amount_specie}, "
	      f"\n\tколичество цветков такого вида: {len(irises_species[min_amount_specie])}\n")


def get_middle_value(irises, length_key, width_key):
	return round(sum([i[length_key] * i[width_key] for i in irises]) / len(irises), 4)


def get_sample_variance(irises, length_key, width_key):
	if len(irises) < 2:
		return 0  # Дисперсия не определяется для одной точки

	mean_value = get_middle_value(irises, length_key, width_key)
	variance = sum((i[length_key] * i[width_key] - mean_value) ** 2 for i in irises) / (len(irises) - 1)

	return round(variance, 4)


def get_median(irises, length_key, width_key):
	a = list(sorted([i[length_key] * i[width_key] for i in irises]))
	n = len(a)
	if n % 2 != 0: return str(a[int(n / 2)])
	return str(round(a[int(n / 2 - 1)], 4)) + " " + str(round(a[int(n / 2)], 4))


def get_sample_quantile(irises, length_key, width_key, p):
	areas = list(sorted([i[length_key] * i[width_key] for i in irises]))
	n = len(areas)

	index = p * (n - 1)  # вычисляем индекс квантиля
	lower = int(index)  # нижний индекс
	upper = min(lower + 1, n - 1)  # верхний индекс (чтобы не выйти за границы массива)

	fraction = index - lower  # дробная часть индекса (для интерполяции)

	return round(areas[lower] + (areas[upper] - areas[lower]) * fraction, 4)


def empirical_cdf(data, title, specie):
	"""
	Строит эмпирическую функцию распределения (ЭФР) для заданных данных.
	:param data: список или numpy-массив с данными
	"""
	hist, edges = np.histogram(data, bins=len(data))
	Y = hist.cumsum()
	for i in range(len(Y)):
		plt.plot([edges[i], edges[i + 1]], [Y[i], Y[i]], c="MediumVioletRed")
	plt.xlabel('Значение')
	plt.ylabel('F(x)')
	plt.title(f'ЭФР {title} {specie}')
	plt.grid()
	plt.legend()
	filename = "efr_sepal_squares" if 'чашелистика' in title else "efr_petal_squares"
	scope = "_whole" if not specie else f"_{specie}"
	plt.savefig(filename + scope + '.png')
	plt.close()


def custom_boxplot(data, title, specie):
	"""
	Строит box-plot с учетом заданного квантиля для усов.
	:param data: список или numpy-массив с данными
	"""
	plt.boxplot(data)
	for i, d in enumerate(data):
		plt.scatter(1, d, c="blue", alpha=0.5)
	plt.xlabel('Данные')
	plt.ylabel('Значение')
	plt.title(f'Box-plot для {title} {specie}')
	plt.grid()
	filename = "boxplot_sepal_squares" if 'чашелистика' in title else "boxplot_petal_squares"
	scope = "_whole" if not specie else f"_{specie}"
	plt.savefig(filename + scope + '.png')
	plt.close()


def draw_histogram(iris_areas, specie=''):
	plt.figure(figsize=(12, 6))

	print(list(sorted(iris_areas)))

	gist_color = 'hotpink'
	gist_title = "гистограмма распределения суммарной площади чашелистиков и лепестков"

	plt.hist(iris_areas, bins=(1 + math.ceil(math.log(len(iris_areas), 2))), color=gist_color, alpha=0.35,
	         edgecolor=gist_color, density=True)

	mu, sigma = np.mean(iris_areas), np.std(iris_areas)
	shape_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(iris_areas, floc=0)
	shape_gamma, loc_gamma, scale_gamma = gamma.fit(iris_areas)
	loc_exp, scale_exp = expon.fit(iris_areas)
	loc_uniform, scale_uniform = uniform.fit(iris_areas)

	x = np.linspace(min(iris_areas), max(iris_areas), 100)

	plt.plot(x, norm.pdf(x, mu, sigma), linewidth=1, color='darkcyan',
	         label=f'нормальное распределение (μ={mu:.2f}, σ={sigma:.2f})')
	plt.plot(x, lognorm.pdf(x, shape_lognorm, loc_lognorm, scale_lognorm), color='MediumVioletRed', linewidth=1,
	         label='логнормальное')
	plt.plot(x, expon.pdf(x, loc_exp, scale_exp), color='DeepSkyBlue', linewidth=1, label='показательное')
	plt.plot(x, uniform.pdf(x, loc_uniform, scale_uniform), 'hotpink', linewidth=1, label='равномерное')

	plt.title(
		f"{gist_title} для вида {specie} " if specie else f"{gist_title} для всей выборки")
	plt.xlabel('суммарная площадь чашелистика и лепестка')
	plt.ylabel('плотность вероятности')
	# plt.xscale('log')
	plt.legend()

	plt.grid(True, color="silver", alpha=0.5)
	filename = "irises_squares"
	scope = "_whole" if not specie else f"_{specie}"
	plt.savefig(filename + scope + '.png')
	plt.close()


if __name__ == "__main__":
	read_data("iris.csv")

	# print("sepal_areas")
	# print(" ".join(list(sorted([str(round(i["sepal_width"]* i['sepal_length'], 2)) for i in irises] ))))
	# print("petal_areas")
	# print(" ".join(list(sorted([str(round(i["petal_width"]* i['petal_length'], 2)) for i in irises] ))))

	print_max_min_specie()
	print(f"выборочное среднее значение площади чашелистика для всей совокупности: "
	      f"{get_middle_value(irises, 'sepal_length', 'sepal_width')}")
	print(f"выборочное среднее значение площади лепестка для всей совокупности: "
	      f"{get_middle_value(irises, 'petal_length', 'petal_width')}")
	print(f"выборочная дисперсия значения площади чашелистика для всей совокупности: "
	      f"{get_sample_variance(irises, 'sepal_length', 'sepal_width')}")
	print(f"выборочная дисперсия значения площади лепестка для всей совокупности: "
	      f"{get_sample_variance(irises, 'petal_length', 'petal_width')}")
	print(f"выборочная медиана значений площади чашелистика для всей совокупности: "
	      f"{get_median(irises, 'sepal_length', 'sepal_width')}")
	print(f"выборочная медиана значений площади лепестка для всей совокупности: "
	      f"{get_median(irises, 'petal_length', 'petal_width')}")
	print(f"выборочная квантиль порядка 2/5 значений площади чашелистика для всей совокупности: "
	      f"{get_sample_quantile(irises, 'sepal_length', 'sepal_width', 0.4)}")
	print(f"выборочная квантиль порядка 2/5 значений площади лепестка для всей совокупности: "
	      f"{get_sample_quantile(irises, 'petal_length', 'petal_width', 0.4)}")
	sepal_areas = [i['sepal_length'] * i['sepal_width'] for i in irises]
	petal_areas = [i['petal_length'] * i['petal_width'] for i in irises]
	irises_areas = [sepal_areas[j] + petal_areas[j] for j in range(len(sepal_areas))]
	draw_histogram(sepal_areas)
	print()
	for k, v in irises_species.items():
		print(f"для вида: {k}")

		print(f"\t-выборочное среднее площади чашелистика: "
		      f"{get_middle_value(v, 'sepal_length', 'sepal_width')}")
		print(f"\t-выборочное среднее площади лепестка: "
		      f"{get_middle_value(v, 'petal_length', 'petal_width')}")
		print(f"\t-выборочная дисперсия значения площади чашелистика: "
		      f"{get_sample_variance(v, 'sepal_length', 'sepal_length')}")
		print(f"\t-выборочная дисперсия значения площади лепестка: "
		      f"{get_sample_variance(v, 'petal_length', 'petal_length')}")
		print(f"\t-выборочная медиана значений площади чашелистика: "
		      f"{get_median(v, 'sepal_length', 'sepal_width')}")
		print(f"\t-выборочная медиана значений площади лепестка: "
		      f"{get_median(v, 'petal_length', 'petal_width')}")
		print(f"\t-выборочная квантиль порядка 2/5 значений площади чашелистика: "
		      f"{get_sample_quantile(v, 'sepal_length', 'sepal_width', 0.4)}")
		print(f"\t-выборочная квантиль порядка 2/5 значений площади лепестка: "
		      f"{get_sample_quantile(v, 'petal_length', 'petal_width', 0.4)}")
		print()
		print(f"\t-выборочное среднее площади чашелистика: "
		      f"{get_middle_value(v, 'sepal_length', 'sepal_width')}")
		print(f"\t-выборочное среднее площади лепестка: "
		      f"{get_middle_value(v, 'petal_length', 'petal_width')}")
		print(f"\t-выборочная дисперсия значения площади чашелистика: "
		      f"{get_sample_variance(v, 'sepal_length', 'sepal_length')}")
		print(f"\t-выборочная дисперсия значения площади лепестка: "
		      f"{get_sample_variance(v, 'petal_length', 'petal_length')}")
		print(f"\t-выборочная медиана значений площади чашелистика: "
		      f"{get_median(v, 'sepal_length', 'sepal_width')}")
		print(f"\t-выборочная медиана значений площади лепестка: "
		      f"{get_median(v, 'petal_length', 'petal_width')}")
		print(f"\t-выборочная квантиль порядка 2/5 значений площади чашелистика: "
		      f"{get_sample_quantile(v, 'sepal_length', 'sepal_width', 0.4)}")
		print(f"\t-выборочная квантиль порядка 2/5 значений площади лепестка: "
		      f"{get_sample_quantile(v, 'petal_length', 'petal_width', 0.4)}")
		sepal_areas = [i['sepal_length'] * i['sepal_width'] for i in v]
		petal_areas = [i['petal_length'] * i['petal_width'] for i in v]
		# irises_areas = [sepal_areas[j] + petal_areas[j] for j in range(len(sepal_areas))]
		# draw_histogram(sepal_areas, True, k)
		draw_histogram(irises_areas, k)
		print()

		# Эмпирические функции распределения для каждого вида цветка
		# Функции для площади чашелистика
		empirical_cdf([i['sepal_length'] * i['sepal_width'] for i in v], 'площади чашелистика для вида', k)
		# Функции для площади лепестка
		empirical_cdf([i['petal_length'] * i['petal_width'] for i in v], 'площади лепестка для вида ', k)

		# boxplot'ы для каждого вида цветка с квантилем 2/5 = 0.4
		# boxplot для площади чашелистика
		custom_boxplot([i['sepal_length'] * i['sepal_width'] for i in v], 'площади чашелистика для вида', k)
		# boxplot для площади лепестка
		custom_boxplot([i['petal_length'] * i['petal_width'] for i in v], 'площади лепестка для вида ', k)

	empirical_cdf(irises_areas, "суммарные площади цветков", "")

	mus = []
	s = sorted(irises_areas)
	for l in range(5, 150):
		new_sample = sample(s, l)

		mu, sigma = np.mean(new_sample), np.std(new_sample)
		mus.append(mu)

	real = np.mean(irises_areas)
	plt.figure(figsize=(10, 5))
	plt.plot(range(5, 150), mus, marker='o', linestyle='-',linewidth=1, color='DarkMagenta', alpha=0.7)
	plt.axhline(y=float(real), color='crimson', linestyle='--', linewidth=1)
	plt.title("График изменения оценки средней площади цветка")
	plt.xlabel("Количество выборки")
	plt.ylabel("Оценка")
	plt.grid()

	filename = "generated_squares_samples_estimates"
	plt.savefig(filename + '.png')
	plt.close()
