

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

import numpy as np
import matplotlib.pyplot as plt


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


def empirical_cdf(data, title):
    """
    Строит эмпирическую функцию распределения (ЭФР) для заданных данных.
    :param data: список или numpy-массив с данными
    """
    hist, edges = np.histogram(data, bins=len(data))
    Y = hist.cumsum()
    for i in range(len(Y)):
        plt.plot([edges[i], edges[i + 1]], [Y[i], Y[i]], c="blue")
    plt.xlabel('Значение')
    plt.ylabel('F(x)')
    plt.title('ЭФР ' + title)
    plt.grid()
    plt.legend()
    plt.show()


def draw_histogram(gist_title, iris_areas, gist_color, sepal=True, specie=''):
    plt.figure(figsize=(8, 6))
    plt.hist(iris_areas, bins=20, edgecolor='black')
    plt.title(f"{gist_title} for {specie}" if specie else f"{gist_title} for whole values")
    plt.xlabel('площадь чашелистика' if sepal else 'площадь лепестка')
    plt.ylabel('частота')
    plt.savefig(gist_title + '.png')
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
    draw_histogram("sepal_squares", sepal_areas, 1)
    draw_histogram("petal_squares", petal_areas, 1, False)

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

        # Эмпирические функции распределения для каждого вида цветка
        # Функции для площади чашелистика
        empirical_cdf([i['sepal_length'] * i['sepal_width'] for i in v], 'площади чашелистика для вида ' + k)
        # Функции для площади лепестка
        empirical_cdf([i['petal_length'] * i['petal_width'] for i in v], 'площади лепестка ' + k)
