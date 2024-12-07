import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import yfinance as yf

# Вхідний файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантажте дані котирувань із Yahoo Finance
start_date = "2003-07-03"
end_date = "2007-05-04"

quotes = [
    yf.download(symbol, start=start_date, end=end_date)
    for symbol in symbols
]

# Обчисліть різницю між котируваннями при відкритті та закритті біржі
# Вилучення котирувань, що відповідають відкриттю та закриттю біржі
opening_quotes = np.array([quote['Open'].values for quote in quotes]).astype(np.float)
closing_quotes = np.array([quote['Close'].values for quote in quotes]).astype(np.float)

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

# Нормалізуйте дані
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створіть модель графа
# Створення моделі графа
edge_model = covariance.GraphLassoCV()

# Навчимо модель
# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Використання AffinityPropagation для кластеризації
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Вивід результатів кластеризації
for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ", ".join(names[labels == i]))

# Візуалізація кластерів
plt.figure(figsize=(10, 8))
for i in range(num_labels + 1):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Кластер {i + 1}")

plt.title("Результати кластеризації")
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.legend()
plt.show()
