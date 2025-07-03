import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt


def polynomial_features(X, degree=2):
    """
    Генерирует полиномиальные признаки для входных данных
    :param X: Входные данные
    :param degree: Максимальная степень полинома
    :return: Массив с расширенными признаками формы
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)


def interaction_features(X):
    """
    Добавляет к исходным признакам все попарные произведения (взаимодействия) признаков
    :param X: Входные данные
    :return: Матрица признаков с добавленными взаимодействиями формы
    """
    # Получаем признаки
    n_feat = X.shape[1]

    # Массив для взаимодействий
    interactions = np.empty((X.shape[0], n_feat * (n_feat - 1) // 2))
    idx = 0

    # Заполняем массив попарными произведениями
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            interactions[:, idx] = X[:, i] * X[:, j]
            idx += 1

    return np.hstack([X, interactions])


def statistical_features(X):
    """
    Добавляет к исходным признакам статистические характеристики
    :param X: Входные данные
    :return: Матрица признаков с добавленными статистиками формы
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return np.hstack([X, mean, std])


def linear_regression_experiment(X_train, y_train, X_test, y_test, feature_set_name):
    """
    Проводит эксперимент линейной регрессии с заданными данными и сохраняет модель
    :param X_train: тренировочные входные данные
    :param y_train: тренировочные целевые данные
    :param X_test: тестирующие входные данные
    :param y_test: тестирующие целевые данные
    :param feature_set_name: Название набора признаков
    :return: Среднеквадратичная ошибка (MSE) на тестовой выборке
    """
    # Подготовка модели и оптимизатора
    model = nn.Linear(X_train.shape[1], 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Подготовка DataLoader для батчевой обработки
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Обучение
    epochs = 100
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # Сохранение модели
    model_name = f"models/linear_regression_{feature_set_name.replace(' ', '_').lower()}.pth"
    torch.save(model.state_dict(), model_name)

    # Подсчет MSE
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test))
        mse = mean_squared_error(y_test, y_pred.numpy())

    return mse


def logistic_regression_experiment(X_train, y_train, X_test, y_test, feature_set_name):
    """
    Проводит эксперимент логистической регрессии с заданными данными и сохраняет модель
    :param X_train: тренировочные входные данные
    :param y_train: тренировочные целевые данные
    :param X_test: тестирующие входные данные
    :param y_test: тестирующие целевые данные
    :param feature_set_name: Название набора признаков
    :return: Среднеквадратичная ошибка (MSE) на тестовой выборке
    """
    #Подготовка модели и оптимизатора
    model = nn.Linear(X_train.shape[1], 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    #Подготовка DataLoader для батчевой обработки
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Обучение
    epochs = 100
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # Сохранение модели
    model_name = f"models/linear_regression_{feature_set_name.replace(' ', '_').lower()}.pth"
    torch.save(model.state_dict(), model_name)

    # Вычисление метрики
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        y_pred = (torch.sigmoid(logits) > 0.5).float()
        acc = accuracy_score(y_test, y_pred.numpy())

    return acc


if __name__ == '__main__':
    np.random.seed(42)
    X, y = np.random.randn(200, 2), np.random.randn(200)
    X_clf, y_clf = np.random.randn(200, 2), (np.random.rand(200) > 0.5).astype(int)

    # Разделение на train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_clf_train, X_clf_test = X_clf[:split], X_clf[split:]
    y_clf_train, y_clf_test = y_clf[:split], y_clf[split:]

    feature_sets = {
        'Базовые': X_train,
        'Полиномиальные (степень 2)': polynomial_features(X_train),
        'Взаимодействия': interaction_features(X_train),
        'Статистические': statistical_features(X_train),
        'Все вместе': statistical_features(interaction_features(polynomial_features(X_train)))
    }

    # Тестовые наборы с теми же преобразованиями
    test_sets = {
        'Базовые': X_test,
        'Полиномиальные (степень 2)': polynomial_features(X_test),
        'Взаимодействия': interaction_features(X_test),
        'Статистические': statistical_features(X_test),
        'Все вместе': statistical_features(interaction_features(polynomial_features(X_test)))
    }

    feature_sets_clf = {k: v for k, v in feature_sets.items()}
    test_sets_clf = {k: v for k, v in test_sets.items()}

    # Эксперименты с линейной регрессией
    print("\nРезультаты линейной регрессии (MSE):")
    results_lin = {}
    for name, X_train_ft in feature_sets.items():
        mse = linear_regression_experiment(X_train_ft, y_train, test_sets[name], y_test, name)
        results_lin[name] = mse
        print(f"{name}: {mse:.4f}")


    # Эксперименты с логистической регрессией
    print("\nРезультаты логистической регрессии (Accuracy):")
    results_log = {}
    for name, X_train_ft in feature_sets_clf.items():
        acc = logistic_regression_experiment(X_train_ft, y_clf_train, test_sets_clf[name], y_clf_test, name)
        results_log[name] = acc
        print(f"{name}: {acc:.4f}")

    # Визуализация
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(results_lin.keys(), results_lin.values())
    plt.title('Линейная регрессия (MSE)')
    plt.xticks(rotation=45)
    plt.ylabel('MSE')

    plt.subplot(1, 2, 2)
    plt.bar(results_log.keys(), results_log.values())
    plt.title('Логистическая регрессия (Accuracy)')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()