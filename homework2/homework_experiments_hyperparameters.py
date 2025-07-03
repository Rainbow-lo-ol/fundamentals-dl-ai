import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tqdm import tqdm
from utils import make_regression_data, mse, log_epoch_lin, RegressionDataset, make_classification_data, accuracy, log_epoch_log, ClassificationDataset


class Experiments:
    def __init__(self):
        self.results = []

    def add_results(self, model, config, train_loss, val_metric):
        """
        обавляет результаты эксперимента в хранилище для последующего анализа.
        :param model: Название модели
        :param config: Конфигурация эксперимента
        :param train_loss: История значений функции потерь на обучении
        :param val_metric: История метрик на валидации
        :return: none
        """
        self.results.append({
            'model': model,
            'config': config,
            'train_loss': train_loss,
            'val_metric': val_metric
        })

    # Визуализация графика
    def plot_results(self):
        models = set([res['model'] for res in self.results])

        for model in models:
            plt.figure(figsize=(15, 5))
            model_results = [res for res in self.results if res['model'] == model]

            # График потерь
            plt.subplot(1, 2, 1)
            for res in model_results:
                plt.plot(res['train_loss'], label=f"{res['config']}")
            plt.title(f'{model} - Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # График метрики
            plt.subplot(1, 2, 2)
            for res in model_results:
                plt.plot(res['val_metric'], label=f"{res['config']}")
            plt.title(f'{model} - Validation {"Accuracy" if "Classification" in model else "MSE"}')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.legend()

            plt.tight_layout()
            plt.show()


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(model, dataset, config, model_type):
    """
    Обучает модель и возвращает метрики обучения и валидации
    :param model: модель
    :param dataset: данные
    :param config: Конфигурация эксперимента
    :param model_type: тип модели
    :return: кортеж списков метрик по эпохам
    """

    # Подготовка данных
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Инициализация criterion
    if model_type == "Linear Regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Инициализация optimizer
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])

    train_losses = []
    val_metrics = []

    # Обучение
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        for feat, targ in train_dataloader:
            optimizer.zero_grad()
            outputs = model(feat)

            if model_type == "Logistic Regression":
                outputs = outputs.squeeze()

            loss = criterion(outputs, targ)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_dataloader))

        # Валидация
        model.eval()
        val_loss = 0
        corr = 0
        total = 0
        with torch.no_grad():
            for feat, targ in val_dataloader:
                outputs = model(feat)

                if model_type == "Linear Regression":
                    val_loss += criterion(outputs, targ).item()
                else:
                    outputs = outputs.squeeze()
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    corr += (preds == targ).sum().item()
                    total += targ.size(0)

        if model_type == "Linear Regression":
            val_metrics.append(val_loss / len(val_dataloader))
        else:
            val_metrics.append(corr / total)

    # Сохранение
    model_name = f"models/{model_type.replace(' ', '_')}_lr{config['lr']}_bs{config['batch_size']}_{config['optimizer']}.pth"
    torch.save(model.state_dict(), model_name)

    return train_losses, val_metrics


# Конфигурация эксперимента
configs = {
    'learning_rates': [0.1, 0.01, 0.001],
    'batch_sizes': [16, 32, 64],
    'optimizers': ['SGD', 'Adam', 'RMSprop'],
    'epochs': 50
}

# Создание данных
X_reg, y_reg = make_regression_data(n=500)
reg_dataset = RegressionDataset(X_reg, y_reg)

X_clf, y_clf = make_classification_data(n=500)
clf_dataset = ClassificationDataset(X_clf, y_clf)

experiment = Experiments()

# Обучение
for model_type, dataset in [("Linear Regression", reg_dataset),
                           ("Logistic Regression", clf_dataset)]:
    print(f"\n=== Running experiments for {model_type} ===")

    if model_type == "Linear Regression":

        model = LinearRegression(X_reg.shape[1])
    else:
        model = LogisticRegression(X_clf.shape[1])

    for lr, batch_size, optimizer in tqdm(product(
            configs['learning_rates'],
            configs['batch_sizes'],
            configs['optimizers']
    ), total=len(configs['learning_rates']) *
             len(configs['batch_sizes']) *
             len(configs['optimizers'])):
        config = {
            'lr': lr,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'epochs': configs['epochs']
        }

        if model_type == "Linear Regression":
            current_model = LinearRegression(X_reg.shape[1])
        else:
            current_model = LogisticRegression(X_clf.shape[1])

        train_loss, val_metric = train_model(current_model, dataset, config, model_type)
        experiment.add_results(model_type, config, train_loss, val_metric)

experiment.plot_results()

# Вывод итогов
print("\n=== Best Configurations ===")
for model_type in ["Linear Regression", "Logistic Regression"]:
    print(f"\nFor {model_type}:")
    model_results = [res for res in experiment.results if res['model'] == model_type]

    # Для линейной регрессии ищем минимальную MSE, для логистической - максимальную точность
    if model_type == "Linear Regression":
        best_results = sorted(model_results, key=lambda x: min(x['val_metric']))[:5]
        metric_name = "Validation MSE"
    else:
        best_results = sorted(model_results, key=lambda x: max(x['val_metric']), reverse=True)[:5]
        metric_name = "Validation Accuracy"

    for res in best_results:
        best_metric = min(res['val_metric']) if model_type == "Linear Regression" else max(res['val_metric'])
        best_epoch = res['val_metric'].index(best_metric)
        print(f"Config: {res['config']} | Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")















