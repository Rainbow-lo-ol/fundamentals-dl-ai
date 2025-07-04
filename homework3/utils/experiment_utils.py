import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import Dict, List


def train_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        device: torch.device = 'cpu',
        lr: float = 0.001
) -> Dict[str, List[float]]:
    """
    функция для обучения модели
    :param model: модель
    :param train_loader: Dataloader для тренировочных данных
    :param test_loader: Dataloader для тестовых данных
    :param epochs: количество эпох
    :param device: cpu или gpu
    :param lr: скорость обучения
    :return: словарь с потерями и количеством верных ответов
    """
    # Инициализация функции потерь и отптимизатора
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Конечный результат
    history = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': []
    }

    for epoch in range(epochs):

        # Обучение модели
        model.train()
        train_loss, train_corr, train_total = 0, 0, 0

        for data, targ in train_loader:
            data, targ = data.to(device), targ.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targ)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdims=True)
            train_corr += pred.eq(targ.view_as(pred)).sum().item()
            train_total += targ.size(0)

        # Работа модели с тестовыми данными
        model.eval()
        test_loss, test_corr, test_total = 0, 0, 0

        with torch.no_grad():
            for data, targ in train_loader:
                data, targ = data.to(device), targ.to(device)

                output = model(data)
                loss = criterion(output, targ)

                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdims=True)
                test_corr += pred.eq(targ.view_as(pred)).sum().item()
                test_total += targ.size(0)

        # Запись результата
        history['train_losses'].append(train_loss / len(train_loader))
        history['train_accs'].append(train_corr / train_total)
        history['test_losses'].append(test_loss / len(test_loader))
        history['test_accs'].append(test_corr / test_total)

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {history["train_losses"][-1]:.4f}, '
              f'Acc: {history["train_accs"][-1]:.4f} | '
              f'Test Loss: {history["test_losses"][-1]:.4f}, '
              f'Acc: {history["test_accs"][-1]:.4f}')

    return history


def save_results(results, path):
    """
    сохранение результата в формате json
    :param results: результат для сохранения
    :param path: путь, по которому будет происходить сохранение
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path):
    """
    загрузка результата в формате json
    :param path: путь к файду
    """
    with open(path, 'w') as f:
        return json.load(f)


































