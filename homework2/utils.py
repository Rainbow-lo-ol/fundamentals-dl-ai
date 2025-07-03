import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')


def make_classification_data(n=100, n_classes=2, n_features=2, source='random'):
    """
    Функция для реализации многоклассовой классификации
    :param n: количество примеров в данных
    :param n_classes: количество классов
    :param n_features: количество признаков
    :param source: источник данных
    :return: Х - тензор признаков размерности, y - тензор меток классов размерности
    """
    if source == 'random':
        # Генерация случайных данных
        X = torch.randn(n, n_features)

        if n_classes == 2:
            # Бинарная классификация
            w = torch.randn(n_features, 1)
            b = torch.randn(1)
            logits = X @ w + b
            y = (logits > 0).float().squeeze()
        else:
            # Многоклассовая классификация
            centers = torch.randn(n_classes, n_features) * 2
            y = torch.randint(0, n_classes, (n,))
            X = centers[y] + torch.randn(n, n_features) * 0.5

        return X, y
    elif source == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')


def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()


def accuracy(y_pred, y_true):
    if len(y_pred.shape) > 1:  # Если y_pred - это вероятности классов (многоклассовый случай)
        y_pred = torch.argmax(y_pred, dim=1)
    elif y_pred.dtype == torch.float32:  # Бинарная классификация с вероятностями
        y_pred = (y_pred > 0.5).float()
    return (y_pred == y_true).float().mean().item()


def log_epoch_log(epoch, loss, acc, prec, rec, f1, roc):
    print(f'Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f} | ROC-AUC: {roc:.4f}')

def log_epoch_lin(epoch, train_loss, val_loss):
    print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')