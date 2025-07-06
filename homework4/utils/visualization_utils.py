import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Optional, Dict, Any
import torch.nn as nn
from torch.utils.data import DataLoader


def plot_learning_curves(history, save_path):
    """Визуализация кривых обучения с проверкой ключей"""
    plt.figure(figsize=(12, 4))

    # Проверка наличия необходимых ключей
    required_keys = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
    for key in required_keys:
        if key not in history:
            raise KeyError(f"Отсутствует ожидаемый ключ '{key}' в history. Доступные ключи: {list(history.keys())}")

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def plot_final_comparison(mnist_results, cifar_results):
    """Визуализация итогового сравнения моделей"""
    plt.figure(figsize=(15, 6))

    # Сравнение на MNIST
    plt.subplot(1, 2, 1)
    for model_name, result in mnist_results.items():
        plt.plot(result['test_accs'], label=model_name)
    plt.title('MNIST Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Сравнение на CIFAR-10
    plt.subplot(1, 2, 2)
    for model_name, result in cifar_results.items():
        plt.plot(result['test_accs'], label=model_name)
    plt.title('CIFAR-10 Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/final_comparison.png')
    plt.close()


def plot_feature_maps(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        layer_name: str,
        save_path: Optional[str] = None,
        num_maps: int = 8,
        figsize: tuple = (15, 5)
) -> None:
    # Переводим модель в режим оценки
    model.eval()

    # Получаем один батч данных
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images.to(device)

    # Словарь для хранения активаций
    activations: Dict[str, torch.Tensor] = {}

    # Функция-хук с аннотацией типа
    def get_activation(name: str) -> Any:
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            activations[name] = output.detach()

        return hook

    # Находим нужный слой
    layer: Optional[nn.Module] = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break

    if layer is None:
        raise ValueError(f"Слой {layer_name} не найден в модели")

    # Регистрируем хук с явным указанием типа
    if isinstance(layer, nn.Module):
        handle = layer.register_forward_hook(get_activation(layer_name))  # type: ignore

        # Пропускаем данные через модель
        with torch.no_grad():
            _ = model(images)

        # Удаляем хук
        handle.remove()
    else:
        raise TypeError(f"Объект {layer_name} не является модулем PyTorch")

    # Получаем активации
    if layer_name not in activations:
        raise RuntimeError(f"Не удалось получить активации для слоя {layer_name}")

    feature_maps = activations[layer_name]

    # Подготовка данных для визуализации
    if feature_maps.dim() != 4:
        raise ValueError(f"Ожидались 4D-тензор активаций, получен {feature_maps.dim()}D")

    sample = feature_maps[0].cpu().numpy()
    num_maps = min(num_maps, sample.shape[0])

    # Визуализация
    plt.figure(figsize=figsize)
    for i in range(num_maps):
        plt.subplot(2, (num_maps + 1) // 2, i + 1)
        plt.imshow(sample[i], cmap='viridis')


def plot_gradient_analysis(results, save_path):
    """Визуализация анализа градиентов"""
    plt.figure(figsize=(10, 6))

    for arch_name, history in results.items():
        if 'gradients' in history:
            gradients = history['gradients']
            plt.plot(gradients, label=arch_name, marker='o', markersize=4)

    plt.title('Анализ градиентов в разных архитектурах')
    plt.xlabel('Шаг обучения')
    plt.ylabel('Норма градиента')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_time_comparison(results, save_path):
    """Сравнение времени обучения разных архитектур"""
    plt.figure(figsize=(10, 5))

    architectures = []
    times = []
    for arch_name, history in results.items():
        if 'training_time' in history:
            architectures.append(arch_name)
            times.append(history['training_time'])

    plt.bar(architectures, times, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Время обучения разных архитектур CNN')
    plt.ylabel('Время (секунды)')
    plt.xticks(rotation=45)

    # Добавляем подписи значений
    for i, v in enumerate(times):
        plt.text(i, v + 5, f"{v:.1f}s", ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()