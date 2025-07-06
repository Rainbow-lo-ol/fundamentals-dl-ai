import torch
import matplotlib.pyplot as plt
from models.cnn_models import KernelAnalysisCNN, SimpleCNN_CIFAR, CNNWithResidual_CIFAR, CIFARCNN
from utils.training_utils import train_model_with_time
from utils.visualization_utils import plot_learning_curves, plot_feature_maps, plot_training_time_comparison,plot_gradient_analysis
from datasets import get_cifar_loaders
import json
import os


def save_results(results, filename):
    """Сохраняет результаты в JSON файл"""
    filepath = os.path.join('results', filename)

    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'train_losses': [float(loss) for loss in model_results.get('train_losses', [])],
            'test_losses': [float(loss) for loss in model_results.get('test_losses', [])],
            'train_accs': [float(acc) for acc in model_results.get('train_accs', [])],
            'test_accs': [float(acc) for acc in model_results.get('test_accs', [])],
            'training_time': float(model_results.get('training_time', 0)),
            'params': int(model_results.get('params', 0)),
            'gradients': model_results.get('gradients', [])
        }

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)


HYPERPARAMS = {
    'batch_size': 32,
    'epochs': 10,
    'lr': 0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

train_dataloader, test_dataloader = get_cifar_loaders(batch_size=HYPERPARAMS['batch_size'])


def kernel_analysis():
    config = {
        '3x3': [3, 3],
        '5x5': [5, 5],
        '7x7': [7, 7],
        '1x1_3x3': [1, 3]
    }

    results = {}
    for name, kernel in config.items():
        print(f"\nTraining {name} model")
        model = KernelAnalysisCNN(kernel).to(HYPERPARAMS['device'])

        history = train_model_with_time(
            model=model,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=HYPERPARAMS['epochs'],
            lr=HYPERPARAMS['lr'],
            device=HYPERPARAMS['device']
        )

        results[name] = {**history}

        plot_feature_maps(
            model=model,
            test_loader=test_dataloader,
            device=HYPERPARAMS['device'],
            layer_name='conv1',
            save_path=f'plots/kernel_{name}_activations.png'
        )

    save_results(results, 'kernel_analysis_results.json')
    return results


def cnn_depth_analysis():
    config = {
        'Shallow': SimpleCNN_CIFAR(input_channels=3, num_classes=10),
        'Medium': CNNWithResidual_CIFAR(input_channels=3, num_classes=10),
        'Deep': CIFARCNN(num_classes=10)
    }

    results = {}
    for name, model in config.items():
        print(f"\nTraining {name} model")
        model = model.to(HYPERPARAMS['device'])

        history = train_model_with_time(
            model=model,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=HYPERPARAMS['epochs'],
            lr=HYPERPARAMS['lr'],
            device=HYPERPARAMS['device']
        )

        results[name] = {**history}

        plot_feature_maps(
            model=model,
            test_loader=test_dataloader,
            device=HYPERPARAMS['device'],
            layer_name='conv1' if 'Shallow' in name else 'res1' if 'Medium' in name else 'conv3',
            save_path=f'plots/depth_{name}_features.png'
        )

    save_results(results, 'depth_analysis_results.json')
    return results


def show_kernel_analysis():
    results = kernel_analysis()

    # Для каждой модели строим отдельные графики
    for model_name, history in results.items():
        print(f"\nГрафики обучения для модели с ядрами {model_name}")
        plot_learning_curves(
            history,  # Передаем непосредственно словарь с метриками
            f'plots/kernel_{model_name}_learning_curves.png'
        )

    # Дополнительно: сводный график точности для всех моделей
    plt.figure(figsize=(10, 5))
    for model_name, history in results.items():
        plt.plot(history['test_acc'], label=f'Ядра {model_name}')
    plt.title('Сравнение точности разных ядер свертки')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.savefig('plots/kernel_all_test_acc.png')
    plt.close()


def show_cnn_depth_analysis():
    results = cnn_depth_analysis()

    # Для каждой модели строим отдельные графики
    for model_name, history in results.items():
        print(f"\nГрафики обучения для архитектуры {model_name}")
        plot_learning_curves(
            history,  # Передаем непосредственно словарь с метриками
            f'plots/depth_{model_name}_learning_curves.png'
        )

    # Сводный график точности для всех архитектур
    plt.figure(figsize=(10, 5))
    for model_name, history in results.items():
        plt.plot(history['test_acc'], label=f'Архитектура {model_name}')
    plt.title('Сравнение точности разных архитектур CNN')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.savefig('plots/depth_all_test_acc.png')
    plt.close()


if __name__ == '__main__':
    show_kernel_analysis()
    show_cnn_depth_analysis()