import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.experiment_utils import train_model, save_results
from utils.model_utils import create_model
from datasets import get_mnist_loaders
from tqdm import tqdm


def run_regularization_experiments():
    config = {
        'input_size': 784,
        'num_classes': 10,
        'base_architecture': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'}
        ],
        'regularizations': [
            {'name': 'no_reg', 'params': {}},
            {'name': 'dropout_0.1', 'params': {'dropout': 0.1}},
            {'name': 'dropout_0.3', 'params': {'dropout': 0.3}},
            {'name': 'dropout_0.5', 'params': {'dropout': 0.5}},
            {'name': 'batchnorm', 'params': {'batchnorm': True}},
            {'name': 'both', 'params': {'dropout': 0.3, 'batchnorm': True}},
            {'name': 'l2', 'params': {'weight_decay': 1e-4}}
        ],
        'adaptive_regs': [
            {'name': 'adaptive_dropout', 'params': {'adaptive_dropout': True}},
            {'name': 'bn_momentum', 'params': {'batchnorm': True, 'bn_momentum': 0.7}},
            {'name': 'combined', 'params': {
                'adaptive_dropout': True,
                'batchnorm': True,
                'weight_decay': 1e-4
            }}
        ],
        'epochs': 20,
        'batch_size': 128,
        'results_dir': 'results/regularization',
        'plots_dir': 'plots/regularization'
    }

    # Создание директорий
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['plots_dir']).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(config['batch_size'])

    # Сравнение техник регуляризации
    print("=== Running Regularization Comparison ===")
    reg_results = []
    for reg in tqdm(config['regularizations'], desc="Regularizations"):
        layers = build_layers(config['base_architecture'], reg['params'])
        model = create_model(
            input_size=config['input_size'],
            num_classes=config['num_classes'],
            layers=layers
        ).to(device)

        optimizer_params = {'lr': 0.001}
        if 'weight_decay' in reg['params']:
            optimizer_params['weight_decay'] = reg['params']['weight_decay']

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['epochs'],
            device=device,
            optimizer_params=optimizer_params
        )

        reg_results.append({
            'name': reg['name'],
            'history': history,
            'weights': get_weight_distribution(model)
        })

        plot_learning_curves(
            history,
            Path(config['plots_dir']) / f"learning_{reg['name']}.png",
            reg['name']
        )

    save_results(reg_results, Path(config['results_dir']) / 'regularization.json')
    plot_reg_comparison(reg_results, Path(config['plots_dir']))
    plot_weight_distributions(reg_results, Path(config['plots_dir']))

    # Адаптивная регуляризация
    print("\n=== Running Adaptive Regularization ===")
    adaptive_results = []
    for reg in tqdm(config['adaptive_regs'], desc="Adaptive Regs"):
        layers = build_layers(config['base_architecture'], reg['params'])
        model = create_model(
            input_size=config['input_size'],
            num_classes=config['num_classes'],
            layers=layers
        ).to(device)

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['epochs'],
            device=device
        )

        adaptive_results.append({
            'name': reg['name'],
            'history': history,
            'params': reg['params']
        })

    save_results(adaptive_results, Path(config['results_dir']) / 'adaptive.json')
    plot_adaptive_results(adaptive_results, Path(config['plots_dir']))


def build_layers(base_arch, reg_params):
    """Собирает слои с учетом параметров регуляризации"""
    layers = []
    for layer in base_arch:
        layers.append(layer.copy())

        if layer['type'] == 'relu':
            if reg_params.get('dropout'):
                layers.append({'type': 'dropout', 'rate': reg_params['dropout']})
            if reg_params.get('adaptive_dropout'):
                # Адаптивный dropout увеличивается к выходу
                layers.append({'type': 'dropout', 'rate': min(0.1 + 0.1 * len(layers) / 10, 0.5)})
            if reg_params.get('batchnorm'):
                bn_params = {}
                if 'bn_momentum' in reg_params:
                    bn_params['momentum'] = reg_params['bn_momentum']
                layers.append({'type': 'batch_norm', **bn_params})
    return layers


def get_weight_distribution(model):
    """Собирает распределение весов для анализа"""
    weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'values': param.data.cpu().numpy().flatten()
            }
    return weights


def plot_learning_curves(history, save_path, title):
    """Визуализация кривых обучения"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['test_losses'], label='Test')
    plt.title(f'Loss ({title})')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train')
    plt.plot(history['test_accs'], label='Test')
    plt.title(f'Accuracy ({title})')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_reg_comparison(results, plots_dir):
    """Сравнение разных методов регуляризации"""
    plt.figure(figsize=(10, 5))

    names = [r['name'] for r in results]
    test_accs = [max(r['history']['test_accs']) for r in results]

    plt.bar(names, test_accs)
    plt.title('Test Accuracy by Regularization Method')
    plt.ylim(0.9, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'reg_comparison.png')
    plt.close()


def plot_weight_distributions(results, plots_dir):
    """Визуализация распределения весов"""
    for result in results:
        plt.figure(figsize=(10, 5))
        for i, (name, stats) in enumerate(result['weights'].items()):
            plt.hist(stats['values'], bins=50, alpha=0.5, label=name)

        plt.title(f"Weight Distribution ({result['name']})")
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(plots_dir / f"weights_{result['name']}.png")
        plt.close()


def plot_adaptive_results(results, plots_dir):
    """Визуализация адаптивных методов"""
    plt.figure(figsize=(12, 5))

    for result in results:
        plt.plot(result['history']['test_accs'], label=result['name'])

    plt.title('Adaptive Regularization Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.savefig(plots_dir / 'adaptive_comparison.png')
    plt.close()


if __name__ == '__main__':
    run_regularization_experiments()