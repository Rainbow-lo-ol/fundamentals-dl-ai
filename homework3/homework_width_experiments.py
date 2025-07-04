import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from homework3.utils.experiment_utils import save_results, train_model
from homework3.utils.model_utils import create_model, count_parameters
from homework3.utils.visualization_utils import plot_width_comparison
from homework3.datasets import get_mnist_loaders


def run_width_experiments():
    config = {
        'input_size': 784,
        'num_classes': 10,
        'hidden_size': 256,
        'width': {
            'narrow': [64, 32, 16],
            'medium': [256, 128, 64],
            'wide': [1024, 512, 256],
            'xwide': [2048, 1024, 512]
        },
        'epochs': 5,
        'batch_size': 32,
        'results_dir': 'results/width_experiments',
        'plots_dir': 'plots/width_experiments',
    }

    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['plots_dir']).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_mnist_loaders(batch_size=config['batch_size'])

    experiments = []

    for name, widths in config['width'].items():
        print(f"\n=== Running {name}, widths {widths} ===")

        layers = []

        for width in widths:
            layers.extend([
                {'type': 'linear', 'size': width},
                {'type': 'relu'}
            ])

        model = create_model(
            input_size=config['input_size'],
            num_classes=config['num_classes'],
            layers=layers
        ).to(device)

        start_time = time.time()
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['epochs'],
            device=device
        )
        train_time = time.time() - start_time

        experiments.append({
            'profile': name,
            'widths': widths,
            'parameters': count_parameters(model),
            'training_time': train_time,
            'max_test_acc': max(history['test_accs']),
            'history': history
        })

        plt.figure()
        plt.plot(history['train_accs'], label='Train Accuracy')
        plt.plot(history['test_accs'], label='Test Accuracy')
        plt.title(f"Width Profile: {name}\n{widths}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(Path(config['plots_dir']) / f'learning_curves_{name}.png')
        plt.close()

    save_results(experiments, Path(config['results_dir']) / 'width_results.json')

    # Визуализация сравнения
    plot_width_comparison(experiments, Path(config['plots_dir']))
    optimize_architecture(config, device, train_loader, test_loader)

    return experiments


def optimize_architecture(config, device, train_loader, test_loader):
    """grid search"""
    print("\n=== Starting Architecture Optimization ===")
    base_widths = [16, 32, 64, 128]
    profiles = {
        'constant': lambda w: [w, w, w],
        'narrowing': lambda w: [w, w // 2, w // 4],
        'expanding': lambda w: [w // 4, w // 2, w]
    }

    results = []
    for width in base_widths:
        for name, items in profiles.items():
            widths = items(width)

            print(f"\nTesting {name} profile with base {width}: {widths}")

            layers = []

            for w in widths:
                layers.extend([
                    {'type': 'linear', 'size': w},
                    {'type': 'relu'}
                ])

            model = create_model(
                input_size=config['input_size'],
                num_classes=config['num_classes'],
                layers=layers
            ).to(device)

            history = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=5,
                device=device
            )

            results.append({
                'base_width': width,
                'profile': name,
                'widths': widths,
                'test_acc': max(history['test_accs'])
            })

        # Визуализация heatmap
        df = pd.DataFrame(results)
        heatmap_data = df.pivot(
            index="base_width",
            columns="profile",
            values="test_acc"
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title("Test Accuracy by Architecture")
        plt.savefig(Path(config['plots_dir']) / 'architecture_heatmap.png')
        plt.close()

        # Сохранение результатов
        best_arch = max(results, key=lambda x: x['test_acc'])
        print(f"\nOptimal architecture: {best_arch['profile']} with base width {best_arch['base_width']}")
        print(f"Widths: {best_arch['widths']}, Accuracy: {best_arch['test_acc']:.4f}")

        save_results(results, Path(config['results_dir']) / 'optimization_results.json')


if __name__ == '__main__':
    run_width_experiments()