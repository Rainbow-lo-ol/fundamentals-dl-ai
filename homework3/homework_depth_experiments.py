import torch
import time
from pathlib import Path
from utils.experiment_utils import save_results, train_model
from utils.visualization_utils import plot_learning_curves, plot_depth_comparison
from utils.model_utils import create_model, count_parameters
from datasets import get_mnist_loaders


def run_depth_experiments():
    # Конфигурация эксперимента
    config = {
        'input size': 784,
        'num_classes': 10,
        'hidden_size': 256,
        'depths': [1, 2, 3, 5, 7],
        'epochs': 5,
        'batch_size': 32,
        'results_dir': 'results/depth_experiments',
        'plots_dir': 'plots/depth_experiments',
        'reg': True
    }

    # Директории для результатов и графиков
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['plots_dir']).mkdir(parents=True, exist_ok=True)

    # устройство для вычислений
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка датасетов
    train_loader, test_loader = get_mnist_loaders(batch_size=config['batch_size'])

    experiments = []

    # проведение эксперимента
    for depth in config['depths']:
        print(f"\n=== Running experiment with depth={depth} ===")

        layers = []

        # добавление слоев
        for _ in range(depth - 1):
            layers.extend([
                {'type': 'linear', 'size': config['hidden_size']},
                {'type': 'relu'}
            ])

            if config['reg'] and depth >= 3:
                layers.extend([
                    {'type': 'dropout', 'rate': 0.2},
                    {'type': 'batch_norm'}
                ])

        # Создание модели
        model = create_model(
            input_size=config['input size'],
            num_classes=config['num_classes'],
            layers=layers
        ).to(device)

        # Вычисление времени
        start_time = time.time()

        # Обучение модели
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['epochs'],
            device=device
        )
        train_time = time.time() - start_time

        # Поиск переобучения
        over_gap = max(history['train_accs']) - max(history['test_accs'])
        over_start = None
        for epoch in range(1, len(history['test_accs'])):
            if history['test_accs'][epoch] < history['train_accs'][epoch-1]:
                over_start = epoch
                break

        experiments.append({
            'depth': depth,
            'parameters': count_parameters(model),
            'training_time': train_time,
            'max_train_acc': max(history['train_accs']),
            'max_test_acc': max(history['test_accs']),
            'overfit_gap': over_gap,
            'overfit_start': over_start,
            'final_test_loss': history['test_losses'][-1],
            'history': history,
            'reg': config['reg'] and depth >= 3
        })

        plot_learning_curves(
            history=history,
            save_path=Path(config['plots_dir']) / f'learning2_curves_depth_{depth}.png',
            title=f'Depth {depth} {"(with reg)" if config["reg"] and depth >=3 else ""}'
        )

        print(f"Results for depth {depth}:")
        print(f"Training time: {train_time:.2f}s")
        print(f"Max Train Accuracy: {max(history['train_accs']):.4f}")
        print(f"Max Test Accuracy: {max(history['test_accs']):.4f}")
        print(f"Overfitting gap: {over_gap:.4f}")
        print(f"Overfitting starts at epoch: {over_start or 'Not detected'}")

    save_results(
        results=experiments,
        path=Path(config['results_dir']) / 'results2.json'
    )

    plot_depth_comparison(
        results=experiments,
        save_path=Path(config['plots_dir']) / 'depth_comparison2.png',
        reg_enabled=config['reg']
    )

    best_model = max(experiments, key=lambda x: x['max_test_acc'])
    print(f"\nOptimal depth: {best_model['depth']} with test accuracy {best_model['max_test_acc']:.4f}")

    return experiments


if __name__ == '__main__':
    run_depth_experiments()


































