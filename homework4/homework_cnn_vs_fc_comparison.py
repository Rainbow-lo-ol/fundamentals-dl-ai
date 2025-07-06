import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from homework4.models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN, SimpleCNN_CIFAR, CNNWithResidual_CIFAR
from homework4.models.fc_models import FullyConnectedModel
from homework4.utils.training_utils import train_model
from homework4.utils.visualization_utils import plot_learning_curves, plot_confusion_matrix, plot_final_comparison
import json
import os
import time


def save_json(results, file_name):
    file_path = os.path.join('results', file_name)

    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'train_losses': [float(loss) for loss in model_results['train_losses']],
            'test_losses': [float(loss) for loss in model_results['test_losses']],
            'train_accs': [float(acc) for acc in model_results['train_accs']],
            'test_accs': [float(acc) for acc in model_results['test_accs']],
            'training_time': float(model_results.get('training_time', 0))
        }

    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)


def load_json(filename):
    """Загружает результаты из JSON файла"""
    filepath = os.path.join('results', filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


HYPERPARAMS = {
    'batch_size': 32,
    'epochs': 10,
    'lr': 0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

FC_CONFIG = {
    'simple': {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.2}
        ]
    },
    'deep': {
        'input_size': 3*32*32,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'batch_norm'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'},
            {'type': 'linear', 'size': 64},
            {'type': 'relu'}
        ]
    }
}

CNN_CONFIG = {
    'simple_cnn_mnist': lambda: SimpleCNN(
        input_channels=1,
        num_classes=10
    ),
    'residual_cnn_mnist': lambda: CNNWithResidual(
        input_channels=1,
        num_classes=10
    ),
    'simple_cnn_cifar': lambda: SimpleCNN_CIFAR(
        input_channels=3,
        num_classes=10
    ),
    'residual_cnn_cifar': lambda: CNNWithResidual_CIFAR(
        input_channels=3,
        num_classes=10
    ),
    'cifar': lambda: CIFARCNN(
        num_classes=10
    )
}


def prepare_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS['batch_size'],
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=HYPERPARAMS['batch_size'],
        shuffle=True
    )

    return train_dataloader, test_dataloader


def prepare_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS['batch_size'],
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=HYPERPARAMS['batch_size'],
        shuffle=True
    )

    return train_dataloader, test_dataloader


def run_experiment(model, train_dataloader, test_dataloader, model_name, dataset_name):
    print(f"\nTraining {model_name} on {dataset_name}...")

    save_path = f"results/{model_name}_{dataset_name}"

    start_time = time.time()

    history = train_model(
        model=model,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        epochs=HYPERPARAMS['epochs'],
        lr=HYPERPARAMS['lr'],
        device=HYPERPARAMS['device']
    )

    training_time = time.time() - start_time
    history['training_time'] = training_time

    plot_learning_curves({
        'train_loss': history['train_losses'],
        'test_loss': history['test_losses'],
        'train_acc': history['train_accs'],
        'test_acc': history['test_accs']
    }, f"{save_path}_learning_curves.png")

    y_true, y_pred = [], []
    with torch.no_grad():
        for data, targ in test_dataloader:
            data, targ = data.to(HYPERPARAMS['device']), targ.to(HYPERPARAMS['device'])
            output = model(data)
            pred = output.argmax(dim=1)
            y_true.extend(targ.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    plot_confusion_matrix(
        y_true,
        y_pred,
        [str(i) for i in range(10)],
        f"{save_path}_confusion_matrix.png"
    )

    torch.save(model.state_dict(), f"{save_path}_model.pth")
    save_json({model_name: history}, f'{model_name}_{dataset_name}_results.json')

    return history


def compare_models_mnist():
    train_dataloader, test_dataloader = prepare_mnist()
    results = {}

    model = FullyConnectedModel(**FC_CONFIG['simple']).to(HYPERPARAMS['device'])
    results["fc_simple"] = run_experiment(
        model,
        train_dataloader,
        test_dataloader,
        "fc_simple",
        "mnist"
    )
    for model_name, model_fn in CNN_CONFIG.items():
        if 'cifar' in model_name:
            continue
        model = model_fn().to(HYPERPARAMS['device'])
        results[f"cnn_{model_name}"] = run_experiment(
            model,
            train_dataloader,
            test_dataloader,
            model_name,
            'mnist')

    return results


def compare_models_cifar10():
    train_dataloader, test_dataloader = prepare_cifar10()
    results = {}

    model = FullyConnectedModel(**FC_CONFIG['deep']).to(HYPERPARAMS['device'])
    results["fc_deep"] = run_experiment(
        model,
        train_dataloader,
        test_dataloader,
        "fc_deep",
        "cifar10"
    )

    for model_name, model_fn in CNN_CONFIG.items():
        if 'cifar' in model_name:
            model = model_fn().to(HYPERPARAMS['device'])
            results[f"cnn_{model_name}"] = run_experiment(
                model,
                train_dataloader,
                test_dataloader,
                model_name,
                'cifar10')

    return results


if __name__ == "__main__":
    mnist_results = compare_models_mnist()
    cifar_results = compare_models_cifar10()
    save_json(mnist_results, "mnist_final_results.json")
    save_json(cifar_results, "cifar_final_results.json")
    plot_final_comparison(mnist_results, cifar_results)