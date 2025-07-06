import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)

    return {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_acc': train_accs,
        'test_acc': test_accs
    }


def train_model_with_time(model, train_loader, test_loader, epochs, lr, device):
    """Модифицированная версия train_model с замером времени и сбором градиентов"""
    start_time = time.time()

    gradients = []

    def hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0].abs().mean().item())

    # Регистрируем хук для первого и последнего слоев
    first_layer = next(model.children())
    if isinstance(first_layer, nn.Sequential):
        first_layer = first_layer[0]
    handle1 = first_layer.register_full_backward_hook(hook_fn)

    last_layer = list(model.children())[-1]
    handle2 = last_layer.register_full_backward_hook(hook_fn)

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        device=device
    )

    # Удаляем хуки
    handle1.remove()
    handle2.remove()

    training_time = time.time() - start_time
    history['training_time'] = training_time
    history['gradients'] = gradients  # Градиенты первого и последнего слоев

    return history