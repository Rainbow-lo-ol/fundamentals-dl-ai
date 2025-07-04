import torch
import torch.nn as nn
from homework3.models import FullyConnectedModel
from pathlib import Path


def create_model(input_size, num_classes, layers):
    """
    Инициализация модели
    :param input_size: исходный размер
    :param num_classes: Количество классов
    :param layers: слои
    :return: модель FullyConnectedModel
    """
    return FullyConnectedModel(
        input_size=input_size,
        num_classes=num_classes,
        layers=layers
    )


def count_parameters(model):
    """количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(layer):
    """Инициализирует веса и смещения для заданного слоя нейронной сети"""
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def save_model(model, optimizer, epoch, metrics, save_dir):
    """Сохранение модели"""
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }

    torch.save(state, Path(save_dir) / f'model_epoch_{epoch}.pth')