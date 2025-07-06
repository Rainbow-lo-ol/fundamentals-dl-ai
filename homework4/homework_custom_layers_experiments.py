import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from homework4.models.custom_layers import CustomConv2d, ChannelAttention, CustomActivation, CustomPool2d, BasicBlock, BottleneckBlock, WideBlock


def test_custom_layers():
    # Тестовые данные
    x = torch.randn(1, 3, 32, 32)

    # Тест CustomConv2d
    conv = CustomConv2d(3, 16, 3, padding=1)
    out = conv(x)
    print(f"CustomConv output shape: {out.shape}")

    #  Тест ChannelAttention
    attn = ChannelAttention(16)
    out = attn(out)
    print(f"Attention output shape: {out.shape}")

    #  Тест CustomActivation
    activation = CustomActivation()
    out = activation(out)
    print(f"CustomActivation output shape: {out.shape}")

    # Тест CustomPool2d
    pool = CustomPool2d(2)
    out = pool(out)
    print(f"CustomPool output shape: {out.shape}")


class TestResNet(nn.Module):
    def __init__(self, block_type, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block_type, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block_type, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block_type, 256, num_blocks[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block_type, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Эксперименты с Residual блоками
def run_residual_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    # Конфигурации моделей
    configs = {
        'Basic': (BasicBlock, [2, 2, 2]),
        'Bottleneck': (BottleneckBlock, [2, 2, 2]),
        'Wide': (WideBlock, [2, 2, 2])
    }

    results = {}
    for name, (block, blocks) in configs.items():
        model = TestResNet(block, blocks).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{name} Residual Model: {params / 1e6:.2f}M parameters")

        # Простой тест forward pass
        x = torch.randn(2, 3, 32, 32).to(device)
        out = model(x)
        print(f"Output shape: {out.shape}")

        results[name] = {
            'params': params,
            'test_output': out.shape
        }

    return results


if __name__ == "__main__":
    test_custom_layers()
    run_residual_experiments()