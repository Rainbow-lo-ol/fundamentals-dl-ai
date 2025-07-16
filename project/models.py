import torch.nn as nn
import torch.nn.functional as F


class LeNet5EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(LeNet5EmotionClassifier, self).__init__()
        # Для входных изображений 32x32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 1x32x32 -> 6x32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 6x32x32 -> 6x16x16
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6x16x16 -> 16x12x12
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x12x12 -> 16x6x6
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 6 * 6)  # Изменено на 6x6
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x