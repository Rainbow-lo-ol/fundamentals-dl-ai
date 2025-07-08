import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CustomImageDataset
import matplotlib.pyplot as plt
from tqdm import tqdm


# 1. Подготовка данных
def prepare_data():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset('data/train', transform=train_transform)
    test_dataset = CustomImageDataset('data/test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    return train_loader, test_loader, train_dataset.get_class_names()


# 2. Подготовка модели
def prepare_model(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Замораживаем все слои, кроме последнего
    for param in model.parameters():
        param.requires_grad = False

    # Заменяем последний слой
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# 3. Обучение
def train_model(model, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Обучение
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}')

    return model, train_losses


# 4. Тестирование
def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    return accuracy, avg_loss


# 5. Визуализация
def plot_metrics(train_losses, test_loss, test_acc):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(['Test'], [test_acc])
    plt.title('Test Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


# Основной процесс
if __name__ == '__main__':
    # Подготовка данных
    train_loader, test_loader, class_names = prepare_data()

    # Подготовка модели
    model = prepare_model(len(class_names))
    print(f"Model prepared for {len(class_names)} classes")

    # Обучение
    model, train_losses = train_model(model, train_loader, num_epochs=10)

    # Тестирование
    test_accuracy, test_loss = test_model(model, test_loader)

    # Визуализация
    plot_metrics(train_losses, test_loss, test_accuracy)

    # Сохранение модели
    torch.save(model.state_dict(), 'results/fine_tuned_resnet18.pth')
    print("Model saved as 'fine_tuned_resnet18.pth'")