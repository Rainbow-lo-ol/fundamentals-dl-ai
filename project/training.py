import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import copy
from sklearn.metrics import classification_report
from project.models import LeNet5EmotionClassifier


class UniversalEmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None, grayscale=False):
        self.data_dir = data_dir
        self.transform = transform
        self.grayscale = grayscale
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    images.append((img_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        # Автоматическое определение режима загрузки
        mode = 'L' if self.grayscale else 'RGB'
        image = Image.open(img_path).convert(mode)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(model_type='vgg', grayscale=False):
    mean = [0.5] if grayscale else [0.485, 0.456, 0.406]
    std = [0.5] if grayscale else [0.229, 0.224, 0.225]

    if model_type == 'lenet':
        base_transforms = [
            transforms.Resize(32),
            transforms.CenterCrop(32)
        ]
    else:
        base_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]

    train_transforms = transforms.Compose([
        *base_transforms,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        *base_transforms,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return {
        'train': train_transforms,
        'test': test_transforms
    }


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Проверка размеров на первом батче
    sample_inputs, _ = next(iter(dataloaders['train']))
    print(f"Input shape: {sample_inputs.shape}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best test Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history


def plot_confusion_matrix(model, dataloader, class_names, save_path='./results'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    report = classification_report(all_labels, all_preds, target_names=class_names)
    return report


def plot_training_history(history, save_path='./results'):
    """
    Визуализирует историю обучения
    """
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Конфигурация
    config = {
        'model_type': 'lenet',
        'grayscale': True,
        'num_classes': 7,
        'batch_size': 32,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'data_dirs': ['./data_without_aug', './data_with_aug']
    }

    model_type = config['model_type']
    print(f"\n\n=== Training {model_type.upper()} model ===")
    transforms_dict = get_transforms(model_type, config['grayscale'])

    for data_dir in config['data_dirs']:
        dataset_name = os.path.basename(data_dir)
        print(f"\nProcessing dataset: {dataset_name}")

        # Создаем датасеты
        image_datasets = {
            'train': UniversalEmotionDataset(
                os.path.join(data_dir, 'train'),
                transform=transforms_dict['train'],
                grayscale=config['grayscale']
            ),
            'test': UniversalEmotionDataset(
                os.path.join(data_dir, 'test'),
                transform=transforms_dict['test'],
                grayscale=config['grayscale']
            )
        }

        # Создаем загрузчики данных
        dataloaders = {
            'train': DataLoader(
                image_datasets['train'],
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=4
            ),
            'test': DataLoader(
                image_datasets['test'],
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4
            )
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        class_names = image_datasets['train'].classes

        # Инициализация модели
        model = LeNet5EmotionClassifier(num_classes=config['num_classes'])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Обучение модели
        model, history = train_model(
            model=model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['num_epochs']
        )

        # Сохранение результатов
        base_filename = f"{model_type}_{dataset_name}"

        # Сохраняем график обучения
        training_plot_path = os.path.join('./results', f'{base_filename}_training.png')
        plot_training_history(history, save_path=training_plot_path)
        print(f"Training plot saved to {training_plot_path}")

        # Сохраняем матрицу ошибок
        cm_path = os.path.join('./results', f'{base_filename}_cm.png')
        report = plot_confusion_matrix(model, dataloaders['test'], class_names, save_path=cm_path)
        print(f"Confusion matrix saved to {cm_path}")

        # Сохраняем отчет классификации
        report_path = os.path.join('./results', f'{base_filename}_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")

        # Сохраняем модель
        model_path = os.path.join('./results', f'{base_filename}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")