import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import make_classification_data, accuracy, log_epoch_log, ClassificationDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


def plot_confusion_matrix(all_targets, all_preds, num_classes):
    cm = confusion_matrix(all_targets.cpu().numpy(), all_preds.cpu().numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    # Генерируем данные
    num_class = 3
    n_features = 2
    epochs = 100
    batch_size = 32

    X, y = make_classification_data(n=200, n_classes=num_class, n_features=n_features)

    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=n_features, num_classes=num_class)
    criterion = nn.CrossEntropyLoss() if num_class > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    for epoch in range(1, epochs + 1):
        all_preds, all_probs, all_targets = [], [], []
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)

            if num_class > 1:
                loss = criterion(logits, batch_y.long())
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)  # Добавляем вероятности для многоклассового случая
            else:
                loss = criterion(logits, batch_y.float())
                preds = (torch.sigmoid(logits) > 0.5).float()
                probs = torch.sigmoid(logits)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(batch_y)

        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)

        avg_loss = total_loss / (i + 1)
        acc = accuracy(all_probs, all_targets)
        prec = precision_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='macro', zero_division=0)
        rec = recall_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='macro', zero_division=0)
        f1 = f1_score(all_targets.cpu().numpy(), all_preds.cpu().numpy(), average='macro', zero_division=0)
        with torch.no_grad():
            if num_class == 1:
            # Бинарная классификация
                roc = roc_auc_score(all_targets.cpu().numpy(), all_probs.cpu().numpy())
            else:
                # Многоклассовая - используем one-vs-rest
                roc = roc_auc_score(all_targets.cpu().numpy(), all_probs.cpu().numpy(), multi_class='ovr')

        if epoch % 10 == 0:
            log_epoch_log(epoch, avg_loss, acc, prec, rec, f1, roc)

    plot_confusion_matrix(all_targets, all_preds, num_class)

    # Сохраняем модель
    torch.save(model.state_dict(), 'models/logreg_torch.pth')