# Задание 1: Модификация существующих моделей (30 баллов)
## 1.1 Расширение линейной регрессии
### Добавление L1 и L2 регуляризаций происходило двумя способами. 
#### 1 способ: расписывать и L1 и L2
добавляем коэффициенты для L1 и L2 регуляризаций до начала обучения модели
```python
l1_lam = 0.1
l2_lam = 0.1
```
после, в обучение модели, во время посчета потерь добавляем регуляризации
```python
l1_reg = torch.zeros(())
for param in model.parameters():
    l1_reg += torch.norm(param, 1)
loss += l1_reg * l1_lam

l2_reg = torch.zeros(())
for param in model.parameters():
    l2_reg += torch.norm(param, 2)
loss += l2_reg * l2_lam
```
#### 2 способ: использовать weight_decay в optimizer
при использование weight_decay всё ещё нужно писать полноценный код для L1
```python
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=l2_lam)
```
При использование данного способо значения loss получаются меньше, так как коэффициент L2_lamещё умножается на lr
### Добавление early stopping
Перед тем как использовать early stopping нужно разделить данные на тренировочные и валидационные. Именно на валидационных будет происходить early stopping. Будет их раделять в соотношение 0,8/0,2
```python
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
```
Напишем парментры для early stopping
```python
best_val_loss = float('inf')
patience = 10
no_improve = 0
min_delta = 0.001
```
Логика early stopping: Во время обучения после каждой эпохи проверяем метрику на валидационных данных. Если метрика ухудшается (или не улучшается) в течение нескольких эпох подряд — обучение останавливается.
```python
if avg_val_loss < best_val_loss - min_delta:
    best_val_loss = avg_val_loss
    no_improve = 0
    torch.save(model.state_dict(), 'models/best_model.pth')
else:
    no_improve += 1

if no_improve >= patience:
    print(f'Early stopping на эпохе {epoch}')
    break
```
### Результаты
```
Размер тренировочного датасета: 160
Размер валидационного датасета: 40
Epoch  10 | Train Loss: 0.2777 | Val Loss: 0.2871
Epoch  20 | Train Loss: 0.2603 | Val Loss: 0.2960
Epoch  30 | Train Loss: 0.2573 | Val Loss: 0.2501
Early stopping на эпохе 31
```
## 1.2 Расширение логистической регрессии
### Добавление поддержки многоклассовой классификации
В функцию make_classification_data из файла utils.py был добавлен аругмет n_classes для подсчета классов и реализовн способ создания данных в зависимости от количества классов
```python
 if n_classes == 2:
    # Бинарная классификация
    w = torch.randn(n_features, 1)
    b = torch.randn(1)
    logits = X @ w + b
    y = (logits > 0).float().squeeze()
else:
    # Многоклассовая классификация
    centers = torch.randn(n_classes, n_features) * 2
    y = torch.randint(0, n_classes, (n,))
    X = centers[y] + torch.randn(n, n_features) * 0.5
```
В класс LogisticRegression так же был добавлен указатель количества классов
```python
def __init__(self, in_features, num_classes):
    super().__init__()
    self.linear = nn.Linear(in_features, num_classes)

def forward(self, x):
    return self.linear(x)
```
### Реализация метрик: precision, recall, F1-score, ROC-AUC
метрики были реализованные с помощью встроенных функция из библиотеки sklearn.metrics
```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
```
Подсчитывались метрики каждую эпоху, выводились вместе со средней потерей каждые 10 эпох
```python
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
```
### Визуализация confusion matrix
```python
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
```
Вывод матрицы после обучения
```python
plot_confusion_matrix(all_targets, all_preds, num_class)
```
### Результаты
```
Размер датасета: 200
Количество батчей: 7
Epoch  10 | Loss: 0.3968 | Accuracy: 0.9300 | Precision: 0.9354 | Recall: 0.9304 | F1-score: 0.9321 | ROC-AUC: 0.9894
Epoch  20 | Loss: 0.2896 | Accuracy: 0.9350 | Precision: 0.9415 | Recall: 0.9332 | F1-score: 0.9368 | ROC-AUC: 0.9909
Epoch  30 | Loss: 0.2478 | Accuracy: 0.9450 | Precision: 0.9489 | Recall: 0.9455 | F1-score: 0.9471 | ROC-AUC: 0.9919
Epoch  40 | Loss: 0.2054 | Accuracy: 0.9500 | Precision: 0.9528 | Recall: 0.9517 | F1-score: 0.9522 | ROC-AUC: 0.9927
Epoch  50 | Loss: 0.2037 | Accuracy: 0.9500 | Precision: 0.9528 | Recall: 0.9517 | F1-score: 0.9522 | ROC-AUC: 0.9928
Epoch  60 | Loss: 0.2002 | Accuracy: 0.9500 | Precision: 0.9528 | Recall: 0.9517 | F1-score: 0.9522 | ROC-AUC: 0.9932
Epoch  70 | Loss: 0.2277 | Accuracy: 0.9500 | Precision: 0.9528 | Recall: 0.9517 | F1-score: 0.9522 | ROC-AUC: 0.9934
Epoch  80 | Loss: 0.1644 | Accuracy: 0.9500 | Precision: 0.9528 | Recall: 0.9517 | F1-score: 0.9522 | ROC-AUC: 0.9936
Epoch  90 | Loss: 0.1681 | Accuracy: 0.9500 | Precision: 0.9528 | Recall: 0.9517 | F1-score: 0.9522 | ROC-AUC: 0.9940
Epoch 100 | Loss: 0.1716 | Accuracy: 0.9500 | Precision: 0.9517 | Recall: 0.9517 | F1-score: 0.9517 | ROC-AUC: 0.9938
```
# Задание 2: Работа с датасетами
## 2.1 Кастомный Dataset класс
В классе реализованы: загрузка данных их файла, предобработка данных, поддрежка различных формтов данных
```python
class CSVread(Dataset):
    def __init__(self, file_path, target_col, num_col=None, cat_col=None, bin_col=None):
        data = pd.read_csv(file_path)
        self.target = torch.FloatTensor(data[target_col].values)

        if num_col is None:
            num_col = data.select_dtypes(include='number').columns.drop([target_col] + (bin_col or [])).tolist()
        if cat_col is None:
            cat_col = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if bin_col:
            data[bin_col] = data[bin_col].astype(int).clip(0, 1)

        preprocessor = make_column_transformer(
            (StandardScaler(), num_col),
            (OneHotEncoder(drop='if_binary', sparse_output=False), cat_col),
            ('passthrough', bin_col or [])
        )

        self.features = torch.FloatTensor(preprocessor.fit_transform(data))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        return self.features[item], self.target[item]
```
## 2.2 Эксперименты с различными датасетами
### Экспримент с моделей линейной регрессии, использовался датасет bigmac.csv
```python
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


bigmac_dataset = CSVread(
    file_path='csv/bigmac.csv',
    target_col='Price in US Dollars'
)

bigmac_dataloader = DataLoader(bigmac_dataset, batch_size=32, shuffle=True)

input_size = bigmac_dataset.features.shape[1]
model = LinearRegression(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    for feat, targ in bigmac_dataloader:
        optimizer.zero_grad()
        outputs = model(feat)
        loss = criterion(outputs, targ.unsqueeze(1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```
### Экспримент с моделей логистической регрессии, использовался датасет Titanic.csv
```python
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Загрузка данных
titanic_dataset = CSVread(
    file_path='csv/Titanic.csv',
    target_col='Survived',
    num_col=['Age', 'Fare', 'Pclass'],
    cat_col=['Sex', 'Embarked'],
    bin_col=None
)

# Инициализация
titanic_dataloader = DataLoader(titanic_dataset, batch_size=32, shuffle=True)
input_size = titanic_dataset.features.shape[1]
model = LogisticRegression(input_size)
criterion = nn.BCEWithLogitsLoss()  # Включает сигмоиду + BCE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 100
for epoch in range(epochs):
    for feat, targ in titanic_dataloader:
        optimizer.zero_grad()

        # Forward pass (возвращает логиты)
        logits = model(feat).squeeze()  # Удаляем лишнюю размерность

        # Вычисление потерь (встроенная сигмоида)
        loss = criterion(logits, targ)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Валидация
    if epoch % 10 == 0:
        with torch.no_grad():
            # Получаем логиты для всего датасета
            logits = model(titanic_dataset.features).squeeze()

            # Применяем сигмоиду для расчета вероятностей
            probs = torch.sigmoid(logits)

            # Расчет метрик
            preds = (probs > 0.5).float()
            acc = (preds == titanic_dataset.target).float().mean()

            print(f'Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}')
```
### Результаты
```
Epoch 0, Loss: 13.9229
Epoch 10, Loss: 9.3317
Epoch 20, Loss: 5.2852
Epoch 30, Loss: 6.1413
Epoch 40, Loss: 3.3893
Epoch 50, Loss: 2.6655
Epoch 60, Loss: 4.7922
Epoch 70, Loss: 1.4634
Epoch 80, Loss: 0.8461
Epoch 90, Loss: 0.9828
Обучение линейной регрессии завершено!

Epoch   0 | Loss: nan | Acc: 0.6162
Epoch  10 | Loss: nan | Acc: 0.6162
Epoch  20 | Loss: nan | Acc: 0.6162
Epoch  30 | Loss: nan | Acc: 0.6162
Epoch  40 | Loss: nan | Acc: 0.6162
Epoch  50 | Loss: nan | Acc: 0.6162
Epoch  60 | Loss: nan | Acc: 0.6162
Epoch  70 | Loss: nan | Acc: 0.6162
Epoch  80 | Loss: nan | Acc: 0.6162
Epoch  90 | Loss: nan | Acc: 0.6162
Обучение логистической регрессии завершено!
```
# Задание 3: Эксперименты и анализ
## 3.1 Исследование гиперпараметров
### Проведение экспериментов с различными: скоростями обучения (learning rate), размерами батчей, оптимизаторами (SGD, Adam, RMSprop)
Код можно посмотреть в файле homework_experiments_hyperparameters.py
### Результат
```
=== Best Configurations ===

For Linear Regression:
Config: {'lr': 0.1, 'batch_size': 32, 'optimizer': 'RMSprop', 'epochs': 50} | Best Validation MSE: 0.0071 at epoch 35
Config: {'lr': 0.01, 'batch_size': 32, 'optimizer': 'RMSprop', 'epochs': 50} | Best Validation MSE: 0.0072 at epoch 37
Config: {'lr': 0.01, 'batch_size': 16, 'optimizer': 'Adam', 'epochs': 50} | Best Validation MSE: 0.0080 at epoch 42
Config: {'lr': 0.1, 'batch_size': 16, 'optimizer': 'RMSprop', 'epochs': 50} | Best Validation MSE: 0.0080 at epoch 26
Config: {'lr': 0.01, 'batch_size': 32, 'optimizer': 'Adam', 'epochs': 50} | Best Validation MSE: 0.0091 at epoch 49

For Logistic Regression:
Config: {'lr': 0.1, 'batch_size': 16, 'optimizer': 'SGD', 'epochs': 50} | Best Validation Accuracy: 1.0000 at epoch 4
Config: {'lr': 0.1, 'batch_size': 16, 'optimizer': 'Adam', 'epochs': 50} | Best Validation Accuracy: 1.0000 at epoch 2
Config: {'lr': 0.1, 'batch_size': 16, 'optimizer': 'RMSprop', 'epochs': 50} | Best Validation Accuracy: 1.0000 at epoch 24
Config: {'lr': 0.1, 'batch_size': 32, 'optimizer': 'SGD', 'epochs': 50} | Best Validation Accuracy: 1.0000 at epoch 4
Config: {'lr': 0.1, 'batch_size': 32, 'optimizer': 'RMSprop', 'epochs': 50} | Best Validation Accuracy: 1.0000 at epoch 2
```
## 3.2 Feature Engineering
### Создание новых признаков для улучшения модели: полиномиальные признаки, взаимодействия между признаками, статистические признаки (среднее, дисперсия)
Код в файле homework_experiments_fe.py
### Результаты
```
Результаты линейной регрессии (MSE):
Базовые: 1.0367
Полиномиальные (степень 2): 0.9770
Взаимодействия: 1.0285
Статистические: 1.0323
Все вместе: 1.0346

Результаты логистической регрессии (Accuracy):
Базовые: 0.4750
Полиномиальные (степень 2): 0.4750
Взаимодействия: 0.4000
Статистические: 0.4500
Все вместе: 0.4250
```
