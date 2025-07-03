import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CSVread import CSVread


# Модель линейной регрессии
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Загрузка данных
bigmac_dataset = CSVread(
    file_path='csv/bigmac.csv',
    target_col='Price in US Dollars'
)

# Инициализация
bigmac_dataloader = DataLoader(bigmac_dataset, batch_size=32, shuffle=True)
input_size = bigmac_dataset.features.shape[1]
model = LinearRegression(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
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

# Сохранение модели
torch.save(model.state_dict(), 'models/bigmac.pth')
print("Обучение линейной регрессии завершено!\n")


# Модель логистической регрессии
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
        logits = model(feat).squeeze()
        loss = criterion(logits, targ)
        loss.backward()
        optimizer.step()

    # Валидация
    if epoch % 10 == 0:
        with torch.no_grad():
            logits = model(titanic_dataset.features).squeeze()
            probs = torch.sigmoid(logits)

            # Расчет метрик
            preds = (probs > 0.5).float()
            acc = (preds == titanic_dataset.target).float().mean()

            print(f'Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}')

# Сохранение модели
torch.save(model.state_dict(), 'models/titanic.pth')
print("Обучение логистической регрессии завершено!")