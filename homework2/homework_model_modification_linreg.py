import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import make_regression_data, mse, log_epoch_lin, RegressionDataset


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Создаём датасет и даталоадер, разделяя их на тренировочные и валидационные данные
    dataset = RegressionDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    print(f'Размер тренировочного датасета: {len(train_dataset)}')
    print(f'Размер валидационного датасета: {len(val_dataset)}')

    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()

    l1_lam = 0.1
    l2_lam = 0.1
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=l2_lam)

    # Параметры для early stopping
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    min_delta = 0.001

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        train_loss = 0

        for i, (batch_X, batch_y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            l1_reg = torch.zeros(())
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += l1_reg * l1_lam

            # l2_reg = torch.zeros(())
            # for param in model.parameters():
            #     l2_reg += torch.norm(param, 2)
            # loss += l2_reg * l2_lam

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                y_pred = model(batch_X)
                val_loss += criterion(y_pred, batch_y)
        avg_val_loss = val_loss / len(val_dataloader)

        # Early stopping логика
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping на эпохе {epoch}')
            break

        if epoch % 10 == 0:
            log_epoch_lin(epoch, avg_train_loss, avg_val_loss)

    # Сохраняем модель
    torch.save(model.state_dict(), 'models/linreg_torch.pth')