import torch


#2.1 Простые вычисления с градиентами
#Создание тензоров с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

#Вычисление функции
f = x**2 + y**2 + z**2 + 2*x*y*z

#Нахождение градиентов по всем перменным
f.backward()
print(f'x.grad = {x.grad.item()}')
print(f'y.grad = {y.grad.item()}')
print(f'z.grad = {z.grad.item()}')


def analytical_gradients(x, y, z):
    """
    Функция для аналитического вычисления градиента
    :param x: значение тензора x
    :param y: значение тензора y
    :param z: значение тензора z
    :return: значения градиентов
    """
    df_dx = 2*x + 2*y*z
    df_dy = 2*y + 2*x*z
    df_dz = 2*z + 2*x*y
    return df_dx, df_dy, df_dz

#Аналитическое вычисление градиента
df_dx, df_dy, df_dz = analytical_gradients(1.0, 2.0, 3.0)

#Сравнение
print(f"df/dx аналитически: {df_dx}, PyTorch: {x.grad.item()}")
print(f"df/dy аналитически: {df_dy}, PyTorch: {y.grad.item()}")
print(f"df/dz аналитически: {df_dz}, PyTorch: {z.grad.item()}")


#2.2 Градиент функции потерь
def mse_loss(w, x, b, y_true):
    """
    Вычисляет MSE и градиенты для линейной модели y_pred = w*x + b
    :param w: вес
    :param x: входные данные
    :param b: смещение
    :param y_true: целевые значения
    :return: градиенты по w и b
    """

    y_pred = w*x + b
    f = torch.mean((y_pred - y_true)**2)
    f.backward()
    return w.grad, b.grad


#Задание входных данных и целевых значений
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([5.0, 6.0, 7.0, 8.0])

#Параметры модели с отслеживанием градиентов
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

#Вычисляем градиенты
df_dw, df_db = mse_loss(x, y_true, w, b)

print(f"df/dw: {df_dw}")
print(f"df/db: {df_db}")


#2.3 Цепное правило
def sin_fuc(x):
    """
    Вычисление функции sin(x^2 + 1)
    :param x: аргумент функции
    :return: значение функции
    """
    return torch.sin(x**2 + 1)


#Точка, в которой будет вычислять градиент
x = torch.tensor(1.0, requires_grad=True)

#Вычисление через backward
f = sin_fuc(x)
z = f.sum()
f.backward()
grad_backward = x.grad

#Вычисление через torch.autograd.grad
grad_autograd = torch.autograd.grad(sin_fuc(x), x)

print(f"Градиент через backward(): {grad_backward}")
print(f"Градиент через autograd.grad: {grad_autograd}")













