# Задание 1: Создание и манипуляции с тензорами 
## 1.1 Создание тензоров
### Тензор размером 3x4, заполненный случайными числами от 0 до 1
```python
x = torch.rand(3, 4)
```
### Тензор размером 2x3x4, заполненный нулями
```python
x = torch.zeros(2, 3, 4)
```
### Тензор размером 5x5, заполненный единицами
```python
x = torch.ones(5, 5)
```
### Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
```python
x = torch.arange(0, 16).reshape(4, 4)
```

## 1.2 Операции с тензорами
### Создание тензоров
```python
A = torch.arange(0, 12).reshape(3, 4)
B = torch.arange(0, 12).reshape(4, 3)
```
### Транспонирование тензора A
```python
A_transposed = A.transpose(0, 1)
```
### Матричное умножение A и B
```python
multiplication_martix = A @ B
```
### Поэлементное умножение A и транспонированного B
```python
B_transposed = B.transpose(0, 1)
multiplication = A * B_transposed
```
### Вычисление суммы всех элементов тензора A
```python
sum_A = A.sum()
```

## 1.3 Индексация и срезы
### Создание тензора размером 5x5x5
```python
C = torch.arange(0, 125).reshape(5, 5, 5)
```
### Извлечение первой строки
```python
print(f'Первая строка: {C[0, 0, :]}')
```
### Извлечение последнео столбца
```python
print(f'Последний столбец: {C[-1, :, -1]}')
```
### Извлечение подматрицы размером 2х2 из цента матрицы
```python
print(f'Подматрица 2x2 из центра тензора: {C[C.shape[0]//2, C.shape[1]//2 - 1:C.shape[1]//2 + 1, C.shape[2]//2 - 1:C.shape[2]//2 + 1]}')
```
### Извлечение всех элементов с четным индексом
```python
print(f'Все элементы с четными индексами: {C[::2, ::2, ::2]}')
```

## 1.4 Работа с формами
### Создание тензора размером 24 элемента
```python
D = torch.arange(24)
```
### Изменение размера тензора 
```python
print(f'Размер тензора 2x12: {D.reshape(2, 12)}')
print(f'Размер тензора 3x8: {D.reshape(3, 8)}')
print(f'Размер тензора 4x6: {D.reshape(4, 6)}')
print(f'Размер тензора 2x3x4: {D.reshape(2, 3, 4)}')
print(f'Размер тензора 2x2x2x3: {D.reshape(2, 2, 2, 3)}')
```


# Задание 2: Автоматическое дифференцирование
## 2.2 Простые вычисления с градиентами
### Создание тензоров с requires_grad=True
```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)
```
### Вычисление функции f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
```python
f = x**2 + y**2 + z**2 + 2*x*y*z
```
### Нахождение градиентов по всем перменным
```python
f.backward()
print(f'x.grad = {x.grad.item()}')
print(f'y.grad = {y.grad.item()}')
print(f'z.grad = {z.grad.item()}')
```
### Проверка результата аналитически
### Функция для аналитического вычисления градиента
```python
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
```
### Аналитическое вычисление градиента
```python
df_dx, df_dy, df_dz = analytical_gradients(1.0, 2.0, 3.0)
```
### Сравнение результатов после двух способов вычисления градиентов
```python
print(f"df/dx аналитически: {df_dx}, PyTorch: {x.grad.item()}")
print(f"df/dy аналитически: {df_dy}, PyTorch: {y.grad.item()}")
print(f"df/dz аналитически: {df_dz}, PyTorch: {z.grad.item()}")
```
### Вывод после сравнения
```python
df/dx аналитически: 14.0, PyTorch: 14.0
df/dy аналитически: 10.0, PyTorch: 10.0
df/dz аналитически: 10.0, PyTorch: 10.0
```
Результаты полностью совпадает, значит можем сделать вывод, что оба способа вычиляют градиент верно и полностью эквиваленты

## 2.2 Градиент функции потерь
### Реализованная функция потерь
```python
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
```
### Задание входных данных и целевых значений
```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([5.0, 6.0, 7.0, 8.0])
```
### Задание параметров модели с отслеживанием градиентов
```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
```
### Вычисление градиентов
```python
df_dw, df_db = mse_loss(x, y_true, w, b)
```
### Результаты
```python
df/dw: None
df/db: 39.0
```

## 2.3 Цепное правило
### Реализация составной функции f(x) = sin(x^2 + 1)
```python
def sin_fuc(x):
    """
    Вычисление функции sin(x^2 + 1)
    :param x: аргумент функции
    :return: значение функции
    """
    return torch.sin(x**2 + 1)
```
### Задание точки, в которой будет вычислять градиент
```python
x = torch.tensor(1.0, requires_grad=True)
```
### Вычисление через backward
```python
f = sin_fuc(x)
z = f.sum()
f.backward()
grad_backward = x.grad
```
### Вычисление через torch.autograd.grad
```python
grad_autograd = torch.autograd.grad(sin_fuc(x), x)
```
### Результаты
```python
Градиент через backward(): -0.832293689250946
Градиент через autograd.grad: (tensor(-0.8323),)
```
Из вывода результатов можем сделать вывод, что backward считает более точное число, чем autograd.grad, но при округление ответ будет одинаковый


# Задание 3: Сравнение производительности CPU vs CUDA
### Создание больших матриц
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m1 = torch.rand(64, 1024, 1024, device=device)
m2 = torch.rand(128, 512, 512, device=device)
m3 = torch.rand(256, 256, 256, device=device)
```
### Реализоция функции измерения времени
```python
def measure_time(operation, device='cuda'):
    """
    Измеряет время выполнения одной операции на указанном устройстве
    :param operation: функция с операцией для измерения
    :param device: 'cuda' или 'cpu'
    :return: время выполнения в миллисекундах
    """
    if device == 'cuda' and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)
    else:
        start_time = time.time()
        operation()
        return (time.time() - start_time) * 1000
```
### Задание операций для тестирования
```python
operations = [
    ("Матричное умножение", lambda: torch.matmul(m1, m1.transpose(-2, -1))),
    ("Поэлементное сложение", lambda: m2 + m2),
    ("Поэлементное умножение", lambda: m3 * m3),
    ("Транспонирование", lambda: m1.transpose(-2, -1)),
    ("Сумма всех элементов", lambda: m2.sum())
]
```
### Измерение производительности
```python
results = []
for op_name, op in operations:
    row = {'Операция': op_name}

    #CPU
    if device.type == 'cuda':
        m1_cpu = m1.cpu()
        m2_cpu = m2.cpu()
        m3_cpu = m3.cpu()

        #Заменяем операции для CPU
        if op_name == "Матричное умножение":
            cpu_op = lambda: torch.matmul(m1_cpu, m1_cpu.transpose(-2, -1))
        elif op_name == "Поэлементное сложение":
            cpu_op = lambda: m2_cpu + m2_cpu
        elif op_name == "Поэлементное умножение":
            cpu_op = lambda: m3_cpu * m3_cpu
        elif op_name == "Транспонирование":
            cpu_op = lambda: m1_cpu.transpose(-2, -1)
        elif op_name == "Сумма всех элементов":
            cpu_op = lambda: m2_cpu.sum()

        cpu_time = measure_time(cpu_op, device='cpu')
    else:
        cpu_time = measure_time(op, device='cpu')

    row['CPU (мс)'] = f"{cpu_time:.3f}"

    #GPU
    if torch.cuda.is_available():
        gpu_time = measure_time(op, device='cuda')
        row['GPU (мс)'] = f"{gpu_time:.3f}"
        row['Ускорение'] = f"{cpu_time / gpu_time:.1f}x"
    else:
        row['GPU (мс)'] = "N/A"
        row['Ускорение'] = "N/A"

    results.append(row)
```
### Результаты
```python
+------------------------+------------+------------+-------------+
| Операция               |   CPU (мс) |   GPU (мс) | Ускорение   |
+========================+============+============+=============+
| Матричное умножение    |   1008.54  |    239.941 | 4.2x        |
+------------------------+------------+------------+-------------+
| Поэлементное сложение  |     53.113 |     30.152 | 1.8x        |
+------------------------+------------+------------+-------------+
| Поэлементное умножение |     19.826 |     15.388 | 1.3x        |
+------------------------+------------+------------+-------------+
| Транспонирование       |      1     |      0.057 | 17.7x       |
+------------------------+------------+------------+-------------+
| Сумма всех элементов   |     18.829 |     25.601 | 0.7x        |
+------------------------+------------+------------+-------------+
```
По результатам тестирования получаются следующие выводы: 
- GPU показывает преимущество в 4 из 5 операций
- Наибольшее ускорение: транспонирование (17.7x) и матричное умножение (4.2x)
- Наименьшее ускорение: операция суммирования выполняется медленнее на GPU (0.7x)

GPU обеспечивает значительный прирост производительности для сложных операций, но для простых операций с малыми данными CPU может быть предпочтительнее.



































