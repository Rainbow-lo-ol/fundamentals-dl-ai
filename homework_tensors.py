import torch

#1.1 Создание тензоров
#Тензор размером 3x4, заполненный случайными числами от 0 до 1
x1 = torch.rand(3, 4)
print(x1)

#Тензор размером 2x3x4, заполненный нулями
x2 = torch.zeros(2, 3, 4)
print(x2)

#Тензор размером 5x5, заполненный единицами
x3 = torch.ones(5, 5)
print(x3)

#Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
x4 = torch.arange(0, 16).reshape(4, 4)
print(x4)


#1.2 Операции с тензорами
A = torch.arange(0, 12).reshape(3, 4)
B = torch.arange(0, 12).reshape(4, 3)

#Транспонирование тензора A
A_transposed = A.transpose(0, 1)
print(f'Исходный тензон: {A}')
print(f'Транспонированный тензор: {A_transposed}')

#Матричное умножение A и B
print(f'Матричное умножение A @ B: {A @ B}')

#Поэлементное умножение A и транспонированного B
B_transposed = B.transpose(0, 1)
print(f'Поэлементное умножение: {A * B_transposed}')

#Вычислите сумму всех элементов тензора A
print(f'Сумма всех элементов тензора A: {A.sum()}')


#1.3 Индексация и срезы
C = torch.arange(0, 125).reshape(5, 5, 5)
print(f'Первая строка: {C[0, 0, :]}')
print(f'Последний столбец: {C[-1, :, -1]}')
print(f'Подматрица 2x2 из центра тензора: {C[C.shape[0]//2, C.shape[1]//2 - 1:C.shape[1]//2 + 1, C.shape[2]//2 - 1:C.shape[2]//2 + 1]}')
print(f'Все элементы с четными индексами: {C[::2, ::2, ::2]}')


#1.4 Работа с формами
D = torch.arange(24)
print(D)
print(f'Размер тензора 2x12: {D.reshape(2, 12)}')
print(f'Размер тензора 3x8: {D.reshape(3, 8)}')
print(f'Размер тензора 4x6: {D.reshape(4, 6)}')
print(f'Размер тензора 2x3x4: {D.reshape(2, 3, 4)}')
print(f'Размер тензора 2x2x2x3: {D.reshape(2, 2, 2, 3)}')