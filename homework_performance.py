import torch
import time
from tabulate import tabulate

#3.1 Подготовка данных
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m1 = torch.rand(64, 1024, 1024, device=device)
m2 = torch.rand(128, 512, 512, device=device)
m3 = torch.rand(256, 256, 256, device=device)


#3.2 Функция измерения времени
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


#Операции для тестирования
operations = [
    ("Матричное умножение", lambda: torch.matmul(m1, m1.transpose(-2, -1))),
    ("Поэлементное сложение", lambda: m2 + m2),
    ("Поэлементное умножение", lambda: m3 * m3),
    ("Транспонирование", lambda: m1.transpose(-2, -1)),
    ("Сумма всех элементов", lambda: m2.sum())
]

#Измерение производительности
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

print(tabulate(results, headers="keys", tablefmt="grid"))
print("\nРазмеры матриц:")
print(f"- Матрица 1: {m1.shape} (~{m1.element_size() * m1.nelement() / 1024 ** 2:.0f} МБ)")
print(f"- Матрица 2: {m2.shape} (~{m2.element_size() * m2.nelement() / 1024 ** 2:.0f} МБ)")
print(f"- Матрица 3: {m3.shape} (~{m3.element_size() * m3.nelement() / 1024 ** 2:.0f} МБ)")
if torch.cuda.is_available():
    print(f"\nУстройство GPU: {torch.cuda.get_device_name(0)}")