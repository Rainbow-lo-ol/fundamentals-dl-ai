import os
import time
import tracemalloc
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import random


class SizeExperiment:
    def __init__(self, dataset_path="data/train"):
        self.dataset_path = dataset_path
        self.sizes = [64, 128, 224, 512]
        self.results = []

    def create_pipeline(self):
        """Создает безопасный пайплайн аугментаций"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            self.SafeAddGaussianNoise(0., 0.1),
            self.SafeRandomErasing(p=0.5),
            self.SafeCutOut(p=0.5, size=32)
        ])

    class SafeAddGaussianNoise:
        """Безопасная версия AddGaussianNoise"""

        def __init__(self, mean=0., std=0.1):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            return tensor + torch.randn_like(tensor) * self.std + self.mean

    class SafeRandomErasing:
        """Безопасная версия RandomErasing"""

        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            if random.random() < self.p:
                c, h, w = tensor.shape
                area = h * w
                erase_area = random.uniform(0.02, 0.2) * area
                erase_w = int(erase_area ** 0.5)
                erase_h = int(erase_area / erase_w)
                x = random.randint(0, w - erase_w)
                y = random.randint(0, h - erase_h)
                tensor[:, y:y + erase_h, x:x + erase_w] = 0
            return tensor

    class SafeCutOut:
        """Безопасная версия CutOut"""

        def __init__(self, p=0.5, size=32):
            self.p = p
            self.size = size

        def __call__(self, tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            if random.random() < self.p:
                c, h, w = tensor.shape
                x = random.randint(0, w - self.size)
                y = random.randint(0, h - self.size)
                tensor[:, y:y + self.size, x:x + self.size] = 0
            return tensor

    def load_images(self, size, num_images=100):
        """Загружает и изменяет размер изображений"""
        images = []
        count = 0

        for class_name in os.listdir(self.dataset_path):
            if count >= num_images:
                break

            class_dir = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if count >= num_images:
                    break

                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((size, size), Image.LANCZOS)
                    images.append(img)
                    count += 1
                except Exception as e:
                    print(f"Ошибка загрузки {img_path}: {str(e)}")
                    continue

        return images

    def run_experiment(self, num_images=100):
        """Проводит эксперимент для всех размеров"""
        for size in self.sizes:
            print(f"\n--- Размер {size}x{size} ---")

            # Загрузка изображений
            start_time = time.time()
            images = self.load_images(size, num_images)
            load_time = time.time() - start_time

            # Создание пайплайна
            pipeline = self.create_pipeline()

            # Измерение памяти и времени
            tracemalloc.start()
            start_time = time.time()

            for img in images:
                try:
                    # Применяем пайплайн
                    result = pipeline(img)
                    # Убедимся, что возвращается только тензор
                    if isinstance(result, (tuple, list)):
                        result = result[0]
                except Exception as e:
                    print(f"Ошибка аугментации: {str(e)}")
                    continue

            aug_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Сохраняем результаты
            self.results.append({
                'size': size,
                'load_time': load_time,
                'aug_time': aug_time,
                'memory_peak_mb': peak / (1024 * 1024),
                'images_processed': len(images)
            })

            print(f"Изображений: {len(images)}")
            print(f"Загрузка: {load_time:.2f} сек")
            print(f"Аугментация: {aug_time:.2f} сек")
            print(f"Память: {peak / (1024 * 1024):.2f} МБ")

    def plot_results(self):
        """Визуализирует результаты"""
        if not self.results:
            print("Нет данных для визуализации")
            return

        df = pd.DataFrame(self.results)

        plt.figure(figsize=(15, 5))

        # График времени загрузки
        plt.subplot(1, 3, 1)
        plt.plot(df['size'], df['load_time'], 'bo-')
        plt.title('Время загрузки')
        plt.xlabel('Размер (px)')
        plt.ylabel('Время (сек)')
        plt.grid(True)

        # График времени аугментации
        plt.subplot(1, 3, 2)
        plt.plot(df['size'], df['aug_time'], 'ro-')
        plt.title('Время аугментации')
        plt.xlabel('Размер (px)')
        plt.ylabel('Время (сек)')
        plt.grid(True)

        # График потребления памяти
        plt.subplot(1, 3, 3)
        plt.plot(df['size'], df['memory_peak_mb'], 'go-')
        plt.title('Пиковая память')
        plt.xlabel('Размер (px)')
        plt.ylabel('Память (МБ)')
        plt.grid(True)

        plt.tight_layout()

        # Сохраняем результаты
        os.makedirs("size_experiment", exist_ok=True)
        plt.savefig("size_experiment/results.png")
        df.to_csv("size_experiment/data.csv", index=False)
        plt.show()


if __name__ == "__main__":
    experiment = SizeExperiment()
    experiment.run_experiment(num_images=100)
    experiment.plot_results()