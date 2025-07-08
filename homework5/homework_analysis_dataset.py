import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict


# Подсчет количества изображений в каждом классе
def count_images_per_class(root_dir):
    class_counts = defaultdict(int)
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            class_counts[class_name] = len(images)
    return class_counts


# Анализ размеров изображений
def analyze_image_sizes(root_dir):
    widths = []
    heights = []

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    with Image.open(img_path) as img:
                        widths.append(img.width)
                        heights.append(img.height)

    return widths, heights


# Визуализация
def visualize_results(class_counts, widths, heights):
    # Гистограмма по классам
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Количество изображений по классам')
    plt.xlabel('Классы')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Распределение размеров
    plt.figure(figsize=(12, 6))
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Распределение размеров изображений')
    plt.xlabel('Ширина (px)')
    plt.ylabel('Высота (px)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Статистика
    print("\n=== Статистика размеров ===")
    print(f"Минимальная ширина: {min(widths)}px")
    print(f"Максимальная ширина: {max(widths)}px")
    print(f"Средняя ширина: {np.mean(widths):.1f}px")
    print(f"\nМинимальная высота: {min(heights)}px")
    print(f"Максимальная высота: {max(heights)}px")
    print(f"Средняя высота: {np.mean(heights):.1f}px")
    print(f"\nМинимальное соотношение сторон: {min(w / h for w, h in zip(widths, heights)):.2f}")
    print(f"Максимальное соотношение сторон: {max(w / h for w, h in zip(widths, heights)):.2f}")


# Основная функция
def analyze_dataset(root_dir):
    class_counts = count_images_per_class(root_dir)

    widths, heights = analyze_image_sizes(root_dir)

    print("=== Количество изображений по классам ===")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} изображений")

    visualize_results(class_counts, widths, heights)


if __name__ == '__main__':
    dataset_path = './data/train'
    analyze_dataset(dataset_path)