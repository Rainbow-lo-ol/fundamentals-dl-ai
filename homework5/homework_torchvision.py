import torch
from torchvision import transforms
from PIL import Image
from datasets import CustomImageDataset
from utils import show_images, show_single_augmentation, show_multiple_augmentations
import matplotlib.pyplot as plt
import numpy as np


# Загрузка датасета без аугментаций и изображений из каждого класса
root = './data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
class_name = dataset.get_class_names()

org_images = []
labels = []
classes_covered = set()

for img, label in dataset:
    if label not in classes_covered and len(classes_covered) != 5:
        org_images.append(img)
        labels.append(label)
        classes_covered.add(label)
    if len(classes_covered) == 5:
        break

standard_augs = [
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
    ("RandomCrop", transforms.RandomCrop(200, padding=20)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
]

for i, (img, label) in enumerate(zip(org_images, labels)):
    print(f"\n=== Класс: {class_name[label]} ===")

    # Показываем оригинал
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Оригинал: {class_name[label]}")
    plt.axis('off')
    plt.show()

    # Применяем и показываем каждую аугментацию отдельно
    augmented_imgs = []
    titles = []

    for name, aug in standard_augs:
        aug_transform = transforms.Compose([
            transforms.ToTensor(),
            aug
        ])
        aug_img = aug_transform(img)
        show_single_augmentation(img, aug_img, name)
        augmented_imgs.append(aug_img)
        titles.append(name)

    # Применяем все аугментации вместе
    combined_aug = transforms.Compose([
        transforms.ToTensor(),
        *[aug for _, aug in standard_augs]
    ])

    combined_img = combined_aug(img)
    show_single_augmentation(img, combined_img, "Все аугментации")

    # Показываем все аугментации в одной сетке
    show_multiple_augmentations(img, augmented_imgs, titles)




















