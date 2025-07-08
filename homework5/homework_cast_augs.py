
from torchvision import transforms
from datasets import CustomImageDataset
from utils import  show_single_augmentation
import matplotlib.pyplot as plt
from extra_augs import (AddGaussianNoise, RandomErasingCustom, CutOut,
                        RandomBlur, RandomPerspective, RandomBrightnessContrast)


root = './data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
original_img, label = dataset[0]
class_names = dataset.get_class_names()

# Создаем пайплайны для новых аугментаций
new_augmentations = [
    ("RandomBlur", RandomBlur(p=1.0)),
    ("RandomPerspective", RandomPerspective(p=1.0)),
    ("RandomBrightnessContrast", RandomBrightnessContrast(p=1.0))
]

# Создаем пайплайны для аугментаций из extra_augs.py
existing_augmentations = [
    ("AddGaussianNoise", AddGaussianNoise(0., 0.2)),
    ("RandomErasingCustom", RandomErasingCustom(p=1.0)),
    ("CutOut", CutOut(p=1.0, size=(32, 32)))
]

# Визуализация
plt.figure(figsize=(10, 5))
plt.imshow(original_img)
plt.title(f"Оригинал: {class_names[label]}")
plt.axis('off')
plt.show()

print("=== Новые аугментации ===")
for name, aug in new_augmentations:
    aug_transform = transforms.Compose([transforms.ToTensor(), aug])
    aug_img = aug_transform(original_img)
    show_single_augmentation(original_img, aug_img, name)

print("\n=== Аугментации из extra_augs.py ===")
for name, aug in existing_augmentations:
    aug_transform = transforms.Compose([transforms.ToTensor(), aug])
    aug_img = aug_transform(original_img)
    show_single_augmentation(original_img, aug_img, name)
