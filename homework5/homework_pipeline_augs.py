import json
import os
import random
from datetime import datetime
from collections import OrderedDict
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps  # Добавляем импорт ImageOps
from extra_augs import AddGaussianNoise, RandomErasingCustom, CutOut, Solarize, Posterize, AutoContrast, \
    ElasticTransform


class AugmentationPipeline:
    def __init__(self):
        self.augmentations = OrderedDict()

    def add_augmentation(self, name, augmentation):
        """Добавляет аугментацию в пайплайн"""
        self.augmentations[name] = augmentation
        return self

    def remove_augmentation(self, name):
        """Удаляет аугментацию из пайплайна"""
        if name in self.augmentations:
            del self.augmentations[name]
        return self

    def apply(self, image):
        """Применяет все аугментации к изображению"""
        # Конвертируем в тензор, если это PIL Image
        if isinstance(image, Image.Image):
            tensor = transforms.ToTensor()(image)
        else:
            tensor = image.clone() if isinstance(image, torch.Tensor) else torch.tensor(image)

        # Применяем все аугментации
        for name, aug in self.augmentations.items():
            try:
                tensor = aug(tensor)
            except Exception as e:
                print(f"Ошибка при применении {name}: {str(e)}")
                continue

        return tensor

    def get_augmentations(self):
        """Возвращает список имен аугментаций"""
        return list(self.augmentations.keys())

    def get_config(self):
        """Возвращает конфигурацию для сохранения"""
        return {
            "augmentations": self.get_augmentations(),
            "timestamp": datetime.now().isoformat()
        }


def create_pipelines():
    """Создает 3 уровня аугментаций с исправленным AutoContrast"""
    return {
        "light": (AugmentationPipeline()
                  .add_augmentation("Flip", transforms.RandomHorizontalFlip(p=0.3))
                  .add_augmentation("Noise", AddGaussianNoise(0., 0.05))),

        "medium": (AugmentationPipeline()
                   .add_augmentation("Flip", transforms.RandomHorizontalFlip(p=0.5))
                   .add_augmentation("Noise", AddGaussianNoise(0., 0.1))
                   .add_augmentation("Erase", RandomErasingCustom(p=0.3, scale=(0.02, 0.1)))),

        "heavy": (AugmentationPipeline()
                  .add_augmentation("Flip", transforms.RandomHorizontalFlip(p=0.7))
                  .add_augmentation("Noise", AddGaussianNoise(0., 0.2))
                  .add_augmentation("CutOut", CutOut(p=0.5, size=(32, 32))))
    }


def process_and_save(dataset_path="data/train", output_dir="aug_results", samples_per_class=2):
    """Основная функция обработки и сохранения"""
    os.makedirs(output_dir, exist_ok=True)
    pipelines = create_pipelines()
    results = {
        "configs": {name: pipe.get_config() for name, pipe in pipelines.items()},
        "examples": []
    }

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in class_images[:samples_per_class]:
            img_path = os.path.join(class_dir, img_name)

            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')

                    sample = {
                        "original": img_name,
                        "class": class_name,
                        "original_size": img.size,
                        "augmented": {}
                    }

                    for pipe_name, pipeline in pipelines.items():
                        augmented = pipeline.apply(img)
                        sample["augmented"][pipe_name] = {
                            "augmentations": pipeline.get_augmentations(),
                            "output_type": "tensor" if isinstance(augmented, torch.Tensor) else str(type(augmented))
                        }

                    results["examples"].append(sample)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {str(e)}")
                continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"augmentations_{timestamp}.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Результаты успешно сохранены в {output_file}")
    return output_file


if __name__ == "__main__":
    process_and_save()