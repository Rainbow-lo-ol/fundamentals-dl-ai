import os
import shutil
import random
from tqdm import tqdm
import cv2
import numpy as np


path = './data'
path_without_aug = './data_without_aug'
path_with_aug = './data_with_aug'


def balance_dataset(
        dataset_type: str = 'both',
        balance_method: str = 'min',  # 'min', 'max' или число
        use_augmentation: bool = False,
        random_seed: int = 42
) -> None:
    """
    Балансирует датасет с возможностью аугментации (без зависимости от imgaug)

    Параметры:
        base_path: Путь к исходным данным
        output_path: Путь для сохранения сбалансированных данных
        dataset_type: 'train', 'test' или 'both'
        balance_method: 'min', 'max' или конкретное число
        use_augmentation: Использовать ли аугментацию
        random_seed: Seed для воспроизводимости
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    for dataset in ['train', 'test']:
        if dataset_type not in ['both', dataset]:
            continue

        input_dir = os.path.join(path, dataset)
        if balance_method == 'max':
            output_dir = os.path.join(path_with_aug, dataset)
        else:
            output_dir = os.path.join(path_without_aug, dataset)

        # Анализ исходных данных
        emotions = [d for d in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, d))]

        counts = {}
        for emotion in emotions:
            emotion_path = os.path.join(input_dir, emotion)
            images = [f for f in os.listdir(emotion_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            counts[emotion] = len(images)

        # Определяем целевое количество
        if isinstance(balance_method, int):
            target_count = balance_method
        elif balance_method == 'max':
            target_count = max(counts.values())
        else:  # 'min' по умолчанию
            target_count = min(counts.values())

        # Обработка каждой эмоции
        for emotion in tqdm(emotions, desc=f"Обработка {dataset}"):
            input_emotion_path = os.path.join(input_dir, emotion)
            output_emotion_path = os.path.join(output_dir, emotion)
            os.makedirs(output_emotion_path, exist_ok=True)

            images = [f for f in os.listdir(input_emotion_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            current_count = len(images)

            if use_augmentation:
                # Копируем все оригинальные изображения
                for img in images:
                    src = os.path.join(input_emotion_path, img)
                    dst = os.path.join(output_emotion_path, img)
                    shutil.copy2(src, dst)

                # Добавляем аугментированные изображения если нужно
                if current_count < target_count:
                    needed = target_count - current_count
                    images_to_augment = random.sample(images, min(len(images), needed))

                    for i, img_name in enumerate(images_to_augment):
                        img_path = os.path.join(input_emotion_path, img_name)
                        img = cv2.imread(img_path)

                        # Простые аугментации с помощью OpenCV
                        for aug_num in range(max(1, needed // len(images_to_augment))):
                            if len(os.listdir(output_emotion_path)) >= target_count:
                                break

                            # Случайные преобразования
                            augmented = img.copy()

                            # Горизонтальное отражение (50% chance)
                            if random.random() > 0.5:
                                augmented = cv2.flip(augmented, 1)

                            # Небольшой поворот
                            angle = random.uniform(-15, 15)
                            h, w = augmented.shape[:2]
                            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                            augmented = cv2.warpAffine(augmented, M, (w, h))

                            # Небольшое размытие
                            if random.random() > 0.5:
                                ksize = random.choice([3, 5])
                                augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

                            # Изменение яркости/контраста
                            alpha = random.uniform(0.8, 1.2)
                            beta = random.uniform(-10, 10)
                            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

                            # Сохранение
                            new_name = f"aug_{i}_{aug_num}_{img_name}"
                            cv2.imwrite(os.path.join(output_emotion_path, new_name), augmented)
            else:
                # Без аугментации - просто выбираем подмножество
                selected = random.sample(images, min(len(images), target_count))
                for img in selected:
                    src = os.path.join(input_emotion_path, img)
                    dst = os.path.join(output_emotion_path, img)
                    shutil.copy2(src, dst)


if __name__ == "__main__":
    # Пример использования без аугментации
    balance_dataset(
        use_augmentation=False
    )

    # Пример использования с аугментацией
    balance_dataset(
        use_augmentation=True,
        balance_method='max'
    )