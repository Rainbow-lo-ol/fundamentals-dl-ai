import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]

    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]

    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow * 2, 2))
    if nrow == 1:
        axes = [axes]

    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_single_augmentation(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)

    # Обработка оригинального изображения
    if isinstance(original_img, Image.Image):  # Если это PIL.Image
        orig_resized = resize_transform(original_img)
        orig_np = np.array(orig_resized)
        if orig_np.max() > 1:  # Если значения в [0, 255]
            orig_np = orig_np / 255.0
    elif torch.is_tensor(original_img):  # Если это тензор
        orig_resized = resize_transform(original_img)
        orig_np = orig_resized.permute(1, 2, 0).numpy()
    else:
        raise TypeError("Неподдерживаемый тип изображения")

    # Обработка аугментированного изображения (ожидается тензор)
    aug_resized = resize_transform(augmented_img)
    aug_np = aug_resized.permute(1, 2, 0).numpy()

    # Отображение
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')

    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def show_multiple_augmentations(original_img, augmented_imgs, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)

    # Оригинальное изображение

    if isinstance(original_img, Image.Image):  # Если это PIL.Image
        orig_resized = resize_transform(original_img)
        orig_np = np.array(orig_resized)
        if orig_np.max() > 1:  # Если значения в [0, 255]
            orig_np = orig_np / 255.0
    elif torch.is_tensor(original_img):  # Если это тензор
        orig_resized = resize_transform(original_img)
        orig_np = orig_resized.permute(1, 2, 0).numpy()
    else:
        raise TypeError("Неподдерживаемый тип изображения")
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()