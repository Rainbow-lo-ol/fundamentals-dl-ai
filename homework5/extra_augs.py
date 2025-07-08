import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms


class AddGaussianNoise:
    """Добавляет гауссов шум к изображению."""

    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class RandomErasingCustom:
    """Случайно затирает прямоугольную область изображения."""

    def __init__(self, p=0.5, scale=(0.02, 0.2)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        c, h, w = img.shape
        area = h * w
        erase_area = random.uniform(*self.scale) * area
        erase_w = int(np.sqrt(erase_area))
        erase_h = int(erase_area // erase_w)
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        img[:, y:y + erase_h, x:x + erase_w] = 0
        return img


class CutOut:
    """Вырезает случайную прямоугольную область из изображения."""

    def __init__(self, p=0.5, size=(16, 16)):
        self.p = p
        self.size = size

    def __call__(self, img):
        if random.random() > self.p:
            return img
        c, h, w = img.shape
        cut_h, cut_w = self.size
        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)
        img[:, y:y + cut_h, x:x + cut_w] = 0
        return img


class Solarize:
    """Инвертирует пиксели выше порога."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img_np = img.numpy()
        mask = img_np > self.threshold / 255.0
        img_np[mask] = 1.0 - img_np[mask]
        return torch.from_numpy(img_np)


class Posterize:
    """Уменьшает количество бит на канал."""

    def __init__(self, bits=4):
        self.bits = bits

    def __call__(self, img):
        img_np = img.numpy()
        factor = 2 ** (8 - self.bits)
        img_np = (img_np * 255).astype(np.uint8)
        img_np = (img_np // factor) * factor
        return torch.from_numpy(img_np.astype(np.float32) / 255.0)


class AutoContrast:
    """Автоматически улучшает контраст изображения."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = img.numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil = ImageOps.autocontrast(img_pil)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        return torch.from_numpy(img_np.transpose(2, 0, 1))


class ElasticTransform:
    """Эластичная деформация изображения."""

    def __init__(self, p=0.5, alpha=1, sigma=50):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = img.numpy().transpose(1, 2, 0)
        h, w = img_np.shape[:2]

        # Создаем случайные смещения
        dx = np.random.randn(h, w) * self.alpha
        dy = np.random.randn(h, w) * self.alpha

        # Сглаживаем смещения
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)

        # Применяем деформацию
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x + dx
        y = y + dy

        # Нормализуем координаты
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        # Применяем трансформацию
        img_deformed = cv2.remap(img_np, x.astype(np.float32), y.astype(np.float32),
                                 cv2.INTER_LINEAR)
        return torch.from_numpy(img_deformed.transpose(2, 0, 1))


class MixUp:
    """Смешивает два изображения."""

    def __init__(self, p=0.5, alpha=0.2):
        self.p = p
        self.alpha = alpha

    def __call__(self, img1, img2):
        if random.random() > self.p:
            return img1
        lam = np.random.beta(self.alpha, self.alpha)
        return lam * img1 + (1 - lam) * img2


class RandomBlur:
    """Случайное размытие изображения"""

    def __init__(self, p=0.5, max_radius=2):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor

        # Конвертируем тензор в PIL Image
        img_pil = transforms.ToPILImage()(tensor)

        # Применяем размытие
        radius = random.uniform(0, self.max_radius)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius))

        # Конвертируем обратно в тензор
        return transforms.ToTensor()(img_pil)


class RandomPerspective:
    """Перспективное преобразование"""

    def __init__(self, p=0.5, distortion=0.3):
        self.p = p
        self.distortion = distortion

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor

        # Конвертируем тензор в numpy array
        img_np = tensor.permute(1, 2, 0).numpy()
        h, w = img_np.shape[:2]

        # Генерируем случайные точки перспективы
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.uniform(-w * self.distortion, w * self.distortion),
             random.uniform(-h * self.distortion, h * self.distortion)],
            [w + random.uniform(-w * self.distortion, w * self.distortion),
             random.uniform(-h * self.distortion, h * self.distortion)],
            [random.uniform(-w * self.distortion, w * self.distortion),
             h + random.uniform(-h * self.distortion, h * self.distortion)],
            [w + random.uniform(-w * self.distortion, w * self.distortion),
             h + random.uniform(-h * self.distortion, h * self.distortion)]
        ])

        # Применяем преобразование
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_transformed = cv2.warpPerspective(img_np, M, (w, h))

        # Конвертируем обратно в тензор
        return torch.from_numpy(img_transformed).permute(2, 0, 1)


class RandomBrightnessContrast:
    """Изменение яркости и контраста"""

    def __init__(self, p=0.5, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor

        # Конвертируем в PIL Image
        img_pil = transforms.ToPILImage()(tensor)

        # Изменяем яркость
        enhancer = ImageEnhance.Brightness(img_pil)
        brightness_factor = random.uniform(*self.brightness_range)
        img_pil = enhancer.enhance(brightness_factor)

        # Изменяем контраст
        enhancer = ImageEnhance.Contrast(img_pil)
        contrast_factor = random.uniform(*self.contrast_range)
        img_pil = enhancer.enhance(contrast_factor)

        # Конвертируем обратно в тензор
        return transforms.ToTensor()(img_pil)




























