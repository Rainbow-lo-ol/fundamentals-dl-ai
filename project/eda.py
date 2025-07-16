import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import mediapipe as mp


path = './data/'
train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'test')
output_dir = "./results"


def visualization_photo():
    fig = plt.figure(figsize=(10, 7))

    rows = 4
    colomns = 2

    img_angry = np.asarray(Image.open(os.path.join(train_path, 'angry/im0.png')))
    img_disgusted = np.asarray(Image.open(os.path.join(train_path, 'disgusted/im0.png')))
    img_fearful = np.asarray(Image.open(os.path.join(train_path, 'fearful/im0.png')))
    img_happy = np.asarray(Image.open(os.path.join(train_path, 'happy/im0.png')))
    img_neutral = np.asarray(Image.open(os.path.join(train_path, 'neutral/im0.png')))
    img_sad = np.asarray(Image.open(os.path.join(train_path, 'sad/im0.png')))
    img_surprised = np.asarray(Image.open(os.path.join(train_path, 'surprised/im0.png')))

    fig.add_subplot(rows, colomns, 1)
    plt.imshow(img_angry, cmap='gray')
    plt.axis('off')
    plt.title('angry')

    fig.add_subplot(rows, colomns, 2)
    plt.imshow(img_disgusted, cmap='gray')
    plt.axis('off')
    plt.title('disgusted')

    fig.add_subplot(rows, colomns, 3)
    plt.imshow(img_fearful, cmap='gray')
    plt.axis('off')
    plt.title('fearful')

    fig.add_subplot(rows, colomns, 4)
    plt.imshow(img_happy, cmap='gray')
    plt.axis('off')
    plt.title('happy')

    fig.add_subplot(rows, colomns, 5)
    plt.imshow(img_neutral, cmap='gray')
    plt.axis('off')
    plt.title('neutral')

    fig.add_subplot(rows, colomns, 6)
    plt.imshow(img_sad, cmap='gray')
    plt.axis('off')
    plt.title('sad')

    fig.add_subplot(rows, colomns, 7)
    plt.imshow(img_surprised, cmap='gray')
    plt.axis('off')
    plt.title('surprised')

    plt.savefig(os.path.join(output_dir, "emotions_grid.png"), dpi=300, bbox_inches='tight')


def class_distribution():
    angry_train = len(os.listdir(os.path.join(train_path, 'angry')))
    disgusted_train = len(os.listdir(os.path.join(train_path, 'disgusted')))
    fearful_train = len(os.listdir(os.path.join(train_path, 'fearful')))
    happy_train = len(os.listdir(os.path.join(train_path, 'happy')))
    neutral_train = len(os.listdir(os.path.join(train_path, 'neutral')))
    sad_train = len(os.listdir(os.path.join(train_path, 'sad')))
    surprised_train = len(os.listdir(os.path.join(train_path, 'surprised')))

    angry_test = len(os.listdir(os.path.join(test_path, 'angry')))
    disgusted_test = len(os.listdir(os.path.join(test_path, 'disgusted')))
    fearful_test = len(os.listdir(os.path.join(test_path, 'fearful')))
    happy_test = len(os.listdir(os.path.join(test_path, 'happy')))
    neutral_test = len(os.listdir(os.path.join(test_path, 'neutral')))
    sad_test = len(os.listdir(os.path.join(test_path, 'sad')))
    surprised_test = len(os.listdir(os.path.join(test_path, 'surprised')))

    d_train = {
        'angry': angry_train,
        'disgusted': disgusted_train,
        'fearful': fearful_train,
        'happy': happy_train,
        'neutral': neutral_train,
        'sad': sad_train,
        'surprised': surprised_train
    }

    d_test = {
        'angry': angry_test,
        'disgusted': disgusted_test,
        'fearful': fearful_test,
        'happy': happy_test,
        'neutral': neutral_test,
        'sad': sad_test,
        'surprised': surprised_test
    }

    plt.clf()

    plt.bar(d_train.keys(), d_train.values(), label='Train')
    plt.bar(d_test.keys(), d_test.values(), label='Test')
    plt.legend()

    plt.savefig(os.path.join(output_dir, "distribution_plt.png"), dpi=300, bbox_inches='tight')


def mean_image():
    emotions = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    mean_images = {}

    # Обрабатываем каждую эмоцию
    for emotion in emotions:
        # Получаем список всех изображений для эмоции (из train и test)
        train_images = [os.path.join(train_path, emotion, img)
                        for img in os.listdir(os.path.join(train_path, emotion))
                        if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        test_images = [os.path.join(test_path, emotion, img)
                       for img in os.listdir(os.path.join(test_path, emotion))
                       if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        all_images = train_images + test_images

        # Загружаем и усредняем изображения
        images = []
        for img_path in all_images:
            try:
                img = np.asarray(Image.open(img_path))
                if len(img.shape) == 2:  # Если изображение в градациях серого
                    img = np.stack([img] * 3, axis=-1)  # Преобразуем в RGB
                images.append(img)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
                continue

        if images:
            mean_images[emotion] = np.mean(images, axis=0).astype(np.uint8)
        else:
            print(f"Нет изображений для эмоции: {emotion}")
            mean_images[emotion] = None

    # Визуализация
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Средние изображения для каждой эмоции", fontsize=16)

    for i, (emotion, img) in enumerate(mean_images.items(), 1):
        if img is None:
            continue

        ax = fig.add_subplot(2, 4, i)
        ax.imshow(img)
        ax.set_title(emotion)
        ax.axis('off')

    # Сохраняем и показываем результат
    output_path = os.path.join(output_dir, "mean_emotions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def mean_points(
    output_csv: str = "unified_landmarks.csv"
):
    # Инициализация MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Словарь для хранения средних значений
    emotion_landmarks = {
        "angry": [], "disgusted": [], "fearful": [],
        "happy": [], "neutral": [], "sad": [], "surprised": []
    }

    # Обработка данных
    for dataset_type in ["train", "test"]:
        dataset_path = os.path.join(path, dataset_type)
        if not os.path.exists(dataset_path):
            continue

        for emotion in emotion_landmarks.keys():
            emotion_dir = os.path.join(dataset_path, emotion)
            if not os.path.isdir(emotion_dir):
                continue

            for img_file in tqdm(os.listdir(emotion_dir)):
                img_path = os.path.join(emotion_dir, img_file)
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if not results.multi_face_landmarks:
                        continue

                    landmarks = []
                    for landmark in results.multi_face_landmarks[0].landmark:
                        landmarks.extend([landmark.x, landmark.y])

                    emotion_landmarks[emotion].append(landmarks)

                except Exception as e:
                    print(f"Ошибка в {img_path}: {e}")
                    continue

    # Вычисление средних landmarks
    mean_landmarks = {}
    for emotion, landmarks_list in emotion_landmarks.items():
        if landmarks_list:
            mean_landmarks[emotion] = np.mean(landmarks_list, axis=0)
        else:
            mean_landmarks[emotion] = None

    # Визуализация средних landmarks
    fig = plt.figure(figsize=(15, 10))

    for idx, (emotion, landmarks) in enumerate(mean_landmarks.items(), 1):
        if landmarks is None:
            continue

        # Создаем "пустое" изображение для визуализации точек
        dummy_image = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # Преобразуем координаты landmarks (468 точек)
        xs = [landmarks[2 * i] * dummy_image.shape[1] for i in range(len(landmarks) // 2)]
        ys = [landmarks[2 * i + 1] * dummy_image.shape[0] for i in range(len(landmarks) // 2)]

        # Рисуем точки
        ax = fig.add_subplot(2, 4, idx)
        ax.imshow(dummy_image)
        ax.scatter(xs, ys, c='red', s=10)

        # Подписываем каждую 20-ю точку для ориентации
        for i in range(0, len(xs), 20):
            ax.annotate(str(i), (xs[i], ys[i]), fontsize=6)

        ax.set_title(f"Mean {emotion}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_points.png"), dpi=300, bbox_inches='tight')
    plt.close()

    face_mesh.close()


if __name__ == "__main__":
    visualization_photo()
    class_distribution()
    mean_image()
    mean_points()

