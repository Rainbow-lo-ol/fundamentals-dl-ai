# Задание 1: Сравнение CNN и полносвязных сетей
## 1.1 Сравнение на MNIST
### Полносвязная сеть
![plot](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/fc_simple_mnist_confusion_matrix.png)
Confusion Matrix показывает, что модель допустила больше ошибок по сравнению с CNN. Общая точность,  ниже, чем у CNN.

![plot](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/fc_simple_mnist_learning_curves.png)
- Loss на обучении и тесте снижается, но на тесте всё же выше, чем на обучении.
- Accuracy достигает значений около 0.97-0.98.

Полносвязные сети обычно имеют больше параметров из-за плотных слоев, что может приводить к переобучению или менее эффективному обучению.

### Простая CNN
![plot](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/simple_cnn_mnist_mnist_confusion_matrix.png)
Confusion Matrix для простой CNN показывает меньше ошибок, чем у FC. Общая точность выше, что подтверждается значениями accuracy около 0.99

![plot](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/simple_cnn_mnist_mnist_learning_curves.png)
- Loss снижается быстрее и достигает низких значений (около 0.04-0.06), что указывает на лучшее обучение.
- Accuracy быстро растет и стабилизируется на высоком уровне (0.99+). 

CNN эффективно используют локальные особенности изображений, что уменьшает количество параметров по сравнению с FC и улучшает обобщение.

### CNN с Residual Block
![plot](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/residual_cnn_mnist_mnist_confusion_matrix.png)
Confusion Matrix демонстрирует меньше ошибок, чем у FC, но больше, чем у Простая CNN. Точность достигает значений около 0.995, что является наивысшим среди трех моделей.

![plot](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/residual_cnn_mnist_mnist_learning_curves.png)
- Loss снижается до очень низких значений (около 0.02-0.04), что указывает на отличную сходимость.
- Accuracy быстро достигает 0.99+ и остается стабильной.

## 1.2 Сравнение на CIFAR-10
### Глубокая полносвязная сеть
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/fc_deep_cifar10_confusion_matrix.png)
- Наибольшие ошибки:Класс 1: 24% ошибок, Класс 5: 66% ошибок.
- Точность: ~50–55% (близко к случайному угадыванию для 10 классов).

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/fc_deep_cifar10_learning_curves.png)
- Accuracy: Останавливается на ~0.55 (test).
- Loss: Высокий (~0.6), что указывает на плохую сходимость.

Полносвязная сеть не подходит для CIFAR-10 из-за отсутствия индуктивных biases, переобучения 

### Простая CNN
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/simple_cnn_cifar_cifar10_confusion_matrix.png)
- Лучшие классы. Класс 6: 77% точности.  Класс 9: 82% точности.
- Проблемные классы. Класс 3: 42% точности. Класс 5: 69% точности.

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/simple_cnn_cifar_cifar10_learning_curves.png)
- Accuracy: Достигает ~0.70–0.75 (test).
- Loss: Снижается до ~0.65 (лучше, чем у FC).

CNN значительно лучше FC благодаря локальному анализу признаков (свертки), пулингу для уменьшения переобучения.

### Residual CNN
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/residual_cnn_mnist_mnist_confusion_matrix.png)
- Наивысшая точность. Класс 6: 92.8% точности. Класс 9: 95.3% точности.
- Улучшение для сложных классов. Класс 3: 73.9% (против 42% у простой CNN). Класс 5: 67% (но меньше ошибок в другие классы).

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/results/residual_cnn_mnist_mnist_learning_curves.png)
- Accuracy: ~0.80–0.85 (test).
- Loss: Снижается до ~0.60 (стабильнее, чем у простой CNN).

Residual CNN превосходит другие модели благодаря глубокой архитектуре без деградации (skip-connections), лучшему обучению сложных признаков (например, текстуры шерсти у животных).

### Общий вывод
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/final_comparison.png)
Для MNIST:
- Достаточно простой CNN для достижения точности >99%.
- Residual Block дают небольшой прирост (~0.5%).

Для CIFAR-10:
- Residual CNN критически важны для приемлемой точности (~75–80%).
- Полносвязные сети почти бесполезны (точность ~50%, как случайный угадывание для 10 классов).

# Задание 2: Анализ архитектур CNN
## 2.1 Влияние размера ядра свертки
- Время обучения: 240.36 секунд.
- Градиенты: Значения градиентов варьируются от ~5.6e-03 до ~3.1e-05, что указывает на устойчивость процесса обучения, так как градиенты не взрываются и не исчезают.

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/kernel_3x3_learning_curves.png)
3x3
- Наилучшая производительность среди всех вариантов.
- Точность на тестовых данных достигает ~0.95.
- Минимальные потери (~0.2), что указывает на хорошую сходимость.

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/kernel_5x5_learning_curves.png)
5x5
- Данные представлены менее наглядно, но видно, что точность ниже, чем у 3x3.
- Потери выше, что говорит о менее эффективном обучении.

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/kernel_7x7_learning_curves.png)
7x7
- Наибольшие потери (~1.4) среди всех вариантов.
- Точность существенно ниже, что может быть связано с избыточной размерностью ядра для данной задачи.

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/kernel_1x1_3x3_learning_curves.png)
1x1+3x3
- Точность ~0.85 на тренировочных данных.
- Виден небольшой разрыв между тренировочной и тестовой кривыми, что может указывать на умеренное переобучение.
  
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/kernel_all_test_acc.png)
- Наивысшая точность достигается при использовании ядер 3x3 (~0.72).
- Ядра 5x5 и 7x7 показывают схожую производительность (~0.64-0.68).
- Комбинация 1x1 и 3x3 ядер демонстрирует промежуточные результаты (~0.66).
- Все модели стабилизируются после 4-6 эпох.

## 2.2 Влияние глубины CNN
- Время обучения: 237.57 секунд.
- Градиенты: Значения градиентов варьируются от ~9.6e-03 до ~1.1e-05, что указывает на устойчивость процесса обучения, так как градиенты не взрываются и не исчезают.

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/depth_Shallow_learning_curves.png)
Shallow
- Низкие потери (~0.4), но точность невысока (~0.75).
- Модель быстро обучается, но недостаточно мощная для сложных паттернов.
  
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/depth_Medium_learning_curves.png)
Medium
- Лучшие результаты: точность ~0.85 на тренировочных данных и ~0.80 на тестовых.
- Потери снижаются до ~0.4, нет явного переобучения.
  
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/depth_Deep_learning_curves.png)
Deep
- Больший разрыв между тренировочной и тестовой точностью (признак переобучения).
- Потери выше (~0.6), что указывает на сложности в обучении.

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework4/plots/depth_all_test_acc.png)
Для данного датасета лучший результат дает средняя по глубине CNN. Глубокая сеть не оправдывает себя без дополнительной настройки.

# Задание 3: Кастомные слои и эксперименты
```
CustomConv output shape: torch.Size([1, 16, 32, 32])
Attention output shape: torch.Size([1, 16, 32, 32])
CustomActivation output shape: torch.Size([1, 16, 32, 32])
CustomPool output shape: torch.Size([1, 16, 16, 16])

Basic Residual Model: 2.78M parameters
Output shape: torch.Size([2, 10])

Bottleneck Residual Model: 0.22M parameters
Output shape: torch.Size([2, 10])

Wide Residual Model: 5.51M parameters
Output shape: torch.Size([2, 10])
```
CustomConv:
- Вход: [1, C, H, W] → Выход: [1, 16, 32, 32]
- Увеличивает количество каналов (если C < 16) или сохраняет пространственные размеры (стрид=1, паддинг=1).

Attention:
- Выход той же формы ([1, 16, 32, 32]), значит, используется механизм внимания (например, SE-block), который модифицирует признаки без изменения размеров.

CustomActivation:
- Не меняет форму — применяется поэлементная функция активации (например, Swish или LeakyReLU).

CustomPool:
- Уменьшает размерность в 2 раза ([1, 16, 16, 16]), вероятно, MaxPool2d или AvgPool2d с ядром 2x2.

Basic Residual:
- Стандартные Residual-блоки (2 свертки 3x3).
- Больше параметров → выше риск переобучения, но лучше подходит для сложных данных.

Bottleneck Residual:
- Использует "бутылочное горлышко" (1x1 → 3x3 → 1x1 свертки).
- В 12.6 раз меньше параметров, чем Basic, но сохраняет выходную форму.
- Оптимален для ограниченных ресурсов.

Wide Residual:
- Увеличенное количество каналов в Residual-блоках.
- Самая большая модель (5.51M параметров) → требует много данных для обучения.
