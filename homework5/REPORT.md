# Задание 1: Стандартные аугментации torchvision
В отчете будет представлен пример только с одним изображнием из пяти, Остальные можной найти в папке results (Figure_1 - 8)
Оригинал
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_1.png)

Результат применения каждой аугментации отдельно
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_2.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_3.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_4.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_5.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_6.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_7.png)

Результат применения всех аугментаций вместе
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_8.png)

# Задание 2: Кастомные аугментации
Кастомные аугментации
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_9.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_10.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_11.png)

Готовые аугментации
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_12.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_13.png)
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_14.png)

В кастомные аугментациях используется разу несколько готовых

# Задание 3: Анализ датасета
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_15.png)
Изображения разделены по класса в равных количествах

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/Figure_16.png)
```
Минимальная ширина: 210px
Максимальная ширина: 736px
Средняя ширина: 538.9px
Минимальная высота: 240px
Максимальная высота: 1308px
Средняя высота: 623.6px
Минимальное соотношение сторон: 0.51
Максимальное соотношение сторон: 2.18
```

# Задание 4: Pipeline аугментаций
Результат сохранен в homework5/results/augmentations_20250708_055650.json

# Задание 5: Эксперимент с размерами
```
--- Размер 64x64 ---   
Изображений: 100       
Загрузка: 0.75 сек    
Аугментация: 0.06 сек
Память: 0.08 МБ

--- Размер 128x128 ---
Изображений: 100
Загрузка: 0.77 сек
Аугментация: 0.08 сек
Память: 0.10 МБ

--- Размер 224x224 ---
Изображений: 100
Загрузка: 0.85 сек
Аугментация: 0.17 сек
Память: 0.29 МБ

--- Размер 512x512 ---
Изображений: 100
Загрузка: 1.19 сек
Аугментация: 1.08 сек
Память: 1.51 МБ
```

![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/results.png)
Время загрузки, аргументации и использования памяти растёт. Это указывает на на зависимость от размера данных.

# Задание 6: Дообучение предобученных моделей
![](https://github.com/Rainbow-lo-ol/fundamentals-dl-ai/blob/main/homework5/results/training_results.png)
- Потери на тренировочные данных стабильно падает с каждой эпохой
- Потери на тестовых данных стабильные, маньше единицы
- Точность на тестовых данных равна пример 0,75















