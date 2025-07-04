# Задание 1: Эксперименты с глубиной сети
## 1.1 Сравнение моделей разной глубины
### Результат
```
=== Running experiment with depth=1 ===
Epoch 1/5: Train Loss: 0.3675, Acc: 0.8943 | Test Loss: 1.7117, Acc: 0.9200
Epoch 2/5: Train Loss: 0.2957, Acc: 0.9155 | Test Loss: 1.6788, Acc: 0.9235
Epoch 3/5: Train Loss: 0.2872, Acc: 0.9187 | Test Loss: 1.7193, Acc: 0.9151
Epoch 4/5: Train Loss: 0.2810, Acc: 0.9223 | Test Loss: 1.6304, Acc: 0.9230
Epoch 5/5: Train Loss: 0.2770, Acc: 0.9231 | Test Loss: 1.5260, Acc: 0.9291
Training time: 151.20s
Max test accuracy: 0.9291

=== Running experiment with depth=2 ===
Epoch 1/5: Train Loss: 0.2033, Acc: 0.9393 | Test Loss: 0.5473, Acc: 0.9717
Epoch 2/5: Train Loss: 0.0899, Acc: 0.9719 | Test Loss: 0.3826, Acc: 0.9811
Epoch 3/5: Train Loss: 0.0632, Acc: 0.9806 | Test Loss: 0.2496, Acc: 0.9866
Epoch 4/5: Train Loss: 0.0484, Acc: 0.9836 | Test Loss: 0.2208, Acc: 0.9879
Epoch 5/5: Train Loss: 0.0380, Acc: 0.9875 | Test Loss: 0.1412, Acc: 0.9923
Training time: 161.97s
Max test accuracy: 0.9923

=== Running experiment with depth=3 ===
Epoch 1/5: Train Loss: 0.2057, Acc: 0.9368 | Test Loss: 0.6216, Acc: 0.9679
Epoch 2/5: Train Loss: 0.0905, Acc: 0.9721 | Test Loss: 0.3119, Acc: 0.9835
Epoch 3/5: Train Loss: 0.0678, Acc: 0.9789 | Test Loss: 0.2470, Acc: 0.9871
Epoch 4/5: Train Loss: 0.0525, Acc: 0.9836 | Test Loss: 0.2347, Acc: 0.9875
Epoch 5/5: Train Loss: 0.0452, Acc: 0.9859 | Test Loss: 0.4061, Acc: 0.9780
Training time: 173.97s
Max test accuracy: 0.9875

=== Running experiment with depth=5 ===
Epoch 1/5: Train Loss: 0.2435, Acc: 0.9251 | Test Loss: 0.6541, Acc: 0.9669
Epoch 2/5: Train Loss: 0.1147, Acc: 0.9668 | Test Loss: 0.4169, Acc: 0.9789
Epoch 3/5: Train Loss: 0.0857, Acc: 0.9753 | Test Loss: 0.4182, Acc: 0.9782
Epoch 4/5: Train Loss: 0.0717, Acc: 0.9786 | Test Loss: 0.4290, Acc: 0.9777
Epoch 5/5: Train Loss: 0.0595, Acc: 0.9826 | Test Loss: 0.2378, Acc: 0.9882
Training time: 158.90s
Max test accuracy: 0.9882

=== Running experiment with depth=7 ===
Epoch 1/5: Train Loss: 0.2905, Acc: 0.9135 | Test Loss: 0.8166, Acc: 0.9608
Epoch 2/5: Train Loss: 0.1398, Acc: 0.9628 | Test Loss: 0.6642, Acc: 0.9709
Epoch 3/5: Train Loss: 0.1055, Acc: 0.9718 | Test Loss: 0.4298, Acc: 0.9810
Epoch 4/5: Train Loss: 0.0871, Acc: 0.9767 | Test Loss: 0.3988, Acc: 0.9827
Epoch 5/5: Train Loss: 0.0708, Acc: 0.9809 | Test Loss: 0.3505, Acc: 0.9830
Training time: 168.27s
Max test accuracy: 0.9830
```
Лучшее время: depth=1
Лучшее acc: depth=2
## 1.2 Анализ переобучения
### Результат
```
=== Running experiment with depth=1 ===
Epoch 1/5: Train Loss: 0.3695, Acc: 0.8940 | Test Loss: 1.7555, Acc: 0.9145
Epoch 2/5: Train Loss: 0.2952, Acc: 0.9162 | Test Loss: 1.6529, Acc: 0.9226
Epoch 3/5: Train Loss: 0.2858, Acc: 0.9198 | Test Loss: 1.6161, Acc: 0.9217
Epoch 4/5: Train Loss: 0.2807, Acc: 0.9213 | Test Loss: 1.6527, Acc: 0.9216
Epoch 5/5: Train Loss: 0.2764, Acc: 0.9231 | Test Loss: 1.6103, Acc: 0.9242
Results for depth 1:
Training time: 132.77s
Max Train Accuracy: 0.9231
Max Test Accuracy: 0.9242
Overfitting gap: -0.0011
Overfitting starts at epoch: Not detected

=== Running experiment with depth=2 ===
Epoch 1/5: Train Loss: 0.2054, Acc: 0.9384 | Test Loss: 0.6030, Acc: 0.9700
Epoch 2/5: Train Loss: 0.0906, Acc: 0.9723 | Test Loss: 0.3475, Acc: 0.9816
Epoch 3/5: Train Loss: 0.0636, Acc: 0.9798 | Test Loss: 0.2827, Acc: 0.9852
Epoch 4/5: Train Loss: 0.0507, Acc: 0.9836 | Test Loss: 0.2633, Acc: 0.9855
Epoch 5/5: Train Loss: 0.0386, Acc: 0.9868 | Test Loss: 0.1563, Acc: 0.9911
Results for depth 2:
Training time: 155.90s
Max Train Accuracy: 0.9868
Max Test Accuracy: 0.9911
Overfitting gap: -0.0043
Overfitting starts at epoch: Not detected

=== Running experiment with depth=3 ===
Epoch 1/5: Train Loss: 0.2957, Acc: 0.9093 | Test Loss: 0.8884, Acc: 0.9629
Epoch 2/5: Train Loss: 0.1913, Acc: 0.9413 | Test Loss: 0.9495, Acc: 0.9684
Epoch 3/5: Train Loss: 0.1662, Acc: 0.9478 | Test Loss: 0.6799, Acc: 0.9773
Epoch 4/5: Train Loss: 0.1446, Acc: 0.9552 | Test Loss: 0.5520, Acc: 0.9786
Epoch 5/5: Train Loss: 0.1318, Acc: 0.9596 | Test Loss: 0.7733, Acc: 0.9820
Results for depth 3:
Training time: 175.73s
Max Train Accuracy: 0.9596
Max Test Accuracy: 0.9820
Overfitting gap: -0.0224
Overfitting starts at epoch: Not detected

=== Running experiment with depth=5 ===
Epoch 1/5: Train Loss: 0.3753, Acc: 0.8854 | Test Loss: 0.9790, Acc: 0.9565
Epoch 2/5: Train Loss: 0.2400, Acc: 0.9280 | Test Loss: 0.6445, Acc: 0.9695
Epoch 3/5: Train Loss: 0.2071, Acc: 0.9378 | Test Loss: 0.5605, Acc: 0.9738
Epoch 4/5: Train Loss: 0.1831, Acc: 0.9456 | Test Loss: 0.4913, Acc: 0.9771
Epoch 5/5: Train Loss: 0.1642, Acc: 0.9502 | Test Loss: 0.5036, Acc: 0.9780
Results for depth 5:
Training time: 176.81s
Max Train Accuracy: 0.9502
Max Test Accuracy: 0.9780
Overfitting gap: -0.0278
Overfitting starts at epoch: Not detected

=== Running experiment with depth=7 ===
Epoch 1/5: Train Loss: 0.4736, Acc: 0.8560 | Test Loss: 1.1657, Acc: 0.9491
Epoch 2/5: Train Loss: 0.3007, Acc: 0.9123 | Test Loss: 0.9846, Acc: 0.9571
Epoch 3/5: Train Loss: 0.2575, Acc: 0.9251 | Test Loss: 0.6620, Acc: 0.9693
Epoch 4/5: Train Loss: 0.2249, Acc: 0.9351 | Test Loss: 0.6107, Acc: 0.9756
Epoch 5/5: Train Loss: 0.2027, Acc: 0.9417 | Test Loss: 0.5203, Acc: 0.9767
Results for depth 7:
Training time: 191.51s
Max Train Accuracy: 0.9417
Max Test Accuracy: 0.9767
Overfitting gap: -0.0350
Overfitting starts at epoch: Not detected

Optimal depth: 2 with test accuracy 0.9911
```
На данной модели переобучения обнаружено не было
# Задание 2: Эксперименты с шириной сети
## 2.1 Сравнение моделей разной ширины
```
Width Experiment Results:
| Profile   | Widths            |   Parameters |   Max Test Acc |   Time (s) |
|:----------|:------------------|-------------:|---------------:|-----------:|
| narrow    | [64, 32, 16]      |        53018 |       0.982267 |    159.864 |
| medium    | [256, 128, 64]    |       242762 |       0.9897   |    175.816 |
| wide      | [1024, 512, 256]  |      1462538 |       0.986283 |    155.934 |
| xwide     | [2048, 1024, 512] |      4235786 |       0.986083 |    176.747 |
```
## 2.2 Оптимизация архитектуры
```
Optimal architecture: constant with base width 16
Widths: [16, 16, 16], Accuracy: 0.9409

Optimal architecture: constant with base width 32
Widths: [32, 32, 32], Accuracy: 0.9745

Optimal architecture: narrowing with base width 64
Widths: [64, 32, 16], Accuracy: 0.9851

Optimal architecture: constant with base width 128
Widths: [128, 128, 128], Accuracy: 0.9886
```
# Задание 3: Эксперименты с регуляризацией
## 3.1 Сравнение техник регуляризации
```

```
## 3.2 Адаптивная регуляризация
```

```























