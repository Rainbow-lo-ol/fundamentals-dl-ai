import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


class CSVread(Dataset):
    def __init__(self, file_path, target_col, num_col=None, cat_col=None, bin_col=None):
        """
        Функция для считвания csv-файла и его преобработки
        :param file_path: путь к файлу
        :param target_col: целевая колонка
        :param num_cols: числовые колонки
        :param cat_cols: категориальные колонки
        :param bin_cols: список бинарных колонок (значения 0/1)
        """
        # Считывание данных
        data = pd.read_csv(file_path)

        # Приведение таргета к тензозу
        self.target = torch.FloatTensor(data[target_col].values)

        # Проверка различных типов данных
        if num_col is None:
            num_col = data.select_dtypes(include='number').columns.drop([target_col] + (bin_col or [])).tolist()
        if cat_col is None:
            cat_col = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if bin_col:
            data[bin_col] = data[bin_col].astype(int).clip(0, 1)

        # Преобработка
        preprocessor = make_column_transformer(
            (StandardScaler(), num_col),
            (OneHotEncoder(drop='if_binary', sparse_output=False), cat_col),
            ('passthrough', bin_col or [])
        )

        self.features = torch.FloatTensor(preprocessor.fit_transform(data))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        return self.features[item], self.target[item]