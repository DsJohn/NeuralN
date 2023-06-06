import numpy as np
import pandas as pd
from tensorflow import keras

# Загрузка модели
model = keras.models.load_model('model_rnn.h5')

# Загрузка обучающих данных
train_data = pd.read_excel('nerual.xlsx')

# Рассчитываем среднее и стандартное отклонение для нормализации новых данных
train_X_mean = train_data.iloc[:, :8].mean()
train_X_std = train_data.iloc[:, :8].std()

# Ввод новых данных для проверки
new_data = input("Введите 8 новых чисел через запятую: ")
new_data = np.array(new_data.split(','), dtype=float)

# Нормализация новых данных
new_X = (new_data - train_X_mean) / train_X_std
new_X = np.reshape(new_X, (1, 1, 8))

# Предсказание результатов на новых данных
prediction = model.predict(new_X)

print(prediction)
