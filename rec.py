import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Загрузка данных из Excel
data = pd.read_excel('nerual.xlsx')

# Разделение данных на тренировочный и тестовый наборы
train_data, test_data = train_test_split(data, test_size=0.2)

# Препроцессинг данных
# Преобразование данных в numpy массивы
train_X = np.array(train_data.iloc[:, :8])
train_y = np.array(train_data.iloc[:, 8])
test_X = np.array(test_data.iloc[:, :8])
test_y = np.array(test_data.iloc[:, 8])

# Нормализация данных
train_X = (train_X - train_X.mean()) / train_X.std()
test_X = (test_X - test_X.mean()) / test_X.std()

# Изменение формы данных для входа в RNN
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Определение модели RNN
model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(1, 8), activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

# Обучение модели
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_split=0.2)

# Оценка модели на тестовых данных
test_loss, test_mae = model.evaluate(test_X, test_y)

# Прогнозирование результатов на новых данных
new_data = pd.DataFrame({'Column 1': [1, 2, 3], 'Column 2': [4, 5, 6], 'Column 3': [7, 8, 9], 'Column 4': [10, 11, 12],
                         'Column 5': [13, 14, 15], 'Column 6': [16, 17, 18], 'Column 7': [19, 20, 21],
                         'Column 8': [22, 23, 24]})
new_X = new_data.values
new_X = (new_X - train_X.mean()) / train_X.std()
new_X = np.reshape(new_X, (new_X.shape[0], 1, new_X.shape[1]))
predictions = model.predict(new_X)
model.save('model_rnn.h5')

plt.plot(history.history['acc'])
plt.title('Model acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.show()

# Построение графика ошибок
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

