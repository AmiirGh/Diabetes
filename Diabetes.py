import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
import keras
import matplotlib.pyplot as plt


df = pd.read_csv(r'diabetes.csv')
data = df.values

X, y = data[:, :-1], data[:, -1]
# print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
N, D = X_train.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = keras.Sequential()
model.add(Dense(1, input_shape=(D,), activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150)
print("_"*100)
print("train score:  ", model.evaluate(X_train, y_train))
print("test score:  ", model.evaluate(X_test, y_test))

print(r.history['loss'])

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.legend()
plt.show()





