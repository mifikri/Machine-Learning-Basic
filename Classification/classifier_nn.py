import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

irisClasses = ['setosa','versicolor','virginica']

seed = 7
np.random.seed(seed)

df = pandas.read_csv("iris.csv")
X = df.values[:,:4]
Y = df.values[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
enY = np_utils.to_categorical(encoded_Y)

xTrain, xTest, yTrain, yTest = train_test_split(X,enY, test_size=0.2, random_state=seed)

model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(xTrain,yTrain,validation_data=(xTest,yTest), batch_size=10,epochs=100, verbose=1)
predictions = np.argmax(model.predict(xTest), axis=1)
print predictions
print encoder.inverse_transform(predictions)

plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model acc')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
