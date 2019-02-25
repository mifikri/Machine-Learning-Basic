import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labelX=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
labelY = ["MEDV"]
df = pd.read_csv('house_price.csv')
X = df.values[:,:13]
Y = df.values[:,13]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=7)

model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

history = model.fit(xTrain,yTrain,validation_data=(xTest,yTest), batch_size=100,epochs=4000, verbose=1)
#predictions = np.argmax(model.predict(xTest), axis=1)
predictions = model.predict(xTest)
#print predictions


a = range(0, len(yTest))

plt.plot(a, list(yTest))
plt.plot(a, list(predictions))
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model acc')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()