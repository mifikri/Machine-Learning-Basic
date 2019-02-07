import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
import pickle

df = pd.read_csv('iris.csv')
print df.head()

X = df.values[:,:4]
Y = df.values[:,4]

xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=0.2, random_state=0)

clf = tree.DecisionTreeClassifier()
clf.fit(xTrain,yTrain)

yPred = clf.predict(xTest)
final_train_accuracy = np.mean((yPred == yTest).astype(np.float32))
scores = cross_val_score(clf, X,Y, cv=10)
print 'Accuracy on the training set:', final_train_accuracy
print 'model validation:', clf.score(xTest,yTest)
print scores
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

filename = 'classifier_tree.sav'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
target_one = loaded_model.predict([xTest[0]])
target = loaded_model.predict(xTest)
print target_one
print target