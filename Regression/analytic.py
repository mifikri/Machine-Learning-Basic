import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('house_price.csv')
X = df.values[:,:13]
Y = df.values[:,13]

labelX=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
labelY = ["MEDV"]

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
    
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('House Price Feature Correlation')
    labels=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    colorRange = list(frange(-1,1,0.05))
    fig.colorbar(cax, ticks=[colorRange])
    plt.show()

correlation_matrix(df)

from scipy.stats import pearsonr

rRow = []
pValue = []
for i in range (X.shape[1]):
    r_row, p_value = pearsonr(X[:,i], Y)
    rRow.append(r_row)
    pValue.append(p_value)

listt = [[]]
for counter, value in enumerate (labelX):
    listt.append([counter, value, rRow[counter]])

dfCorrelation = pd.DataFrame(listt,columns=['No','Features','Value'])
dfCorrelation = dfCorrelation[np.isfinite(dfCorrelation['Value'])]
final_df = dfCorrelation.drop(dfCorrelation[(dfCorrelation['Value'] > -0.45) & (dfCorrelation['Value'] < 0.5)].index)
print final_df
#x = df[list(final_df.Features)]
x = df[labelX]
y = df[list(labelY)]

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.0, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#regressor = SVR(kernel='linear')
regressor = LinearRegression()
Lreg = regressor.fit(xTrain,yTrain)
pred=regressor.predict(xTrain)

#predY = regressor.predict(xTest)

print regressor.score(xTrain,yTrain)
print Lreg.coef_,
print Lreg.intercept_
print r2_score(yTrain, pred)

a = range(0,len(yTrain))
plt.plot(a, yTrain, label='actual')
plt.plot(a, pred, label='predict')
plt.ylabel('harga rumah')
plt.xlabel('jumlah sampel')
plt.title('harga rumah actual and prediction')
plt.legend()
plt.show()
'''
b = range(0,len(yTest))
plt.plot(b, yTest, label='train')
plt.plot(b, predY, label='predict')
plt.show()
'''
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
X1 = sm.add_constant(x)
model1=sm.OLS(y,X1)

result=model1.fit()

print(result.summary())


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

vif = pd.DataFrame()
aX = xx.assign(const=1)
vif['VIF Factor'] = [variance_inflation_factor(aX.values, i) for i in range(aX.shape[1])]
vif['features'] = aX.columns

print vif