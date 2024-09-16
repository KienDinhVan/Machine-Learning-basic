import matplotlib.pyplot as plt
import skimage.io
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from lazypredict.Supervised import LazyRegressor
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('data/FuelConsumption.csv')
cdf = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


msk = np.random.rand(len(data)) < 0.8
train = cdf[msk]
test = cdf[~msk]


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine size')
plt.ylabel('Emissions')
plt.show()

regr = LinearRegression()
# regr = KNeighborsRegressor(n_neighbors=7)
x_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train, y_train)

x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
predictions = regr.predict(x_test)
comp = pd.DataFrame(predictions.flatten(),y_test.flatten())
print(comp)
print('MAE: %.2f' % np.mean(np.absolute(predictions- y_test)))
print('R2Score: %.2f'% r2_score(y_test,predictions))

# clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train,x_test, y_train,y_test)
# print(models)

# Ks = 20
# r2 = np.zeros((Ks-1))
#
# for k in range(1,Ks):
#     neigh = KNeighborsRegressor(n_neighbors=k)
#     neigh.fit(x_train,y_train)
#     y_pred = neigh.predict(x_test)
#     r2[k-1]= r2_score(y_test,y_pred)
# print(r2)
# print('Best R2_score is: ', r2.max(), ' with k= ',r2.argmax()+1)