import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/data_KNN.csv')
# print(data.info())

target = 'custcat'
X= data.drop([target],axis= 1)
y=data[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state= 100)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Ks = 25
# mean_acc = np.zeros((Ks-1))
# for k in range(1,Ks):
#     neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
#     predictions = neigh.predict(X_test)
#     mean_acc[k-1] = accuracy_score(y_test, predictions)
# print(mean_acc)
# print("The best accuracy is: ",mean_acc.max(), " with k= ", mean_acc.argmax()+1)

k= 19
neigh19 = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)
predictions = neigh19.predict(X_test)
comp = pd.DataFrame(predictions.flatten(),y_test.values)
print(comp)
