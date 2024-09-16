import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.tree as tree

data = pd.read_csv('data/drug200.csv')
# print(data.head())

label = LabelEncoder()
lb_sex = label.fit(['M','F'])
data['Sex']= lb_sex.transform(data['Sex'])
lb_BP = label.fit(['HIGH','NORMAL','LOW'])
data['BP'] = lb_BP.transform(data['BP'])
lb_Cho = label.fit(['HIGH','NORMAL'])
data['Cholesterol'] = lb_Cho.transform(data['Cholesterol'])
# print(data['Cholesterol'].value_counts())
# print(data.head())

target = 'Drug'
x = data.drop([target],axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8, random_state= 100)

model = DecisionTreeClassifier(criterion= 'entropy', max_depth= 4)
model.fit(x_train,y_train)

predictions = model.predict(x_test)
comp = pd.DataFrame({'Predicted': predictions,'Actual' : y_test.values})
print(comp)
print("Decision Tree accuracy: ", accuracy_score(y_test,predictions))

tree.plot_tree(model)
plt.show()
