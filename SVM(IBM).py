import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report, jaccard_score
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv('data/cell_samples.csv')
# print(data.columns) #['ID', 'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit',
#                     # 'Class'],
# print(data['Class'].value_counts())
# print(data.dtypes)

# #Visualize
# ax = data[data['Class'] == 4][0:100].plot(
#     kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='Malignant')
# data[data['Class'] == 2][0:100].plot(
#     kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='Benign', ax=ax)
# plt.show()

target = 'Class'
x= data.drop([target,'ID','BareNuc'], axis=1)
y= data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state= 4)

model = SVC(kernel='rbf')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
comp = pd.DataFrame({'Predicted':y_pred.flatten(),'Actual': y_test.values})
print(comp)
print(classification_report(y_test, y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
print('F1-score: ', f1_score(y_test,y_pred, average= 'weighted'))
print('Jaccard accuracy: ', jaccard_score(y_test,y_pred, pos_label=2))

# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# print(models)