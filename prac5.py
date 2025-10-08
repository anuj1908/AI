import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
col_name = ['reservation', 'raining', 'badservice', 'satur', 'result']
hoteldata = pd.read_csv("dtree.csv", header=None, names=col_name)
feature_cols=['reservation', 'raining','badservice','satur']
x=hoteldata[feature_cols]
y=hoteldata.result
x_train , x_test , y_train, y_test = train_test_split(x, y , test_size=0.3, random_state=1)
adahotel = AdaBoostClassifier(n_estimators = 6 , learning_rate = 6)
adahotel = adahotel.fit(x_train, y_train)
y_pred=adahotel.predict(x_test)
print("ytest = \n",y_test)
print("ypres =\n", y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))