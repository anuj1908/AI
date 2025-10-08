import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from six import StringIO
from IPython.display import Image
import pydotplus

# Load dataset
col_name=["resolution","raining","badservice","satur","result"]
hoteldata = pd.read_csv("hotel.csv", names=col_name)

# Features and target
features_col = ["resolution","raining","badservice","satur"]
X = hoteldata[features_col]
Y = hoteldata["result"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Train model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(X_train, Y_train)

# Predict
y_pred = clf.predict(X_test)

print("Y_test =\n", Y_test.values)
print("Y_pred =\n", y_pred)
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))

# Visualization
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                feature_names=features_col, class_names=['leave',"wait"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("hotel.png")
Image(graph.create_png())
