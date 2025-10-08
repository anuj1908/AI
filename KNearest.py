import numpy as np
from collections import Counter

def knn(x_train, y_train, x_test, k=3, task='classification'):
    def euclidean(x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

    predictions = []
    for test_point in x_test:
        distances = [euclidean(test_point, x) for x in x_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest = [y_train[i] for i in k_indices]

        if task == 'classification':
            pred = Counter(k_nearest).most_common(1)[0][0]
        else:
            pred = np.mean(k_nearest)
        predictions.append(pred)
    return predictions

x_train_cls = [[1, 2], [2, 3], [3, 3], [6, 5]]
y_train_cls = ['A', 'A', 'B', 'B']
x_test_cls = [[2, 2], [6, 6]]
y_test_cls = ['A', 'B']

result_cls = knn(x_train_cls, y_train_cls, x_test_cls, k=3, task='classification')
print("Classification Result:", result_cls)
correct_cls = sum(p == a for p, a in zip(result_cls, y_test_cls))
accuracy_cls = correct_cls / len(y_test_cls)
print('Classification Accuracy:', accuracy_cls)

x_train_reg = [[1], [2], [3], [4]]
y_train_reg = [1.5, 2.0, 3.5, 4.0]
x_test_reg = [[2.5], [3.5]]
y_test_reg = [2.2, 3.8]

result_reg = knn(x_train_reg, y_train_reg, x_test_reg, k=2, task='regression')
print('Regression Result:', result_reg)

mse = np.mean((np.array(result_reg) - np.array(y_test_reg)) ** 2) 
print('Regression Mean Square Error (MSE):', mse)
