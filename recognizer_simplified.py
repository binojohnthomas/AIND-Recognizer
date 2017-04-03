import numpy as np
import pandas as pd
from asl_data import AslDb


from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(2)
kf.get_n_splits()

print(kf)

for train_index, test_index in kf.split(X):
 print("TRAIN:", train_index, "TEST:", test_index)
 X_train, X_test = X[train_index], X[test_index]
 print(X_train)
 print(X_test)
 y_train, y_test = y[train_index], y[test_index]
