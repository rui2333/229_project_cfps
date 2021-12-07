import os
base_path = r'/Users/~/code_dir'

os.chdir(base_path)
import PerformMetrics  # for loop below

import numpy as np
import time
from sklearn.linear_model import LogisticRegression

def run_softmax_reg(x_train, y_train, x_val, y_val):

    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg = LogisticRegression(C=0.01, max_iter =10000)  # Lambda = 1/C
    logreg.fit(x_train, y_train)

    y_pred_train = logreg.predict(x_train)
    y_pred_val = logreg.predict(x_val)

    return y_pred_train, y_pred_val

