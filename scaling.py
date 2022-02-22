from sklearn import preprocessing
import numpy as np
X_train = np.random.rand(1,20)
scaler = preprocessing.StandardScaler().fit(X_train)
# scaler
#
#
# scaler.mean_
#
#
# scaler.scale_


X_scaled = scaler.transform(X_train)
