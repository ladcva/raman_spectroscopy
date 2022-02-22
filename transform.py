# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
from numpy import mean
from numpy import std
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from BaselineRemoval import BaselineRemoval
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


def crop_range_to_numpy(df):
    # Dropping columns : ['patientID', 'has_DM2']
    df_dropped_index = df.drop(labels=['patientID', 'has_DM2'], axis=1, inplace=False)

    # Cropping raman shift from range 800 cm-1 to 1800cm-1
    cropped_signal = df_dropped_index.loc[:, 'Var802':'Var1801']

    # Convert data to numpy.ndarray
    cropped_signal_np = cropped_signal.to_numpy()

    # Dropping the first index corresponding to label : ramanShift
    cropped_signal_np = cropped_signal_np[1:]

    return cropped_signal_np

def polyfit(input_array):

    new_array = np.empty(shape=(1000,))

    polynomial_degree = 5
    gradient = 0.05

    for row in input_array:
        baseObj = BaselineRemoval(row)
        Imodpoly_output = baseObj.IModPoly(polynomial_degree, gradient=gradient)
        new_array = np.vstack([new_array, Imodpoly_output])

    return pd.DataFrame(new_array)

if __name__ == '__main__':
    ages_df = pd.read_csv("AGEs.csv")
    arm_df = pd.read_csv("innerArm.csv")
    thumbnail_df = pd.read_csv("thumbNail.csv")
    earlobe_df = pd.read_csv("earLobe.csv")
    vein_df = pd.read_csv("vein.csv")

    final_df = polyfit(crop_range_to_numpy(vein_df))
    final_df.to_csv('raw_raman_spectroscopy/vein_df.csv', sep=',')
    print(final_df)