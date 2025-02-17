import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_excel("GRA.xls")
def dimensionlessProcessing(df_values, df_colums):
    scaler = StandardScaler()
    res = scaler.fit_transform(df_values)
    return pd.DataFrame(res, columns=df_colums)

def GRA_ONE(data, m=0):
    data = dimensionlessProcessing(data.values, data.columns)
    std = data.iloc[:, m]
    ce = data.copy()
    n = ce.shape[0]
    m = ce.shape[1]
    grap = np.zeros([n, m])
    for i in range(m):
        for j in range(n):
            grap[j, i] = abs(ce.iloc[j, i] - std[j])
    mmax = np.amax(grap)
    mmin = np.amin(grap)
    p = 0.5
    grap = pd.DataFrame(grap).applymap(lambda x: (mmin + p * mmax) / (x + p * mmax))
    RT = grap.mean(axis=0)
    return pd.Series(RT)

def GRA(data):
    list_columns = np.arange(data.shape[1])
    df_local = pd.DataFrame(columns=list_columns)
    for i in np.arange(data.shape[1]):
        df_local.iloc[:,i] = GRA_ONE(data, m=i)
    return df_local

data_gra = GRA(data)
def ShowGRAHeatMap(data, x_labels, y_labels, save_path=None):
    colormap = plt.cm.RdYlBu
    plt.figure(figsize=(18, 16))
    sns.heatmap(data.astype(float), linewidths=0.1, vmax=1.0, square=True,\
                cmap=colormap, linecolor='white', annot=True,\
                xticklabels=x_labels, yticklabels=y_labels)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
feature_labels = ['spindle speed', 'feed speed', 'cutting width','cutting depth',\
                                       'overhang elongation','machining dimensions']
ShowGRAHeatMap(data_gra, feature_labels, feature_labels, save_path='GRA_HeatMap_1.png')





