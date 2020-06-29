import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def damage_distributions(col_name, df):
    data = [df[df['damage_grade']==1][col_name], 
            df[df['damage_grade']==2][col_name], 
            df[df['damage_grade']==3][col_name]]
    return data

def damage_scatter_two_Components(x_col, y_col, low_df, med_df, high_df, save=False):
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    ax.scatter(low_df[x_col], low_df[y_col], color='yellow', alpha=.5, label='Low')
    ax.scatter(med_df[x_col], med_df[y_col], color='orange', alpha=.5, label='Medium')
    ax.scatter(high_df[x_col], high_df[y_col], color='red', alpha=.01, label='High')
    ax.set_title(f'{x_col} vs {y_col}', fontsize=20)
    ax.set_ylabel(f'{y_col}')
    ax.set_xlabel(f'{x_col}')
    ax.legend()
    if save:
        plt.savefig(f'images/{x_col} vs {y_col}', dpi=80)
    plt.show()

if __name__ == "__main__":
    df_train = pd.read_csv('data/train_values.csv')
    df_train_labels = pd.read_csv('data/train_labels.csv')
    df_combined = df_train.merge(df_train_labels, left_index=True, right_index=True)

    one = df_combined[df_combined['damage_grade']==1]
    two = df_combined[df_combined['damage_grade']==2]
    three = df_combined[df_combined['damage_grade']==3]

    X = df_combined[['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage']]
    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    p_components = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(data = p_components, columns = ['pc1', 'pc2', 'pc3'], index=df_combined.index)
    df_pca = pd.concat([df_pca, df_combined[['damage_grade']]], axis=1)

    targets = [1, 2, 3]
    color = ['yellow', 'orange', 'red']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for t, c in zip(targets, color):
        idx = pca_df['damage_grade'] == t
        ax.scatter(pca_3_df.loc[idx, 'pc1'], pca_3_df.loc[idx, 'pc2'], pca_3_df.loc[idx, 'pc2'], color=c)

