import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from PIL import Image,ImageFilter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

plt.style.use('ggplot')

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

def tts(X, y, strat=True):
    if strat:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test

def standardize(X):
    ss = StandardScaler(with_mean=True, with_std=True)
    X_scaled = ss.fit_transform(X)
    return X_scaled

def feat_importance_plot(model, file_name, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_indices = feature_names[indices]
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax.set_title("Feature Importances", fontsize=18)
    ax.bar(range(X.shape[1]), importances[indices],
            color="teal", align="center")
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(feat_indices, rotation=90)
    ax.set_ylabel('Feature Importance', fontsize=14) 
    ax.set_xlabel('Feature', fontsize=14)
    plt.tight_layout(pad=1)
    plt.savefig(f'{file_name}.png', dpi=100)

if __name__ == "__main__":
    pca = False
    plot_3d_gif = False
    rf = False
    lr = False
    gb = False
    df_train = pd.read_csv('data/train_values.csv')
    df_train_labels = pd.read_csv('data/train_labels.csv')
    df_train.set_index(df_train['building_id'], inplace=True)
    df_train.drop(columns='building_id', inplace=True)
    df_train_labels.set_index(df_train_labels['building_id'], inplace=True)
    df_train_labels.drop(columns='building_id', inplace=True)
    df_combined = df_train.merge(df_train_labels, left_index=True, right_index=True)

    # df_dummies = df_combined.drop(columns=['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'])
    # df_dummies = pd.get_dummies(df_dummies, columns=['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
    #             'position', 'plan_configuration', 'legal_ownership_status'], drop_first=True)

    one = df_combined[df_combined['damage_grade']==1]
    two = df_combined[df_combined['damage_grade']==2]
    three = df_combined[df_combined['damage_grade']==3]

    if pca:
        X = df_combined[['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage']]
        df_int = pd.get_dummies(df_combined, columns=[], drop_first=True)
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)
        pca = PCA(n_components=3, random_state=42)
        p_components = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(data = p_components, columns = ['pc1', 'pc2', 'pc3'], index=df_combined.index)
        df_pca = pd.concat([df_pca, df_combined[['damage_grade']]], axis=1)

        if plot_3d_gif:
            targets = [1, 2, 3]
            labels = ['Low', 'Moderate', 'High']
            color = ['yellow', 'orange', 'red']
            frames = 36
            for n in range(frames):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for t, c in zip(targets, color):
                    idx = df_pca['damage_grade'] == t
                    ax.scatter(df_pca.loc[idx, 'pc1'], df_pca.loc[idx, 'pc2'], df_pca.loc[idx, 'pc3'], color=c, label=labels[t-1])
                ax.set_title('PCA with 3 Components', fontsize=20, loc= 'left')
                ax.set_xlabel('PC 1', fontsize=13)
                ax.set_ylabel('PC 2', fontsize=13)
                ax.set_zlabel('PC 3', fontsize=13)
                ax.legend()
                ax.view_init(30, (n*10))
                plt.draw()
                plt.pause(.001)
                plt.savefig(str(n)+'.png')
                plt.close()

            images = []
            for n in range(frames):
                exec('a'+str(n)+'=Image.open("'+str(n)+'.png")')
                images.append(eval('a'+str(n)))
            images[0].save('pca_july25.gif',
                        save_all=True,
                        append_images=images[1:],
                        duration=150,
                        loop=0)
    df_model = pd.get_dummies(df_combined, columns=['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 
            'other_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status'], drop_first=True)
    y = df_model.pop('damage_grade')
    X = df_model.values
    X_train, X_test, y_train, y_test = tts(X, y)
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.fit_transform(X_test)

    if rf:
        rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-2)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_f1 = f1_score(y_test, rf_preds, average='micro')
        rf_feat_imp = rf.feature_importances_
        print(rf_f1)
    if lr:
        lr = LogisticRegression(C=.001, class_weight='balanced', random_state=42, n_jobs=-2)
        lr.fit(X_train_scaled, y_train)
        lr_preds = lr.predict(X_test_scaled)
        lr_f1 = f1_score(y_test, lr_preds, average='micro')
        print(lr_f1)
    if gb:
        gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
        gb.fit(X_train, y_train)
        gb_preds = gb.predict(X_test)
        gb_f1 = f1_score(y_test, gb_preds, average='micro')
        gb_feat_imp = gb.feature_importances_
        print(gb_f1)