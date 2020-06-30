import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from PIL import Image,ImageFilter

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

if __name__ == "__main__":
    pca = False
    plot_3d_gif = False
    df_train = pd.read_csv('data/train_values.csv')
    df_train_labels = pd.read_csv('data/train_labels.csv')
    df_combined = df_train.merge(df_train_labels, left_index=True, right_index=True)

    one = df_combined[df_combined['damage_grade']==1]
    two = df_combined[df_combined['damage_grade']==2]
    three = df_combined[df_combined['damage_grade']==3]

    if pca:
        X = df_combined[['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage']]
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
                    ax.scatter(df_pca.loc[idx, 'pc1'], df_pca.loc[idx, 'pc2'], df_pca.loc[idx, 'pc2'], color=c, label=labels[t-1])
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
            images[0].save('pca.gif',
                        save_all=True,
                        append_images=images[1:],
                        duration=150,
                        loop=0)
