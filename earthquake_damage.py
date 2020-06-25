import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def damage_distributions(col_name, df):
    data = [df[df['damage_grade']==1][col_name], 
            df[df['damage_grade']==2][col_name], 
            df[df['damage_grade']==3][col_name]]
    return data

if __name__ == "__main__":
    df_train = pd.read_csv('data/train_values.csv')
    df_train_labels = pd.read_csv('data/train_labels.csv')
    df_combined = df_train.merge(df_train_labels, left_index=True, right_index=True)
    