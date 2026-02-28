
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os

RANDOM_SEED = 42

def normalize_hand(row):
    # wrist (landmark 0)
    x0, y0 = row["x1"], row["y1"]

    # middle finger tip (landmark 13)
    xm, ym = row["x13"], row["y13"]
    
    new_row = row.copy()
    for i in range(1, 22):
        new_row[f"x{i}"] = (row[f"x{i}"] - x0) / xm
        new_row[f"y{i}"] = (row[f"y{i}"] - y0) / ym

    return new_row

def read_data(file_path:Path):
    df = pd.read_csv(file_path)
    return df

def data_preprocess(df:pd.DataFrame):
    #==========Spliting Data===================================
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    #===========Encoding Data===================================
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    #===========Scaling Data===================================
    X_train = X_train.apply(normalize_hand, axis=1)
    X_test = X_test.apply(normalize_hand, axis=1)


    return X_train, X_test, y_train, y_test