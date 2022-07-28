import pandas as pd


def cat_var_to_onehot_encoded(df, col):
    series = df[col]
    one_hot = pd.get_dummies(series)
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(col, axis = 1)
    return df

