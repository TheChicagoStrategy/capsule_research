from function_bank import *
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns = 20


def main():
    data = pd.read_csv('/Users/quintonaguilera/Desktop/Aederide & Toledo/capsule_research/data/Data.xlsx - MUPS.csv',
                       delimiter=",")

    mups = data[data['Tablet'] == 'MUPS'] # isolate mups

    relevant_vars = ['Excipient', 'CP', 'Family', 'AoR', 'Span', 'MPS', 'BD', 'HR', 'Y', 'E']

    mups = mups[relevant_vars]
    fam_avgs = mups.groupby('Family')[['E', 'BD', 'MPS', 'Span']].mean()
    exp_avgs = mups.groupby('Excipient')[['E', 'BD', 'MPS', 'Span']].mean()
    com_avgs = mups.groupby('CP')[['E', 'BD', 'MPS', 'Span']].mean()

    mups = cat_var_to_onehot_encoded(mups, 'Excipient')
    mups = cat_var_to_onehot_encoded(mups, 'Family')

    # sns.lmplot(x="CP", y="E", data=mups, order=2, ci=None)
    # print(mups)

    regr = LinearRegression()
    regr.fit(mups[['CP']], mups[['E']])
    print('Predict E using Compaction Pressure')
    print(regr.score(mups[['CP']], mups[['E']]))
    print(com_avgs)
    print()

    regr = LinearRegression()
    regr.fit(mups[['CP','KG','PH','UF']], mups[['E']])
    print('Predict E using Compaction Pressure & Family')
    print(regr.score(mups[['CP','KG','PH','UF']], mups[['E']]))
    print(fam_avgs)
    print()

    regr = LinearRegression()
    regr.fit(mups[['CP', 'KG-1000', 'KG-802', 'PH-101', 'PH-102', 'UF-702', 'UF-711']], mups[['E']])
    print('Predict E using Compaction Pressure & Excipient')
    print(regr.score(mups[['CP','KG-1000', 'KG-802', 'PH-101', 'PH-102', 'UF-702', 'UF-711']], mups[['E']]))
    print(exp_avgs)
    print()


if __name__ == "__main__":
    main()
