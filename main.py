from eda import EDA
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# AI Roles augmented and replaced
def main():
    pd.set_option('display.max_columns', None)
    try:
        df = pd.read_csv('global_ai.csv')
    except FileNotFoundError as e:
        print("Could not find dataset: " + e)
        return
    
    eda = EDA(df)
    eda.showBasicInfo()
    eda.performEDA()
    for col in df.columns:
        eda.showHistogram(col)

    

if __name__ == '__main__':
    main()