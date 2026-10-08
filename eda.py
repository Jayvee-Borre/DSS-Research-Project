import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# AI Roles augmented and replaced
def main():
    pd.set_option('display.max_columns', None)
    try:
        df = pd.read_csv('global_ai.csv') # Same directory just change value if needed
    except FileNotFoundError as e:
        print("File isn't found in your directory!")
        return 

    null_vals = df.isnull().sum()
    print(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")
    print(f"Null Column Values: {null_vals[null_vals > 1]}\n\n") # Null values

    # DROPPING COMPANY ID
    df.drop(columns=['Company_ID'], inplace=True)

    # Encode object type to int
    le = LabelEncoder()
    df['Company_Size'] = le.fit_transform(df['Company_Size']) # Medium -> 2, Large -> 1, Small -> 3, Enterprise -> 0
    df['Has_Strict_AI_Governance'] = le.fit_transform(df['Has_Strict_AI_Governance'])  # True -> 1, False -> 0
    df['Industry'] = le.fit_transform(df['Industry'])
    df['Primary_AI_Agent_Role'] = le.fit_transform(df['Primary_AI_Agent_Role'])

    # showHistPlot(df, 'Months_To_Positive_ROI')
    # showBoxPlot(df, 'Months_To_Positive_ROI')
    # showBasicInfo()

    # Compute Z_Scores for the dataset for outlier detection
    df_num = df.select_dtypes(include='number')
    for col in df_num:
        z_score = np.abs(stats.zscore(df_num[col]))
        outliers = df_num[z_score > 3]
        if outliers.empty:
            print(f"No Outliers at {col}")
        else:
            print(f"Outliers at {col}: {outliers.shape[0]}")

    df_corr = df.corr()
    # showCorrMatrix(df_corr)

def showBasicInfo(data):
    print(f"{data.info()}\n\n")
    print(f"{data.head()}\n\n")

def showHistPlot(data, colummName):
    sns.histplot(data[colummName], kde=True) # THE GRAPH IS FAT
    plt.show()

def showBoxPlot(data, columnName):
    sns.boxplot(x=data[columnName])
    plt.show()

def showCorrMatrix(data):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(data, annot=True, square=True, robust=True, fmt='.1f', 
                        cmap='coolwarm', annot_kws={"size": 9}, linewidths=0)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.title("Correlational Matrix of AI Roles Dataset")
    plt.show()

if __name__ == '__main__':
    main()