from Modeling import Modeling
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class EDA(Modeling):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)
    
    def performEDA(self):
        le = LabelEncoder()
        minmax = MinMaxScaler()

        # DROPPING COMPANY ID and Renaming Employee sentiment score feature
        self.dataset.drop(columns=['Company_ID'], inplace=True)
        self.renameColumn('Employee_Sentiment_Score_1_to_10', 'Emp_SentimentScore')
    
        # Encode object type to int
        self.dataset['Company_Size'] = le.fit_transform(self.dataset['Company_Size']) # Medium -> 2, Large -> 1, Small -> 3, Enterprise -> 0
        self.dataset['Has_Strict_AI_Governance'] = le.fit_transform(self.dataset['Has_Strict_AI_Governance'])  # True -> 1, False -> 0
        self.dataset['Industry'] = le.fit_transform(self.dataset['Industry'])
        self.dataset['Primary_AI_Agent_Role'] = le.fit_transform(self.dataset['Primary_AI_Agent_Role'])

        # Log transform skewed features
        self.logTransformFeature('Autonomous_Agents_Deployed')
        self.logTransformFeature('Human_Roles_Augmented')

        # Fix skewness of employee sentiment score, makes it so its now just
        # A little bit left skewed
        self.dataset['Emp_SentimentScore'] = minmax.fit_transform(self.dataset[['Emp_SentimentScore']])

        # Create new column for roles repalced
        self.dataset['Has_Roles_Replaced'] = (self.dataset['Human_Roles_Replaced'] > 0).astype(int)
        self.logTransformFeature('Human_Roles_Replaced')

    def checkNullValues(self):
        null_values = self.dataset.isnull().sum()
        print(f"Null Column Values: {null_values[null_values > 1]}\n\n") # Null values

    def checkOutliers(self):
        # Compute Z_Scores for the dataset for outlier detection
        df_num = self.dataset.select_dtypes(include='number')
        for col in df_num:
            z_score = np.abs(stats.zscore(df_num[col]))
            outliers = df_num[z_score > 3]
            if outliers.empty:
                print(f"No Outliers at {col}")
            else:
                print(f"Outliers at {col}: {outliers.shape[0]}")

    