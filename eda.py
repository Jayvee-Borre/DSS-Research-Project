from Modeling import Modeling
from scipy import stats
import numpy as np
import pandas as pd


class EDA(Modeling):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)

    def checkNullValues(self):
        null_counts = self.dataset.isnull().sum()
        active_nulls = null_counts[null_counts > 0]
        if active_nulls.empty:
            print("No missing values found across dataset columns.")
        else:
            print(f"Null Column Values:\n{active_nulls}\n")

    def checkOutliers(self):
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.dataset[col].dropna()))
            outlier_count = (z_scores > 3).sum()
            if outlier_count == 0:
                print(f"No Z-score outliers (> 3 std) in: {col}")
            else:
                print(f"Outliers detected in {col}: {outlier_count} rows")

    def prepareData(self):
        if 'Company_ID' in self.dataset.columns:
            self.dataset.drop(columns=['Company_ID'], inplace=True)

        self.renameColumn('Employee_Sentiment_Score_1_to_10',
                          'Emp_SentimentScore')

        size_mapping = {
            'Small (1-50)': 0,
            'Medium (51-500)': 1,
            'Large (501-5000)': 2,
            'Enterprise (5000+)': 3
        }
        if 'Company_Size' in self.dataset.columns:
            self.dataset['Company_Size'] = self.dataset['Company_Size'].map(
                size_mapping)
            if self.dataset['Company_Size'].isna().any():
                raise ValueError("Unmapped Company_Size values found")

        if 'Has_Strict_AI_Governance' in self.dataset.columns:
            self.dataset['Has_Strict_AI_Governance'] = self.dataset['Has_Strict_AI_Governance'].astype(
                int)

        self.dataset['Has_Roles_Replaced'] = (
            self.dataset['Human_Roles_Replaced'] > 0).astype(int)

        nominal_cols = ['Industry', 'Primary_AI_Agent_Role']
        existing_nominals = [
            c for c in nominal_cols if c in self.dataset.columns]
        if existing_nominals:
            self.dataset = pd.get_dummies(
                self.dataset, columns=existing_nominals, drop_first=True, dtype=int)

        return self.dataset
