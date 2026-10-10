from sklearn.feature_selection import mutual_info_regression
from Modeling import Modeling
from sklearn import feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FeatureSelect(Modeling):
    def __init__(self, dataset):
        super().__init__(dataset)

    def performVarianceThreshold(self):
        vt = feature_selection.VarianceThreshold(threshold=.8)
        vt.fit_transform(self.dataset)

    def plotMutualInformation(self, target_col: str, discrete_features: list = None):
        """
        Calculates and plots Mutual Information scores for all features against a target.
        """
        X = self.dataset.drop(columns=[target_col]).copy() # features list
        y = self.dataset[target_col] # target

        # Identify discrete/categorical columns for scikit-learn
        if discrete_features is None:
            discrete_mask = (X.dtypes == 'int64') | (X.dtypes == 'bool')
        else:
            discrete_mask = [col in discrete_features for col in X.columns]

        # Compute mutual information
        mi_scores = mutual_info_regression(
            X, y, 
            discrete_features=discrete_mask, 
            random_state=42
        )
        
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)

        # Plot horizontal bar chart
        plt.figure(figsize=(10, 6))
        mi_series.plot(kind='barh', color='skyblue', edgecolor='black')
        plt.title(f"Mutual Information Scores (Target: {target_col})")
        plt.xlabel("Mutual Information Score (Higher = Stronger Association)")
        plt.tight_layout()
        plt.show()

        return mi_series.sort_values(ascending=False)

    def checkMutualInformation(self, discrete_cols):
        # Run MI for Human_Roles_Replaced
        print("--- Mutual Information for Human Roles Replaced ---")
        mi_replaced = self.plotMutualInformation(
            target_col='Human_Roles_Replaced', 
            discrete_features=discrete_cols
        )
        print(mi_replaced)
    
        # Run MI for Human_Roles_Augmented
        print("\n--- Mutual Information for Human Roles Augmented ---")
        mi_augmented = self.plotMutualInformation(
            target_col='Human_Roles_Augmented', 
            discrete_features=discrete_cols
        )
        print(mi_augmented)
        # Run MI for Human_Roles_Augmented
        print("\n--- Mutual Information for Has Roles Replaced ---")
        mi_hasroles = self.plotMutualInformation(
            target_col='Has_Roles_Replaced', 
            discrete_features=discrete_cols
        )
        print(mi_hasroles)