from Modeling import Modeling
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FeatureSelect(Modeling):
    def __init__(self, dataset):
        super().__init__(dataset)

    def performVarianceThreshold(self, threshold=0.01):
        numeric_df = self.dataset.select_dtypes(include=[np.number])
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(numeric_df)
        retained_columns = numeric_df.columns[vt.get_support()]
        dropped_columns = list(set(numeric_df.columns) - set(retained_columns))
        if dropped_columns:
            self.dataset.drop(columns=dropped_columns, inplace=True)
            print(f"Dropped low-variance features: {dropped_columns}")
        else:
            print(f"No features below variance threshold of {threshold}")
        return self.dataset

    def plotMutualInformation(self, target_col: str, drop_cols: list = None,
                              discrete_features: list = None, classification: bool = False):
        exclude_list = [target_col]
        if drop_cols:
            exclude_list.extend(
                [c for c in drop_cols if c in self.dataset.columns])

        X = self.dataset.drop(columns=exclude_list).copy()
        y = self.dataset[target_col]

        discrete_features = discrete_features or []
        # Treat listed columns and any 0/1 columns (e.g. dummies) as discrete
        discrete_mask = [
            (col in discrete_features) or X[col].dropna().isin([0, 1]).all()
            for col in X.columns
        ]

        mi_fn = mutual_info_classif if classification else mutual_info_regression
        mi_scores = mi_fn(
            X, y, discrete_features=discrete_mask, random_state=42)

        mi_series = pd.Series(
            mi_scores, index=X.columns).sort_values(ascending=True)

        plt.figure(figsize=(10, 6))
        mi_series.plot(kind='barh', color='skyblue', edgecolor='black')
        plt.title(f"Mutual Information Scores (Target: {target_col})")
        plt.xlabel("Mutual Information Score (Higher = Stronger Association)")
        plt.tight_layout()
        plt.show()

        return mi_series.sort_values(ascending=False)

    def checkMutualInformation(self, discrete_cols):
        print("--- Mutual Information for Human Roles Replaced ---")
        if 'Human_Roles_Replaced' in self.dataset.columns:
            mi_replaced = self.plotMutualInformation(
                target_col='Human_Roles_Replaced',
                drop_cols=['Has_Roles_Replaced'],
                discrete_features=discrete_cols
            )
            print(mi_replaced)

        print("\n--- Mutual Information for Human Roles Augmented ---")
        if 'Human_Roles_Augmented' in self.dataset.columns:
            mi_augmented = self.plotMutualInformation(
                target_col='Human_Roles_Augmented',
                drop_cols=['Has_Roles_Replaced', 'Human_Roles_Replaced'],
                discrete_features=discrete_cols
            )
            print(mi_augmented)

        print("\n--- Mutual Information for Has Roles Replaced ---")
        if 'Has_Roles_Replaced' in self.dataset.columns:
            mi_hasroles = self.plotMutualInformation(
                target_col='Has_Roles_Replaced',
                drop_cols=['Human_Roles_Replaced'],
                discrete_features=discrete_cols,
                classification=True
            )
            print(mi_hasroles)
