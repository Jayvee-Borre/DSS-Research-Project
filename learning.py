from Modeling import Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
import numpy as np


class ModelLearn(Modeling):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.scaler = StandardScaler()
        self.incidence_model = None
        self.regression_model = None
        self.y_pred = None
        self.is_log_target = False

    def splitDataset(self, target_col: str, drop_cols: list = None, test_size: float = 0.3,
                     log_transform_target: bool = False, stratify: bool = False):
        drop_list = [target_col]
        if drop_cols:
            drop_list.extend(
                [c for c in drop_cols if c in self.dataset.columns])

        X = self.dataset.drop(columns=drop_list)
        y = self.dataset[target_col]
        self.is_log_target = log_transform_target

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if stratify else None
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def trainIncidence(self):
        self.incidence_model = LogisticRegression(max_iter=1000)
        self.incidence_model.fit(self.X_train, self.Y_train)

    def predictIncidence(self):
        self.y_pred = self.incidence_model.predict(self.X_test)
        return self.y_pred

    def evaluateIncidence(self):
        acc = accuracy_score(self.Y_test, self.y_pred)
        y_prob = self.incidence_model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.Y_test, y_prob)

        print("--- Classification Evaluation ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC:  {auc:.4f}\n")
        print("Confusion Matrix:")
        print(confusion_matrix(self.Y_test, self.y_pred))
        print("\nClassification Report:")
        print(classification_report(self.Y_test, self.y_pred))

    def trainRegression(self):
        self.regression_model = LinearRegression()
        if self.is_log_target:
            self.regression_model.fit(self.X_train, np.log1p(self.Y_train))
        else:
            self.regression_model.fit(self.X_train, self.Y_train)

    def predictRegression(self):
        raw_pred = self.regression_model.predict(self.X_test)
        if self.is_log_target:
            self.y_pred = np.clip(np.expm1(raw_pred), 0, None)
        else:
            self.y_pred = raw_pred
        return self.y_pred

    def evaluateRegression(self):
        r2 = r2_score(self.Y_test, self.y_pred)
        mse = mean_squared_error(self.Y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.Y_test, self.y_pred)

        print("--- Regression Evaluation (Original Scale) ---")
        print(f"R2-Score: {r2:.4f}")
        print(f"MSE:      {mse:.4f}")
        print(f"MAE:      {mae:.4f}")
        print(f"RMSE:     {rmse:.4f}")
