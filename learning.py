from Modeling import Modeling
from sklearn.model_selection import train_test_split
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
        self.incidence_model = None
        self.regression_model = None
        self.y_pred = None

    def splitDataset(self, target_col: str, drop_cols: list = None, test_size: float = 0.2):
        drop_list = [target_col]
        if drop_cols:
            drop_list.extend(drop_cols)

        X = self.dataset.drop(columns=drop_list)
        y = self.dataset[target_col]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    def trainIncidence(self):
        self.incidence_model = LogisticRegression(max_iter=1000)
        self.incidence_model.fit(self.X_train, self.Y_train)

    def predictIncidence(self):
        self.y_pred = self.incidence_model.predict(self.X_test)
        return self.y_pred

    def evaluateIncidence(self):
        self.acc = accuracy_score(self.Y_test, self.y_pred)
        self.y_prob = self.incidence_model.predict_proba(self.X_test)[:, 1]
        self.auc = roc_auc_score(self.Y_test, self.y_prob)

        print("--- Incidence Classification Evaluation ---")
        print(f"Accuracy: {self.acc:.4f}")
        print(f"ROC-AUC:  {self.auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.Y_test, self.y_pred))

        print("\nClassification Report:")
        print(classification_report(self.Y_test, self.y_pred))

    def trainRegression(self):
        self.regression_model = LinearRegression()
        self.regression_model.fit(self.X_train, self.Y_train)

    def predictRegression(self):
        self.y_pred = self.regression_model.predict(self.X_test)
        return self.y_pred        

    def evaluateRegression(self):
        self.r2_score = r2_score(self.Y_test, self.y_pred)
        self.mse = mean_squared_error(self.Y_test, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(self.Y_test, self.y_pred)
        
        print("\nRegression Model Performance")
        print(f"{'-' * 15}")
        print("R2-Score:",self.r2_score)
        print("Mean-Squared-Error:",self.mse)
        print("Mean-Absolute-Error:",self.mae)
        print("RMSE:",self.rmse)