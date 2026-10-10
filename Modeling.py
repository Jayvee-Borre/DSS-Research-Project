import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Modeling:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    def showBasicInfo(self):
        print(f"Rows: {self.dataset.shape[0]}\nColumns: {self.dataset.shape[1]}")
        print(f"{self.dataset.info()}\n\n")
        print(f"{self.dataset.head()}")

    def showHistogram(self, columnName):
        sns.histplot(self.dataset[columnName], kde=True)
        plt.show()

    def showBoxPlotX(self, columnName):
        sns.boxplot(x=self.dataset[columnName])
        plt.show()

    def showBoxPlotY(self, columnName):
        sns.boxplot(y=self.dataset[columnName])
        plt.show()

    def showCorrMatrix(self, title: str):
        corr_data = self.dataset.corr()
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(corr_data, annot=True, square=True, robust=True, fmt='.1f',
                            cmap='coolwarm', annot_kws={"size": 6}, linewidths=0)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.title(title)
        plt.show()

    def logTransformFeature(self, columnName):
        self.dataset[columnName] = np.log1p(self.dataset[columnName])

    def renameColumn(self, columnName, newName):
        self.dataset.rename(columns={f'{columnName}': f'{newName}'}, inplace=True)