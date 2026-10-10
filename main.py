from eda import EDA
from feature_select import FeatureSelect
from learning import ModelLearn
import pandas as pd


def main():
    pd.set_option('display.max_columns', None)
    try:
        df = pd.read_csv('global_ai.csv')
    except FileNotFoundError as e:
        print("Could not find dataset: " + str(e))
        return

    eda = EDA(df)

    print("=== Diagnostic Checks ===")
    eda.checkNullValues()
    eda.checkOutliers()

    processed_df = eda.prepareData()

    featureselect = FeatureSelect(processed_df)
    featureselect.performVarianceThreshold(threshold=0.01)

    discrete_cols = ['Company_Size', 'Has_Strict_AI_Governance',
                     'Cybersecurity_Incidents_YTD', 'Has_Roles_Replaced']
    featureselect.checkMutualInformation(discrete_cols)

    drop_for_incidence = [
        "Human_Roles_Replaced",
        "Avg_Agent_Cost_Per_Month_USD",
        "Months_To_Positive_ROI",
        "Emp_SentimentScore"
    ]

    drop_for_roles_replaced = [
        "Has_Roles_Replaced",
        "Avg_Agent_Cost_Per_Month_USD",
        "Productivity_Gain_Percent",
        "Months_To_Positive_ROI",
        "Has_Strict_AI_Governance"
    ]

    print("\n=== 1. Classification: Has_Roles_Replaced ===")
    clf_model = ModelLearn(featureselect.getDataset())
    clf_model.splitDataset('Has_Roles_Replaced',
                           drop_cols=drop_for_incidence, test_size=0.3, stratify=True)
    clf_model.trainIncidence()
    clf_model.predictIncidence()
    clf_model.evaluateIncidence()

    print("\n=== 2. Regression: Human_Roles_Replaced ===")
    reg_model = ModelLearn(featureselect.getDataset())
    reg_model.splitDataset('Human_Roles_Replaced', drop_cols=drop_for_roles_replaced,
                           test_size=0.3, log_transform_target=True)
    reg_model.trainRegression()
    reg_model.predictRegression()
    reg_model.evaluateRegression()


if __name__ == '__main__':
    main()
