from eda import EDA
from feature_select import FeatureSelect
from learning import ModelLearn
import pandas as pd

# AI Roles augmented and replaced
def main():
    pd.set_option('display.max_columns', None)
    try:
        df = pd.read_csv('global_ai.csv')
    except FileNotFoundError as e:
        print("Could not find dataset: " + str(e))
        return
    
    eda = EDA(df)
    eda.performEDA()

    # Human Roles Augmented
    dropped_columns_for_HumanRolesAugmented = [
        "Industry",
        "Primary_AI_Agent_Role",
        "Avg_Agent_Cost_Per_Month_USD",
        "Months_To_Positive_ROI",
        "Has_Strict_AI_Governance"
    ]

    # Has Roles Replaced
    dropped_columns_for_HasRolesReplaced = [
        "Human_Roles_Replaced",
        "Industry",
        "Avg_Agent_Cost_Per_Month_USD",
        "Months_To_Positive_ROI",
        "Emp_SentimentScore"
    ]

    dropped_columns_for_HumanRolesReplaced = [
        "Has_Roles_Replaced",
        "Industry",
        "Avg_Agent_Cost_Per_Month_USD",
        "Productivity_Gain_Percent",
        "Months_To_Positive_ROI",
        "Has_Strict_AI_Governance"
    ]

    discrete_cols = [
        'Company_Size',
        'Industry',
        'Primary_AI_Agent_Role',
        'Has_Strict_AI_Governance',
        'Cybersecurity_Incidents_YTD',
        'Has_Roles_Replaced'
    ]

    featureselect = FeatureSelect(eda.getDataset())
    # featureselect.checkMutualInformation(discrete_cols) # Uncomment this if you need to check

    model = ModelLearn(featureselect.getDataset())
    
    # Choose One Target Variable To Run
    # Logistic Regression (Incidence)
    # model.splitDataset('Has_Roles_Replaced', dropped_columns_for_HasRolesReplaced, test_size=0.3)
    # model.trainIncidence()
    # model.predictIncidence()
    # model.evaluateIncidence()

    # Linear Regression (Magnitude - Replaced)
    # model.splitDataset('Human_Roles_Replaced', dropped_columns_for_HumanRolesReplaced, test_size=0.3)
    # model.trainRegression()
    # model.predictRegression()
    # model.evaluateRegression()

    # Linear Regression (Magnitude - Augmented)
    # model.splitDataset('Human_Roles_Augmented', dropped_columns_for_HumanRolesAugmented, test_size=0.3)
    # model.trainRegression()
    # model.predictRegression()
    # model.evaluateRegression()

if __name__ == '__main__':
    main()