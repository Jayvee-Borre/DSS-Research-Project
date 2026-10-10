from eda import EDA
from feature_select import FeatureSelect
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

    discrete_cols = [
        'Company_Size', 
        'Industry', 
        'Primary_AI_Agent_Role', 
        'Has_Strict_AI_Governance', 
        'Cybersecurity_Incidents_YTD',
        'Has_Roles_Replaced'
    ]

    featureselect = FeatureSelect(df)
    featureselect.checkMutualInformation(discrete_cols)

if __name__ == '__main__':
    main()