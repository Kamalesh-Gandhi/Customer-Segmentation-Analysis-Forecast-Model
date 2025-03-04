import pandas as pd
import logging
from scipy.stats import chi2_contingency,ttest_ind
from scipy.stats import f_oneway, pearsonr
from typing import Tuple
from typing_extensions import Annotated

def Feature_Selection_Classification(df_train:pd.DataFrame,
                                     df_test:pd.DataFrame, 
                                     continuous_cols:list,
                                     categorical_cols:list, 
                                     target_col:str) ->tuple[pd.DataFrame,pd.DataFrame]:
    """
    

    """

    try:

        logging.info("Feature Selection for classification Problem")
        logging.info(continuous_cols)
        logging.info(categorical_cols)
        logging.info(target_col)

        # Remove target column from feature lists
        if target_col in continuous_cols:
            continuous_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        results = []

        # Perform T-Test for continuous variables
        for col in continuous_cols:
            group1 = df_train[df_train[target_col] == df_train[target_col].unique()[0]][col]
            group2 = df_train[df_train[target_col] == df_train[target_col].unique()[1]][col]
            
            stat, p_value = ttest_ind(group1, group2, equal_var=False)  # T-Test
            results.append({"Feature": col, "Test": "T-Test", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Perform Chi-Square Test for categorical variables
        for col in categorical_cols:
            contingency_table = pd.crosstab(df_train[col], df_train[target_col])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            results.append({"Feature": col, "Test": "Chi-Square", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Convert to DataFrame and display results
        results_df = pd.DataFrame(results)

        print(results_df)

        # Filter the results to get only the features with significance < 0.05
        significant_features1 = results_df[results_df["Significant (<0.05)"] == True]["Feature"].tolist()
        significant_features1.append(target_col)

        classification_feature_selected_dfTrain = df_train[significant_features1]
        classification_feature_selected_dfTest = df_test[significant_features1]

        if classification_feature_selected_dfTrain is None:
            raise ValueError("🚨 classification_feature_selected_df Train is empty. Check the significant features value.")
        
        logging.info(f'Classification train data shape: {classification_feature_selected_dfTrain.shape[0]} rows, {classification_feature_selected_dfTrain.shape[1]} columns.')

        if classification_feature_selected_dfTest is None:
            raise ValueError("🚨 classification_feature_selected_df Test is empty. Check the significant features value.")

        logging.info(f'Classification test data shape: {classification_feature_selected_dfTest.shape[0]} rows, {classification_feature_selected_dfTest.shape[1]} columns.')
        

        return classification_feature_selected_dfTrain, classification_feature_selected_dfTest
    
    except Exception as e:
        logging.error('Error in Feature Selection for Classification Problem')
        raise e


def Feature_Selection_Regression(df_train:pd.DataFrame, 
                                 df_test:pd.DataFrame,
                                continuous_cols:list,
                                categorical_cols:list, 
                                target_col:str) -> tuple[pd.DataFrame,pd.DataFrame]: 
    
    """
    
    
    """
    
    try:

        logging.info('Feature Selection for Regression Problem')


        # Remove target column from feature lists
        if target_col in continuous_cols:
            continuous_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        results = []

        # Perform ANOVA F-Test for categorical variables
        for col in categorical_cols:
            groups = [df_train[df_train[col] == cat][target_col] for cat in df_train[col].unique()]
            f_stat, p_value = f_oneway(*groups)
            results.append({"Feature": col, "Test": "ANOVA", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Perform Pearson Correlation Test for continuous variables
        for col in continuous_cols:
            corr_coeff, p_value = pearsonr(df_train[col], df_train[target_col])
            results.append({"Feature": col, "Test": "Pearson Correlation", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        print(results_df)

        # Extract only significant features
        significant_features2 = results_df[results_df["Significant (<0.05)"] == True]["Feature"].tolist()
        significant_features2.append(target_col)

        Regression_feature_selected_dfTrain = df_train[significant_features2]
        Regression_feature_selected_dfTest = df_test[significant_features2]

        if Regression_feature_selected_dfTrain is None:
            raise ValueError("🚨 Regression_feature_selected_df Train is empty. Check the significant features value.")
        
        logging.info(f'Regression train data shape: {Regression_feature_selected_dfTrain.shape[0]} rows, {Regression_feature_selected_dfTrain.shape[1]} columns.')

        if Regression_feature_selected_dfTest is None:
            raise ValueError("🚨 Regression_feature_selected_df Test is empty. Check the significant features value.")

        logging.info(f'Regression test data shape: {Regression_feature_selected_dfTest.shape[0]} rows, {Regression_feature_selected_dfTest.shape[1]} columns.')
        logging.info(Regression_feature_selected_dfTrain.columns)
        logging.info(Regression_feature_selected_dfTest.columns)

        return Regression_feature_selected_dfTrain , Regression_feature_selected_dfTest 

    except Exception as e:
        logging.error('Error in Feature Selection for Regression Problem')
        raise e
