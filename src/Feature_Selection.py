import pandas as pd
import logging
from scipy.stats import chi2_contingency,ttest_ind
from scipy.stats import f_oneway, pearsonr

def Feature_Selection_Classification(df:pd.DataFrame, 
                                     continuous_cols:list,
                                     categorical_cols:list, 
                                     target_col:str) -> list:
    """
    

    """

    try:

        logging.info("Feature Selection for classification Problem")

        # Remove target column from feature lists
        if target_col in continuous_cols:
            continuous_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        results = []

        # Perform T-Test for continuous variables
        for col in continuous_cols:
            group1 = df[df[target_col] == df[target_col].unique()[0]][col]
            group2 = df[df[target_col] == df[target_col].unique()[1]][col]
            
            stat, p_value = ttest_ind(group1, group2, equal_var=False)  # T-Test
            results.append({"Feature": col, "Test": "T-Test", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Perform Chi-Square Test for categorical variables
        for col in categorical_cols:
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            results.append({"Feature": col, "Test": "Chi-Square", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Convert to DataFrame and display results
        results_df = pd.DataFrame(results)

        # Filter the results to get only the features with significance < 0.05
        significant_features1 = results_df[results_df["Significant (<0.05)"] == True]["Feature"].tolist()

        return significant_features1
    
    except Exception as e:
        logging.error('Error in Feature Selection for Classification Problem')
        raise e


def Feature_Selection_Regression(df:pd.DataFrame, 
                                     continuous_cols:list,
                                     categorical_cols:list, 
                                     target_col:str) -> list :
    
    """
    
    
    """
    
    try:

        logging.info('Feature Selection for Regression Problem')



        target_col = 'price' 

        # Remove target column from feature lists
        if target_col in continuous_cols:
            continuous_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        results = []

        # Perform ANOVA F-Test for categorical variables
        for col in categorical_cols:
            groups = [df[df[col] == cat][target_col] for cat in df[col].unique()]
            f_stat, p_value = f_oneway(*groups)
            results.append({"Feature": col, "Test": "ANOVA", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Perform Pearson Correlation Test for continuous variables
        for col in continuous_cols:
            corr_coeff, p_value = pearsonr(df[col], df[target_col])
            results.append({"Feature": col, "Test": "Pearson Correlation", "P-Value": p_value, "Significant (<0.05)": p_value < 0.05})

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Extract only significant features
        significant_features2 = results_df[results_df["Significant (<0.05)"] == True]["Feature"].tolist()

        return significant_features2

    except Exception as e:
        logging.error('Error in Feature Selection for Regression Problem')
        raise e
