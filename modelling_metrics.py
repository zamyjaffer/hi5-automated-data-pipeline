import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_modelling_metrics(dataframe, column_list):
    """
    Generate modeling metrics based on the provided DataFrame and column list.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the data for calculating metrics.
        column_list (list): List of columns to calculate metrics for.

    Returns:
        pandas.DataFrame: The DataFrame containing the calculated modeling metrics.
    """
    print('Generating Data to Model')
    result_data = []
    
    # Loop through each column and calculate metrics
    for column in tqdm(column_list, desc='Processing Rows'):
        # Calculate absolute difference
        abs_difference = dataframe.groupby(['id', 'Trial'])[column].transform(lambda x: abs(x.diff().iloc[1]))
        
        # Calculate mean of the column
        mean_column = dataframe.groupby(['id', 'Trial'])[column].transform(lambda x: x.mean())
        
        # Calculate cross correlation
        cross_correlation_results = dataframe.groupby(['id', 'Trial'])[column].apply(lambda x: np.correlate(x, x, 'full'))
        cross_correlation_results = cross_correlation_results.apply(lambda x: x[0])
        
        # Create a DataFrame for cross correlation results
        cross_correlation_df = pd.DataFrame({
            'id': cross_correlation_results.index.get_level_values('id'),
            'trial': cross_correlation_results.index.get_level_values('Trial'),
            column.lower() + '_crosscorr': cross_correlation_results.values
        })
        
        # Append the calculated metrics to the result_data list
        result_data.append(pd.DataFrame({
            'id': dataframe['id'],
            'trial': dataframe['Trial'],
            'pairingtype': dataframe['PairingType'],
            column.lower() + '_difference': abs_difference,
            column.lower() + '_mean': mean_column
        }))
    
    # Merge the calculated metrics DataFrames
    modelling_data = pd.concat(result_data, axis=1)
    
    # Drop any duplicate columns
    modelling_data = modelling_data.loc[:, ~modelling_data.columns.duplicated()]
    # Drop any duplicate rows
    modelling_data = modelling_data.drop_duplicates()

    # Filter to relevant trials
    modelling_data = modelling_data[(modelling_data['trial'] >= 10) & (modelling_data['trial'] <= 30)]
    modelling_data = modelling_data.replace({"pairingtype": {"CC": 0, "CA": 1, "AA": 2}})
    modelling_data.drop(['id', 'trial'], axis=1, inplace=True)
    modelling_data.reset_index(drop=True, inplace=True)

    # Plot and save pairplot
    numerical_features = modelling_data.columns.to_list()
    numerical_features = numerical_features[1:]
    sns.pairplot(modelling_data, vars=numerical_features, hue='pairingtype', markers=["o", "s", "D"])
    plt.savefig('code/plots/modelling_data_pairplot.png')
    plt.close()
    print("Modelling Data Successfully Created.\n")
    # Return the merged DataFrame
    return modelling_data

def normalise_data(dataframe):
    """
    Normalize the given DataFrame by subtracting mean and dividing by standard deviation.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the data to be normalized.

    Returns:
        pandas.DataFrame: The normalized DataFrame.
    """
    print('Normalising Data for Modelling')
    def norm_set(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        z = (x - mu) / sigma
        return z
    
    # Extract the 'pairingtype' column and remove it from the DataFrame
    pairingtype_column = dataframe.pop('pairingtype')
    
    columns_to_normalise = dataframe.columns
    
    mean_data = dataframe[columns_to_normalise].mean().to_numpy()
    std_data = dataframe[columns_to_normalise].std().to_numpy()
    data = dataframe[columns_to_normalise].to_numpy()

    normalised_data = norm_set(data, mean_data, std_data)
    
    # Create a DataFrame with normalized columns
    output_data = pd.DataFrame(normalised_data, columns=columns_to_normalise)
    
    # Insert the 'pairingtype' column at the front of the DataFrame
    output_data.insert(0, 'pairingtype', pairingtype_column)
    print("Data Successfully Normalised.\n")
    return output_data