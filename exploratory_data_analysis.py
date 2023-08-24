import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_visualisation_data(dataframe):
    """
    Filter a DataFrame to retrieve data for visualisation based on specified conditions.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the data to be filtered.

    Returns:
        pandas.DataFrame: A filtered DataFrame containing data for visualisation.
    """
    print('Acessing Data for Visualisation')
    # Define the filter conditions
    stiffness_condition = dataframe['Stiffness'] == 0.0297
    connected_condition = dataframe['Connected'] == 1.0
    
    # Combine the filter conditions using the logical AND operator
    combined_filter = stiffness_condition & connected_condition
    
    # Apply the filter and return the filtered DataFrame
    filtered_dataframe = dataframe[combined_filter]

    print("Visualisation Data Sucessfully Accessed.\n")
    return filtered_dataframe

def generate_and_save_plots(dataframe):
    """
    Generate and save plots based on the data in the provided DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the data for plotting.

    Returns:
        None
    """
    print('Plotting Data.')
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Define the order of pairing types for boxplots
    order = ['CC', 'CA', 'AA']
    
    # Create and save the RMSE Boxplot by Pairing Type
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='PairingType', y='RMSE', data=dataframe, order=order)
    plt.title('RMSE Boxplot by Pairing Type for Connected Trials')
    plt.xlabel('Pairing Type')
    plt.ylabel('RMSE')
    plt.savefig('outputs/visualisations/rmse_boxplot.png')
    plt.close()  # Close the plot to avoid overlapping plots
    
    # Calculate average RMSE for each trial and pairing type
    avg_rmse_by_trial = dataframe.groupby(['PairingType', 'Trial'])['RMSE'].mean().reset_index()
    
    # Create and save the Linechart of RMSE averaged over trials
    plt.figure(figsize=(10, 5))
    for pairing_type in order:
        subset = avg_rmse_by_trial[avg_rmse_by_trial['PairingType'] == pairing_type]
        plt.plot(subset['Trial'], subset['RMSE'], marker='o', linestyle='-', label=pairing_type)
    plt.xlabel('Trial')
    plt.ylabel('RMSE')
    plt.title('RMSE Averaged over Trials')
    plt.legend()
    plt.savefig('outputs/visualisations/rmse_linechart.png')
    plt.close()  # Close the plot
    
    # Create and save the SPARC Boxplot by Pairing Type
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='PairingType', y='SPARC', data=dataframe, order=order)
    plt.title('SPARC Boxplot by Pairing Type for Connected Trials')
    plt.xlabel('Pairing Type')
    plt.ylabel('SPARC')
    plt.savefig('outputs/visualisations/sparc_boxplot.png')
    plt.close()  # Close the plot
    
    # Calculate average SPARC for each trial and pairing type
    avg_sparc_by_trial = dataframe.groupby(['PairingType', 'Trial'])['SPARC'].mean().reset_index()
    
    # Create and save the Linechart of SPARC averaged over trials
    plt.figure(figsize=(10, 5))
    for pairing_type in order:
        subset = avg_sparc_by_trial[avg_sparc_by_trial['PairingType'] == pairing_type]
        plt.plot(subset['Trial'], subset['SPARC'], marker='o', linestyle='-', label=pairing_type)
    plt.xlabel('Trial')
    plt.ylabel('SPARC')
    plt.title('SPARC Averaged over Trials')
    plt.legend()
    plt.savefig('outputs/visualisations/sparc_linechart.png')
    plt.close()  # Close the plot
    print("EDA Plots Successfully Saved.\n")
