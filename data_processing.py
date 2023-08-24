import numpy as np
import pandas as pd
from tqdm import tqdm

def create_connected_metric(dataframe):
    """
    Create a new column in the DataFrame to identify connected/disconnected trials based on trial values.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to which the 'Connected' column will be added.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'Connected' column that indicates
                          whether each trial is connected (1) or disconnected (0) based on trial values.
    """
    print('Identifying Connected Trials')
    # Creating a new column to identify connected/disconnected trials
    dataframe['Connected'] = 0

    # Adding column to identify whether participants are connected or not
    # Iterate through each row of the DataFrame
    for index, row in tqdm(dataframe.iterrows(), desc="Processing Rows"):
        trial_value = row['Trial']

        # Check if the value in 'trial' is between 1 and 10
        if 1 <= trial_value <= 10:
            dataframe.at[index, 'Connected'] = 0
        # Check if the value in 'trial' is above 10 and even
        elif trial_value > 10 and trial_value % 2 == 0:
            dataframe.at[index, 'Connected'] = 0
        # Check if the value in 'trial' is above 10 and odd
        elif trial_value > 10 and trial_value % 2 != 0:
            dataframe.at[index, 'Connected'] = 1
    
    print("Connection Data Successfully Added.\n")
    return dataframe

def create_pairing_metric(dataframe):
    """
    Create a new column in the DataFrame to determine the pairing type based on participant ages in trials.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing participant and trial information.

    Returns:
        pandas.DataFrame: The input DataFrame with an additional 'PairingType' column that indicates
                          the pairing type for each trial based on the ages of participants.
    """
    print('Identifying Pairing Types.')

    for id in tqdm(dataframe['id'].unique(), desc='Processing Rows'):
        id_data = dataframe[dataframe['id'] == id]
        trials = id_data['Trial'].unique()

        # Iterate over the unique trials for the current ID
        for trial in trials:
            trial_data = id_data[id_data['Trial'] == trial]

            # Check if both participants are present in the current trial
            if 'G' in trial_data['Participant'].values and 'Y' in trial_data['Participant'].values:
                # Get the age values for both participants in the current trial
                age_participant_G = trial_data.loc[trial_data['Participant'] == 'G', 'Age'].values[0]
                age_participant_Y = trial_data.loc[trial_data['Participant'] == 'Y', 'Age'].values[0]

                # Compare the age values and assign the pairing type accordingly
                if age_participant_G < 18 and age_participant_Y < 18:
                    dataframe.loc[(dataframe['id'] == id) & (dataframe['Trial'] == trial), 'PairingType'] = 'CC'
                elif age_participant_G >= 18 and age_participant_Y < 18:
                    dataframe.loc[(dataframe['id'] == id) & (dataframe['Trial'] == trial), 'PairingType'] = 'CA'
                elif age_participant_G < 18 and age_participant_Y >= 18:
                    dataframe.loc[(dataframe['id'] == id) & (dataframe['Trial'] == trial), 'PairingType'] = 'CA'
    print("Pairing Type Data Successfully Added.\n")    
    return dataframe

def calculate_measure_metrics(dataframe, fs=1/0.001, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calculate measurement metrics including SPARC and RMSE for each trial in the DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing trial data.
        fs (float): Sampling frequency. Default is 1/0.001.
        padlevel (int): Padding level for FFT. Default is 4.
        fc (float): Cutoff frequency for SPARC calculation. Default is 10.0.
        amp_th (float): Amplitude threshold for SPARC calculation. Default is 0.05.

    Returns:
        pandas.DataFrame: A DataFrame containing calculated metrics (SPARC, RMSE) for each trial,
                          along with additional information about experiment ID, participant,
                          age, and stiffness.
    """
    print('Calculating SPARC & RMSE values.')

    def rmse(y_true, predictions):
        """
        Calculate the Root Mean Squared Error (RMSE) between two sets of values.

        Parameters:
            y_true (array-like): The true values.
            predictions (array-like): The predicted values.

        Returns:
            float: The RMSE value.
        """
        # Convert the inputs to NumPy arrays
        y_true, predictions = np.array(y_true), np.array(predictions)
        
        # Calculate the mean squared error
        mse_val = np.mean((y_true - predictions)**2)
        
        # Calculate the square root of the mean squared error to get RMSE
        rmse_val = np.sqrt(mse_val)
        
        return rmse_val
    
    def sparc(movement, fs=1/0.001, padlevel=4, fc=10.0, amp_th=0.05):
        """
        Calculate the Spectral ARC (SPARC) measure for a given movement signal.

        Parameters:
            movement (array-like): The movement signal data.
            fs (float): Sampling frequency of the movement signal. Default is 1/0.001.
            padlevel (int): Padding level for FFT computation. Default is 4.
            fc (float): Cutoff frequency for SPARC calculation. Default is 10.0.
            amp_th (float): Amplitude threshold for SPARC calculation. Default is 0.05.

        Returns:
            float: The calculated SPARC value.
        """
        # Calculate the number of points for FFT
        nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))
        
        # Generate the frequency range
        f = np.arange(0, fs, fs / nfft)
        
        # Compute the FFT of the movement signal
        Mf = abs(np.fft.fft(movement, nfft))
        
        # Normalise the FFT values
        Mf = Mf / max(Mf)
        
        # Find the indices of frequency components within the specified cutoff frequency
        fc_inx = ((f <= fc) * 1).nonzero()
        
        # Select the frequency components within the cutoff frequency
        f_sel = f[fc_inx]
        Mf_sel = Mf[fc_inx]
        
        # Find the indices of amplitude values above the threshold
        inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
        
        # Calculate the SPARC value based on selected frequency and amplitude components
        if len(inx) == 0:
            new_sal = 0  # Set to a default value when inx is empty
        else:
            fc_inx = range(inx[0], inx[-1] + 1)
            f_sel = f_sel[fc_inx]
            Mf_sel = Mf_sel[fc_inx]
            new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                                pow(np.diff(Mf_sel), 2)))
        
        return new_sal

         
    metrics = []
    for experiment_id in tqdm(dataframe['id'].unique(), desc='Processing Rows'):
        experiment_data = dataframe[dataframe['id'] == experiment_id]
        for participant in experiment_data['Participant'].unique():
            participant_data = experiment_data[experiment_data['Participant'] == participant]
            for trial in participant_data['Trial'].unique():
                trial_data = participant_data[participant_data['Trial'] == trial]
                velocity_data = trial_data['Velocity']
                angle_data = trial_data['Angle']
                target_angle_data = trial_data['TargetAngleH']
                sparc_values = sparc(velocity_data)
                rmse_values = rmse(angle_data, target_angle_data)
                metrics.append({
                    'id': experiment_id,
                    'Participant': participant,
                    'Trial': trial,
                    'SPARC': sparc_values,
                    'RMSE': rmse_values
                })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.sort_values(by=['id', 'Trial'], inplace=True)
    
    clean_dataframe = dataframe[['id', 'Participant', 'Trial', 'Age', 'Stiffness']]
    clean_dataframe.drop_duplicates(inplace=True)
    clean_dataframe.sort_values('Trial', inplace=True)
    clean_dataframe.reset_index(drop=True, inplace=True)
    
    clean_data = pd.merge(metrics_df, clean_dataframe, on=['id', 'Participant', 'Trial'], how='left')
    
    print("RMSE & SPARC Data Successfully Calculated.\n")
    return clean_data
