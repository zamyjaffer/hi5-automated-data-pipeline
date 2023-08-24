import os
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm


def ingest_csvs(folder_path):
    """
    Ingest CSV files from subdirectories, process them, and return a list of dataframes.

    Parameters:
        folder_path (str): Path to the main folder containing subdirectories with CSV files.

    Returns:
        list of pandas.DataFrame: A list of dataframes, each containing processed data from a CSV file.
    """
    print('Beginning Data Ingestion.')
    # Get a list of all subdirectories in the given folder path
    subdirectories = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # List to hold all the dataframes
    all_dataframes = []

    for folder in tqdm(subdirectories, desc= 'Ingesting Data'):
        folder_path_with_id = os.path.join(folder_path, folder)
        
        # Get a list of all files ending with 'Datalog.csv' in the folder
        csv_files = [file for file in os.listdir(folder_path_with_id) if file.endswith('Datalog.csv')]
        
        # Check if there is exactly one 'Datalog.csv' file in the folder
        if len(csv_files) == 1:
            csv_file_path = os.path.join(folder_path_with_id, csv_files[0])

            # Read the CSV file using pandas
            data = pd.read_csv(csv_file_path)

            # Get the id number from the folder name
            id_number = int(folder[2:])  # Assuming the folder name starts with 'id' followed by the number

            # Add the 'id' column to the front of the dataframe with the corresponding id number
            data.insert(0, 'id', id_number)

            # Rename the 'Datalog.csv' file to match the folder name
            new_csv_file_name = folder + '.csv'
            new_csv_file_path = os.path.join(folder_path_with_id, new_csv_file_name)

            # Save the updated data to the new file with the updated name
            data.to_csv(new_csv_file_path, index=False)

            # Append the dataframe to the list
            all_dataframes.append(data)

            #print(f"Processed '{csv_file_path}' and added to the list of dataframes.")
        #else:
            #print(f"Expected exactly one 'Datalog.csv' file in folder '{folder}', but found {len(csv_files)} files.")
    print("Data Ingestion Complete.\n")
    return all_dataframes

def format_data(dataframes):
    """
    Concatenate and reformat a list of DataFrames into a long-format DataFrame.

    Parameters:
        dataframes (list): A list of pandas DataFrames containing wide-format data.

    Returns:
        pandas.DataFrame: The concatenated and reformatted long-format DataFrame.
    """
    print('Beginning Data Conversion from Wide to Long.')
    # Check the input is a list
    if not isinstance(dataframes, list):
        raise ValueError("Input must be a list of DataFrames.")

    if len(dataframes) == 0:
        raise ValueError("Input list is empty.")
    
    # Concatenate all dataframes
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Separate DataFrames for G and Y data
    df_g = concatenated_df[['id', 'TimeG', 'Trial', 'TargetAngleH', 'TargetAngleR', 'AngleG', 'VelocityG',
            'Torque(motor)G', 'Torque(sensor)G', 'RecordedTorqueG', 'HumanErrorG',
            'FCRTnormG', 'ECRLTnormG', 'FCRCCnormG', 'ECRLCCnormG', 'FCRrectG',
            'ECRrectG', 'AccelerationG', 'ModeG', 'Stiffness', 'VisualMode',
            'MirroredParallel', 'SoloHand']].copy()
    df_y = concatenated_df[['id', 'TimeY', 'Trial', 'TargetAngleH', 'TargetAngleR', 'AngleY', 'VelocityY',
            'Torque(motor)Y', 'Torque(sensor)Y', 'RecordedTorqueY', 'HumanErrorY',
            'FCRTnormY', 'ECRLTnormY', 'FCRCCnormY', 'ECRLCCnormY', 'FCRrectY',
            'ECRrectY', 'AccelerationY', 'ModeY', 'Stiffness', 'VisualMode',
            'MirroredParallel', 'SoloHand']].copy()
    
    # Rename the columns for G and Y data
    df_g.columns = ['id', 'Time', 'Trial', 'TargetAngleH', 'TargetAngleR', 'Angle',
                    'Velocity', 'TorqueMotor', 'TorqueSensor', 'RecordedTorque', 'HumanError',
                    'FCRTnorm', 'ECRLTnorm', 'FCRCCnorm', 'ECRLCCnorm', 'FCRrect',
                    'ECRrect', 'Acceleration', 'Mode', 'Stiffness', 'VisualMode',
                    'MirroredParallel', 'SoloHand']

    df_y.columns = ['id', 'Time', 'Trial', 'TargetAngleH', 'TargetAngleR', 'Angle',
                    'Velocity', 'TorqueMotor', 'TorqueSensor', 'RecordedTorque', 'HumanError',
                    'FCRTnorm', 'ECRLTnorm', 'FCRCCnorm', 'ECRLCCnorm', 'FCRrect',
                    'ECRrect', 'Acceleration', 'Mode', 'Stiffness', 'VisualMode',
                    'MirroredParallel', 'SoloHand']
    
    # Add column to identify participant
    df_g.insert(1, 'Participant', 'G')
    df_y.insert(1, 'Participant', 'Y')

    # Concatenate the G and Y DataFrames
    df_long = pd.concat([df_g, df_y], ignore_index=True)

    # Sort the DataFrame by Participant and Time
    df_long = df_long.sort_values(by=['id', 'Participant', 'Trial']).reset_index(drop=True)

    # Save the new long-format CSV
    with tqdm(total=1, desc="Saving New Data Format") as pbar:
        df_long.to_csv('data/long_format.csv', index=False)
        pbar.update(1)

    print("Data Formatting Complete.\n")
    return df_long

def send_to_sql(dataframe, table_name, db_config, chunk_size=10000):
    """
    Save a DataFrame as a new table in a PostgreSQL database using psycopg2 and sqlalchemy in batches.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to be saved as a new table.
        table_name (str): The name of the table to be created in the database.
        db_config (dict): A dictionary containing the database connection details:
                          - 'host': PostgreSQL host address
                          - 'database': Name of the database
                          - 'user': Username
                          - 'password': Password (optional if not required)
        chunk_size (int, optional): Number of rows to send in each batch. Default is 10000.

    Returns:
        None
    """
    print('Sending Data to SQL.')
    try:
        # Connect to the database
        connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
        engine = create_engine(connection_string)

        # Calculate the number of chunks required
        total_rows = len(dataframe)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size

        # Save the DataFrame in batches
        for i in tqdm(range(num_chunks), 'Sending Data to SQL'):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)

            # Extract the chunk of rows from the DataFrame
            chunk_df = dataframe.iloc[start_idx:end_idx]
            # Save the chunk to the new table
            chunk_df.to_sql(table_name, engine, index=False, if_exists='replace')
            #print(start_idx, end_idx)

        print(f"DataFrame saved as table '{table_name}' in PostgreSQL.\n")

    except Exception as e:
        print(f"An error occurred: {e}")


def execute_sql_query(query, db_config):
    """
    Execute an SQL query using SQLAlchemy.

    Parameters:
        query (str): The SQL query to be executed.
        db_config (dict): A dictionary containing the database connection details:
                          - 'host': PostgreSQL host address
                          - 'database': Name of the database
                          - 'user': Username
                          - 'password': Password (optional if not required)

    Returns:
        result (list or None): The result of the query as a list of rows, or None if the query is not a SELECT statement.
    """
    print(f"Executing query: '{query}'")
    try:
        # Create a SQLAlchemy engine to connect to the database
        connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
        engine = create_engine(connection_string)

        # Execute the SQL query
        with engine.connect() as connection:
            result = connection.execute(query)
            print("Query Successfully Executed.\n")
            # If the query is a SELECT statement, fetch all rows and return them as a list
            if query.lstrip().upper().startswith('SELECT'):
                return result.fetchall()
            else:
                return None
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_from_sql(table_name, columns_list, db_config):
    """
    Fetch data from a PostgreSQL table and load it into a Pandas DataFrame.

    Parameters:
        table_name (str): The name of the table to retrieve data from.
        columns_list (list): A list of column names to retrieve from the table.
        db_config (dict): A dictionary containing the database connection details:
                          - 'host': PostgreSQL host address
                          - 'database': Name of the database
                          - 'user': Username
                          - 'password': Password (optional if not required)

    Returns:
        pandas.DataFrame: A DataFrame containing the retrieved data from the specified table
                         and columns. If the 'columns_list' contains ['id', 'Participant', 'Trial'],
                         the DataFrame is sorted by these columns in ascending order.
    """
    print('Fetching Data from SQL.')
    # Create a SQLAlchemy engine to connect to the database
    connection_string = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
    engine = create_engine(connection_string)

    # Load the table into a Pandas DataFrame
    columns_str = ", ".join(columns_list)
    query = f"""
    SELECT {columns_str}
    FROM {table_name}
    """
    dataframe = pd.read_sql(query, engine)

    if all(col in columns_list for col in ['id', 'Participant', 'Trial']):
        dataframe.sort_values(['id', 'Participant', 'Trial'], inplace=True)

    print(f"Table {table_name} Successfully Accessed.\n")
    return dataframe
