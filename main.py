from data_ingestion import ingest_csvs, send_to_sql, execute_sql_query, get_from_sql, format_data
from data_processing import create_connected_metric, create_pairing_metric, calculate_measure_metrics
from exploratory_data_analysis import get_visualisation_data, generate_and_save_plots
from modelling_metrics import generate_modelling_metrics, normalise_data
from kmeans_clustering import perform_kmeans_clustering
from svm_classification import perform_svm_classification
from results import plot_results
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

def main():
    # Data Ingestion
    folder_path = 'code\data'
    all_data = ingest_csvs(folder_path)

    # Data Processing
    long_data = format_data(all_data)

    # Sending Data to SQL
    database_config = {
        'host': '', # Add host name 
        'database': '', # Add database name 
        'user': '', # Add username 
        'password': '' # Add password 
    }
    table_name = 'trialData'
    send_to_sql(long_data, table_name, database_config)

    # Load demographic data from Excel file
    table_name = 'demographicData'
    demographic_data = pd.read_excel('extractedData\Demographics.xlsx')
    # Send demographic data to SQL database
    send_to_sql(demographic_data, table_name, database_config)

    # Drop 'Stiffness' column from demographic data
    query1 = """
    ALTER TABLE "demographicData"
    DROP COLUMN "Stiffness"
    """
    execute_sql_query(query1, database_config)

    # Fix age data in demographic data
    query2 = """
    UPDATE "demographicData"
    SET "Age" = CASE WHEN LENGTH("Age") > 2 THEN NULL ELSE "Age" END;
    """
    execute_sql_query(query2, database_config)

    # Change data type of 'Age' column to INTEGER in demographic data
    query3 = """
    ALTER TABLE "demographicData"
    ALTER COLUMN "Age" TYPE INT USING "Age"::integer;
    """
    execute_sql_query(query3, database_config)

    # Join tables 'trialData' and 'demographicData' into a new table 'allData'
    query4 = """
    CREATE TABLE "allData" AS
    SELECT "trialData".*, "demographicData".*
    FROM "trialData"
    INNER JOIN "demographicData" 
    ON "trialData"."id" = "demographicData"."ID"
    AND "trialData"."Participant" = "demographicData"."Side"
    """
    execute_sql_query(query4, database_config)

    table = '"allData"'
    columns = ['"id"', '"Participant"', '"Trial"', '"Stiffness"', '"TargetAngleH"', '"Angle"', '"Velocity"', '"Age"']
    allData = get_from_sql(table, columns, database_config)
    
    # Calculate measure metrics on the combined data
    data = calculate_measure_metrics(allData)
    data.head()
    # Create connected metric
    create_connected_metric(data)
    # Create pairing metric
    create_pairing_metric(data)

    # Exploratory Data Analysis
    eda_data = get_visualisation_data(data)
    generate_and_save_plots(eda_data)

    # Modelling Metrics
    metrics_to_model = ['SPARC', 'RMSE'] 
    modelling_data = generate_modelling_metrics(eda_data, metrics_to_model)

    # Data Normalization
    normalised_data = normalise_data(modelling_data)

    # K-Means Clustering
    kmeans_model, kmeans_ypred = perform_kmeans_clustering(normalised_data, 2)

    # SVM Classification
    svm_model, svm_ypred = perform_svm_classification(normalised_data)

    # Plotting Results
    plot_results(svm_model, kmeans_model, normalised_data)

if __name__ == "__main__":
    main()