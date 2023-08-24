from sklearn.cluster import KMeans
import pandas as pd

def perform_kmeans_clustering(dataframe, n_clusters=3):
    """
    Perform KMeans clustering on the given DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the data to be clustered.
        n_clusters (int): The number of clusters to create. Default is 3.

    Returns:
        tuple: A tuple containing the trained KMeans model and the cluster labels for the data.
    """
    # Split the data into train and test sets (3:1 split)
    X = dataframe.iloc[:, 1:] 
    y = dataframe.iloc[:, 0].values
    
    # Initialise the KMeans model
    kmeans = KMeans(n_clusters,
                    init='k-means++',
                    max_iter=300,
                    n_init=10,
                    random_state=42)
    
    # Fit & predict the KMeans model 
    y_pred = kmeans.fit_predict(X)
    
    # Return the trained model and test cluster labels
    print("K-means Model Successfully Built.\n")
    return kmeans, y_pred
