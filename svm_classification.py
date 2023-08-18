from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

def perform_svm_classification(dataframe):
    """
    Perform SVM (Support Vector Machine) classification on the given DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the data to be classified.

    Returns:
        tuple: A tuple containing the trained SVM model and the predicted labels for the test data.
    """
    # Split the data into train and test sets (3:1 split)
    X = dataframe.iloc[:, 1:] 
    y = dataframe.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Initialize the SVM model
    svm = SVC(kernel='linear', random_state=42)
    
    # Fit the SVM model to the training data
    svm.fit(X_train, y_train)
    
    # Predict labels for the test data
    y_pred = svm.predict(X_test)
    
    print("SVM Model Sucessfully Built.\n")
    # Return the trained model and predicted labels
    return svm, y_pred