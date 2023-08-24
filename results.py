import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(svm_model, kmeans_model, dataset):
    """
    Generate and display scatter plots of ground truth, SVM predictions, and KMeans predictions.

    Parameters:
        svm_model (SVC): Trained Support Vector Machine classification model.
        kmeans_model (KMeans): Trained KMeans clustering model.
        dataset (pandas.DataFrame): The DataFrame containing the data for visualisation.
    """
    print('Plotting Modelling Results.')
    def calculate_accuracy(predictions, ground_truth):
        """
        Calculate the accuracy of predictions by comparing them to the ground truth.

        Parameters:
            predictions (numpy.ndarray): Predicted values.
            ground_truth (numpy.ndarray): Ground truth values.

        Returns:
            float: The accuracy of the predictions as a value between 0 and 1.
        """
        correct_predictions = sum(predictions == ground_truth)
        total_samples = len(ground_truth)
        accuracy = correct_predictions / total_samples
        return accuracy

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0].values

    # Original Data Plot
    for pairing_type, group in dataset.groupby("pairingtype"):
        axes[0].scatter(group["rmse_mean"], group["rmse_difference"], label=pairing_type)
    axes[0].set_xlabel("Mean RMSE")
    axes[0].set_ylabel("Variability")
    axes[0].set_title("Ground truth")
    axes[0].legend()

    # SVM Predictions Plot
    svmy_pred = svm_model.predict(X)
    svm_accuracy = calculate_accuracy(svmy_pred, y)
    svm_predictions = pd.DataFrame({'rmse_mean': X['rmse_mean'], 'rmse_difference': X['rmse_difference'], 'pairingtype': svmy_pred})
    for pairing_type, group in svm_predictions.groupby("pairingtype"):
        axes[1].scatter(group["rmse_mean"], group["rmse_difference"], label=pairing_type)
    axes[1].set_xlabel("Mean RMSE")
    axes[1].set_ylabel("Variability")
    axes[1].set_title(f"SVM Predictions (Accuracy: {svm_accuracy:.2f})")
    axes[1].legend()

    # KMeans Predictions Plot
    ky_pred = kmeans_model.fit_predict(X)
    kmeans_accuracy = calculate_accuracy(ky_pred, y)
    kmeans_predictions = pd.DataFrame({'rmse_mean': X['rmse_mean'], 'rmse_difference': X['rmse_difference'], 'pairingtype': ky_pred})
    for pairing_type, group in kmeans_predictions.groupby("pairingtype"):
        axes[2].scatter(group["rmse_mean"], group["rmse_difference"], label=pairing_type)
    axes[2].set_xlabel("Mean RMSE")
    axes[2].set_ylabel("Variability")
    axes[2].set_title(f"KMeans Predictions (Accuracy: {kmeans_accuracy:.2f})")
    axes[2].legend()

    # Adjust layout
    plt.suptitle("Modelling Results - Mean RMSE vs RMSE Variability", fontsize=16)
    plt.tight_layout(rect=[0, 0.07, 1, 1])  # Adjust the bottom margin for classification report
    plt.savefig('outputs/results/modelling_results.png')
    plt.close()

    print("Results Have Been Successfully Saved.\n")
