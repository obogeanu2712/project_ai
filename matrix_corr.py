import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(data, columns, output_file='correlation_matrix.png', figsize=(10, 8), cmap='coolwarm', dpi=300):
    """
    Plots, saves, and returns a correlation matrix.

    Parameters:
        data (pd.DataFrame): The scaled data to compute the correlation matrix.
        columns (list): The column names for the DataFrame.
        output_file (str): The name of the output PNG file.
        figsize (tuple): The size of the figure.
        cmap (str): The colormap for the heatmap.
        dpi (int): The resolution of the saved PNG file.

    Returns:
        pd.DataFrame: The computed correlation matrix.
    """
    # Create a correlation matrix
    correlation_matrix = pd.DataFrame(data, columns=columns).corr()

    # Plot and save the correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.savefig(output_file, dpi=dpi)
    plt.close()

    # Return the correlation matrix
    return correlation_matrix