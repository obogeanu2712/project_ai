import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def encode(data, target_column):
    """
    Encodes categorical features and target column in the dataset inplace.

    Parameters:
        data (pd.DataFrame): The dataset to encode.
        target_column (str): The name of the target column.

    Returns:
        dict: A dictionary containing label encoders for features and the target encoder.
    """
    label_encoders = {}
    for col in data.columns:
        if col != target_column:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    target_encoder = LabelEncoder()
    data[target_column] = target_encoder.fit_transform(data[target_column])

    return {
        'label_encoders': label_encoders,
        'target_encoder': target_encoder
    }

