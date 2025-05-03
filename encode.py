import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def encode_columns(data, categorical_columns, target_column):
    """
    Encodes specified categorical and numerical features and the target column in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset to encode.
        categorical_columns (list): List of categorical column names to encode.
        numerical_columns (list): List of numerical column names to scale.
        target_column (str): The name of the target column.

    Returns:
        dict: A dictionary containing label encoders for categorical features, 
                the target encoder, and the scaler for numerical features.
    """
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    data[target_column] = target_encoder.fit_transform(data[target_column])

    return {
        'label_encoders': label_encoders,
        'target_encoder': target_encoder
    }

