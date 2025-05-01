import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from encode import encode
from matrix_corr import correlation_matrix
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error


# Load the dataset
file_path = 'post-operative.data'

columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'ADM-DECS']

data = pd.read_csv(file_path, header=None, names=columns, na_values='?')

# Drop rows with missing values
data.dropna(inplace=True)

# Call the encode function to encode the data
data_encoder, target_encoder = encode(data, target_column='ADM-DECS')

# Write the current DataFrame to a CSV file
data.to_csv('processed_data.csv', index=False)

# Scale the features

# scaler = StandardScaler()
# data = scaler.fit_transform(data)

data = pd.DataFrame(data)

# print(type(data))

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the processed data for the MLP classifier
processed_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'label_encoders': data_encoder,
    'target_encoder': target_encoder,
    # 'scaler': scaler
}

print("Data has been processed and is ready for the MLP classifier.")

# Define the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=1000, random_state=42)

# Train the MLP classifier

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the MLP classifier: {accuracy:.2f}")

