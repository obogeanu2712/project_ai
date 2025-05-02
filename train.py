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

data = pd.DataFrame(data)

# print(type(data))

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

perceptrons = [(12, ), (12, 6), (12, 24), (12, 12)]

# Initialize a dictionary to store results
results = {}

# Loop through the perceptrons and evaluate each configuration
for perceptron in perceptrons:
    # Define the MLP classifier with the current perceptron configuration
    mlp = MLPClassifier(hidden_layer_sizes=perceptron, max_iter=1000, random_state=42, learning_rate='constant', learning_rate_init=0.1, solver='sgd', activation='logistic')
    
    # Train the MLP classifier
    
    mlp.fit(X_train, y_train)
    
    # Make predictions
    y_pred = mlp.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Store the results in the dictionary
    results[perceptron] = {'accuracy': accuracy, 'mse': mse}

# Print the results
for perceptron, metrics in results.items():
    print(f"Perceptron {perceptron}: Accuracy = {metrics['accuracy']:.12f}, MSE = {metrics['mse']:.12f}")

