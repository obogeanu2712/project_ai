from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# fetch dataset

post_operative_patient = fetch_ucirepo(id=82)

# data (as pandas dataframes)
X_raw = post_operative_patient.data.features # raw - not encoded

y_raw = post_operative_patient.data.targets

# counts_y = y_raw['']

# print(y_raw.columns)

for column in y_raw :
    print(column)
    print(y_raw[column].value_counts())

# PREPROCESSING : ENCODING DATA

# for the features X :

ordinal_cols = X_raw.columns[:7]
numeric_col = X_raw.columns[7]

categories = [
    ['low', 'mid', 'high'],                # L-CORE
    ['low', 'mid', 'high'],                # L-SURF
    ['poor', 'fair', 'good', 'excellent'], # L-O2
    ['low', 'mid', 'high'],                # L-BP
    ['unstable', 'mod-stable', 'stable'],  # SURF-STBL
    ['unstable', 'mod-stable', 'stable'],  # CORE-STBL
    ['unstable', 'mod-stable', 'stable']   # BP-STBL
]



encoder = OrdinalEncoder(categories=categories)

X_encoded_ordinal = encoder.fit_transform(X_raw[ordinal_cols])

X_encoded_ordinal_df = pd.DataFrame(X_encoded_ordinal, columns=ordinal_cols, index=X_raw.index)

X = pd.concat([X_encoded_ordinal_df, X_raw[[numeric_col]]], axis=1)

X['COMFORT'] = X['COMFORT'].fillna(-1)

# and for the labels :

# categories1 = [
#     ['I', 'S', 'A ', 'A'] # seems that 'A' and 'A ' have a problem
# ]

# y_raw['ADM-DECS'].replace('A ', 'A', inplace = True)

y_raw.loc[:, 'ADM-DECS'] = y_raw['ADM-DECS'].str.strip()


# for column in y_raw :
#     print(column)
#     print(y_raw[column].value_counts())

categories1 = [
    ['I', 'S', 'A'] # solved the whitespace problem
]

encoder1 = OrdinalEncoder(categories=categories1)

y = encoder1.fit_transform(y_raw)

# SPLIT DATA BETWEEN TRAIN AND TEST (ALSO APPLIES SHUFFLE - random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = MLPClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print(accuracy)