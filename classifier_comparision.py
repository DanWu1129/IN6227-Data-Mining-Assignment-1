import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the datasets
df_train = pd.read_csv("adult.data", sep=', ', header=None, engine='python')
df_test = pd.read_csv("adult.test", sep=', ', skiprows=1, header=None, engine='python')

# Replace missing values denoted by "?" and drop rows with missing values
df_train.replace("?", None, inplace=True)
df_test.replace("?", None, inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# Helper function for preprocessing (scaling numerical features, encoding categorical ones)
def preprocess(df):
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = StandardScaler().fit_transform(df[[column]])
        else:
            df[column] = LabelEncoder().fit_transform(df[column])
    return df

# Preprocess the training and test data
df_train = preprocess(df_train)
df_test = preprocess(df_test)

# Separate features and labels
X_train, y_train = df_train.drop(columns=[df_train.columns[-1]]), df_train[df_train.columns[-1]]
X_test, y_test = df_test.drop(columns=[df_test.columns[-1]]), df_test[df_test.columns[-1]]

# Function to evaluate a model (training time, prediction time, accuracy, F1 score)
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    start_time = time.time()
    predictions = model.predict(X_test)
    predict_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    return training_time, predict_time, accuracy, f1

# K-Nearest Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_train_time, knn_predict_time, knn_accuracy, knn_f1 = evaluate_model(knn_classifier, X_train, y_train, X_test, y_test)

# Decision Tree
tree_classifier = DecisionTreeClassifier()
tree_train_time, tree_predict_time, tree_accuracy, tree_f1 = evaluate_model(tree_classifier, X_train, y_train, X_test, y_test)

# Print results
print(f"KNN - Training Time: {knn_train_time:.4f}s, Prediction Time: {knn_predict_time:.4f}s, Accuracy: {knn_accuracy:.4f}, F1-Score: {knn_f1:.4f}")
print(f"Decision Tree - Training Time: {tree_train_time:.4f}s, Prediction Time: {tree_predict_time:.4f}s, Accuracy: {tree_accuracy:.4f}, F1-Score: {tree_f1:.4f}")

# Optionally check class balance
# print(y_train.value_counts())
