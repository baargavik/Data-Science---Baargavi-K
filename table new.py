import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocessing: Define target variable and predictors
df['good_bad_grade'] = np.where(df['Curricular units 1st sem (approved)'] >= 5, 1, 0)
X = df[['Mother\'s qualification', 'Father\'s qualification']]
y = df['good_bad_grade']

# Split data into 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform automatic forward selection (example pipeline)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SequentialFeatureSelector(LogisticRegression(), direction='forward')),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Compute metrics for training set
train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
train_precision = precision_score(y_train, pipeline.predict(X_train))
train_recall = recall_score(y_train, pipeline.predict(X_train))

# Compute metrics for validation set
val_accuracy = accuracy_score(y_test, y_pred_test)
val_precision = precision_score(y_test, y_pred_test)
val_recall = recall_score(y_test, y_pred_test)

# Print metrics
print("Metrics for Training Set:")
print(f"Accuracy: {train_accuracy:.3f}")
print(f"Precision: {train_precision:.3f}")
print(f"Recall: {train_recall:.3f}")

print("\nMetrics for Validation Set:")
print(f"Accuracy: {val_accuracy:.3f}")
print(f"Precision: {val_precision:.3f}")
print(f"Recall: {val_recall:.3f}")
