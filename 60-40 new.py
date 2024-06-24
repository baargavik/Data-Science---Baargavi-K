import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocessing: Define target variable and predictors
df['good_bad_grade'] = np.where(df['Curricular units 1st sem (approved)'] >= 5, 1, 0)
X = df[['Mother\'s qualification', 'Father\'s qualification']]
y = df['good_bad_grade']

# Define a range of model complexities (for demonstration purposes)
complexities = [1, 2, 3, 4, 5]

# Splitting the data into 60/40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Undersampling to balance the training data
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

# Initialize lists to store accuracies for training and validation sets
train_accuracies = []
val_accuracies = []

# Iterate over different model complexities
for complexity in complexities:
    # Initialize and train logistic regression model on undersampled training data
    mylr = LogisticRegression(max_iter=complexity, random_state=42)
    mylr.fit(X_train_under, y_train_under)
    
    # Training accuracy
    y_train_pred = mylr.predict(X_train_under)
    train_acc = accuracy_score(y_train_under, y_train_pred)
    train_accuracies.append(train_acc)
    
    # Validation accuracy
    y_test_pred = mylr.predict(X_test)
    val_acc = accuracy_score(y_test, y_test_pred)
    val_accuracies.append(val_acc)

# Plotting training and validation accuracy vs model complexity
plt.figure(figsize=(10, 6))
plt.plot(complexities, train_accuracies, marker='o', color='blue', label='Training Accuracy')
plt.plot(complexities, val_accuracies, marker='o', color='red', label='Validation Accuracy')
plt.title('Training and Validation Accuracy vs Model Complexity (60/40 Split)')
plt.xlabel('Model Complexity (Max Iterations)')
plt.ylabel('Accuracy')
plt.xticks(complexities)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

