import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
qual = pd.read_csv('dataset.csv')

# Preprocessing
qual['good_bad_grade'] = np.where(qual['Curricular units 1st sem (approved)'] >= 5, 1, 0)

# Define predictors and target
X_mother = qual[['Mother\'s qualification']]
X_father = qual[['Father\'s qualification']]
y = qual['Curricular units 1st sem (approved)']  # Assuming this is the column for first semester credits

# Initialize and train linear regression models
regressor_mother = LinearRegression()
regressor_father = LinearRegression()
regressor_mother.fit(X_mother, y)
regressor_father.fit(X_father, y)

# Predictions
y_pred_mother = regressor_mother.predict(X_mother)
y_pred_father = regressor_father.predict(X_father)

# Plotting
plt.figure(figsize=(10, 6))

# Plot for Mother's qualification
plt.subplot(2, 1, 1)
plt.scatter(X_mother, y, color='blue', label='Actual data')
plt.plot(X_mother, y_pred_mother, color='red', linewidth=2, label='Linear regression line')
plt.title('Mother\'s Qualification vs. 1st Semester Credits')
plt.xlabel('Mother\'s qualification')
plt.ylabel('Credits')
plt.legend()

# Plot for Father's qualification
plt.subplot(2, 1, 2)
plt.scatter(X_father, y, color='blue', label='Actual data')
plt.plot(X_father, y_pred_father, color='green', linewidth=2, label='Linear regression line')
plt.title('Father\'s Qualification vs. 1st Semester Credits')
plt.xlabel('Father\'s qualification')
plt.ylabel('Credits')
plt.legend()

plt.tight_layout()
plt.show()
