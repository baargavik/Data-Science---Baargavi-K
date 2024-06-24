import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv')

# Columns of interest
columns_of_interest = ['Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)']

# Dictionary to store results
results = {}

# Loop through each column
for col in columns_of_interest:
    data = df[col].dropna()  # Drop NaN values if any
    
    # Compute mean
    mean_val = np.mean(data)
    
    # Compute median
    median_val = np.median(data)
    
    # Compute range
    range_val = np.ptp(data)
    
    # Compute quartiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # Store results in dictionary
    results[col] = {
        'Mean': mean_val,
        'Median': median_val,
        'Range': range_val,
        'Q1': Q1,
        'Q3': Q3
    }

# Print results
for col, metrics in results.items():
    print(f"Column: {col}")
    print(f"  Mean: {metrics['Mean']}")
    print(f"  Median: {metrics['Median']}")
    print(f"  Range: {metrics['Range']}")
    print(f"  Q1 (Lower Quartile): {metrics['Q1']}")
    print(f"  Q3 (Upper Quartile): {metrics['Q3']}")
    print()
