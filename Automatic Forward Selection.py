import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score  # Import confusion_matrix here
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocessing: Define target variable and predictors
df['good_bad_grade'] = np.where(df['Curricular units 1st sem (approved)'] >= 5, 1, 0)
X = df[['Mother\'s qualification', 'Father\'s qualification']]
y = df['good_bad_grade']

# Function for automatic forward selection
def forward_selection(X, y, clf, threshold=0.05):
    selected_features = []
    features = list(X.columns)
    while len(features) > 0:
        best_pval = np.inf
        for feature in features:
            X_temp = X[selected_features + [feature]]
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.3, random_state=42)
            clf.fit(X_train, y_train)
            pvals = clf.coef_[0]
            if pvals[-1] < best_pval:
                best_pval = pvals[-1]
                best_feature = feature
        if best_pval < threshold:
            selected_features.append(best_feature)
            features.remove(best_feature)
        else:
            break
    return selected_features

# Perform forward selection
selected_features = forward_selection(X, y, LogisticRegression())

# Print selected features
print("Selected features:", selected_features)

# Undersampling to balance the dataset
undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(X[selected_features], y)

# Initialize and train logistic regression model on undersampled data
mylr = LogisticRegression()
mylr.fit(X_under, y_under)

# Class for model summary
class ModelSummary:
    """ This class extracts a summary of the model
    
    Methods
    -------
    get_se()
        computes standard error
    get_ci(SE_est)
        computes confidence intervals
    get_pvals()
        computes p-values
    get_summary(name=None)
        prints the summary of the model
    """
    
    def __init__(self, clf, X, y):
        """
        Parameters
        ----------
        clf: class
            the classifier object model
        X: numpy array
            matrix of predictors
        y: numpy array
            matrix of variable
        """
        self.clf = clf
        self.X = X
        self.y = y
    
    def get_se(self):
        predProbs = self.clf.predict_proba(self.X)
        X_design = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
        V = np.diagflat(np.product(predProbs, axis=1))
        covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
        return np.sqrt(np.diag(covLogit))

    def get_ci(self, SE_est):
        p = 0.975
        df = len(self.X) - 2
        crit_t_value = stats.t.ppf(p, df)
        coefs = np.concatenate([self.clf.intercept_, self.clf.coef_[0]])
        upper = coefs + (crit_t_value * SE_est)
        lower = coefs - (crit_t_value * SE_est)
        cis = np.zeros((len(coefs), 2))
        cis[:,0] = lower
        cis[:,1] = upper
        return cis
    
    def get_pvals(self):
        p = self.clf.predict_proba(self.X)
        n = len(p)
        m = len(self.clf.coef_[0]) + 1
        coefs = np.concatenate([self.clf.intercept_, self.clf.coef_[0]])
        se = self.get_se()
        t =  coefs/se  
        p = (1 - stats.norm.cdf(abs(t))) * 2
        return p
    
    def get_summary(self, names=None):
        ses = self.get_se()
        cis = self.get_ci(ses)
        lower = cis[:, 0]
        upper = cis[:, 1]
        pvals = self.get_pvals()
        coefs = np.concatenate([self.clf.intercept_, self.clf.coef_[0]])
        data = []
        for i in range(len(coefs)):
            currlist = []
            currlist.append(np.round(coefs[i], 3))
            currlist.append(np.round(ses[i], 3))
            currlist.append(np.round(pvals[i], 3))
            currlist.append(np.round(lower[i], 3))
            currlist.append(np.round(upper[i], 3))
            data.append(currlist)
        cols = ['coefficient', 'std', 'p-value', '[0.025', '0.975]']
        sumdf = pd.DataFrame(columns=cols, data=data)
        if names is not None:
            new_names = ['intercept'] + [i for i in names]
            sumdf.index = new_names
        else:
            try:
                names = list(self.X.columns)
                new_names = ['intercept'] + [i for i in names]
                sumdf.index = new_names
            except:
                pass
        print(sumdf)
        acc = accuracy_score(self.y, self.clf.predict(self.X))
        confmat = confusion_matrix(self.y, self.clf.predict(self.X))  # Calculate confusion matrix
        print('-'*60)
        print('Confusion Matrix (total:{}) \t Accuracy: \t  {}'.format(len(self.X), np.round(acc, 3)))
        print('  TP: {} | FN: {}'.format(confmat[1][1], confmat[1][0]))
        print('  FP: {} | TN: {}'.format(confmat[0][1], confmat[0][0]))

# Create instance of ModelSummary for detailed model interpretation
model_summary = ModelSummary(mylr, X_under, y_under)
model_summary.get_summary(names=selected_features)

# Predictions on the full dataset for plotting purposes
y_pred = mylr.predict(X[selected_features])

# Plotting the selected features for visualization
plt.figure(figsize=(8, 6))

# Plot Mother's qualification
plt.scatter(df['Mother\'s qualification'], y_pred, color='blue', label="Mother's qualification")

# Plot Father's qualification
plt.scatter(df['Father\'s qualification'], y_pred, color='red', label="Father's qualification")

plt.xlabel('Qualification')
plt.ylabel('Predicted Probability of Passing')
plt.title('Logistic Regression: Predicted Probability vs Qualification')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
