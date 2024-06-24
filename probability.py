import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

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
        confmat = confusion_matrix(self.y, self.clf.predict(self.X))
        print('-'*60)
        print('Confusion Matrix (total:{}) \t Accuracy: \t  {}'.format(len(self.X), np.round(acc, 3)))
        print('  TP: {} | FN: {}'.format(confmat[1][1], confmat[1][0]))
        print('  FP: {} | TN: {}'.format(confmat[0][1], confmat[0][0]))

# Load data
qual = pd.read_csv('dataset.csv')

# Preprocessing
qual['good_bad_grade'] = np.where(qual['Curricular units 1st sem (approved)'] >= 5, 1, 0)

# Define predictors and target
X = qual[['Mother\'s qualification', 'Father\'s qualification']]
y = qual['good_bad_grade']

# Initialize and train logistic regression model
mylr = LogisticRegression()
mylr.fit(X, y)

# Create instance of ModelSummary
model_summary = ModelSummary(mylr, X, y)
model_summary.get_summary()

# Define ranges for mother's and father's qualifications
mother_qual_range = np.linspace(qual["Mother's qualification"].min(), qual["Mother's qualification"].max(), 100)
father_qual_range = np.linspace(qual["Father's qualification"].min(), qual["Father's qualification"].max(), 100)

# Calculate mean values to keep constant
mean_father_qual = qual["Father's qualification"].mean()
mean_mother_qual = qual["Mother's qualification"].mean()

# Predict probabilities for varying mother's qualification (father's qualification constant)
mother_qual_probs = mylr.predict_proba(np.c_[mother_qual_range, np.full(mother_qual_range.shape, mean_father_qual)])[:, 1]

# Predict probabilities for varying father's qualification (mother's qualification constant)
father_qual_probs = mylr.predict_proba(np.c_[np.full(father_qual_range.shape, mean_mother_qual), father_qual_range])[:, 1]

# Create line plots
plt.figure(figsize=(12, 8))
plt.plot(mother_qual_range, mother_qual_probs, label="Mother's Qualification", color='blue')
plt.plot(father_qual_range, father_qual_probs, label="Father's Qualification", color='green')

# Add labels and title
plt.xlabel('Qualification')
plt.ylabel('Probability of Passing (Credits ≥ 5)')
plt.title('Probability of Passing vs. Parents\' Qualifications')
plt.legend()
plt.grid(True)
plt.show()
