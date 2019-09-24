# LendingClubAnalysis

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:55:07 2019

@author: arun_meghna
"""

#Lenidng Club modeling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer, roc_auc_score,accuracy_score, roc_curve
from scipy.stats import ks_2samp
#from treeinterpreter import treeinterpreter as ti

from sklearn.preprocessing import Imputer, StandardScaler
#from sklearn import cross_validation
from sklearn import metrics
"""
from sklearn import metrics
from sklearn import linear_model

from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import preprocessing
"""

# ignore Deprecation Warning
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning,RuntimeWarning) 
warnings.filterwarnings("ignore") 

#plt.style.use('fivethirtyeight') # Good looking plots
pd.set_option('display.max_columns', None) # Display any number of columns
 

 

#read the csv file

df = pd.read_csv(r"/Volumes/Arun/Python Practice/LoanStats3d-Modified.csv",sep=',',low_memory=False) #reading the dataset in a dataframe using pandas

df.info()
df.head(10) # checking the first 10 rows of the complete data
df.describe() #gives the summary stats
df.sample(3)

 

#DATA CLEANING
##Finding the the count and percentage of values that are missing in the dataframe.

df_null = pd.DataFrame({'Count': df.isnull().sum(), 'Percent': 100*df.isnull().sum()/len(df)})

##printing columns with null count more than 0
df_null[df_null['Count'] > 0]

#Drop the null values where there is 80% of null
df1 = df.dropna(axis=1,thresh=int(0.80*len(df)))
df1.head(5)
df1.info()
df1.shape


#Remvoing variables that are not available before lending a loan.

df1.drop(['acc_now_delinq', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths',
             'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag',
             'delinq_2yrs', 'delinq_amnt',  'funded_amnt', 'funded_amnt_inv', 'hardship_flag', 'inq_last_6mths',
             'last_credit_pull_d',  'last_pymnt_amnt', 'last_pymnt_d', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
             'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
             'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts',
             'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 
             'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pymnt_plan', 'recoveries', 'tax_liens',
             'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
             'total_pymnt', 'total_pymnt_inv', 'total_rec_int',
             'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim'],axis=1, inplace = True)

df1.info()

#Exploratory Analysis of Each variable

def plot_var(col_name, full_name, continuous):
    """
    Visualize a variable with/without faceting on the loan status.
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True for continuous variables
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(15,3))
    # plot1: counts distribution of the variable   
    if continuous: 
        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(full_name)
    ax1.set_ylabel('Count')
    ax1.set_title(full_name)
    # plot2: bar plot of the variable grouped by loan_status
    if continuous:
        sns.boxplot(x=col_name, y='loan_status', data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(full_name + ' by Loan Status')
    else:
        Charged_Off_rates = df.groupby(col_name)['loan_status'].value_counts(normalize=True)[:,'Charged Off']
        sns.barplot(x=Charged_Off_rates.index, y=Charged_Off_rates.values, color='#5975A4', saturation=1, ax=ax2)
        ax2.set_ylabel('Fraction of Loans Charged Off')
        ax2.set_title('Charged Off Rate by ' + full_name)
        ax2.set_xlabel(full_name)
    # plot3: kde plot of the variable gropued by loan_status
    if continuous: 
        facet = sns.FacetGrid(df, hue = 'loan_status', size=3, aspect=4)
        facet.map(sns.kdeplot, col_name, shade=True)
        #facet.set(xlim=(df[col_name].min(), df[col_name].max()))
        facet.add_legend() 
    else:
        fig = plt.figure(figsize=(12,3))
        sns.countplot(x=col_name, hue='loan_status', data=df, order=sorted(df[col_name].unique()) )
plt.tight_layout() 



#######

df1['loan_status'].value_counts()
m =df1['loan_status'].value_counts()
m = m.to_frame()
m.reset_index(inplace=True)
m.columns = ['Loan Status','Count']
plt.subplots(figsize=(20,8))
sns.barplot(y='Count', x='Loan Status', data=m)
plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Distribution of Loan Status in our Dataset")
plt.show()

df1['loan_amnt'].describe()
plot_var('loan_amnt', 'Loan Amount', continuous=True)

#Term
df1['term'].sample(5)
df1['term'].value_counts(dropna=False)
df1['term'] = df['term'].apply(lambda s: np.int8(s.split()[0]))

plot_var('term', 'Term', continuous=False)
df1['term'].value_counts(normalize=True)

#Compare the charge-off rate by loan period

df1.groupby('term')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']

 #int_rate
 df1['int_round'] = df1['int_rate'].replace("%","", regex=True).astype(float)
 df1['int_round'].sample(10)
 plot_var('int_round', 'Interest Rate', continuous=True)
 
#Outliers
#outliers can be exactly what we want to learn about, e.g., anomaly detection.
#In this project, however, outliers may distort the picture of the data in both statistical analysis and visualization.
#Below, I use the modified Z-score method and the IQR method. Note that the variable must be continuous, not categorical, for any of these functions to make sense.
#Outliers
#outliers can be exactly what we want to learn about, e.g., anomaly detection.
#In this project, however, outliers may distort the picture of the data in both statistical analysis and visualization.
#Below, I use the modified Z-score method and the IQR method. Note that the variable must be continuous, not categorical, for any of these functions to make sense.



#####

def outliers_modified_z_score(dataframe, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices corresponding to the observations containing more than n outliers according to the modified z-score Method
    """
    threshold = 3.5
    outlier_indices = []
    for col in features:
        median_y = np.median(dataframe[col])
        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in dataframe[col]])
        modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in dataframe[col]]
        outlier_list_col = dataframe[np.abs(modified_z_scores) > threshold].index
       # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers

#####
    
def outliers_iqr(dataframe, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    for col in features:
        # 1st quartile (25%) & # 3rd quartile (75%)
        quartile_1, quartile_3 = np.percentile(dataframe[col], [25,75])
        #quartile_3 = np.percentile(dataframe[col], 75)
      
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        # Determine a list of indices of outliers for feature col
        outlier_list_col = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)].index
       # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers

df1.groupby('loan_status')['int_rate'].describe()


df1.loc[(df1.int_round > 15.61) & (df1.loan_status == 'Fully Paid')].shape[0]
(df1.loc[(df1.int_round > 15.61) & (df1.loan_status == 'Fully Paid')].shape[0])/df1['loan_status'].value_counts(normalize=False, dropna=False)[0]
df1.loc[(df1.int_round >18.55) & (df1.loan_status == 'Charged Off')].shape[0]/df1['loan_status'].value_counts(normalize=False, dropna=False)[1]

sorted(df1['grade'].unique())
print(sorted(df1['sub_grade'].unique()))

plot_var('sub_grade','Subgrade',continuous=False)
plot_var('grade','Grade',continuous=False)

df1['emp_length'].value_counts(dropna=False).sort_index()
df1['emp_length'].replace('10+ years', '10 years', inplace=True)
df1['emp_length'].replace('< 1 year', '0 years', inplace=True)
df1['emp_length'].value_counts(dropna=False).sort_index()
df1.emp_length.map( lambda x: str(x).split()[0]).value_counts(dropna=True).sort_index()
df1['emp_length'] = df1.emp_length.map( lambda x: float(str(x).split()[0]))
df1['emp_length'].sample(5)

plot_var('emp_length', 'Employment length', continuous=False)

df1['home_ownership'].replace(['NONE','ANY'],'OTHER', inplace=True)
df1['annual_inc'] = df1['annual_inc'].apply(lambda x:np.log10(x+1))

plot_var('annual_inc', 'Log10 Annual income', continuous=True)

drop_cols('title')
drop_cols('zip_code')

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
sns.distplot(df1.loc[df1['dti'].notnull() & (df1['dti'] < 60), 'dti'], kde=False, ax=ax1)
ax1.set_xlabel('dti')
ax1.set_ylabel('Count')
ax1.set_title('debt to income')
sns.boxplot(x=df1.loc[df1['dti'].notnull() & (df1['dti'] < 60), 'dti'], y='loan_status', data=df1, ax=ax2)
ax2.set_xlabel('DTI')
ax2.set_ylabel('Fraction of Loans fully paid')
ax2.set_title('Fully paid rate by debt to income')
ax2.set_title('DTI by loan status')

###
from datetime import datetime

df1.earliest_cr_line = pd.to_datetime(df1.earliest_cr_line, errors = 'coerce')
df1 = df1.dropna(subset=['earliest_cr_line'])
dttoday = datetime.now().strftime('%Y-%m-%d')
df1.earliest_cr_line = df1.earliest_cr_line.apply(lambda x:(np.timedelta64((x - pd.Timestamp(dttoday)),'D').astype(int))/-365)

df1.earliest_cr_line.shape

plot_var('earliest_cr_line', 'Length of of the earliest Credit Line (Months to today)', continuous=True)

df1.pub_rec = df1.pub_rec.map(lambda x: 3 if x >2.0 else x)
df1['revol_bal'] = df1['revol_bal'].apply(lambda x:np.log10(x+1))
drop_cols('policy_code')
df1.mort_acc = df.mort_acc.map(lambda x: 6.0 if x > 6.0 else x)

######

#Feature Selection and Statistical Overvie : Convert target variable to 0/1 indicator

# Next, I will convert the "loan_status" column to a 0/1 "charged off" column. Fully Paid:0 Charged Off: 1
df1['Charged_Off'] = df1['loan_status'].apply(lambda s: np.float(s == 'Charged Off'))

list_float = df1.select_dtypes(exclude=['object']).columns

def run_KS_test(feature):
    dist1 = df1.loc[df1.Charged_Off == 0,feature]
    dist2 = df1.loc[df1.Charged_Off == 1,feature]
    print(feature+':')
    print(ks_2samp(dist1,dist2),'\n')
    
from statsmodels.stats.proportion import proportions_ztest
def run_proportion_Z_test(feature):
    dist1 = df1.loc[df1.Charged_Off == 0, feature]
    dist2 = df1.loc[df1.Charged_Off == 1, feature]
    n1 = len(dist1)
    p1 = dist1.sum()
    n2 = len(dist2)
    p2 = dist2.sum()
    z_score, p_value = proportions_ztest([p1, p2], [n1, n2])
    print(feature+':')
    print('z-score = {}; p-value = {}'.format(z_score, p_value),'\n')


from scipy.stats import chi2_contingency
def run_chi2_test(df1, feature):

    dist1 = df1.loc[df1.loan_status == 'Fully Paid',feature].value_counts().sort_index().tolist()
    dist2 = df1.loc[df1.loan_status == 'Charged Off',feature].value_counts().sort_index().tolist()
    chi2, p, dof, expctd = chi2_contingency([dist1,dist2])
    print(feature+':')
    print("chi-square test statistic:", chi2)
    print("p-value", p, '\n')

for i in list_float:
    run_KS_test(i)
    
df1.info()

list_float = df1.select_dtypes(exclude=['object']).columns

list_float

fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches
cm_df = sns.heatmap(df1[list_float].corr(),annot=True, fmt = ".2f", cmap = "coolwarm", ax=ax)

#The linearly correlated features are:

#"installment" vs "loan_amnt" (0.94) "total_acc" vs "open_acc" (0.71) "pub_rec_bankruptcies" vs "pub_rec" (0.69) "mo_sin_old_rev_tl_op" vs "earliest_cr_line" (0.42)

#(*) with null values Dependence of Charged-off on the predictors: "int_rate" is the most correlated one.
cor = df1[list_float].corr()
cor.loc[:,:] = np.tril(cor, k=-1) # below main lower triangle of an array
cor = cor.stack()
cor[(cor > 0.1) | (cor < -0.1)]

df1[["installment","loan_amnt","mo_sin_old_rev_tl_op","earliest_cr_line","total_acc","open_acc", "pub_rec_bankruptcies", "pub_rec"]].isnull().any()


linear_corr = pd.DataFrame()

# Pearson coefficients
for col in df1[list_float].columns:
    linear_corr.loc[col, 'pearson_corr'] = df1[col].corr(df1['Charged_Off'])
linear_corr['abs_pearson_corr'] = abs(linear_corr['pearson_corr'])

linear_corr.reset_index(inplace=True)
#linear_corr.rename(columns={'index':'variable'}, inplace=True)

linear_corr

df1.shape

#Drop the null values where there is 80% of null
df1 = df1.dropna(axis=1,thresh=int(0.80*len(df1)))

df1.shape
dummy_list =['sub_grade','home_ownership','verification_status','purpose','addr_state','initial_list_status','application_type']

df1[dummy_list].isnull().any()

df1 = pd.get_dummies(df1, columns=dummy_list, drop_first=True)

def drop_cols(cols):
    df.drop(labels=cols, axis=1, inplace=True)

drop_cols('revol_util')
drop_cols('grade')

df1['int_round'].describe()

X_train = df1.drop(['Charged_Off'], axis=1)
y_train = df1.loc[:, 'Charged_Off']

X_test = df1.drop(['Charged_Off'], axis=1)
y_test = df1['Charged_Off']

X_all = df1.drop(['Charged_Off'], axis=1)
Y_all = df1.loc[:, 'Charged_Off']


#Logistic Regression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# CV model with Kfold stratified cross val
kfold = 3
random_state = 42

pipeline_sgdlr = Pipeline([
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=random_state, warm_start=False))
])

param_grid_sgdlr  = {
    'model__alpha': [10**-5, 10**-1, 10**2],
    'model__penalty': ['l1', 'l2']
}

grid_sgdlr = GridSearchCV(estimator=pipeline_sgdlr, param_grid=param_grid_sgdlr, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=kfold, verbose=1, return_train_score=False)

grid_sgdlr.fit(X_train, y_train)
sgdlr_estimator = grid_sgdlr.best_estimator_
print('Best score: ', grid_sgdlr.best_score_)
print('Best parameters set: \n', grid_sgdlr.best_params_)
y_pred_sgdlr = sgdlr_estimator.predict(X_test)
y_prob_sgdlr = sgdlr_estimator.predict_proba(X_test)[:,1]
y_train_pred_sgdlr = sgdlr_estimator.predict(X_train)
y_train_prob_sgdlr = sgdlr_estimator.predict_proba(X_train)[:,1]























































