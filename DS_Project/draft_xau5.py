'''
@author: Jimmy Liang

## Would be good idea to have an 'element' (oil or metal) to reflect against 
## all currencies for prediction and linear/logistic regression?

# Possible Model to do: Support Vector Machines, Decision Trees, Random forests

# Figuring out how correlation and dependence are relevant between XAU and 
# currency purchase power!

# Find events which result dip or gain. *** 
# Apply sentiments to correlate with plots.

# Create feature with 15 days linear regression with row index

Four Ways to Forecast Currency Changes
1) Purchasing Power Parity (PPP) 
2) Relative Economic Strength Approach
3) Econometric Models
        USD/CAD(1-year) = z + a(INT) + b(GDP) + c(IGR)
            z, a, b, c: coefficient on how much a certain factor affects the 
                        exchange rate and direction of the effect (postive or negative) 
            INT: interest rate differential between US and Canada
            GDP: GDP growth rates
            IGR: Income growth rate
4) Time Series Model


# Try to make new mean data set so you have more control with plots...

!! Do this ARMA: Autoregressive moving  average 

Confidence intervals
p-values
rsquared


'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf  # import formula api as alias smf 
from sklearn import tree    # Decision Tree
from sklearn import metrics # Decision Tree
from sklearn.grid_search import GridSearchCV # Decision Tree
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

XAU = pd.read_csv('~jimmy/desktop/datascience/sf_dat_17_work/ds_project/XAUCurrency.csv')

# explore first 11 data
XAU.head(11)

# delete first 4 rows and set up clean data
XAU_data = XAU.drop([0, 1, 2, 3, 4])
#clean_data = data_currency.ix[4:]   # does the same as above syntax...

# rename columns
XAU_data = XAU_data.rename(columns={'Results':'Date'})

XAU_data.head()   # Inspect dataframe
XAU_data.dtypes   #Inspect data types

# convert Date from object to datetime64
XAU_data['Date'] = pd.to_datetime(XAU_data['Date'])

XAU_data.dtypes   #Inspect data types

# Convert columns data to float value
for column in XAU_data.columns[1:]:
    XAU_data[column] = XAU_data[column].astype(float)

XAU_data.dtypes     # check to confirm XAU_data types --> datetime64 and float64
XAU_data.head()     # Inspect dataframe
XAU_data.shape      # (180, 6)

XAU_data['EURUSD'] = XAU_data['XAUEUR']/XAU_data['XAUUSD']
XAU_data['GBPUSD'] = XAU_data['XAUGBP']/XAU_data['XAUUSD']
XAU_data['CADUSD'] = XAU_data['XAUCAD']/XAU_data['XAUUSD']
XAU_data['CHFUSD'] = XAU_data['XAUCHF']/XAU_data['XAUUSD']

XAU_data.head()     # Inspect dataframe with new columns
XAU_data.shape      # (180, 10)
XAU_data.describe() # show data details
# show same data details as above...
XAU_data.mean()     # calculate average value
XAU_data.min()      # look up for minimum value
XAU_data.max()      # look up for maximum value
np.std(XAU_data)    # calculate sample standard deviation of data

# plots
XAU_data.plot()

XAU_data.head()     # Inspect dataframe to verify


XEUR2 = pd.stats.moments.rolling_mean(XAU_data['EURUSD'], 2)
XEUR3 = pd.stats.moments.rolling_mean(XAU_data['EURUSD'], 3)
XEUR4 = pd.stats.moments.rolling_mean(XAU_data['EURUSD'], 4)
XEUR5 = pd.stats.moments.rolling_mean(XAU_data['EURUSD'], 5)
XEUR6 = pd.stats.moments.rolling_mean(XAU_data['EURUSD'], 6)

# shift data to line up with original data for linear/logistic regression purpose?
# Possible to shift Simple Moving Average to the left and see if data can be predict?
XEUR2 = XEUR2.shift(-1)
XEUR3 = XEUR3.shift(-2)
XEUR4 = XEUR4.shift(-3)
XEUR5 = XEUR5.shift(-4)
XEUR6 = XEUR6.shift(-5)

# put XEUR into dataframe
XAU_data['XEUR2'] = XEUR2
XAU_data['XEUR3'] = XEUR3
XAU_data['XEUR4'] = XEUR4
XAU_data['XEUR5'] = XEUR5
XAU_data['XEUR6'] = XEUR6

# check for any null values
XAU_data.isnull().sum()     # there are nulls within XEUR* columns

# replace null to mean value <- viable value?
XAU_data = XAU_data.fillna(XAU_data.mean().fillna(0))
XAU_data.isnull().sum()     # check to confirm there is no nulls

#XAU_data.to_csv("DetailXAU.csv", sep='\t', encoding='utf-8')

# Data Set with date for Linear Regression...
X_data = XAU_data

# set Resuls into the right DataFrame
XAU_data = XAU_data.set_index("Date")

X_data.head()       # Inspect data set with index
XAU_data.head()     # Inspect data set with Date index


fig = plt.figure()
XAU_data['EURUSD'].plot()
fig.suptitle('EUR / USD Currency', fontsize=15)
fig.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('EUR / USD')
#fig.savefig('EUR/USD.jpg')

# Mixed U.S. economic data and Greek debt crisis 
# U.S. GDP for the 1st Quarter was revised downward to -0.7%
XAU_data['2015-05-15':'2015-05-28']['EURUSD'].plot(color='black')
XAU_data['2015-05-15':'2015-05-28']['XEUR2'].plot(color='red')
XAU_data['2015-05-15':'2015-05-28']['XEUR3'].plot(color='green')
XAU_data['2015-05-15':'2015-05-28']['XEUR4'].plot(color='purple')
XAU_data['2015-05-15':'2015-05-28']['XEUR5'].plot(color='fuchsia')
XAU_data['2015-05-15':'2015-05-28']['XEUR6'].plot(color='skyblue')

#pd.stats.moments.rolling_mean(XAU_data['2015-05-13':'2015-05-28']['EUR/USD'], 3).plot()
#pd.stats.moments.rolling_mean(XAU_data['2015-05-11':'2015-05-28']['EUR/USD'], 5).plot()


# European Central Bank chief Mario Draghi's EURO stimulus and 
# Greece debt crisis: 3rd bailout repayment by August 20, 2015.
XAU_data['2015-08-04':'2015-08-26']['EURUSD'].plot(color='black')
XAU_data['2015-08-04':'2015-08-26']['XEUR2'].plot(color='red')
XAU_data['2015-08-04':'2015-08-26']['XEUR3'].plot(color='green')
XAU_data['2015-08-04':'2015-08-26']['XEUR4'].plot(color='purple')
XAU_data['2015-08-04':'2015-08-26']['XEUR5'].plot(color='fuchsia')
XAU_data['2015-08-04':'2015-08-26']['XEUR6'].plot(color='skyblue')

#pd.stats.moments.rolling_mean(XAU_data['2015-08-02':'2015-08-26']['EUR/USD'], 3).plot()
#pd.stats.moments.rolling_mean(XAU_data['2015-07-31':'2015-08-26']['EUR/USD'], 5).plot()


# check exchange rate USD to EUR.
#USD_EUR = 1/XAU_data['2015-08-04':'2015-08-26']['EUR/USD']
#USD_EUR.plot()

# Linear Regression between XAU_data['EUR/USD'] and XEURs
linreg = LinearRegression()
feature_cols = ['XEUR2', 'XEUR3', 'XEUR4', 'XEUR5', 'XEUR6']
feature_cols = ['XEUR2']   # better predictor without other variables
x = XAU_data[feature_cols]
y = XAU_data['EURUSD']

plt.plot(x)     # see plots
plt.plot(y)

scores = cross_val_score(linreg, x, y, cv=20, scoring='mean_squared_error')
scores.mean()   # -3.36767e-06
np.sqrt(-1*scores.mean()) # == rmse == root mean squared error: 0.001835
# it means that your error is quite small


#print linreg.intercept_
#print linreg.coef_

### Linear Regression Date vs Currency
feature_cols = ['EURUSD']
xd = XAU_data[feature_cols]  
yd = XAU_data.index.day
linreg.fit(xd, yd)
print linreg.intercept_     # -1.42786031805
print linreg.coef_          # 18.94113462

xd_pred = linreg.predict(x)
print linreg.predict(1)
print linreg.predict(2)
print linreg.predict(3)


feature_cols = ['EURUSD']
xw = XAU_data[feature_cols]  
yw = XAU_data.index.week
linreg.fit(xw, yw)
print linreg.intercept_     # 135.139986013
print linreg.coef_          # -116.37727308

xw_pred = linreg.predict(x)

feature_cols = ['EURUSD']
xm = XAU_data[feature_cols]  
ym = XAU_data.index.month
linreg.fit(xm, ym)
print linreg.intercept_     # 31.7673676586
print linreg.coef_          # -27.16222528

xm_pred = linreg.predict(x)


plt.scatter(XAU_data['EURUSD'], xd_pred, color='red')
plt.scatter(XAU_data['EURUSD'], xw_pred, color='blue')
plt.scatter(XAU_data['EURUSD'], xm_pred, color='green')
# x-axis shows EURUSD rate while y-axis presents prediction...
# How do I present this plot?
# X-axis shows rate of EUR and USD
# Y-axis shows linear regression prediction


sm.graphics.tsa.plot_acf(y, lags=30)
# further the plot heads, the less correlated signal it would be.
# 0 = signal is random
# y = XAU_data['EURUSD']
# lags measure in the day

# pair the feature names with the coefficients
#   zip(feature_cols, linreg.coef_)
# Output
# XEUR2 =  1.7420490700099842       relationship? why + or - coefficient?
# XEUR3 = -0.81425425262460049
# XEUR4 =  0.026389727902237113
# XEUR5 = -0.022849806531411141
# XEUR6 =  0.049268236133893861

EUR2 = linreg.predict(x)
plt.scatter(XAU_data['EURUSD'], XAU_data['XEUR2'], color='blue')
plt.scatter(XAU_data['EURUSD'], EUR2, color='red')
# result: the plots seem to match up with linear regression prediciton.

sns.pairplot(XAU_data, x_vars=['XEUR2'], y_vars=['EURUSD'], size=5, aspect=0.85, kind='reg')


sns.pairplot(XAU_data, x_vars=['XEUR2', 'XEUR3', 'XEUR4'], y_vars=['EURUSD'], size=4.5, aspect=0.7, kind='reg')
sns.pairplot(XAU_data, x_vars=['XEUR5', 'XEUR6'], y_vars=['EURUSD'], size=4.5, aspect=0.7, kind='reg')
# Plot Result: XEUR2 seems to be most closely match with original data while 
# other has high variances and cluster closer together as more days calculated.

# Pearson Score (correlation) 
# Model suggest: Ordinary Least Squares
# Look at R-squared: higher percentage the better variability between datas
Xpred = smf.ols(formula='EURUSD ~ XEUR2 + XEUR3 + XEUR4 + XEUR5 + XEUR6', data=XAU_data).fit()
Xpred.summary()

# Logistic Regression between XAU_data['EUR/USD'] and XEURs
logreg = LogisticRegression()
feature_cols = ['XEUR2', 'XEUR3', 'XEUR4', 'XEUR5', 'XEUR6']
x = XAU_data[feature_cols]
y = XAU_data['EURUSD']
logreg.fit(x, y)

# print the XEURs predictions
XEURLog = logreg.predict(x)
print XEURLog

plt.scatter(XAU_data['EURUSD'], XAU_data['XEUR2'], color='blue')
plt.plot(XAU_data['EURUSD'], XEURLog, color='red')
# result: plots didn't match up with predictions at all!


fig2 = plt.figure()
XAU_data['GBP/USD'].plot()
fig2.suptitle('GBP / USD Currency', fontsize=15)
fig2.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('GBP / USD')
#fig2.savefig('GBP/USD.jpg')


# Canadian Dollars weakened due to Oil and Energy prices.
# CAD performed worst against USD.
fig3 = plt.figure()
XAU_data['CAD/USD'].plot()
fig3.suptitle('CAD / USD Currency', fontsize=15)
fig3.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('CAD / USD')
#fig3.savefig('CAD/USD.jpg')

fig4 = plt.figure()
XAU_data['CHF/USD'].plot()
fig4.suptitle('CHF / USD Currency', fontsize=15)
fig4.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('CHF / USD')
#fig4.savefig('CHF/USD.jpg')


# Explore data with Linear Regression...
XAU_data.head()      # inspect linear_data
XAU_data.dtypes      # check to confirm the clean data

# scatter matrix in Seaborn
sns.pairplot(XAU_data)

# scatter matrix in Pandas -> Not so pretty figures....
#pd.scatter_matrix(linear_data, figsize=(12, 10))

# Use a **correlation matrix** to visualize the correlation 
# between all numerical variables.
# Compute correlation matrix
XAU_data.corr()

# display correlation matrix in Seaborn using a heatmap
sns.heatmap(XAU_data.corr())

'''
# Need to edit variables to match up with data sets...
# Holt-Winters Exponentially weighted moving averages smoothing
ewma = pandas.stats.moments.ewma
# make a hat function, and add noise
x = np.linspace(0,1,100)
x = np.hstack((x,x[::-1]))
x += np.random.normal( loc=0, scale=0.1, size=200 )
plot( x, alpha=0.4, label='Raw' )
 
# take EWMA in both directions with a smaller span term
fwd = ewma( x, span=15 ) # take EWMA in fwd direction
bwd = ewma( x[::-1], span=15 ) # take EWMA in bwd direction
c = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
c = np.mean( c, axis=0 ) # average
 
# regular EWMA, with bias against trend
plot( ewma( x, span=20 ), 'b', label='EWMA, span=20' )
 
# "corrected" (?) EWMA
plot( c, 'r', label='Reversed-Recombined' )
 
legend(loc=8)
savefig( 'ewma_correction.png', fmt='png', dpi=100 )


def holt_winters_second_order_ewma( x, span, beta ):
    N = x.size
    alpha = 2.0 / ( 1 + span )
    s = np.zeros(( N, ))
    b = np.zeros(( N, ))
    s[0] = x[0]
    for i in range( 1, N ):
        s[i] = alpha * x[i] + ( 1 - alpha )*( s[i-1] + b[i-1] )
        b[i] = beta * ( s[i] - s[i-1] ) + ( 1 - beta ) * b[i-1]
    return s

# make a hat function, and add noise
x = np.linspace(0, 1, 100)
x = np.hstack((x, x[::-1]))
x += np.random.normal( loc=0, scale=0.1, size=200)

# take EWMA in both directions with a smaller span term
fwd = ewma( x, span=15 ) # take EWMA in fwd direction
bwd = ewma( x[::-1], span=15 ) # take EWMA in bwd direction
c = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
c = np.mean( c, axis=0 ) # average
 
# regular EWMA, with bias against trend
plot( ewma( x, span=20 ), 'b', label='EWMA, span=20' )
 
# "corrected" (?) EWMA
plot( c, 'r', label='Reversed-Recombined' )
 
legend(loc=8)
savefig( 'ewma_correction.png', fmt='png', dpi=100 )
'''

################################
#
# XAU/USD, XAU/EUR, XAU/GBP, XAU/CAD, XAU/CHF 
# EUR/USD, GBP/USD, CAD/USD, CHF/USD
#
# Explore data with Logistic Regression...
XAU_Log = XAU_data  # Create new dataframe
XAU_Log.head()      # inspect logistic_data
XAU_Log.dtypes      # check to confirm the clean data

# Visualize **scatter plot**  with the relationship between currencies
# scatter plot in Seaborn
sns.lmplot(x='XAU/USD', y='XAU/EUR', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/USD', y='XAU/GBP', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/USD', y='XAU/CAD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/USD', y='XAU/CHF', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/USD', y='EUR/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/USD', y='GBP/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/USD', y='CAD/USD', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/USD', y='CHF/USD', data=XAU_Log, ci=None)     # relevant

sns.lmplot(x='XAU/EUR', y='XAU/USD', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/EUR', y='XAU/GBP', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/EUR', y='XAU/CAD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/EUR', y='XAU/CHF', data=XAU_Log, ci=None)     # Try to create 2 lines?
sns.lmplot(x='XAU/EUR', y='EUR/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/EUR', y='GBP/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/EUR', y='CAD/USD', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/EUR', y='CHF/USD', data=XAU_Log, ci=None)     # 

sns.lmplot(x='XAU/GBP', y='XAU/USD', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/GBP', y='XAU/EUR', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/GBP', y='XAU/CAD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/GBP', y='XAU/CHF', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/GBP', y='EUR/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/GBP', y='GBP/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/GBP', y='CAD/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/GBP', y='CHF/USD', data=XAU_Log, ci=None)     # 

sns.lmplot(x='XAU/CAD', y='XAU/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CAD', y='XAU/EUR', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CAD', y='XAU/GBP', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CAD', y='XAU/CHF', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CAD', y='EUR/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CAD', y='GBP/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CAD', y='CAD/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CAD', y='CHF/USD', data=XAU_Log, ci=None)     # 

sns.lmplot(x='XAU/CHF', y='XAU/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CHF', y='XAU/EUR', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CHF', y='XAU/CAD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CHF', y='XAU/GBP', data=XAU_Log, ci=None)     # relevant
sns.lmplot(x='XAU/CHF', y='EUR/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CHF', y='GBP/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CHF', y='CAD/USD', data=XAU_Log, ci=None)     # 
sns.lmplot(x='XAU/CHF', y='CHF/USD', data=XAU_Log, ci=None)     # 


# similar plots as above
sns.pairplot(XAU_Log)

# scatter plot using Pandas
XAU_Log.plot(kind='scatter', x='XAU/EUR', y='EUR/USD')
# scatter plot using Matplotlib
plt.scatter(XAU_Log['XAU/EUR'], XAU_Log['XAU/USD'])

# fit a linear regression model
linreg = LinearRegression()
feature_cols = ['XAU/EUR']
x = XAU_Log[feature_cols]
y = XAU_Log['XAU/USD']
linreg.fit(x, y)

# explore coefficients to get the equation for the line
print linreg.intercept_
print linreg.coef_

# examine predictions for arbitrary points
print linreg.predict(1)
print linreg.predict(2)
print linreg.predict(3)

EUR_pred = linreg.predict(x)
plt.scatter(XAU_Log['XAU/EUR'], XAU_Log['XAU/USD'], color='blue')
plt.plot(XAU_Log['XAU/EUR'], EUR_pred, color='red')


# fit a logistic regression model and store the class predictions
logreg = LogisticRegression()
feature_cols = ['XAU/EUR']
x = XAU_Log[feature_cols]
y = XAU_Log['XAU/USD']
logreg.fit(x, y)

# class prediction output predict(x): predict class labels for samples in X
assorted_predx = logreg.predict(x)

# explore logistic predictions (predict_proba(x): Probability estimates)
logreg.predict_proba(1)
logreg.predict_proba(2)
logreg.predict_proba(3)

# syntax below doesn't make sense with predicting data?
# plot with prediction output line
plt.scatter(XAU_Log['XAU/EUR'], XAU_Log['EUR/USD'])
plt.plot(XAU_Log['XAU/USD'], assorted_predx, color='red')
plt.plot(XAU_Log['XAU/USD'], XAU_Log['XAU/EUR'])

# *** Need to make another plot with USD vs EUR based on exchange rate?



# Decision Tree creation
feature = XAU_data['XEUR2']
DXAU_data = XAU_data.drop('XEUR2', 1)

# split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(DXAU_data, feature, random_state=1)

ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=3)

# Fit the decision tree classifer
ctree.fit(x_train, y_train)

# create a feature vector
featureV = DXAU_data.columns.tolist()

featureV

# interpret Tree diagram
ctree.classes_

# check which features are the most important
ctree.feature_importances_

# combine featureV and ctree.feature_importances for better observation
pd.DataFrame(zip(featureV, ctree.feature_importances_)).sort_index(by=1, ascending=False)

# make predictions on the test set
preds = ctree.predict(x_test)

# calculate accuracy  RMSE: 0.014165  better than before
np.sqrt(metrics.mean_squared_error(y_test, preds))

# conduct a grid serach for the best tree depth
ctree = tree.DecisionTreeRegressor(random_state=1)
depth_range = range(1, 50)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='mean_squared_error')
grid.fit(DXAU_data, feature)

# check out the scores of grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

grid_mean_scores

# find the best estimator
best = grid.best_estimator_

cross_val_score(best, DXAU_data, feature, cv=10, scoring='mean_squared_error').mean()
# result: -6.2486109559843741e-06
np.sqrt(-1*-6.2486109559843741e-06)     # 0.0024997221757596133 

cross_val_score(linreg, DXAU_data, feature, cv=10, scoring='mean_squared_error').mean()
# result: -8.8979923444580203e-07
np.sqrt(-1*-8.8979923444580203e-07)     # 0.00094329170167334873

# conclusion: linear regression will give better insight with 
# currency purchase power analysis.

day_of_month = [z.day for z in x.index]
# Set that in X variable and compare that in Y (EUR2, ....)


