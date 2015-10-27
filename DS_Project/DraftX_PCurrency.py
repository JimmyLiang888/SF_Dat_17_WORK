import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

data_currency = pd.read_csv('~jimmy/desktop/datascience/sf_dat_17_work/ds_project/exchangeGVIForex.csv')

# explore first 11 data
data_currency.head(11)

# delete first 6 rows and set up clean data
clean_data = data_currency.drop([0, 1, 2, 3, 4, 5])
#clean_data = data_currency.ix[6:]   # does the same as above syntax...

clean_data.head()   # Inspect dataframe

# rename columns
clean_data = clean_data.rename(columns={'Results':'Date','EUR/USD Close':'EUR/USD', 'USD/JPY Close':'USD/JPY', 'USD/CHF Close':'USD/CHF', 'GBP/USD Close':'GBP/USD', 'USD/CAD Close':'USD/CAD'})
clean_data.head()
clean_data.dtypes   #Inspect data types

# convert Date from object to datetime64
# Edit Date object to datetime64
clean_data['Date'] = clean_data.Date.apply(lambda x:pd.to_datetime(x, format='%Y-%m-%d'))
#clean_data['Date'] = pd.to_datetime(clean_data['Date'])

# Convert columns data to float value
for column in clean_data.columns[1:]:
    clean_data[column] = clean_data[column].astype(float)

clean_data.dtypes   # check to confirm clean_data types --> datetime64 and float64


# reciprocal EUR and GBP column to have uniform USD value
clean_data['EUR/USD'] = 1. / clean_data['EUR/USD']
clean_data['GBP/USD'] = 1. / clean_data['GBP/USD']
clean_data['EUR/USD'].head()
clean_data['GBP/USD'].head()

# Relabel EUR and GBP
clean_data = clean_data.rename(columns={'EUR/USD':'USD/EUR', 'GBP/USD':'USD/GBP'})

clean_data.head()   # inspect clean_data for accurancy

# Set up data for linear regression
linear_data = clean_data

# Check USD/EUR Close data details
clean_data['USD/EUR'].describe()

clean_data.describe()   # show all data details

# show same data details as above...
clean_data.mean()   # calculate average value
clean_data.min()    # look up for minimum value
clean_data.max()    # look up for maximum value
np.std(clean_data)  # calculate sample standard deviation of data


len(clean_data.columns)     # check how many columns: 5
type(clean_data)            # pandas.core.frame.DataFrame
data_currency.shape         # (788, 6)
clean_data.shape            # (782, 6)

clean_data.head()           # explore clean_data's first 5 data 

# set Results into the right DataFrame
clean_data = clean_data.set_index("Date")


# plots
clean_data.plot()

fig = plt.figure()
clean_data['USD/EUR'].plot()
fig.suptitle('USD / EUR Currency', fontsize=15)
fig.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('USD / EUR')
#fig.savefig('USD/EUR.jpg')

fig2 = plt.figure()
clean_data['USD/JPY'].plot()
fig2.suptitle('USD / JPY Currency', fontsize=15)
fig2.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('USD / JPY')
#fig2.savefig('USD/JPY.jpg')

fig3 = plt.figure()
clean_data['USD/CHF'].plot()
fig3.suptitle('USD / CHF Currency', fontsize=15)
fig3.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('USD / CHF')
#fig3.savefig('USD/CHF.jpg')

fig4 = plt.figure()
clean_data['USD/GBP'].plot()
fig4.suptitle('USD / GBP Currency', fontsize=15)
fig4.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('USD / CHF')
#fig4.savefig('USD/GBP.jpg')

fig5 = plt.figure()
clean_data['USD/CAD'].plot()
fig5.suptitle('USD / CAD', fontsize=15)
fig5.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('USD / CAD')
#fig5.savefig('USD/CAD.jpg')


################################
# Explore different syntax plot styles
# doesn't include Results date on x-axis? Inspect Index location?
plt.plot(clean_data['USD/EUR'])
plt.ylabel('USD / EUR')
plt.xlabel('Date')
plt.title('USD / EUR Currency', fontsize=15)
plt.show()

################################
clean_data.dtypes
clean_data.info()
clean_data.head()

clean_data.plot(y = 'USD/EUR', figsize=(9,5))
plt.ylabel('USD / EUR')
plt.xlabel('Date')
plt.title('USD / EUR Currency', fontsize=15)

################################


# Explore data with Linear Regression...
linear_data.head()      # inspect linear_data
linear_data.dtypes      # check to confirm the clean data

# Use a **scatter plot** to visualize the relationship between currencies
# scatter plot in Seaborn
sns.pairplot(linear_data, x_vars=['USD/EUR', 'USD/JPY'], y_vars='USD/CAD', size=4.5, aspect=0.7)
sns.pairplot(linear_data, x_vars=['USD/CHF', 'USD/GBP'], y_vars='USD/CAD', size=4.5, aspect=0.7)

# include a "regression line"
sns.pairplot(linear_data, x_vars=['USD/EUR', 'USD/JPY'], y_vars='USD/CAD', size=4.5, aspect=0.7, kind='reg')
sns.pairplot(linear_data, x_vars=['USD/CHF', 'USD/GBP'], y_vars='USD/CAD', size=4.5, aspect=0.7, kind='reg')

# scatter matrix in Seaborn
sns.pairplot(linear_data)

# scatter matrix in Pandas -> Not so pretty figures....
#pd.scatter_matrix(linear_data, figsize=(12, 10))

# Use a **correlation matrix** to visualize the correlation 
# between all numerical variables.
# Compute correlation matrix
linear_data.corr()

# display correlation matrix in Seaborn using a heatmap
sns.heatmap(linear_data.corr())

'''
## Would be good idea to have an 'element' (oil or metal) to reflect against 
## all currencies for prediction and linear/logistic regression?

### STATSMODELS ###

# create a fitted model
ls = smf.ols(formula='USD/EUR ~ USD/CAD', linear_data=linear_data).fit()

# print the coefficients
ls.params
'''

### SCIKIT-LEARN ###

# create x and y
feature_cols = ['USD/EUR']
x = linear_data[feature_cols]
y = linear_data['USD/CAD']

# instantiate and fit
linreg = LinearRegression()
linreg.fit(x, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_
