# %% [markdown]
# # Dragon Real Estate Price Predictor

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
housing = pd.read_csv('data.csv')

# %%
housing.head()

# %%
housing.info()

# %%
housing['CHAS'].unique()

# %%
housing['CHAS'].value_counts()

# %%
housing.describe()

# %%
fig, ax = plt.subplots()
ax.scatter(housing.index,housing['CRIM'])
ax.set_ylabel('CRIM')
ax.set_xlabel('Index')
plt.show()

# %%
housing.hist(bins=50, figsize=(20,15))
plt.show()

# %% [markdown]
# ## Train-Test Splitting

# %%
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# %%
train_set, test_set = split_train_test(housing, 0.2)

# %%
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

# %%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

# %%
# sklearn values mught be different because of rounding off error

# %%
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# %%
strat_train_set['CHAS'].value_counts()

# %%
housing_train = strat_train_set.copy()
housing_test = strat_test_set.copy()

# %%
housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV']

# %% [markdown]
# ## Looking for Correlations

# %%
housing1=pd.read_csv('data.csv')
corr_matrix = housing1.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# %%
from pandas.plotting import scatter_matrix

# %%
attributes = ['RM', 'ZN', 'MEDV', 'LSTAT']
scatter_matrix(housing1[attributes], figsize = (12,8))
plt.show()

# %%
housing1.plot(kind="scatter", x = "RM", y = "MEDV", alpha=0.4)
plt.show()

# %% [markdown]
# ## Trying out Attribute Combinations

# %%
housing1["TAXRM"]=housing1["TAX"]/housing["RM"]

# %%
housing1["TAXRM"].head()

# %%
corr_matrix = housing1.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# %%
housing1.plot(kind="scatter", x = "TAXRM", y = "MEDV", alpha=0.4)
plt.show()

# %% [markdown]
# ## Missing Attributes

# %% [markdown]
# To take care of missing attributes, you have three options:
# 1. Get rid of the missing data points
# 2. Get rid of the whole attribute
# 3. Set the value to some value(0, mean, median) <br>
# We are assuming few RM values to be missing as it has high correlation.

# %%
a = housing.dropna(subset=["RM"]) # Option1
a.shape
# a will not change as we dont have na values

# %%
housing.drop("RM", axis=1) #OPtion2
#note that there is no RM column but will not affect original dataframe

# %%
median = housing["RM"].median() # Compute median Option 3
housing["RM"].fillna(median)
#Original dataframe remains unchanged

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy ="median")
imputer.fit(housing)

# %%
print('Values for imputer:',imputer.statistics_)
print('Shape:',imputer.statistics_.shape)

# %%
X = imputer.transform(housing)

# %%
housing_tr = pd.DataFrame(X, columns=housing.columns)

# %%
housing_tr.describe()

# %% [markdown]
# ## Scikit-learn Design

# %% [markdown]
# Primarily, 3 types of objects:
# 1. Estimators
# - They estimate parameters based on a database. eg. Imputer
# - It has a fit method and transform method.
# - Fit methods - Fits the dataset and calculates internal parameters. 
# 2. Transformers
# - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.
# 3. Predictors
# - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also has a score() function which will evaluate the predictions.

# %% [markdown]
# ## Feature Scaling

# %% [markdown]
# Two types of feature-scaling methods primarily exist:
# 1. Min-max scaling (Normalisation)
#   (value-min)/(max-min)
#   Sklearn proveds a class called MinMaxScaler for this.
# 
# 2. Standardisation
#    (value-mean)/std
#    Sklearn provides a class called StandardScaler for this.
# 

# %% [markdown]
# ## Creating a Pipeline

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    # ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

# %%
housing_num_tr = my_pipeline.fit_transform(housing_tr)

# %%
housing_num_tr.shape

# %% [markdown]
# ## Selecting a desired model for Dragon Real Estate

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# %%
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
# model = LinearRegression()

# %%
model.fit(housing_num_tr, housing_labels)

# %%
some_data = housing.iloc[:5]

# %%
some_labels = housing_labels.iloc[:5]

# %%
prepared_data = my_pipeline.transform(some_data)

# %%
model.predict(prepared_data)

# %%
list(some_labels)

# %% [markdown]
# ## Evaluating the model

# %%
from sklearn.metrics import mean_squared_error

# %%
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# %%
mse

# %% [markdown]
# ## Using better evaluation technique - Cross Validation

# %%
from sklearn.model_selection import cross_val_score

# %%
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring='neg_mean_squared_error', cv = 10)
rmse_scores = np.sqrt(-scores)

# %%
rmse_scores

# %%
def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

# %%
print_scores(rmse_scores)

# %%



