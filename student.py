import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("datasets.csv")
df = df.drop('PlaceofBirth', axis=1)  # Drop irrelevant column

# Print the dataset summary statistics
print(df.describe())

# Visualize categorical variables
ls = ['gender', 'Relation', 'Topic', 'Section ID', 'Grade ID',
      'Nationality', 'Class', 'Stage ID', 'Semester', 
      'Parent Answering Survey', 'Parent school Satisfaction', 'Student Absence Days']

# Adjusted plotting for better readability
for feature in ls:
    g = sns.catplot(x=feature, data=df, kind='count', height=4, aspect=1.5)
    g.set_axis_labels(feature, 'Count')  # Add axis labels
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

# Prepare data for training
target = df.pop('Class')
X = pd.get_dummies(df)
le = LabelEncoder()
y = le.fit_transform(target)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)  # Use transform instead of fit_transform on test data

# Removing irrelevant features based on domain knowledge
X_train_new = X_train[['VisITedResources', 'raisedhands', 'AnnouncementsView',
                       'Student Absence Days_Above-7', 'Student Absence Days_Under-7', 'Discussion']]

X_test_new = X_test[['VisITedResources', 'raisedhands', 'AnnouncementsView',
                     'Student Absence Days_Above-7', 'Student Absence Days_Under-7', 'Discussion']]

# Spot checking algorithms
models = [
    ('LR', LinearRegression()),
    ('LASSO', Lasso()),
    ('EN', ElasticNet()),
    ('KNN', KNeighborsRegressor()),
    ('CART', DecisionTreeRegressor()),
    ('SVR', SVR())
]

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})")

# Scaling and comparing ensemble models
pipelines = [
    ('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])),
    ('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])),
    ('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])),
    ('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])),
    ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])),
    ('ScaledSVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())]))
]

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})")

# Boxplot for comparison
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
plt.show()

# Tuning Lasso Regression
scaler = StandardScaler().fit(X_train_new)
rescaledX = scaler.transform(X_train_new)
k_values = np.array([.1, .11, .12, .13, .14, .15, .16, .09, .08, .07, .06, .05, .04])
param_grid = dict(alpha=k_values)
model = Lasso()
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

# Print GridSearchCV results
print(f"Best: {grid_result.best_score_:.6f} using {grid_result.best_params_}")
for mean, stdev, param in zip(grid_result.cv_results_['mean_test_score'], 
                               grid_result.cv_results_['std_test_score'], 
                               grid_result.cv_results_['params']):
    print(f"{mean:.6f} ({stdev:.6f}) with: {param}")

# Using ensemble models
ensembles = [
    ('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])),
    ('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])),
    ('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])),
    ('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())]))
]

results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})")

# Boxplot for ensemble model comparison
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.xticks(rotation=45, ha='right')
plt.show()

# Tune AdaBoost Regressor
param_grid = dict(n_estimators=np.array([50, 100, 150, 200, 250, 300, 350, 400]))
model = AdaBoostRegressor(random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

# Print GridSearchCV results for AdaBoost
print(f"Best: {grid_result.best_score_:.6f} using {grid_result.best_params_}")
for mean, stdev, param in zip(grid_result.cv_results_['mean_test_score'], 
                               grid_result.cv_results_['std_test_score'], 
                               grid_result.cv_results_['params']):
    print(f"{mean:.6f} ({stdev:.6f}) with: {param}")

# Final model: GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=7, n_estimators=400)
model.fit(rescaledX, y_train)

# Predictions on test set
rescaledValidationX = scaler.transform(X_test_new)
predictions = model.predict(rescaledValidationX)
print(f"Mean Squared Error on test set: {mean_squared_error(y_test, predictions):.6f}")
