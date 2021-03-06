import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import f_regression, SelectKBest, RFE
from sklearn.preprocessing import PolynomialFeatures

from math import sqrt


def visualize_distribution(df, column):
    '''
    This function takes in a dataframe and columns and creates a histogram with each column
    '''
    for col in column:
        plt.hist(df[col])
        plt.title(f"{col} distribution")
        plt.show()


def distplot(df, column):
    '''
    This functions takes in a dataframe and the columns to plot. It then weeds out any string columns and plots the
    distribution of numerical columns
    '''
    for i in column:
        if df[i].dtypes != 'object':
            sns.distplot(df[i])
            plt.xticks(fontsize= 12)
            plt.yticks(fontsize=12)
            plt.ylabel("Count", fontsize= 13, fontweight="bold")
            plt.xlabel(i, fontsize=13, fontweight="bold")
            plt.title('Distribution of '+i)
            plt.show()
    
    else:
        print('')


### Scatterplot

def scatterplot(X, y, train):
    '''
    Takes in an X and y and a dataframe and creates a loop for a scatterplot
    Tip: Add y as string or assign a variable
    assign X as a variable
    '''

    for i in X:
        sns.scatterplot(x=i, y=y, data=train)
        plt.title('Correlation of '+ i+ ' with '+y)
        plt.show()


### Lmplot

def lmplot(X, y, train):
    '''
    Takes in an an X, y and dataframe and returns a loop of lmplots.
    Tip: Add y as string or assign a variable
    assign X as a variable
    '''
    for i in X:
        sns.lmplot(x=i, y=y, data=train, line_kws={'color': 'red'})
        plt.title('Correlation of '+ i+ ' with '+y)
        plt.show()






def rmse(algo, X_train, X_validate, y_train, y_validate, target, model_name):
    '''
    This function takes in an algorithm name, X_train, X_validate, y_train, y_validate, target and a model name
    and returns the RMSE score for train and validate dataframes.
    '''

    # enter target and model_name as a string
    # algo is algorithm name, enter with capitals for print statement
    
    # fit the model using the algorithm
    algo.fit(X_train, y_train[target])

    # predict train
    y_train[model_name] = algo.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train[target], y_train[model_name])**(1/2)

    # predict validate
    y_validate[model_name] = algo.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate[target], y_validate[model_name])**(1/2)

    print("RMSE for", model_name, "using", algo, "\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    print()
    
    return rmse_train, rmse_validate



# KBEST FUNCTION

def select_kbest(X_train_scaled, y_train, no_features):
    '''
    This function takes in scaled data and number of features and returns the top features
    '''
    
    # using kbest
    f_selector = SelectKBest(score_func=f_regression, k=no_features)
    
    # fit
    f_selector.fit(X_train_scaled, y_train)

    # display the two most important features
    mask = f_selector.get_support()
    
    return X_train_scaled.columns[mask]



## RFE FUNCTION

def rfe(X_train_scaled, y_train, no_features):
    '''
    This function takes in scaled data and number of features and returns the top features
    '''
    
    # now using recursive feature elimination
    lm = LinearRegression()
    rfe = RFE(estimator=lm, n_features_to_select=no_features)
    rfe.fit(X_train_scaled, y_train)

    # returning the top chosen features
    return X_train_scaled.columns[rfe.support_]