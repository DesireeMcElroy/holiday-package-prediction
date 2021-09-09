import pandas as pd
import numpy as np

import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# This wrangle file holds all of my acquisition and data preparation functions for my notebook.



def get_info(df):
    '''
    This function takes in a dataframe and prints out information about the dataframe.
    '''

    print(df.info())
    print()
    print('------------------------')
    print()
    print('This dataframe has', df.shape[0], 'rows and', df.shape[1], 'columns.')
    print()
    print('------------------------')
    print()
    print('Null count in dataframe:')
    print('------------------------')
    print(df.isnull().sum())
    print()
    print('------------------------')
    print(' Dataframe sample:')
    print()
    return df.sample(3)


