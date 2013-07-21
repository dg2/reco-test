# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:25:29 2013

@author: dario
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from os.path import join
DATA_FOLDER= '/home/dario/Projects/RecommenderTest/ml-100k'
DATA_FILE = 'u.data'

def load_ml100k(max_ratings = None, normalize = 'user'):
    # Read tab separated data into a dataframe
    df = pd.read_csv(join(DATA_FOLDER, DATA_FILE), sep = '\t', header = False)
    df.columns = ['user','item','rating','ts']
    if (max_ratings!=None):
        df = df.ix[:max_ratings]
    if (normalize != None):
        gr = df.groupby(normalize)
        df['rating'] = gr.transform(lambda(x): x - x.mean())['rating']

    # Now transform it into a sparse matrix
    A = csr_matrix((df.rating.values,df[['user','item']].values.T))
    return A        
