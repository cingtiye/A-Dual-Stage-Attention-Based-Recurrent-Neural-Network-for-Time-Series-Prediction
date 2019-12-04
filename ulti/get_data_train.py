# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import h5py
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def get_data(data_path,T, n):
    input_X = []
    input_Y = []
    label_Y = []

    df = pd.read_csv(data_path)
    row_length = len(df)
    column_length = df.columns.size
    for i in tqdm(range(row_length-T+1)):
        X_data = df.iloc[i:i+T, 0:column_length-1]
        Y_data = df.iloc[i:i+T-1,column_length-1]
        label_data = df.iloc[i+T-1,column_length-1]
        input_X.append(np.array(X_data))
        input_Y.append(np.array(Y_data))
        label_Y.append(np.array(label_data))
    input_X = np.array(input_X).reshape(-1,T,n)
    input_Y = np.array(input_Y).reshape(-1,T-1,1)
    label_Y = np.array(label_Y).reshape(-1,1)
    return input_X,input_Y,label_Y

def write_cache(fname,input_X, input_Y, label_Y):
    h5 = h5py.File(fname,'w')
    #h5.create_dataset('num',data=len(input_X))
    h5.create_dataset('input_X',data=input_X)
    h5.create_dataset('input_Y',data=input_Y)
    h5.create_dataset('label_Y',data=label_Y)
    h5.close()

def read_cache(fname):
    f = h5py.File(fname,'r')
    #num = int(f['num'].value)
    input_X = f['input_X'].value
    input_Y = f['input_Y'].value
    label_Y = f['label_Y'].value
    f.close()
    return input_X,input_Y,label_Y

def run(path=None):
    print('read data ...')
    T = 10
    n = 81
    if path == None:
        path = './data/nasdaq100/small/'
    data_path = path + 'nasdaq100_padding.csv'
    cache_path = os.path.join(path, 'cache')
    fname_model = 'model_T{}.h5'.format(T)
    CACHEDATA = True

    if CACHEDATA and os.path.isdir(cache_path) is False:
        os.mkdir(cache_path)
    fname = os.path.join(cache_path, 'nasdaq_T{}.h5'.format(T))
    if os.path.exists(fname) and CACHEDATA:
        input_X, input_Y, label_Y = read_cache(fname)
        print('load %s successfully' % fname)
    else:
        input_X, input_Y, label_Y = get_data(data_path, T, n)
        if CACHEDATA:
            write_cache(fname, input_X, input_Y, label_Y)

    print(input_X.shape)
    print(input_Y.shape)
    print(label_Y.shape)

    train_size = 35091
    valid_size = 128*42

    input_X_train = input_X[:train_size, :, :]
    input_Y_train = input_Y[:train_size, :, :].reshape([train_size, -1])
    label_Y_train = label_Y[:train_size, :].reshape([-1])

    input_X_valid = input_X[train_size:train_size + valid_size, :, :]
    input_Y_valid = input_Y[train_size:train_size + valid_size, :, :].reshape([valid_size, -1])
    label_Y_valid = label_Y[train_size:train_size + valid_size, :].reshape([-1])

    input_X_test = input_X[train_size + valid_size:, :, :]
    input_Y_test = input_Y[train_size + valid_size:, :, :].reshape([input_X.shape[0]-(train_size + valid_size), -1])
    label_Y_test = label_Y[train_size + valid_size:, :].reshape([-1])

    print('input_X_train shape: {}'.format(input_X_train.shape))
    print('input_Y_train shape: {}'.format(input_Y_train.shape))
    print('label_Y_train shape: {}'.format(label_Y_train.shape))
    print('input_X_valid shape: {}'.format(input_X_valid.shape))
    print('input_Y_valid shape: {}'.format(input_Y_valid.shape))
    print('label_Y_valid shape: {}'.format(label_Y_valid.shape))
    print('input_X_test shape: {}'.format(input_X_test.shape))
    print('input_Y_test shape: {}'.format(input_Y_test.shape))
    print('label_Y_test shape: {}'.format(label_Y_test.shape))

    return (input_X_train, input_Y_train, label_Y_train),\
           (input_X_valid, input_Y_valid, label_Y_valid),\
           (input_X_test, input_Y_test, label_Y_test)

if __name__ == '__main__':
    train_data, valid_data, test_data = run('../data/nasdaq100/small/')

