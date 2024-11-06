## Train deep learning model for chemical-disease data in CTD
## 10 fold results, with 1:1 positive/negative ratio, (or validation_data=0.2)
##   --> positive: 71187 CTD_wDE   (with DirectEvidence) "marker/mechanism" or "therapeutic"
##   --> positive: 26789 CTD_thera (only DirectEvidence="therapeutic")
##                  (chemical: with MACCS, properties)
##                  (disease : nodes in MeSH tree and not all zero similarity with 277 repDiseases)
##   --> negative: 8011328 CTD_thera chemical-disease data not in CTD (CTD_thera chemical-disease pair not in CTD) 

# feature: 182/228/311+277=459/505/588
    # 1. affinity with 182/228/311 repProteins (4157 CTD_chemical)
    # 2. similarity with 277 repDiseases (2149 CTD_disease)
    

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, concatenate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow.keras.metrics as km
import pandas as pd
import time
import random
import joblib
import os

os.chdir("D:/drug_disease_model/HEDDI-Net")

from metric_fn import get_metrics

## [train_type]
# reserved 0.8 train/0.2 test
# 1: 10-fold pos_CTD_wDE and neg_CTD_thera
# 2: 10-fold pos_CTD_thera and neg_CTD_thera
# 3: (reserved options)
# 4: 10-fold compared B-dataset
#exe_path   = "/mnt/nvme2/nvme1"
exe_path   = "./data"
train_type = 2
k_fold_CV  = True
batch_size = 11000
epochs     = 1600
num_folds  = 10
lr         = 1e-3

ge5,ge4,ge3 = 182,228,311
pfeature, pfdata = ge4, "ge4"
## 1. load .npy file and create X,y
##    [0-181/227/310:repProtein ge5/4/3 affinity, 311-587:repDisease similarity]
#     71187*459/505/588   CTD_wDE;
#       pos_{ge5/ge4/ge3}_wDE.npy
#     26789*459/505/588   CTD_thera;
#       pos_{ge5/ge4/ge3}_thera.npy
#     8011328*459/505/588 CTD_thera_notin_CTD; (excluding inferred associations)
#       neg_{ge5/ge4/ge3}.npy

df_result_all = pd.DataFrame(columns=['AUPR','AUC','F1','ACC','Recall','Specificity','Precision'])

for batch_size in range(11000,12000,1000):
    for epochs in range(1600,1700,100):
#for run in range(0,1):
#    for tt in range(0,100):
        # give positive data => X_1
        if train_type == 1:
            X_1 = np.load(exe_path+ '/npy/pos_' +pfdata+ '_wDE.npy')
            neg_data = np.load(exe_path+ '/npy/neg_' +pfdata+ '.npy')
        elif train_type == 2:
            X_1 = np.load(exe_path+ '/npy/pos_' +pfdata+ '_thera.npy')
            neg_data = np.load(exe_path+ '/npy/neg_' +pfdata+ '.npy')
        elif train_type == 3: # reserved item
            X_1 = np.load(exe_path+ '/npy/pos_' +pfdata+ '.npy')
            neg_data = np.load(exe_path+ '/npy/neg_' +pfdata+ '.npy')
        elif train_type == 4:
            X_1 = np.load('./B-dataset/npy/pos_' +pfdata+ '.npy')
            neg_data = np.load('./B-dataset/npy/neg_' +pfdata+ '.npy')
        
        print("negative data=", neg_data.shape)
        print("positive data=", X_1.shape)
    
        # give the number of negative data (unobserved association pair) => X_0
        if train_type!=3:
            ind = random.sample(range(len(neg_data)), len(X_1))
            X_0 = neg_data[ind]
            #X_0 = neg_data
            X = np.append(X_1, X_0, axis=0).astype('float32')
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
            y = np.append(np.ones((len(X_1),1)), np.zeros((len(X_0),1)), axis=0).astype('bool')
            # split and shuffle data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True, random_state=None, stratify=y)
        else:  # [reserved options]
            ind = random.sample(range(len(neg_data)), len(X_1)+71187)
            X_0 = neg_data[ind[:len(X_1)]]
            X = np.append(X_1, X_0, axis=0).astype('float32')
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
            y = np.append(np.ones((len(X_1),1)), np.zeros((len(X_0),1)), axis=0).astype('bool')   
            # split and shuffle data
            p = np.random.permutation(len(y))
            x_train, y_train = X[p], y[p]
            x1_test = np.load('./data/npy/pos_' +pfdata+ '_wDE.npy')
            x0_test = neg_data[ind[len(X_1):]]
            x_test  = np.append(x1_test, x0_test, axis=0).astype('float32')
            x_test  = scaler.transform(x_test)
            y_test  = np.append(np.ones((len(x1_test),1)), np.zeros((len(x0_test),1)), axis=0).astype('bool')
        
        del X,y,X_0,X_1,neg_data
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)
        
        
        ## 2. DNN Model description
        def DNN_model():
            # Chemical part
            chemical_in = Input(shape=(pfeature,))
            model_chemical = Dense(128, activation='tanh')(chemical_in)
            model_chemical = Dense(64, activation='tanh')(model_chemical)
            model_chemical = Dropout(0.25)(model_chemical)
            # Disease part
            disease_in = Input(shape=(277,))
            model_disease = Dense(128, activation='tanh')(disease_in)
            model_disease = Dense(64, activation='tanh')(model_disease)
            model_disease = Dropout(0.25)(model_disease)
            # Concated model
            concate = concatenate([model_chemical, model_disease])
            model_combine = Dense(128, activation='sigmoid')(concate)
            model_combine = Dense(64, activation='sigmoid')(model_combine)
            model_combine = Dropout(0.5)(model_combine)
            model_combine = Dense(2, activation='softmax')(model_combine)
            
            model = Model(inputs=[chemical_in, disease_in], outputs=model_combine)
            #model.summary()
            opti = keras.optimizers.Adam(learning_rate=lr)
            model.compile(loss="categorical_crossentropy", optimizer=opti,
                          metrics=[km.Recall(), km.Precision(), km.CategoricalAccuracy(),  
                                   km.AUC(curve='PR', name='aupr'), km.AUC(curve='ROC', name='auc')])
            return model

        
        df_result1 = pd.DataFrame(columns=['Recall', 'Precision', 'ACC', 'AUPR', 'AUC'])
        df_result = pd.DataFrame(columns=['AUPR','AUC','F1','ACC','Recall','Specificity','Precision'])
        fold_no = 1
        
        if(k_fold_CV and train_type!=3):
            skf  = StratifiedKFold(n_splits = num_folds, random_state = None, shuffle = True)
            X = np.append(x_train, x_test, axis=0)
            y = np.append(y_train, y_test, axis=0)
            model = DNN_model()

            for train, test in skf.split(X, np.argmax(y, axis=1)):
                print(f"====== train fold {fold_no}/{num_folds} ======")        
                hist = model.fit([X[train][:,:pfeature], X[train][:,pfeature:]], y[train],
                                  batch_size=batch_size, epochs=epochs, verbose=2)
                results = model.evaluate([X[test][:,:pfeature], X[test][:,pfeature:]], y[test], verbose=0)
                y_pred = model.predict([X[test][:,:pfeature], X[test][:,pfeature:]])
                metric = get_metrics(y[test][:,1], y_pred[:,1])
                df_result1.loc[fold_no] = results[1:]
                df_result.loc[fold_no] = metric
              
                fold_no += 1
            ## save weights to HDF5 file for further use
            #model.save_weights('./data/scaler/DNN_t2_'+str(tt)+'_weight.h5')
            
        else: # 20% validation or train_type=3 (reserved)
            hist = model.fit([x_train[:,:pfeature], x_train[:,pfeature:]], y_train, batch_size=batch_size,
                              epochs=epochs, validation_data=([x_test[:,:pfeature], x_test[:,pfeature:]], y_test))
            results = model.evaluate([x_test[:,:pfeature], x_test[:,pfeature:]], y_test, verbose=0)
        
        
        df_result.loc['median'] = df_result.loc[1:num_folds,:].median(axis=0)
        df_result.loc['mean'] = df_result.loc[1:num_folds,:].mean(axis=0)
        df_result.to_excel('./newtype'+str(train_type)+'_'+pfdata+'_1to1_e'+str(epochs)+'_bat'+str(batch_size)+'.xlsx')
        
        '''
        df_result_all.loc[run] = df_result.loc['median']
        df_result.to_excel('./newtype'+str(train_type)+'_'+pfdata+'_1to1_e'+str(epochs)+'_bat'+str(batch_size)+'_'+str(run)+'.xlsx')
        #save the neg index for negative data
        np.save('./newtype'+str(train_type)+'_'+pfdata+'_1to1_negindex_'+str(run)+'.npy', np.array(ind))
            #ind = np.load('negindex.npy').tolist()
        print(df_result_all)
        df_result.to_excel('./data/scaler/DNN_t2_'+str(tt)+'.xlsx')
        '''


##  HDF5 file --> './data/scaler/DNN_t2_'+str(tt)+'_weight.h5'
##    Which can be applied to predict new candidates for designated disease based on 100 saved model_weights
      #model.load_weights('./data/scaler/DNN_t2_'+str(tt)+'_weight.h5')

