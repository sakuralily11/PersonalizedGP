

#################################################################
## Wrapper for predicting adas13, based on features for time t ## 
#################################################################

from call_pgp import * 

import os 
import csv
import numpy as np
import itertools 

######################
##### PREP DATA ######
######################

print('----- PREPARING DATA -----')

features = ['fl1']

for f in features: 
    
    #define directories
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_%s_l4.csv'%(f)) #save data CSV file to current directory 
    ID_DIR = os.path.join(CURRENT_DIR, 'Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv') #save ID CSV file to current directory 
    
    #create list of IDs
    with open(ID_DIR, 'r') as id_csv:
        reader = csv.reader(id_csv)
        ID_all = list(reader)
        ID_all = list(itertools.chain.from_iterable(ID_all)) #flatten list
        ID_all = list(map(int, ID_all)) #cast each element as int
    
    #create X_all, Y_all, value_all
    with open(CSV_DATA_DIR, 'r', encoding = 'utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = " ", quotechar = '|')
        reader = list(reader)[0:]
        X_all = {}
        Y_all = {}
        indicator_all = {}
        for line in reader:
            line = line[0].split(',')
            line = list(map(float, line))
            ID = int(line[0])
            x_line = list(map(float, line[1:-8])) 
            y_line = list(map(float, line[-8:-4])) # EDITED 
            #y_line = list(map(float, line[-8:-7]))
#            y_line = list(map(float, line[-8:-4]))
            #indicator_line = list(map(int, line[-4:-3]))
            indicator_line = list(map(int, line[-4:])) # EDITED 
            if ID in ID_all: #check if ID is in ID_all
                if ID in X_all:
                    X_all[ID] = np.vstack((X_all[ID], x_line))
                    Y_all[ID] = np.vstack((Y_all[ID], y_line))
                    indicator_all[ID] = np.vstack((indicator_all[ID], indicator_line))
                else:
                    X_all[ID] = x_line
                    Y_all[ID] = y_line
                    indicator_all[ID] = indicator_line
    
    X_array = np.vstack(tuple(X_all.values()))
    Y_array = np.vstack(tuple(Y_all.values()))
    
    results = np.array([])
    
    #loop for 10 folds
    for i in range(0, 10): #0 to 9
        tst_ind = ID_all[i*10:i*10+10]
        tr_ind_source = np.setdiff1d(ID_all, tst_ind)
        
        #create x_s, y_s
        x_dict_s = {key:value for key, value in X_all.items() if key in tr_ind_source}
        y_dict_s = {key:value for key, value in Y_all.items() if key in tr_ind_source}
        
        x_list_s = tuple(x_dict_s.values())
        y_list_s = tuple(y_dict_s.values())
    
        x_s = np.vstack(x_list_s)
        y_s = np.vstack(y_list_s)
        
        #create x_a, y_a 
        x_a = {key:value for key, value in X_all.items() if key in tst_ind}
        y_a = {key:value for key, value in Y_all.items() if key in tst_ind}
        
        #ground truth 
        g_t = y_a 
        
        #create g_t_all 
        g_list_t = tuple(g_t.values())
        g_t_all = np.vstack(g_list_t) ##
        
        #create xtest_all, ytest_all
        x_dict_test_all = {key:value for key, value in X_all.items() if key in tst_ind}
        y_dict_test_all = {key:value for key, value in Y_all.items() if key in tst_ind}
        
        x_list_test_all = tuple(x_dict_test_all.values())
        y_list_test_all = tuple(y_dict_test_all.values())
        
        xtest_all = np.vstack(x_list_test_all)
        ytest_all = np.vstack(y_list_test_all)
        
        #creating RBF kernel 
        d = np.shape(x_s)[1]
        k = New_RBF(input_dim = d, slices = [d])
        
        #TRAINING SOURCE MODEL 
        #creating gp model m with x_s, y_s variables and kernel k 
        m = gpflow.models.GPR(x_s, y_s, kern = k)
        #initialize hyperparameters 
        m.likelihood.variance = np.exp(2*np.log(np.sqrt(0.1*np.var(y_s))))
        max_x = np.amax(x_s, axis=0)
        min_x = np.amin(x_s, axis=0)
        m.kern.lengthscales = np.array([np.median(max_x-min_x)])
        m.kern.variance = np.var(y_s)
        print('Parameters Before Optimization:', m.read_trainables())
        #optimizing model 
        m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(m, maxiter=30)
        print('Parameters After Optimization:', m.read_trainables())
    
        #TRAIN ADAPTATION AND TARGET MODELS
        
        print('----- TRAINING ADAPTATION AND TARGET MODELS -----')
        
        for ID in tst_ind: 
            
            print('----- TEST PATIENT: %s -----'%(ID))
            
            x_a_patient = x_a[ID][:-1,:]
            y_a_patient = y_a[ID][:-1,:]
            xtest = x_a[ID]
            
            #predictions 
            out = call_pgp(m, x_s, y_s, x_a_patient, y_a_patient, xtest, k)
            
            ID_array = np.full((len(y_a[ID]), 1), ID)
            ytest = y_a[ID] #21x4, ground truth labels 
            m_s = out['source model mu'] #21x4 
            s_s = out['source model sigma'] #21x1 
            m_a = out['adapted model mu'] #21x4 
            s_a = out['adapted model sigma'] #21x1 
            m_t = out['target model mu'] #21x4 
            s_t = out['target model sigma'] #21x1 
            
            sub_results = np.concatenate((ID_array, ytest, m_s, s_s, m_a, s_a, m_t, s_t), axis = 1)
            
            if len(results) == 0:
                results = sub_results 
            else:
                results = np.vstack((results, sub_results))
    
    #once done with whole loop... add indicators numpy array to last column 
    #results = np.hstack((results, indicator_all_modified))
    
    #dump results array to csv file 
    
    #create csv files 
    output_csv_name = 'adni_adas13_%s_output_mod.csv'%(f) #EDITED 
    OUTPUT_CSV_DIR = os.path.join(CURRENT_DIR, output_csv_name)
    np.savetxt(OUTPUT_CSV_DIR, results, delimiter = ',')
