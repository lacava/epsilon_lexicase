import sys
from ellyn import ellyn
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import itertools
import pdb
import time
from joblib import Parallel, delayed

dataset = sys.argv[1]
output_file = sys.argv[2]
savename = output_file.split('/')[-1][:-4]
problem = sys.argv[3]
scale_these = ['enh','enc','housing','airfoil','concrete','yacht','crime']


# print('savename:',savename)
# Read the data set into memory
input_data = pd.read_csv(dataset, sep=None, engine='python')

#header
with open(output_file,'w') as out:
    out.write('dataset\tmethod\ttrial\tmse\tr2\ttime\n')
print('dataset\tmethod\ttrial\tmse\tr2\ttime')

# data
sc_y = StandardScaler()
if problem in scale_these:
    ops = 'n,v,+,-,*,/,sin,cos,exp,log'
    ops_w = '6,6,1,1,1,1,0.5,0.5,0.5,0.5'
    X = StandardScaler().fit_transform(input_data.drop('label', axis=1).values.astype(float))
    y = sc_y.fit_transform(input_data['label'].values.reshape(-1,1))
else:
    ops = 'n,v,+,-,*,/,sin,cos'
    ops_w = '6,6,1,1,1,1,0.5,0.5'
    X = input_data.drop('label', axis=1).values.astype(float)
    y = input_data['label'].values

sss = ShuffleSplit(n_splits=50,train_size=0.7,test_size=0.3,random_state=63)

# fit estimator function
def fit_est(i,train,test):

    # Create the pipeline for the model
    est = ellyn(g=1000,popsize=1000,selection='rand',
                            fit_type='MSE',max_len=50,
                            islands=False,
                            num_islands=20,
                            island_gens=100,
                            verbosity=0, print_novelty=True,
                            print_data=True,savename=savename+'_trial_'+str(i),
                            resultspath='/home/lacava/results/ep-lex-benchmark/trials/rand/',
                            elitism=True,
                            ops=ops,
                            ops_w=ops_w, pHC_on=True)

    #fit model
    # pdb.set_trace()
    t0 = time.time()
    est.fit(X[train],y[train])
    #get fit time
    runtime = time.time()-t0
    # print("training done")
    # pdb.set_trace()
    # predict on test set

    if problem in scale_these:
        test_mse = mean_squared_error(sc_y.inverse_transform(est.predict(X[test])),
                                      sc_y.inverse_transform(y[test]))
        test_r2 = r2_score(sc_y.inverse_transform(est.predict(X[test])),
                                    sc_y.inverse_transform(y[test]))
    else:
        test_mse = mean_squared_error(est.predict(X[test]),y[test])
        test_r2 = r2_score(est.predict(X[test]),y[test])

    # print results
    out_text = '\t'.join([dataset.split('/')[-1][2:-4],
                          'rand',
                          str(i),
                          str(test_mse),
                          str(test_r2),
                          str(runtime)])

    with open(output_file,'a') as out:
        out.write(out_text+'\n')
    print(out_text)
    sys.stdout.flush()
# run parallel trials
Parallel(n_jobs=50)(delayed(fit_est)(i,train,test) for i,(train,test) in enumerate(sss.split(X,y)))
