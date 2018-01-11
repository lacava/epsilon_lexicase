import sys
from ellyn import ellyn
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoLarsCV

import itertools
import pdb
import time

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
    X = StandardScaler().fit_transform(input_data.drop('label', axis=1).values.astype(float))
    y = sc_y.fit_transform(input_data['label'].values.reshape(-1,1))
else:
    X = input_data.drop('label', axis=1).values.astype(float)
    y = input_data['label'].values

sss = ShuffleSplit(n_splits=50,train_size=0.7,test_size=0.3,random_state=63)

for i,(train,test) in enumerate(sss.split(X,y)):

    # Create the pipeline for the model
    est = LassoLarsCV()

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
                          'lasso',
                          str(i),
                          str(test_mse),
                          str(test_r2),
                          str(runtime)])

    with open(output_file,'a') as out:
        out.write(out_text+'\n')
    print(out_text)
    sys.stdout.flush()
