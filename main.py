from utils import *
#import argparse
import numpy as np
import pandas as pd

import os
import shutil

args = parse_args()
print(args)
print(args.learningrate)
train_params = dict()
train_params['lr'] = args.learningrate
train_params['epochs'] = args.epochs
train_params['batch_size'] = args.batchsize

data = args.input
labels = np.loadtxt(args.labels, dtype=np.float, delimiter=',')
out_dir = args.outputdir
random_state=args.randomstate

data_array, data_df = seq_2_array(data, data_format='fasta')

print(data_df)
print(data_array)
print(type(data_array))
labels = np.random.random((data_array.shape[0],))*2//1
models = train_model(x_train=data_array, y_train=labels, train_params=train_params, weight_dir=args.weightdir, log_dir=args.logdir, out_dir=out_dir, random_state=random_state)
print(models)

#y_pred = predict(models=models, data=data_array)
y_pred = save_predict(models=models, data=data_array, out_dir=out_dir)
metric_score = get_prediction_score(y_true=labels, y_pred=y_pred)

print(y_pred)
print(metric_score)

