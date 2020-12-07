from utils import *
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

#training data
data = np.loadtxt(args.input, dtype=np.float, delimiter=',')
labels = np.loadtxt(args.labels, dtype=np.float, delimiter=',')
out_dir = args.outputdir
random_state=args.randomstate

#test data
test_data = np.loadtxt('dataset/test_data.csv', dtype=np.float, delimiter=',')
test_labels = np.loadtxt('dataset/test_labels.csv', dtype=np.float, delimiter=',')
#data_array, data_df = seq_2_array(data, data_format='fasta')

models = train_model(x_train=data, y_train=labels, train_params=train_params, weight_dir=args.weightdir, log_dir=args.logdir, out_dir=out_dir, random_state=random_state)
print(models)

#y_pred = predict(models=models, data=data_array)
y_pred = save_predict(models=models, data=test_data, out_dir=out_dir)
metric_score = get_prediction_score(y_true=test_labels, y_pred=y_pred)

print(y_pred)
print(metric_score)

