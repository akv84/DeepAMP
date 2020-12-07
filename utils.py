import numpy as np
import pandas as pd
from sys import argv
import argparse
import csv
import os
import shutil
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from Bio import SeqIO
import Bio

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Embedding, MaxPool1D, Flatten, ZeroPadding1D, Dropout, Concatenate
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, CSVLogger

from sklearn.metrics import accuracy_score, f1_score, fbeta_score, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, auc, roc_auc_score
from imblearn.metrics import sensitivity_score, specificity_score

def parse_args():
    parser = argparse.ArgumentParser(description='DeepAMP-Antimicrobial Peptides Prediction')
    parser.add_argument('input', type=str,
			help='Input data is in fasta/csv format')
    parser.add_argument('labels', type=str,
			help='Input labeles is csv format')
    parser.add_argument('outputdir', type=str, 
			help='The path of the output directory for prediction results')
    parser.add_argument('-w', '--weightdir', type=str, default='./', 
			help='The path of directory to save weights')
    parser.add_argument('-l', '--logdir', type=str, default='./', 
			help='The path of directory to save training logs')
    # training params
    parser.add_argument('--randomstate', type=int, default=0,
            help="random state (default:0)")
    parser.add_argument('-e', '--epochs', type=int, default=500,
            help="Number of training epochs (default:500)")
    parser.add_argument('-b', '--batchsize', type=int, default=64,
            help="Batch size (default:32)")
    parser.add_argument('-d', '--dropoutrate', type=float, default=0.2,
            help="Dropout rate (default: 0)")
    parser.add_argument('-r', '--learningrate', type=float, default=0.01,
            help="Learning rate (default: 0.001)")

    return parser.parse_args()

def get_prediction_score(y_true=None, y_pred=None):
	result_score_dict = dict()
	result_score_dict['accuracy_score'] = accuracy_score(y_true, np.round(y_pred))
	result_score_dict['roc_auc_score'] = roc_auc_score(y_true, y_pred)
	result_score_dict['accuracy_score'] = accuracy_score(y_true, np.round(y_pred))
	result_score_dict['matthews_corrcoef'] = matthews_corrcoef(y_true, np.round(y_pred))
	result_score_dict['precision_score'] = precision_score(y_true, np.round(y_pred))
	result_score_dict['sensitivity_score'] = sensitivity_score(y_true, np.round(y_pred))
	
	return result_score_dict

def predict(models=None, data=None):
	if isinstance(models, list) == False:
		return
	pred = 0
	for model in models:
		pred = pred + model.predict(data)/len(models)
	return pred

def save_predict(models=None, data=None, out_dir=None):
	if os.path.isdir(out_dir) == False:
		out_dir = './'
		return
	pred = 0
	for model in models:
		pred = pred + model.predict(data)/len(models)
	np.savetxt(os.path.join(out_dir,'prediction.csv'), pred, fmt='%s', delimiter=',')
	return pred

def create_model(lr=None, input_data=None):
    if lr==None:
        lr=0.01
    inp_layer = Input(shape=input_data.shape[1:])
    embedding = Embedding(input_data.shape[1], 21, input_length=input_data.shape[1])(inp_layer)
    cnn_layer_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='valid', activation='relu')(embedding)
    dropout_layer_1 = Dropout(rate=0.2)(cnn_layer_1)
    maxpool_layer_1 = MaxPool1D(pool_size=5, strides=2, padding='valid')(dropout_layer_1)
    cnn_layer_2 = Conv1D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu')(maxpool_layer_1)
    dropout_layer_2 = Dropout(rate=0.2)(cnn_layer_2)
    maxpool_layer_2 = MaxPool1D(pool_size=5, strides=2, padding='valid')(dropout_layer_2)
    
    flatten_layer = Flatten()(maxpool_layer_2)
    dense_cnn_1 = Dense(64, activation='relu')(flatten_layer)
    dropout_layer_3 = Dropout(rate=0.2)(dense_cnn_1)
    
    out_layer = Dense(1, activation='sigmoid')(dropout_layer_3)
    model = Model(inputs=inp_layer, outputs=out_layer)
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=["accuracy"])
    model.summary()
    
    return model

def callbacks_fun(weight_file=None, log_file=None):
	call_back_list = list()
	if isinstance(weight_file, str):
		model_check_point = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True, save_weights_only=True)
		call_back_list.append(model_check_point)
	#early_stop = EarlyStopping(patience=25, restore_best_weights=True)
	#call_back_list.append(early_stop)
	if isinstance(log_file, str):
		csv_log = CSVLogger(filename=log_file)
		call_back_list.append(csv_log)
	#lr_schedular = ReduceLROnPlateau(factor=0.1, patience=10, verbose=1, min_lr=0.0000001)
	#call_back_list.append(lr_schedular)
	
	return call_back_list

def get_params(lr=None, batch_size=None, epochs=None):
	train_params = dict()
	train_params['lr'] = 0.01
	train_params['epochs'] = 500
	train_params['batch_size'] = 64
	if lr != None:
		train_params['lr'] = lr
	if lr != None:
		train_params['batch_size'] = batch_size
	if lr != None:
		train_params['epochs'] = epochs
	
	return train_params
	
def train_model(x_train=None, y_train=None, train_params=None, weight_dir=None, log_dir=None, out_dir=None, random_state=0):
	
	if isinstance(out_dir, str):
		if os.path.isdir(out_dir) == False:
			os.mkdir(out_dir)
	else:
		out_dir = './'

	if isinstance(weight_dir, str):
		weight_dir = os.path.join(out_dir, weight_dir)
		if os.path.isdir(weight_dir) == False:
			os.mkdir(weight_dir)
	else:
		weight_dir = out_dir
	
	if isinstance(log_dir, str):
		log_dir = os.path.join(out_dir, log_dir)
		if os.path.isdir(log_dir) == False:
			os.mkdir(log_dir)
	else:
		log_dir = out_dir
		
	if isinstance(train_params, dict) == False:
		return
		
	n_folds = 10
	kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
	models = list()
	for fold, (train_index, test_index) in enumerate(kf.split(x_train)):
		
		np.random.seed(random_state)
		
		model = create_model(lr=train_params['lr'], input_data=x_train)
		
		weight_file = os.path.join(weight_dir, 'weights_for_fold'+str(fold)+'.h5')
		
		log_file = os.path.join(log_dir, 'log_for_fold'+str(fold)+'.csv')
		
		call_back_list = callbacks_fun(weight_file=weight_file, log_file=log_file)
		
		x_tr, x_te = x_train[train_index], x_train[test_index]
		y_tr, y_te = y_train[train_index], y_train[test_index]
		
		model.fit(x_tr, y_tr,
			batch_size=train_params['batch_size'],
			epochs=train_params['epochs'],
			validation_data=(x_te, y_te),
			shuffle=True,
			callbacks=call_back_list)
		
		models.append(model)
	
	return models

def seq_2_array(data_file_path=None, max_len=47, data_format=None):
	
	amino_acid_list = ['A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z']
	amino_acid_dict = dict()
	for s in amino_acid_list:
		amino_acid_dict[s] = amino_acid_list.index(s) + 1
	
	df = pd.DataFrame(columns=['Seq_ID','Seq','Seq_len'])
	if data_format=='fasta':
		id_list = list()
		seq_list = list()
		seq_len_list = list()
		warning_list = list()
		for seq_record in SeqIO.parse(data_file_path, "fasta"):
			seq = seq_record.seq.upper()
			id_list.append(seq_record.id)
			seq_list.append(''.join(seq))
			seq_len_list.append(len(seq_record))
			if len(set(list(seq)).difference(amino_acid_dict)) > 0:
				warning_list.append('Sequence contains invalid symbol')
		df['Seq_ID'] = id_list
		df['Seq'] = seq_list
		df['Seq_len'] = seq_len_list
	else:
		id_list = list()
		seq_list = list()
		seq_len_list = list()
		warning_list = list()
		f = open(data_file_path,'r')
		for seq_record in f:
			id_list.append(None)
			seq = seq_record.strip().upper()
			seq_list.append(seq)
			seq_len_list.append(len(seq))
			if len(set(list(seq)).difference(amino_acid_dict)) > 0:
				warning_list.append('Sequence contains invalid symbol')
		f.close()
		df['Seq_ID'] = id_list
		df['Seq'] = seq_list
		df['Seq_len'] = seq_len_list
	Seq_list = [seq if (len(seq) <= max_len) else seq[:max_len] for seq in df['Seq']]
	Seq_list = [[amino_acid_dict[s] for s in list(seq)] for seq in Seq_list]
	Seq_zero_pad = np.array([s+[0]*(max_len-len(s)) for s in Seq_list], dtype=np.int)    
	
	return Seq_zero_pad, df

if __name__=='__main__':
	
	args = parse_args()
	print(args)
	print(args.learningrate)
	train_params = dict()
	train_params['lr'] = args.learningrate
	train_params['epochs'] = args.epochs
	train_params['batch_size'] = args.batchsize
	
	data = args.input
	
	data = seq_2_array(data, data_format=None)
