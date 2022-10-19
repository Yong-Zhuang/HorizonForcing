'''
Created on Sat Jan 28 2020
Updated on Oct 15 2022
@author: Yong Zhuang
'''
import numpy as np
import pandas as pd
import os
import argparse
import config
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from model.nns import *
from pickle import dump



def generate_train_test_indices(training_samples, x_steps, dataset_name):
    kfold = KFold(n_splits=config.FOLDS, shuffle=True, random_state=3)
    total_steps = x_steps+1
    train_data = training_samples[:,:total_steps]
    fold_idx = 0
    indices_folder = f"{config.DATA_FOLDER}/{dataset_name}/indices"
    if not os.path.exists(indices_folder):
        os.makedirs(indices_folder)
    for train_index, val_index in kfold.split(train_data[:,:x_steps,0], train_data[:,x_steps,0]):
        path = os.path.join(indices_folder,f'{config.FOLDS}fold.{fold_idx}_train_index')
        np.save(path, train_index) 
        path = os.path.join(indices_folder,f'{config.FOLDS}fold.{fold_idx}_val_index')
        np.save(path, val_index) 
        fold_idx +=1

def training(training_samples, x_steps, dataset_name,fold_idx,x_y_lag,norm):
    total_steps = x_steps+x_y_lag
    train_data = training_samples[:,:total_steps]
   

    #for fold_idx in [1]:
    train_index = np.load(os.path.join(f"{config.DATA_FOLDER}/{dataset_name}/indices",f'{config.FOLDS}fold.{fold_idx}_train_index.npy'))
    val_index = np.load(os.path.join(f"{config.DATA_FOLDER}/{dataset_name}/indices",f'{config.FOLDS}fold.{fold_idx}_val_index.npy'))

    #name_prefix = prefix+f"_xsteps.{x_steps}_fold.{fold_idx}"
    saved_folder = f"{config.MODEL_FOLDER}/{dataset_name}/fold_{fold_idx}/{norm}/{x_y_lag}"
    X_train, y_train = train_data[train_index][:,:-x_y_lag], train_data[train_index][:,x_y_lag:]
    X_val, y_val = train_data[val_index][:,:-x_y_lag], train_data[val_index][:,x_y_lag:] 
    pred_dim = training_samples.shape[2] 
    print (f"==Start the training")        
    for estimator_name, setting in config.ESTIMATORS.items():
        train_X, val_X, train_Y, val_Y = [X_train], [X_val], [y_train], [y_val]
        lr =  pow(10,setting["lr"])
        if estimator_name == f"GRU_baseline": 
            estimator = TeacherForcing(pred_dim, saved_folder,x_y_lag=x_y_lag)
        elif estimator_name == "GRU_k_20":
            estimator = HorizonForcing(pred_dim, saved_folder,x_y_lag=x_y_lag, k=setting["k"], lr=lr)
            for j in np.arange(setting["k"]):
                train_Y = train_Y + [y_train[:,1+j:]] 
                val_Y = val_Y + [y_val[:,1+j:]] 
            train_Y = [train_Y[-1]]
            val_Y = [val_Y[-1]]
        elif  estimator_name == f"GRU_ss": 
            estimator = DiscontinuousScheduledSampling(pred_dim, saved_folder,x_y_lag=x_y_lag,ts = x_steps, batch_size = config.BATCH_SIZE, decay=setting["decay"])
        elif "GRU_hf" in estimator_name:
            if estimator_name == "GRU_hf_5":
                base_model = TeacherForcing(pred_dim, saved_folder,x_y_lag=x_y_lag)
                base_model.load_weights()
                gru = base_model.gru
                output_dense = base_model.output_dense
            elif estimator_name == "GRU_hf_10":
                base_model = HorizonForcing(pred_dim, saved_folder,x_y_lag=x_y_lag, k=5, lr=lr)
                base_model.load_weights()
                gru = base_model.time_gru.gru
                output_dense = base_model.time_gru.output_dense
            elif estimator_name == "GRU_hf_15":
                base_model = HorizonForcing(pred_dim, saved_folder,x_y_lag=x_y_lag, k=10, lr=lr)
                base_model.load_weights()
                gru = base_model.time_gru.gru
                output_dense = base_model.time_gru.output_dense
            elif estimator_name == "GRU_hf_20":
                base_model = HorizonForcing(pred_dim, saved_folder,x_y_lag=x_y_lag, k=15, lr=lr)
                base_model.load_weights()
                gru = base_model.time_gru.gru
                output_dense = base_model.time_gru.output_dense
            W_gru = gru.get_weights()
            W_output_dense = output_dense.get_weights()  
            estimator = HorizonForcing(pred_dim, saved_folder,x_y_lag=x_y_lag, k=setting["k"], lr=lr)
            estimator.time_gru.gru.set_weights(W_gru)
            estimator.time_gru.output_dense.set_weights(W_output_dense) 
            for j in np.arange(setting["k"]):
                train_Y = train_Y + [y_train[:,1+j:]] 
                val_Y = val_Y + [y_val[:,1+j:]] 
            train_Y = [train_Y[-1]]
            val_Y = [val_Y[-1]]
        else:
            estimator =  MusicTransformer(
                pred_dim, saved_folder,x_y_lag=x_y_lag, latent_dim=setting["num_head"]*64, 
                num_layer=setting["num_encoder"], max_seq=x_steps)
        _, history = estimator.training(train_X, train_Y, val_X, val_Y, config.BATCH_SIZE, config.EPOCHS)
        hist_df = pd.DataFrame(history.history) 
        hist_csv_file = f'{saved_folder}/{estimator.model_name}.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
parser = argparse.ArgumentParser (description='horizon_forcing')
parser.add_argument('-gpu', metavar='int', type=str, default='0', required=True, help='gpu index, using which GPU')
parser.add_argument(
    "-s",
    "--system",
    type=str.lower,
    choices=["lorenz", "accelerometer","gait_force", "roaming_worm","electricity"],
    default="lorenz",
    required=True, 
    help="System to build data for."
)
parser.add_argument('-norm', dest='norm', action='store_true')
parser.add_argument('-no-norm', dest='norm', action='store_false')
parser.set_defaults(norm=False)
parser.add_argument('-mode', metavar='int', type=int, default=0, help='0: train; 1: generate indices')
parser.add_argument('-fold', metavar='int', type=int, default=1, help='fold idx: 0,1,2,3,or 4')
parser.add_argument('-lag', metavar='int', type=int, default=1, help='a lag time steps between input and output, value should be less or equal than the input steps xs')

if __name__== "__main__": 
    args = parser.parse_args()  
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu  
    # Collect arguments
    setting = config.EXP_SETTING[args.system] 
    dataset_name = config.get_dataset_name(args.system)
    x_steps = setting["input_steps"]

    if config.PRETRAINED:
        train_data = np.load(f"{config.DATA_FOLDER}/pre_generated/{dataset_name}/train.npy")
    else:
        train_data = np.load(f"{config.DATA_FOLDER}/{dataset_name}/train.npy")
   

    if args.norm:   
        train_shape = train_data.shape
        n_variables = train_shape[-1]             
        train_reshaped = train_data.reshape((-1,n_variables))
        scaler = StandardScaler()
        scaler.fit(train_reshaped)
        train_data = scaler.transform(train_reshaped).reshape(train_shape)
        dump(scaler, open(f'{config.DATA_FOLDER}/{dataset_name}/stanard_scaler.pkl', 'wb'))
    if args.mode == 0:
        training(train_data, x_steps,dataset_name, args.fold,args.lag,args.norm)
    else:
        generate_train_test_indices(train_data, x_steps, dataset_name)
 