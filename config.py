

MODEL_FOLDER = "saved_models"
RESULT_FOLDER = "results"
DATA_FOLDER = "data"
BATCH_SIZE = 50
EPOCHS = 2
FOLDS = 5
PRETRAINED = True

EXP_SETTING = {"lorenz":{"file": "lorenz_0.05", "sub": 0.05, "total_steps": 58000, "n_training":8500, "burn_steps":8000, "delta-t":0.05, "window": 1500, "stride": 5, "input_steps":100,
                        "title":"$Lorenz$" , "inference_steps": 200, "gamma": 3.1065, "zeta": 0.228, "mu":0.1105, "y_lim":{"ett":[(0,10),(0,0.73),(0,0.36)],"hf":[(0,9.2),(0,0.72),(0,0.35)],"ett_hf":[(0,9.2),(0,0.72),(0,0.35)],"benchmark":[(0,11),(0,0.82),(0,0.42)],"all":[(0,11),(0,0.82),(0,0.42)],"transformers":[(0,11),(0,0.82),(0,0.42)],"recurrent":[(0,11),(0,0.82),(0,0.42)]}, 
},
"accelerometer":{"file":"accelerometer_subject3.csv.gz", "sub": 3, "n_training":5500, "window": 400, "stride": 2, "input_steps":100,
                        "title":"$Accelerometer$", "inference_steps": 300, "gamma": 0.2935, "zeta": 0.3635, "mu":0.182, "y_lim":{"ett":[(0,0.5),(0,0.6),(0,0.3)],"hf":[(0,0.38),(0,0.46),(0,0.23)],"ett_hf":[(0,0.38),(0,0.46),(0,0.23)],"benchmark":[(0,0.84),(0,0.8),(0,0.43)],"all":[(0,0.84),(0,0.84),(0,0.43)],"transformers":[(0,0.84),(0,0.8),(0,0.43)],"recurrent":[(0,0.84),(0,0.8),(0,0.43)]}, 
},
"gait_force":{"file":"gait_force_patient1_speed2.csv.gz","sub": 2, "n_training":5600, "window": 400, "stride": 9, "input_steps":100,
                        "title":"$Gait$ $Force$", "inference_steps": 300, "gamma": 158.6875, "zeta": 0.501, "mu":0.2565, "y_lim":{"ett":[(0,320),(0,1),(0,0.65)],"hf":[(0,290),(0,0.92),(0,0.58)],"ett_hf":[(0,290),(0,0.92),(0,0.58)],"benchmark":[(0,350.7),(0,1.1),(0,1)],"all":[(0,310.7),(0,1.1),(0,1.1)],"transformers":[(0,350.7),(0,1.1),(0,1)],"recurrent":[(0,350.7),(0,1.1),(0,1)]}, 
},
"roaming_worm":{"file":"roaming_worm2.csv.gz","sub": "", "n_training":5600, "window": 400, "stride": 4, "input_steps":100,
                        "title":"$Roaming$ $Worm$",  "inference_steps": 160, "gamma": 2.265, "zeta": 0.8575, "mu":0.4615, "y_lim":{"ett":[(0,4.2),(0,1.7),(0,0.7)],"hf":[(0,3.3),(0,1.23),(0,0.65)],"ett_hf":[(0,3.3),(0,1.23),(0,0.65)],"benchmark":[(0,5.8),(0,2.5),(0,0.8)],"all":[(0,6.5),(0,2.5),(0,0.85)],"transformers":[(0,5.8),(0,2.5),(0,0.8)],"recurrent":[(0,5.8),(0,2.5),(0,0.8)]},
},
"electricity":{"file":"electricity_train_test.csv.gz","sub": "", "n_training":5000, "window": 400, "stride": 23, "input_steps":100,
                        "title":"$Electricity$", "inference_steps": 300, "gamma": 121.54, "zeta": 0.219, "mu":0.0595, "y_lim":{"ett":[(0,300),(0,0.50),(0,0.22)],"hf":[(0,211),(0,0.38),(0,0.16)],"ett_hf":[(0,211),(0,0.38),(0,0.16)],"benchmark":[(0,301),(0,0.55),(0,0.42)],"all":[(0,305),(0,0.55),(0,0.42)],"transformers":[(0,301),(0,0.55),(0,0.42)],"recurrent":[(0,301),(0,0.55),(0,0.42)]},
},
}


ESTIMATORS = {
            "GRU_baseline":{"lr":-2},
            "MT":{"num_head":1, "num_encoder":1, "lr":-2},
            "GRU_ss":{"decay":"is", "lr":-2},
            # "GRU_ss":{"decay":"exp", "lr":-2},
            # "GRU_ss":{"decay":"linear", "lr":-2},
            "GRU_hf_5":{"k":5, "lr":-4},
            "GRU_hf_10":{"k":10, "lr":-5},
            "GRU_hf_15":{"k":15, "lr":-6},
            "GRU_hf_20":{"k":20, "lr":-7},
            "GRU_k_20":{"k":20, "lr":-2},
            }


COLOR_BANK = {"BLS":"#8C8C8C","ELM":"#8C564B","MTF":"#00A86B","Teacher Forcing":"#E377C2","SSFS":"#1F77B4", "SSES":"#17BECF",  "HF5":"#BCBD22",
"HF10":"#FFC400","HF15":"#FF7F0E","HF20":"#D62728", "IF":"#E377C2", "AF":"#B5338A","ETT Direct":"#B5338A"}


def get_dataset_name(system):
    setting = EXP_SETTING[system]
    return f"{system}/{setting['sub']}" if {setting["sub"]} else f"{system}"

def get_model_name(dataset_name, fold, normalization, x_y_lag):
    if PRETRAINED:
        return f"pre_trained/{dataset_name}/fold_{fold}/{normalization}/{x_y_lag}"
    return  f"{dataset_name}/fold_{fold}/{normalization}/{x_y_lag}"