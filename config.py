PRETRAINED = False
DATA_FOLDER = "data"
MODEL_FOLDER = "saved_models"
RESULT_FOLDER = "results"
BATCH_SIZE = 50
EPOCHS = 150
FOLDS = 5

EXP_SETTING = {
    # -s lorenz_s -ns 58000 -bs 8000 -dt 0.01 -n 8500   ; 1201
    # -s lorenz_s -ns 28000 -bs 8000 -dt 0.05 -n 8500   ; 1201
    "lorenz": {
        "0.05": {
            "file": "lorenz_0.05",
            "norm": False,
            "total_steps": 58000,
            "n_training": 8500,
            "burn_steps": 8000,
            "delta-t": 0.05,
            "window": 1500,
            "stride": 5,
            "input_steps": 100,
            "title": "$Lorenz$",
            "inference_steps": 200,
            "gamma": 3.1065,
            "zeta": 0.228,
            "mu": 0.1105,
            "y_lim": {
                "ett": [(0, 10), (0, 0.73), (0, 0.36)],
                "hf": [(0, 9.2), (0, 0.72), (0, 0.35)],
                "ett_hf": [(0, 9.2), (0, 0.72), (0, 0.35)],
                "benchmark": [(0, 11), (0, 0.82), (0, 0.42)],
                "all": [(0, 11), (0, 0.82), (0, 0.42)],
                "transformers": [(0, 11), (0, 0.82), (0, 0.42)],
                "recurrent": [(0, 11), (0, 0.82), (0, 0.42)],
                "no-auto-informer": [(0, 11), (0, 0.82), (0, 0.42)],
            },
        }
    },
    # -s rossler_s -ns 368000 -bs 8000 -dt 0.01 -n 7500   ;  1401
    "rossler": {
        "0.05": {
            "file": "rossler_0.05",
            "norm": False,
            "total_steps": 368000,
            "n_training": 7500,
            "burn_steps": 8000,
            "delta-t": 0.05,
            "window": 3000,
            "stride": 5,
            "input_steps": 100,
            "title": "$Lorenz$",
            "inference_steps": 1300,
            "gamma": 3.1065,
            "zeta": 0.228,
            "mu": 0.1105,
            "y_lim": {
                "ett": [(0, 10), (0, 0.73), (0, 0.36)],
                "hf": [(0, 9.2), (0, 0.72), (0, 0.35)],
                "ett_hf": [(0, 9.2), (0, 0.72), (0, 0.35)],
                "benchmark": [(0, 11), (0, 0.82), (0, 0.42)],
                "all": [(0, 11), (0, 0.82), (0, 0.42)],
                "transformers": [(0, 11), (0, 0.82), (0, 0.42)],
                "recurrent": [(0, 11), (0, 0.82), (0, 0.42)],
                "no-auto-informer": [(0, 11), (0, 0.82), (0, 0.42)],
            },
        }
    },
    # -s accelerometer -sub 3 -n 5500 -mt multiple; 13000
    # -sub 1: stride 4; -sub 3: stride 2
    "accelerometer": {
        "1": {  # training samples: (5500, 400, 3); test samples: (901, 400, 3)
            "file": "accelerometer_subject1.csv.gz",
            "norm": True,
            "n_training": 5500,
            "window": 400,
            "stride": 4,
            "input_steps": 100,
            "title": "$Accelerometer_sub1$",
            "inference_steps": 300,
            "gamma": 0.2935,
            "zeta": 0.3635,
            "mu": 0.182,
            "y_lim": {
                "ett": [(0, 0.5), (0, 0.6), (0, 0.3)],
                "hf": [(0, 0.38), (0, 0.46), (0, 0.23)],
                "ett_hf": [(0, 0.38), (0, 0.46), (0, 0.23)],
                "benchmark": [(0, 0.84), (0, 0.8), (0, 0.43)],
                "all": [(0, 0.84), (0, 0.84), (0, 0.43)],
                "transformers": [(0, 0.84), (0, 0.8), (0, 0.43)],
                "recurrent": [(0, 0.84), (0, 0.8), (0, 0.43)],
                "no-auto-informer": [(0, 0.84), (0, 0.84), (0, 0.43)],
            },
        },
        "3": {  # training samples: (5500, 400, 3); test samples: (801, 400, 3)
            "file": "accelerometer_subject3.csv.gz",
            "norm": True,
            "n_training": 5500,
            "window": 400,
            "stride": 2,
            "input_steps": 100,
            "title": "$Accelerometer_sub3$",
            "inference_steps": 300,
            "gamma": 0.2935,
            "zeta": 0.3635,
            "mu": 0.182,
            "y_lim": {
                "ett": [(0, 0.5), (0, 0.6), (0, 0.3)],
                "hf": [(0, 0.38), (0, 0.46), (0, 0.23)],
                "ett_hf": [(0, 0.38), (0, 0.46), (0, 0.23)],
                "benchmark": [(0, 0.84), (0, 0.8), (0, 0.43)],
                "all": [(0, 0.84), (0, 0.84), (0, 0.43)],
                "transformers": [(0, 0.84), (0, 0.8), (0, 0.43)],
                "recurrent": [(0, 0.84), (0, 0.8), (0, 0.43)],
                "no-auto-informer": [(0, 0.84), (0, 0.84), (0, 0.43)],
            },
        },
    },
    # -s dwelling_worm -sub 2 -n 5600 -mt multiple; 26998, 5
    "dwelling_worm": {
        "1": {
            # training samples: (5600, 400, 5); test samples: (1050, 400, 5)
            "file": "dwelling_worm1.csv.gz",
            "norm": False,
            "n_training": 5600,
            "window": 400,
            "stride": 4,
            "input_steps": 100,
            "title": "$Dwelling$ $Worm$ sub1",
            "inference_steps": 160,
            "gamma": 2.265,
            "zeta": 0.8575,
            "mu": 0.4615,
            "y_lim": {
                "ett": [(0, 4.2), (0, 1.7), (0, 0.9)],
                "hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "ett_hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "benchmark": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "all": [(0, 6.5), (0, 2.5), (0, 0.85)],
                "transformers": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "recurrent": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "no-auto-informer": [(0, 6.5), (0, 2.5), (0, 0.85)],
            },
        },
        "2": {
            # training samples: (5600, 400, 5); test samples: (1050, 400, 5)
            "file": "dwelling_worm2.csv.gz",
            "norm": False,
            "n_training": 5600,
            "window": 400,
            "stride": 4,
            "input_steps": 100,
            "title": "$Dwelling$ $Worm$ sub2",
            "inference_steps": 160,
            "gamma": 2.265,
            "zeta": 0.8575,
            "mu": 0.4615,
            "y_lim": {
                "ett": [(0, 4.2), (0, 1.7), (0, 0.9)],
                "hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "ett_hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "benchmark": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "all": [(0, 6.5), (0, 2.5), (0, 0.85)],
                "transformers": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "recurrent": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "no-auto-informer": [(0, 6.5), (0, 2.5), (0, 0.85)],
            },
        },
    },
    # -s ecg -sub 1 -n 5500 -mt multiple; 67499, 1
    "ecg": {  #  training samples: (6710, 400, 1); test samples: (6710, 400, 1)
        "default": {
            "file": {"test": "ecg_test.csv.gz", "train": "ecg_train.csv.gz"},
            "norm": False,
            "n_training": 5500,
            "window": 400,
            "stride": 10,
            "input_steps": 100,
            "title": "$ECG_1$",
            "inference_steps": 160,
            "gamma": 2.265,
            "zeta": 0.8575,
            "mu": 0.4615,
            "y_lim": {
                "ett": [(0, 4.2), (0, 1.7), (0, 0.9)],
                "hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "ett_hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "benchmark": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "all": [(0, 6.5), (0, 2.5), (0, 0.85)],
                "transformers": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "recurrent": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "no-auto-informer": [(0, 6.5), (0, 2.5), (0, 0.85)],
            },
        },
    },
    # -s ecosystem -n 5200 -mt multiple; 19000, 9
    "ecosystem": {  # training samples: (5200, 400, 9); test samples: (1001, 400, 9)
        "default": {
            "file": "ecosystem.csv.gz",
            "norm": True,
            "n_training": 5200,
            "window": 400,
            "stride": 3,
            "input_steps": 100,
            "title": "$Ecosystem$",
            "inference_steps": 150,
            "gamma": 1.243,
            "zeta": 0.066,
            "mu": 0.305,
            "y_lim": {
                "ett": [(0, 4.2), (0, 1.7), (0, 0.9)],
                "hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "ett_hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "benchmark": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "all": [(0, 6.5), (0, 2.5), (0, 0.85)],
                "transformers": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "recurrent": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "no-auto-informer": [(0, 6.5), (0, 2.5), (0, 0.85)],
            },
        }
    },
    # -s electricity -n 5000 -mt multiple; 140255, 1
    "electricity": {  # training samples: (5000, 400, 1); test samples: (1081, 400, 1)
        "default": {
            "file": "electricity_train_test.csv.gz",
            "norm": False,
            "n_training": 5000,
            "window": 400,
            "stride": 23,
            "input_steps": 100,
            "title": "$Electricity$",
            "inference_steps": 300,
            "gamma": 121.54,
            "zeta": 0.219,
            "mu": 0.0595,
            "y_lim": {
                "ett": [(0, 300), (0, 0.60), (0, 0.22)],
                "hf": [(0, 211), (0, 0.38), (0, 0.16)],
                "ett_hf": [(0, 211), (0, 0.38), (0, 0.16)],
                "benchmark": [(0, 301), (0, 0.55), (0, 0.42)],
                "all": [(0, 305), (0, 0.55), (0, 0.42)],
                "transformers": [(0, 301), (0, 0.55), (0, 0.42)],
                "recurrent": [(0, 301), (0, 0.55), (0, 0.42)],
                "no-auto-informer": [(0, 305), (0, 0.55), (0, 0.42)],
            },
        }
    },
    # -s gait_force -sub 2 -n 5600 -mt multiple; 60000,6
    "gait_force": {
        "1": {  # training samples: (5600, 400, 6); test samples: (1023, 400, 6)
            "file": "gait_force_patient1_speed2.csv.gz",
            "norm": True,
            "n_training": 5600,
            "window": 400,
            "stride": 9,
            "input_steps": 100,
            "title": "$Gait$ $Force$",
            "inference_steps": 300,
            "gamma": 158.6875,
            "zeta": 0.501,
            "mu": 0.2565,
            "y_lim": {
                "ett": [(0, 320), (0, 12), (0, 0.65)],
                "hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "ett_hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "benchmark": [(0, 350.7), (0, 1.1), (0, 1)],
                "all": [(0, 310.7), (0, 1.1), (0, 1.1)],
                "transformers": [(0, 350.7), (0, 1.1), (0, 1)],
                "recurrent": [(0, 350.7), (0, 1.1), (0, 1)],
                "no-auto-informer": [(0, 310.7), (0, 1.1), (0, 1.1)],
            },
        },
        "2": {  # training samples: (5600, 400, 6); test samples: (1023, 400, 6)
            "file": "gait_force_patient1_speed2.csv.gz",
            "norm": True,
            "n_training": 5600,
            "window": 400,
            "stride": 9,
            "input_steps": 100,
            "title": "$Gait$ $Force$",
            "inference_steps": 300,
            "gamma": 158.6875,
            "zeta": 0.501,
            "mu": 0.2565,
            "y_lim": {
                "ett": [(0, 320), (0, 12), (0, 0.65)],
                "hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "ett_hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "benchmark": [(0, 350.7), (0, 1.1), (0, 1)],
                "all": [(0, 310.7), (0, 1.1), (0, 1.1)],
                "transformers": [(0, 350.7), (0, 1.1), (0, 1)],
                "recurrent": [(0, 350.7), (0, 1.1), (0, 1)],
                "no-auto-informer": [(0, 310.7), (0, 1.1), (0, 1.1)],
            },
        },
    },
    # -s gait_trackers -sub 2 -n 5000 -mt multiple; 12000,24
    "gait_marker_tracker": {
        "1": {  # training samples: (5000, 400, 24); test samples: (801, 400, 24)
            "file": "gait_marker_trackers_patient1_speed1.csv.gz",
            "norm": True,
            "n_training": 5000,
            "window": 400,
            "stride": 2,
            "input_steps": 100,
            "title": "$Gait$ $Marker$ $Tracker$ sub 1",
            "inference_steps": 300,
            "gamma": 158.6875,
            "zeta": 0.501,
            "mu": 0.2565,
            "y_lim": {
                "ett": [(0, 320), (0, 12), (0, 0.65)],
                "hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "ett_hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "benchmark": [(0, 350.7), (0, 1.1), (0, 1)],
                "all": [(0, 310.7), (0, 1.1), (0, 1.1)],
                "transformers": [(0, 350.7), (0, 1.1), (0, 1)],
                "recurrent": [(0, 350.7), (0, 1.1), (0, 1)],
                "no-auto-informer": [(0, 310.7), (0, 1.1), (0, 1.1)],
            },
        },
        "2": {  # training samples: (5000, 400, 24); test samples: (801, 400, 24)
            "file": "gait_marker_trackers_patient1_speed2.csv.gz",
            "norm": True,
            "n_training": 5000,
            "window": 400,
            "stride": 2,
            "input_steps": 100,
            "title": "$Gait$ $Marker$ $Tracker$ sub 2",
            "inference_steps": 300,
            "gamma": 158.6875,
            "zeta": 0.501,
            "mu": 0.2565,
            "y_lim": {
                "ett": [(0, 320), (0, 12), (0, 0.65)],
                "hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "ett_hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "benchmark": [(0, 350.7), (0, 1.1), (0, 1)],
                "all": [(0, 310.7), (0, 1.1), (0, 1.1)],
                "transformers": [(0, 350.7), (0, 1.1), (0, 1)],
                "recurrent": [(0, 350.7), (0, 1.1), (0, 1)],
                "no-auto-informer": [(0, 310.7), (0, 1.1), (0, 1.1)],
            },
        },
    },
    # -s geyser -n 5000 -mt multiple; 140255, 1
    "geyser": {  # training samples: (5000, 400, 1); test samples: (1115, 400, 1)
        "default": {
            "file": "geyser_train_test.csv.gz",
            "norm": False,
            "n_training": 5000,
            "window": 400,
            "stride": 23,
            "input_steps": 100,
            "title": "$Old Faithful Geyser$",
            "inference_steps": 300,
            "gamma": 158.6875,
            "zeta": 0.501,
            "mu": 0.2565,
            "y_lim": {
                "ett": [(0, 320), (0, 12), (0, 0.65)],
                "hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "ett_hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "benchmark": [(0, 350.7), (0, 1.1), (0, 1)],
                "all": [(0, 310.7), (0, 1.1), (0, 1.1)],
                "transformers": [(0, 350.7), (0, 1.1), (0, 1)],
                "recurrent": [(0, 350.7), (0, 1.1), (0, 1)],
                "no-auto-informer": [(0, 310.7), (0, 1.1), (0, 1.1)],
            },
        }
    },
    # -s mouse -n 5400 -mt multiple; 45726, 1
    "mouse": {  # training samples: (5400, 400, 1); test samples: (1076, 400, 1)
        "default": {
            "file": "mouse.csv.gz",
            "norm": False,
            "n_training": 5400,
            "window": 400,
            "stride": 7,
            "input_steps": 100,
            "title": "$Mouse$",
            "inference_steps": 300,
            "gamma": 158.6875,
            "zeta": 0.501,
            "mu": 0.2565,
            "y_lim": {
                "ett": [(0, 320), (0, 12), (0, 0.65)],
                "hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "ett_hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "benchmark": [(0, 350.7), (0, 1.1), (0, 1)],
                "all": [(0, 310.7), (0, 1.1), (0, 1.1)],
                "transformers": [(0, 350.7), (0, 1.1), (0, 1)],
                "recurrent": [(0, 350.7), (0, 1.1), (0, 1)],
                "no-auto-informer": [(0, 310.7), (0, 1.1), (0, 1.1)],
            },
        }
    },
    # -s pendulum -n 5000 -mt multiple; 5927, 3
    "pendulum": {  # training samples: (5528, 400, 3); test samples: (5584, 400, 3)
        "default": {
            "file": {"train": "pendulum_train.csv.gz", "test": "pendulum_test.csv.gz"},
            "norm": True,
            "n_training": 5000,
            "window": 400,
            "stride": 1,
            "input_steps": 100,
            "title": "$Pendulum_1$",
            "inference_steps": 300,
            "gamma": 158.6875,
            "zeta": 0.501,
            "mu": 0.2565,
            "y_lim": {
                "ett": [(0, 320), (0, 12), (0, 0.65)],
                "hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "ett_hf": [(0, 290), (0, 0.92), (0, 0.58)],
                "benchmark": [(0, 350.7), (0, 1.1), (0, 1)],
                "all": [(0, 310.7), (0, 1.1), (0, 1.1)],
                "transformers": [(0, 350.7), (0, 1.1), (0, 1)],
                "recurrent": [(0, 350.7), (0, 1.1), (0, 1)],
                "no-auto-informer": [(0, 310.7), (0, 1.1), (0, 1.1)],
            },
        },
    },
    # -s roaming_worm -n 5600 -mt multiple; 26995, 5
    "roaming_worm": {  # training samples: (5600, 400, 5); test samples: (1049, 400, 5)
        "default": {
            "file": "roaming_worm2.csv.gz",
            "norm": True,
            "n_training": 5600,
            "window": 400,
            "stride": 4,
            "input_steps": 100,
            "title": "$Roaming$ $Worm$",
            "inference_steps": 160,
            "gamma": 2.265,
            "zeta": 0.8575,
            "mu": 0.4615,
            "y_lim": {
                "ett": [(0, 4.2), (0, 1.7), (0, 0.9)],
                "hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "ett_hf": [(0, 3.3), (0, 1.23), (0, 0.65)],
                "benchmark": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "all": [(0, 6.5), (0, 2.5), (0, 0.85)],
                "transformers": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "recurrent": [(0, 5.8), (0, 2.5), (0, 0.8)],
                "no-auto-informer": [(0, 6.5), (0, 2.5), (0, 0.85)],
            },
        }
    },
}


ESTIMATORS = {
    "GRU_baseline": {"lr": -2},
    "MT": {"num_head": 1, "num_encoder": 1, "lr": -2},
    "GRU_ss": {"decay": "is", "lr": -2},
    # "GRU_ss":{"decay":"exp", "lr":-2},
    # "GRU_ss":{"decay":"linear", "lr":-2},
    "GRU_hf_5": {"k": 5, "lr": -4},
    "GRU_hf_10": {"k": 10, "lr": -5},
    "GRU_hf_15": {"k": 15, "lr": -6},
    "GRU_hf_20": {"k": 20, "lr": -7},
    "GRU_k_20": {"k": 20, "lr": -2},
}


COLOR_BANK = {
    "BLS": "#8C8C8C",
    "ELM": "#8C564B",
    "MTF": "#00A86B",
    "Teacher Forcing": "#BCBD22",
    "SSFS": "#1F77B4",
    "SSES": "#17BECF",
    "HF5": "#BCBD22",
    "HF10": "#FFC400",
    "HF15": "#FF7F0E",
    "HF20": "#D62728",
    "IF": "#E377C2",
    "AF": "#B5338A",
    "ETT Direct": "#8C564B",
}


# def get_dataset_name(system):
#     setting = EXP_SETTING[system]
#     return f"{system}/{setting['sub']}" if {setting["sub"]} else f"{system}"


def get_dataset_name(system, sub):
    setting = EXP_SETTING[system]
    return f"{system}/{sub}" if {sub} else f"{system}"


def get_model_name(dataset_name, fold, normalization, x_y_lag, pretrained=PRETRAINED):
    if pretrained:
        return f"pre_trained/{dataset_name}/fold_{fold}/{normalization}/{x_y_lag}"
    return f"{dataset_name}/fold_{fold}/{normalization}/{x_y_lag}"
