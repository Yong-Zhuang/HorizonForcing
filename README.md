# HorizonForcing

Use the experiments_results notebook to regenrate the experimental results in the paper.

run following command to regenrate data samples:
    python generate_dataset.py -s electricity
where -s is the system to build data for, optional values are ["lorenz", "accelerometer", "gait_force", "roaming_worm", "electricity"]

run following command to genrate the index of the samples:   
    python training.py -gpu 1 -s electricity -mode 1


run following command to retrain all models:
    python training.py -gpu 1 -s electricity
where -s is the system to build data for, optional values are ["lorenz", "accelerometer", "gait_force", "roaming_worm", "electricity"]