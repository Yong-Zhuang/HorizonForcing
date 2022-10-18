# HorizonForcing

make sure that the code directory has following sub folders: data, results, saved_models, logs.

run following command to regenrate evaluation results in the paper:
python evaluation.py -data 1
where -data: 1 means lorenz63 system, 2 means rossler system

run following command to retrain all models:
python training.py -gpu 0 -data 1
where -data: 1 means lorenz63 system, 2 means rossler system