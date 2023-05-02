# HorizonForcing

This repository contains the code for the HorizonForcing project, which explores the application of forecasting techniques to time series data from various systems, including "lorenz", "accelerometer", "gait_force", "roaming_worm", and "electricity". The experiments_results notebook can be used to regenerate the experimental results presented in the paper [WideningtheTimeHorizon:Predictingthe Long-TermBehaviorofChaoticSystems](https://yong-zhuang.github.io/assets/pdf/zhuang2022hf/paper.pdf).

To regenerate the data samples, run the following command:

```
python generate_dataset.py -s electricity
```

Note that the "-s" option can be used to specify the system to build data for, with optional values including ["lorenz", "accelerometer", "gait_force", "roaming_worm", "electricity"].

To generate the index of the samples, run the following command:

```
python training.py -gpu 1 -s electricity -mode 1
```

To retrain all models, run the following command:

```
python training.py -gpu 1 -s electricity
```

Again, note that the "-s" option can be used to specify the system to build data for. Enjoy exploring the code and trying out your own experiments!