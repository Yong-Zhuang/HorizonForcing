# HorizonForcing

This repository contains the code for the HorizonForcing project, which explores the application of forecasting techniques to time series data from various systems, including "lorenz", "rossler", "accelerometer", "dwelling_worm", "ecg", "ecosystem", "electricity", "gait_force", "gait_marker_tracker", "geyser", "mouse", "pendulum", and "roaming_worm".

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
