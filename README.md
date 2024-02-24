# HorizonForcing

This repository contains the code for the HorizonForcing project, which explores the application of forecasting techniques to time series data from various systems, including "lorenz", "rossler", "accelerometer", "dwelling_worm", "ecg", "ecosystem", "electricity", "gait_force", "gait_marker_tracker", "geyser", "mouse", "pendulum", and "roaming_worm".

## Supported Systems

| System Name         | Subject(s) |
| ------------------- | ---------- |
| lorenz              | 0.05       |
| rossler             | 0.05       |
| accelerometer       | 1, 3       |
| dwelling_worm       | 1, 2       |
| ecg                 |            |
| ecosystem           |            |
| electricity         |            |
| gait_force          | 1, 2       |
| gait_marker_tracker | 1, 2       |
| geyser              |            |
| mouse               |            |
| pendulum            |            |
| roaming_worm        |            |

## Data Generation

To generate data samples for a specific system, use the command below, replacing {system-name} and {subject} with your target system and subject(s):

```
python generate_dataset.py -s {system-name} -sub {subject}
```

## Index Generation

Generate the index of the samples for structured access and analysis:

```
python training.py -gpu {gpu-index} -mode 1  -s {system-name} -sub {subject}
```

Ensure to replace {gpu-index} with the index of the GPU you wish to use (e.g., 1), if applicable.

## Model Training

To retrain models with the generated data:

```
python training.py -gpu {gpu-index}  -s {system-name} -sub {subject}
```

## Contribution

We welcome contributions, experiments, and discussions from the community. Feel free to dive into the code, try out your own experiments, and share your findings or improvements through pull requests or issues.

Enjoy exploring the vast potential of time series forecasting with HorizonForcing!

## Task

Try

```
python generate_dataset.py -s {system-name} -sub {subject}
```

```
python training.py -gpu {gpu-index} -mode 1  -s {system-name} -sub {subject}
```

```
python training.py -gpu {gpu-index}  -s {system-name} -sub {subject}
```

for

| System Name         | Subject(s) |
| ------------------- | ---------- |
| rossler             | 0.05       |
| accelerometer       | 1          |
| dwelling_worm       | 1, 2       |
| ecg                 |            |
| gait_force          | 1          |
| gait_marker_tracker | 1, 2       |
| geyser              |            |
| mouse               |            |
| pendulum            |            |
