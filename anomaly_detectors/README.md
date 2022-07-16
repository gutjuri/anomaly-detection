# Anomaly Detectors

This directory contains the scripts used to test the proposed anomaly detectors and determine the performance of different available algorithms.
The files themselves contain configuration parameters.

Run experiments:

Anomalous Logins: `python3 detect_anomalous_logins.py`
Anomalous Movement: not implemented
Anomalous Log Sequences:
- DeepLog: `cd DeepLog; python3 LogKeyModel_train.py -training_dataset /path/to/x_train` for training and `cd DeepLog; python3 LogKeyModel_predict.py -dataset path/to/x_test -label_path /path/to/label/file` for prediction.
- Other models: `cd loglizer/benchmarks; python3 benchmarks.py /path/to/x_train /path/to/x_test /path/to/x_validation /path/to/labelfile`
Anomalous Jobs: `python3 detect_anomalous_jobs.py /path/to/slurm/output`

Visualising the jobs of numerous research groups as done in the Thesis:

`python3 visualize_jobs.py /path/to/slurm/output`