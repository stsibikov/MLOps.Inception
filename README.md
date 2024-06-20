# [MLOps.Inception](https://stepik.org/course/181476/syllabus)

HSE university, spring 2024

Implementing automatic data processing, model training and experiment tracking with Airflow and MLflow.

The goal of the course was to write a DAG that trains 3 different regressor models and stores its code and metrics in a MLflow experiment.

The DAG:
* Retrieves data from locally-running PostgreSQL server, and stores it into S3
* Retrieves data from S3 and runs preprocessing. The results are also stored into S3
* Initializes the MLflow experiment, providing experiment id to be used for training tasks
* Runs 3 model training tasks in parallel, logging the model and storing regressor metrics into MLflow
* Saves timestamps of the tasks into S3 

DAG graph:
![DAG_graph](assets/DAG_graph.png)

MLflow metrics of all models:
![MLflow_metrics](assets/MLflow_metrics.png)

Metadata of one of the models - HistGB:
![MLflow_HistGB_artifacts](assets/MLflow_HistGB_artifacts.png)

Code for the DAG is available [here](mlops_final.py).
