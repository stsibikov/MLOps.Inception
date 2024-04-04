#imports
import os
import json
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Literal

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

import mlflow
from mlflow.models import infer_signature
#/imports



#var and DAG setup
S3_REGION_NAME = os.environ.get('S3_REGION_NAME')

BUCKET = os.environ.get('BUCKET')

S3_MLFLOW_ARTIFACTS_LOC = os.environ.get('S3_MLFLOW_ARTIFACTS_LOC')

FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
            'AveOccup', 'Latitude', 'Longitude']

TARGET = 'MedHouseVal'

DEFAULT_ARGS = {
    'owner' : 'Sergei Tsibikov',
    'email_on_failure' : False,
    'email_on_retry' : False,
    'retry' : 3,
    'retry_delay' : timedelta(minutes=1)
}

models = {'RandomForest' : RandomForestRegressor(),
          'LinearRegression' : LinearRegression(),
          'HistGB' : HistGradientBoostingRegressor()}

dag = DAG(dag_id='mlops_final',
          schedule_interval='0 1 * * *',
          start_date=days_ago(2),
          catchup=False,
          tags=['mlops'],
          default_args=DEFAULT_ARGS)

DATETIME_FORMAT = '%d.%m.%Y %H:%M:%S'
#/var and DAG setup



#pipeline (functions) setup
def init() -> Dict[str, Any]:
    '''Initialize airflow pipeline'''    
    dag_kwargs = {}
    dag_kwargs['init_start'] = datetime.now().strftime(DATETIME_FORMAT)
    return dag_kwargs


def get_data(**kwargs) -> Dict[str, Any]:
    '''Get data from PostgreSQL and save it into S3'''
    task_instance = kwargs['ti']
    dag_kwargs = task_instance.xcom_pull(task_ids='init')
    dag_kwargs['data_download_start'] = datetime.now().strftime(DATETIME_FORMAT)

    #collect data from Postgres
    pg_hook = PostgresHook('pg_connection')
    con = pg_hook.get_conn()
    data = pd.read_sql_query('SELECT * FROM california_housing', con)
    
    #connect to S3
    s3_hook = S3Hook('s3_connection')
    session = s3_hook.get_session(S3_REGION_NAME)
    resource = session.resource('s3')
    
    #store raw data on S3
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, 'datasets/california_housing.pkl').put(Body=pickle_byte_obj)

    return dag_kwargs


def process_data(**kwargs) -> Dict[str, Any]:
    '''Data processing, processed data is uploaded to S3'''
    task_instance = kwargs['ti']
    dag_kwargs = task_instance.xcom_pull(task_ids='get_data')
    dag_kwargs['data_processing_start'] = datetime.now().strftime(DATETIME_FORMAT)
    
    #read raw data on S3
    s3_hook = S3Hook('s3_connection')
    file = s3_hook.download_file(key='datasets/california_housing.pkl', bucket_name=BUCKET)
    data = pd.read_pickle(file)

    #data processing
    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    #store processed data on S3
    session = s3_hook.get_session(S3_REGION_NAME)
    resource = session.resource('s3')

    for name, data in zip(['X_train', 'X_test', 'y_train', 'y_test'],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f'datasets/{name}.pkl').put(Body=pickle_byte_obj)

    return dag_kwargs


def init_experiment(**kwargs) -> Dict[str, Any]:
    '''Init mlflow experiment'''
    task_instance = kwargs['ti']
    dag_kwargs = task_instance.xcom_pull(task_ids='get_data')
    
    experiment_name = 'mlops_stepik_final'
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=f'{S3_MLFLOW_ARTIFACTS_LOC}{experiment_name}')
    mlflow.set_experiment(experiment_name)
    dag_kwargs['mlflow_experiment_id'] = experiment_id

    dag_kwargs['mlflow_experiment_initialized'] = datetime.now().strftime(DATETIME_FORMAT)
    
    return dag_kwargs


def train_model(**kwargs) -> None:
    '''Train a model using given model alias and model callable, log the model into mlflow, evaluate its metrics'''
    task_instance = kwargs['ti']
    dag_kwargs = task_instance.xcom_pull(task_ids='init_experiment')
    experiment_id = dag_kwargs['mlflow_experiment_id']
    
    model_alias = kwargs['model_alias'] #this is provided with op_kwargs at task initialization,
                                        #so this refers to kwargs, not dag_kwargs
    model = kwargs['model_uri'] #same
    
    dag_kwargs[f'train_{model_alias}_start'] = datetime.now().strftime(DATETIME_FORMAT)
        
    #load processed data from S3
    s3_hook = S3Hook('s3_connection')
    data_dir = {}
    for name in ['X_train', 'X_test', 'y_train', 'y_test']:
        file = s3_hook.download_file(key=f'datasets/{name}.pkl', bucket_name=BUCKET)
        data_dir[name] = pd.read_pickle(file)

    #train models
    with mlflow.start_run(run_name = model_alias, experiment_id = experiment_id):
        model.fit(data_dir['X_train'], data_dir['y_train'])

        prediction = model.predict(data_dir['X_test'])

        signature = infer_signature(data_dir['X_test'], prediction)

        model_info = mlflow.sklearn.log_model(model, model_alias, signature=signature)

        mlflow.evaluate(
            model_info.model_uri,
            data=data_dir['X_test'],
            targets=data_dir['y_test'].values,
            model_type='regressor',
            evaluators=['default'])

    dag_kwargs[f'train_{model_alias}_end'] = datetime.now().strftime(DATETIME_FORMAT)

    return dag_kwargs


def save_results(**kwargs) -> None:
    '''Store timestamps into S3'''
    task_instance = kwargs['ti']
    dag_kwargs = task_instance.xcom_pull(task_ids=[f'train_{model_alias}' for model_alias in models.keys()])
    
    metrics = {}
    for metric, value in dag_kwargs.items():
        metrics[metrics] = value
    
    metrics['dag_end'] = datetime.now().strftime(DATETIME_FORMAT)
    
    #store metrics into S3
    s3_hook = S3Hook('s3_connection')
    session = s3_hook.get_session(S3_REGION_NAME)
    resource = session.resource('s3')
    
    datetime_now = datetime.now().strftime(DATETIME_FORMAT)
    json_byte_object = json.dumps(metrics)
    resource.Object(BUCKET, f'results/metrics_{datetime_now}.json').put(Body=json_byte_object)
    return None

#/pipeline (functions) setup



#tasks setup
task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id='get_data',
                               python_callable=get_data,
                               dag=dag,
                               provide_context=True)

task_process_data = PythonOperator(task_id='process_data',
                                   python_callable=process_data,
                                   dag=dag,
                                   provide_context=True)

task_init_experiment = PythonOperator(task_id='init_experiment',
                                   python_callable=init_experiment,
                                   dag=dag,
                                   provide_context=True)

task_train_models = [PythonOperator(task_id=f'train_{model_alias}',
                    		    python_callable=train_model,
                    		    dag=dag,
                    		    provide_context=True,
                    		    op_kwargs={'model_alias': model_alias, 'model_uri': model_uri})
		     for model_alias, model_uri in models.items()]

task_save_results = PythonOperator(task_id='save_results',
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)
#/tasks setup



#DAG run
task_init >> task_get_data >> task_process_data >> task_init_experiment >> task_train_models >> task_save_results