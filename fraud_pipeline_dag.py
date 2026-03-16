from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

def make_task(task_id, cmd):
    return BashOperator(task_id=task_id, bash_command=cmd)

with DAG(
    dag_id='fraud_pipeline_dag',
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=False,
    default_args={'retries': 0}
):
    features = make_task('features', 'python -m src.ingest && python -m src.features')
    train = make_task('train', 'python -m src.train')
    score = make_task('score', 'python -m src.score')
    features >> train >> score
