.PHONY: features train score airflow

features:
	python ingest.py && python features.py

train:
	python train.py

score:
	python score.py

airflow:
	AIRFLOW_HOME=$$(pwd)/airflow airflow standalone
