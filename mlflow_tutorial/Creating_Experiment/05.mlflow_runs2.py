import mlflow

if __name__ == '__main__':
    with mlflow.start_run(run_name='mlflow_runs') as run:
        mlflow.log_param('learning_rate', 0.01)
        print(f'RUN ID: {run.info.run_id}')
        print(run.info)
