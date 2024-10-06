import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":
    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print(f'Name: {experiment.name}')

    with mlflow.start_run(
        run_name='logging_metrics',
        experiment_id=experiment.experiment_id
    ) as run:
        mlflow.log_metric('mse', 0.01)

        metrics = {
            'mse': 0.01,
            'mae': 0.01,
            'rmse': 0.01,
            'r2': 0.01
        }

        mlflow.log_metrics(metrics)

        print(
            f'''
run_id: {run.info.run_id}
experiment_id: {run.info.experiment_id}
status: {run.info.status}
start_time: {run.info.start_time}
end_time: {run.info.end_time}
lifecycle_stage: {run.info.lifecycle_stage}
'''
        )
