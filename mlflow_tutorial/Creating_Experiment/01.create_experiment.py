# import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__ == '__main__':
    # mlflow.create_experiment(
    #     name='testing_mlflow01',
    #     artifact_location='testing_mlflow01_artifacts',
    #     tags={'env': 'dev', 'version': '1.0.0'},
    # )

    experiment_id = create_mlflow_experiment(
        experiment_name='testing_mlflow1',
        artifact_location='test_mlflow1_artifacts',
        tags={'env': 'dev', 'version': '1.0.0'},
    )

    print(f'Experiment id: {experiment_id}')
