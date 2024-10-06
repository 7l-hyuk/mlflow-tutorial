from mlflow_utils import get_mlflow_experiment

if __name__ == '__main__':
    experiment = get_mlflow_experiment(experiment_name='testing_mlflow02')
    print(
        f'''
Name: {experiment.name}
Experiment id: {experiment.experiment_id}
Artifact Location: {experiment.artifact_location}
Tags: {experiment.tags}
Lifecycle_stage: {experiment.lifecycle_stage}
Creating timestamp: {experiment.creation_time}
'''
    )
