import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print(f'Name: {experiment.name}')

    with mlflow.start_run(
        run_name='logging_models',
        experiment_id=experiment.experiment_id
    ) as run:

        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=5,
            random_state=42
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
            )

        mlflow.sklearn.autolog()

        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        # mlflow.sklearn.log_model(
        #     sk_model=rfc,
        #     artifact_path="random_forest_classifier"
        #     )

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
