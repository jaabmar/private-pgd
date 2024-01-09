import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import wandb
from inference.dataset import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

import json
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pdb
from experiments.post_processing.utils import get_filtered_runs, get_paths

import json
import pickle


def process_dataset(api,set_runs, dataset, entity_name, project_name, models_to_evaluate=[] ):
    ref_path = os.path.join(data_path, dataset)
    print(f"Start with the data set {dataset}")
    
    try:
        train_data = pd.read_csv(os.path.join(ref_path, "data_disc.csv"))
        test_data = pd.read_csv(os.path.join(ref_path, "testdata_disc.csv"))
        train_data.columns = train_data.columns.astype(str)
        test_data.columns = test_data.columns.astype(str)
    except Exception as e:
        print(f"Error loading domain or inverse mapping for dataset {dataset}: {e}")
        return
    

    orig_evaluated = False


    #print(f"{model_type} on original data: Train accuracy: {train_accuracy_orig}, Test accuracy: {test_accuracy_orig}")
    for run_id in set_runs:

        run = api.run(f"{project_name}/{run_id}")

        try:
            train_data_synth = pd.read_csv(os.path.join(ref_path, f"synth_data{run_id}.csv"))
            train_data_synth.columns = train_data_synth.columns.astype(str)
            stop = False
        except Exception as e:
            print(f"Error reading synthesized data for run_id {run_id}: {e}")
            stop = True

        if not stop:
            

            models_to_evaluate_now = []
            for model  in models_to_evaluate:
                if f"{model}_eval" not in run.summary:
                    models_to_evaluate_now.append(model)
                    run.summary[f"{model}_eval"] = True

            print("models:")
            print(models_to_evaluate_now)
            if models_to_evaluate is None:
                continue


            # ... (rest of your run evaluation code)
            model_synth = Classifier(train_data_synth, test_data)
            temp_synth, is_regression = model_synth.evaluate_all_models(name="synth", models_to_evaluate=models_to_evaluate_now)
            if not orig_evaluated:
                model_orig = Classifier(train_data, test_data)
                temp,_= model_orig.evaluate_all_models(name="orig", models_to_evaluate=models_to_evaluate_now)
                orig_evaluated = True

            for key, value in temp_synth.items():
                run.summary[key] = value
            for key, value in temp.items():
                run.summary[key] = value

            if is_regression:
                run.summary['type'] = "regression"
            else:
                run.summary['type'] = "classification"
            run.summary[f"dataset_test_n"] = test_data.shape[0]
            run.summary["evaluated_full"] = True

            run.update()

    print(f"Done with the data set {dataset}")




class Classifier:
    def __init__(self, train_data, test_data):
        self.X_train = train_data.iloc[:, :-1].values
        self.X_test = test_data.iloc[:, :-1].values
        self.y_train = train_data[train_data.columns[-1]]
        self.y_test = test_data[test_data.columns[-1]]
        
        self.is_regression = len(self.y_train.unique()) > 2





    def train_test(self, model_type):
        if self.is_regression:
            if model_type == "sgdl1":
                model = SGDRegressor(penalty='l1', max_iter=1000, tol=1e-3)  # Modified line
            elif model_type == "ridge_sgd":
                model = SGDRegressor(loss='squared_loss', penalty='l2', max_iter=1000, tol=1e-3)
            elif model_type == "elastic_net_sgd":
                model = SGDRegressor(loss='squared_loss', penalty='elasticnet', max_iter=1000, tol=1e-3)
            elif model_type == "gradboost":
                model = GradientBoostingRegressor()
            elif model_type == "catboost":
                model = cb.CatBoostRegressor(verbose=0)
            elif model_type == "hist_gradient_boosting":
                model = HistGradientBoostingRegressor()
            elif model_type == "mlp":
                model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                                     solver='sgd', verbose=0, learning_rate_init=.1)
            else:
                raise ValueError(f"Model type {model_type} not recognized for regression")

            metric = mean_squared_error
        else:
            if model_type == "sgdl1":
                model = SGDClassifier(loss='log_loss', penalty='l1', max_iter=1000, tol=1e-3)  # Correct the loss to 'log' and added L1 penalty
            elif model_type == "linear_svm":
                model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
            elif model_type == "gradboost":
                model = GradientBoostingClassifier()
            elif model_type == "catboost":
                model = cb.CatBoostClassifier(verbose=0)
            elif model_type == "hist_gradient_boosting":
                model = HistGradientBoostingClassifier()
            elif model_type == "mlp":
                model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                                     solver='sgd', verbose=0, learning_rate_init=.1)
            else:
                raise ValueError(f"Model type {model_type} not recognized for classification")

            metric = accuracy_score

        model.fit(self.X_train, self.y_train)

        train_predictions = model.predict(self.X_train)
        test_predictions = model.predict(self.X_test)

        train_accuracy = metric(self.y_train, train_predictions)
        test_accuracy = metric(self.y_test, test_predictions)

        return train_accuracy, test_accuracy

    def evaluate_all_models(self, name, models_to_evaluate=None):
        results = {}
        models_to_evaluate_reg = ["sgdl1", "gradboost", "mlp"]
        models_to_evaluate_class = ["sgdl1", "gradboost", "mlp"]

        if models_to_evaluate is None:
            if self.is_regression:
                models_to_evaluate = models_to_evaluate_reg
            else:
                models_to_evaluate = models_to_evaluate_class
        else:
            if self.is_regression:
                models_to_evaluate = [model for model in models_to_evaluate if model in models_to_evaluate_reg]
            else:
                models_to_evaluate = [model for model in models_to_evaluate if model in models_to_evaluate_class]




        for model_type in models_to_evaluate:
            print(f"done with {model_type}")
            train_accuracy, test_accuracy = self.train_test(model_type)
            results[f"{name}_{model_type}_train"] = train_accuracy
            results[f"{name}_{model_type}_test"] = test_accuracy

        return results, self.is_regression







import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Retrieve project name from command line")
    parser.add_argument("--project_name", type=str, default="advanced_particle",
                        help="Name of the project")
    parser.add_argument("--filters_adv", type=str, default="./experiments/post_processing/filters_adv.json",
                        help="Name of the project")
    parser.add_argument("--filters_relational", type=str, default="./experiments/post_processing/relational_filters.json",
                        help="Name of the project")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    project_name = args.project_name

    print(f"Project Name: {project_name}")
    

    paths = get_paths()
    data_path = paths['data_path']
    entity_name = paths['entity_name']
    with open(args.filters_adv, 'r') as file:
        filters_adv = json.load(file)
    with open(args.filters_relational, 'r') as file:
        relational_filters = {k: tuple(v) for k, v in json.load(file).items()}
    api = wandb.Api()

    runs = get_filtered_runs(api, project_name, entity_name,filters_adv,relational_filters)

    MODELS_TO_EVALUATE =  ["gradboost"]

    # Fetch the runs based on filters
    datasets = set(run.config.get("dataset") for run in runs)
    dataset_runs = {dataset: [] for dataset in datasets}
    for run in runs:
        run_id = run.id
        dataset = run.config.get("dataset", None)
        if dataset:
            dataset_runs[dataset].append(run_id)

    def process_dataset_setup(dataset):
        process_dataset(api, dataset_runs[dataset], dataset, entity_name, project_name , MODELS_TO_EVALUATE)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_dataset_setup, dataset_runs.keys()))

      