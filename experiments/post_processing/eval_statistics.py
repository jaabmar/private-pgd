import os
import pandas as pd
import wandb
import json
import pickle
import numpy as np
from itertools import combinations
import argparse
import torch
from inference.dataset import Dataset
from inference.embedding  import Embedding

from ot import sliced_wasserstein_distance

from experiments.post_processing.utils import get_filtered_runs, get_paths
from experiments.post_processing.evaluators import DatasetEvaluator



def process_dataset(api,dataset, run_set, project_name, main_path, metrics_to_evaluate=[]):
    ref_path = os.path.join(main_path, dataset)
    print(f"Start with the data set {dataset}")
    try:
        train_data = Dataset.load(os.path.join(ref_path, "data_disc.csv"),os.path.join(ref_path, "domain.json")) 
    except Exception as e:
        print(f"Fail for data {dataset}")
        return

    for run_id in run_set:
        run = api.run(f"{project_name}/{run_id}")

        try:
            synth_data = Dataset.load(os.path.join(ref_path, f"synth_data{run_id}.csv"),os.path.join(ref_path, "domain.json")) 


        except Exception as e:
            print(f"Error reading synthesized data for run_id {run_id}: {e}")
            continue


        already_evaluated_metrics = [key for key in metrics_to_evaluate if key in run.summary]
        metrics_to_evaluate_now = list(set(metrics_to_evaluate) - set(already_evaluated_metrics))

        if metrics_to_evaluate_now:
            #extract from the ref_path the folder name  
            #and the remaining path
            evaluator = DatasetEvaluator(train_data, synth_data, dataset, ref_path)
            metrics = evaluator.evaluate(metrics_to_evaluate_now)
            for key, value in metrics.items():
                run.summary[key] = value
            run.update()

    print(f"Done with the data set {dataset}")








def get_args():
    parser = argparse.ArgumentParser(description="Retrieve project name from command line")
    parser.add_argument("--project_name", type=str, default="advanced_particle", help="Name of the project")
    parser.add_argument("--filters_adv", type=str, default="./experiments/post_processing/filters_adv.json", help="Name of the project")
    parser.add_argument("--filters_relational", type=str, default="./experiments/post_processing/relational_filters.json", help="Name of the project")
    parser.add_argument("--metrics", nargs='+', default = [], help="Metrics to evaluate")
   
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_args()
    project_name = args.project_name
    print(f"Project Name: {project_name}")

    paths = get_paths()
    main_path = paths['main_path']
    entity_name = paths['entity_name']

    with open(args.filters_adv, 'r') as file:
        filters_adv = json.load(file)
    with open(args.filters_relational, 'r') as file:
        relational_filters = {k: tuple(v) for k, v in json.load(file).items()}
    api = wandb.Api()
    runs = get_filtered_runs(api, project_name, entity_name,filters_adv,relational_filters)

    datasets = set(run.config.get("dataset") for run in runs)
    dataset_runs = {dataset: [] for dataset in datasets}
    for run in runs:
        run_id = run.id
        dataset = run.config.get("dataset", None)
        if dataset:
            dataset_runs[dataset].append(run_id)


    def process_dataset_setup(dataset):
        return process_dataset(api, dataset, dataset_runs[dataset], project_name, main_path, args.metrics)


    # with ProcessPoolExecutor() as executor:
    #     results = list(executor.map(process_dataset_setup, dataset_runs.keys()))

    results = []
    for key in dataset_runs.keys():
        result = process_dataset_setup(key)
        results.append(result)