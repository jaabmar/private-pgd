import wandb
from collections import defaultdict
import os
import pdb
import pickle
import json
import numpy as np
from experiments.post_processing.utils import get_filtered_runs, get_paths, add_stat






project_name = "benchmark_table"
paths = get_paths()
main_path = paths['main_path']
entity_name = paths['entity_name']
script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script



# with open(os.path.join(script_folder, "filters_adv.json"), 'r') as file:
#     filters_adv = json.load(file)
# with open(os.path.join(script_folder, "relational_filters.json"), 'r') as file:
#     relational_filters = {k: tuple(v) for k, v in json.load(file).items()}
api = wandb.Api()
runs = get_filtered_runs(api, project_name, entity_name)





# otheralgos =["gem", "private_gsd", "rap"]
otheralgos =[ "private_gsd"]




k = 32
name = "workload_3way" 
statistics = ['wdist_', "synth_", "cov_" ,"rand_", "orig_", "elapsed", "cormat", "dataset", "w_", "l", "new"]
map_statistics = {
              "rand_thrs_query_mean_dist": "rand_thrs_query_orig",
            "rand_coun_new_mean_dist": "rand_coun_new_orig", 
                        "cov_fixed_spectral_norm":      "cov_fixed_spectral_base" ,   
                                                   "cov_fixed_frobenius_norm":      "cov_fixed_frobenius_base"       }









filtered_data = defaultdict(lambda: {})

for run in runs:
    config = run.config

    summary = run.summary
    dataset_name = summary.get('dataset_name')
    epsilon = float(config.get('epsilon'))
    inference_type = config.get('inference_type')
    mechanism = config.get('mechanism')
    if not(dataset_name and epsilon and inference_type):
        continue
    p_mask = config.get('p_mask', 0)

    if inference_type =="privpgd":
        if p_mask != 80:
            continue

    name_run = f"{epsilon}+{inference_type}+{mechanism}"
    if name_run not in filtered_data[dataset_name]:
        filtered_data[dataset_name][name_run] = {}

    add_stat(filtered_data, statistics, run, dataset_name, name_run, map_statistics)



for otheralg in otheralgos:
    runs_2 = get_filtered_runs(api, otheralg, entity_name,{},{})
    for run in runs_2:
        config = run.config
        summary = run.summary
        if not  str(summary.get('hp_best_15_01')) == "True":
            continue

        dataset_name = config.get('dataset_name')
        epsilon = float(config.get('epsilon'))
        inference_type = config.get('inference_type')
        mechanism = config.get('mechanism')


        name_run = f"{epsilon}+{inference_type}+{mechanism}"


        if name_run not in filtered_data[dataset_name]:
            filtered_data[dataset_name][name_run] = {}
        

        add_stat(filtered_data, statistics, run, dataset_name, name_run, map_statistics)






with open(os.path.join(script_folder, f"save_stats"), "wb") as file:
    pickle.dump(dict(filtered_data), file)



