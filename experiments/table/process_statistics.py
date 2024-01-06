import wandb
from collections import defaultdict
import os
import pdb
import pickle
import json
import numpy as np
from experiments.post_processing.utils import get_filtered_runs, get_paths



def add_stat(filtered_data, statistics, run, dataset_name, name_run, map_statistics):
    for stat in statistics:
        for key_run in run.summary.keys():
            if stat in key_run:
                if key_run not in filtered_data[dataset_name][name_run]:
                    filtered_data[dataset_name][name_run][key_run] = []
                if stat in map_statistics.keys():
                    filtered_data[dataset_name][name_run][key_run].append(run.summary[key_run]/run.summary[map_statistics[stat]])
                elif key_run == "elapsed_time":
                    filtered_data[dataset_name][name_run][key_run].append(run.summary[key_run]/60)
                else:
                    filtered_data[dataset_name][name_run][key_run].append(run.summary[key_run])




project_name = "experiment_table_new"
print(f"Project Name: {project_name}")
base_path = "/cluster/work/yang/donhausk/privacy-ot/data"
store_eval = "/cluster/work/yang/donhausk/privacy-ot/"
entity_name = "eth-sml-privacy-project"
script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script



with open(os.path.join(script_folder, "filters_adv.json"), 'r') as file:
    filters_adv = json.load(file)
with open(os.path.join(script_folder, "relational_filters.json"), 'r') as file:
    relational_filters = {k: tuple(v) for k, v in json.load(file).items()}
api = wandb.Api()
runs = get_filtered_runs(api, project_name, entity_name,filters_adv,relational_filters)



with open(os.path.join(script_folder, "filters_adv.json"), 'r') as file:
    filters_adv = json.load(file)
with open(os.path.join(script_folder, "relational_filters.json"), 'r') as file:
    relational_filters = {k: tuple(v) for k, v in json.load(file).items()}



otheralgos =["gem", "private_gsd", "rap"]




k = 32
name = "workload_3way" 
statistics = ['wdist_', "synth_", "cov_" ,"rand_", "orig_", "elapsed", "cormat", "dataset", "w_", "l", "new"]
#statistics = ['w_mean', 'w_max']
map_statistics = {
              "rand_thrs_query_mean_dist": "rand_thrs_query_orig",
            "rand_coun_new_mean_dist": "rand_coun_new_orig", 
                        "cov_fixed_spectral_norm":      "cov_fixed_spectral_base" ,   
                                                   "cov_fixed_frobenius_norm":      "cov_fixed_frobenius_base"       }

statistics_map = {
              "rand_thrs_query_mean_dist": "thresholding q.",
            "synth_gradboost_test": "class. acc.",
            "cov_fixed_frobenius_norm": "cov. mat. fro.", 
            "rand_coun_new_mean_dist": "counting q."    , 
                   "dataset_n": "n data points", 
                                "dataset_d": "d dimensions", 
                                 "l1_2Way_avg": "avg TV dist" }



# statistics_map = {
#             "synth_gradboost_test": "class. acc."       }

inference_methods = {"pgm_euclid":"PGM", "advanced_extended":"PrivPGD", "private_gsd": "PrivGSD", "gem":"GEM", "rap":"RAP", "localPGM":"AP-PGM"}







filtered_data = defaultdict(lambda: {})


runs_2 = get_filtered_runs(api, "experiment_table_marginals" , entity_name,{},{})
for run in runs_2:

    config = run.config

    summary = run.summary

    dataset_type = config.get('dataset')
    parts = dataset_type.split('_')
    
    if len(parts) >= 3:  # Ensure there are at least three parts
        # The last part is the k-value
        disc_k = parts[-1]
        
        # The second to last part is the type
        disc_type = parts[-2]
        
        # All preceding parts make up the dataset name
        dataset_name = '_'.join(parts[:-2])
    else:
        continue

    epsilon = float(config.get('epsilon'))

    if not(dataset_name):
        continue
    inference_type = config.get('inference_type')
    mechanism = config.get('mechanism')
    if inference_type != "localPGM":
        continue
    else:
        "juhuu there"


    name_run = f"{epsilon}+{inference_type}+{mechanism}"


    if name_run not in filtered_data[dataset_name]:
        filtered_data[dataset_name][name_run] = {}
    
    add_stat(filtered_data, statistics, run, dataset_name, name_run, map_statistics)



for run in runs:
    config = run.config

    summary = run.summary
    dataset_name = summary.get('dataset_name')
    epsilon = float(config.get('epsilon'))
    inference_type = config.get('inference_type')
    mechanism = config.get('mechanism')

    if not(dataset_name and epsilon and inference_type):
        continue
    if inference_type == "advanced_extended" and str(config['proper_normalization']) == "0": 
        name_run = f"{epsilon}+advanced_basic+{mechanism}"
    else:
    
        name_run = f"{epsilon}+{inference_type}+{mechanism}"

    if name_run not in filtered_data[dataset_name]:
        filtered_data[dataset_name][name_run] = {}
    add_stat(filtered_data, statistics, run, dataset_name, name_run, map_statistics)



for otheralg in otheralgos:
    runs_2 = get_filtered_runs(api, otheralg, entity_name,{},{})
    for run in runs_2:
        config = run.config
        summary = run.summary
        if not  str(summary.get('hp_best_12_12')) == "True":
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



