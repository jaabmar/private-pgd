

import pickle
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pdb
import pickle
import json

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np



statistics = {            "rand_counting_query_mean_dist": "counting queries",

                "synth_gradboost_test": "class/reg test error",
                "cov_fixed_frobenius_norm": "covariance matrix", 
            "rand_thresholding_query_mean_dist": "thresholding queries",
                    "wdist_1_2Way_avg": "$SW_1$ distance" , 
                    "newl1_2Way_avg" : "TV distance",
            }


script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script


epsilons = [0.2,1.0, 2.5]
methods = ["privpgd+KWay", "pgm_euclid+AIM", "pgm_euclid+MST", "private_gsd+KWay", "gem+KWay", "rap+KWay"]
inference_types = {"pgm_euclid":"PGM", "privpgd":"PrivPGD", "private_gsd": "PrivGSD", "gem":"GEM", "rap":"RAP "}
corr = "dataset_n"

with open(os.path.join(script_folder, f"save_stats"), "rb") as file:
    filtered_data = pickle.load(file)

for epsilon in epsilons:
    print(f"start with {epsilon}")

    ycompare ={}
    for plot_stat in statistics.keys():
        ycompare[plot_stat] = {}

        for (j,method) in enumerate(methods):
            ycompare[plot_stat][method] = {}
            yratio = []
            xvalues = []
            for dataset in filtered_data.keys():
                if "ans" in dataset:
                    continue
                if epsilon >2.5:
                    epsilon = int(epsilon)

                name_run= f"{epsilon}+{method}"

                if name_run not in filtered_data[dataset] or plot_stat not in filtered_data[dataset][name_run] :
                    continue
                if  plot_stat == "synth_gradboost_test" and "regression" not in dataset:
                    ycompare[plot_stat][method][dataset] = 1- np.mean(filtered_data[dataset][name_run][plot_stat])
                else:
                    ycompare[plot_stat][method][dataset] = np.mean(filtered_data[dataset][name_run][plot_stat])


    # Initialize dictionaries to store counts
    winners = {method: {plot_stat: 0 for plot_stat in statistics.keys()} for method in methods}
    second_winners = {method: {plot_stat: 0 for plot_stat in statistics.keys()} for method in methods}

    # Process data
    for plot_stat in statistics.keys():
        temp = ycompare[plot_stat]
        for dataset in filtered_data.keys():
            best_method = None
            second_best_method = None
            best_score = 10000
            second_best_score = 10000

            for method in methods:
                if method in temp and dataset in temp[method]:
                    score = temp[method][dataset]
                    if best_score > score:
                        second_best_score = best_score
                        second_best_method = best_method
                        best_score = score
                        best_method = method
                    elif second_best_score > score and method != best_method:
                        second_best_score = score
                        second_best_method = method

            if best_method is not None:
                winners[best_method][plot_stat] += 1
            if second_best_method is not None:
                second_winners[second_best_method][plot_stat] += 1



    latex_table = "\\begin{table}[t!]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{c" + "c" * len(statistics.values()) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "Method & " + " & ".join(statistics.values()) + " \\\\\n"
    latex_table += "\\midrule\n"

    for method in methods:
        inference_type, mechanism = method.split('+')

        if "AIM" in mechanism:
            mechanism = "AIM"#-" + mechanism.split('_')[1]
        elif "MST" not in mechanism:
            mechanism  = mechanism + "-2"
            
        if "advanced_extended" in inference_type:
            mechanism = mechanism.split('_')[0]

        row = [" " + inference_types[inference_type] + " " + mechanism]
        for plot_stat in statistics.keys():
            win_count = winners[method][plot_stat]
            second_count = second_winners[method][plot_stat]
            row.append(f"{win_count} ({second_count})")
        latex_table += " & ".join(row) + " \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Your Caption Here}\n"
    latex_table += "\\label{tab:your_label}\n"
    latex_table += "\\end{table}"


    print(latex_table)
    # Saving to a .tex file
    with open(os.path.join(script_folder, f"counts_{epsilon}.tex"), 'w') as file:
        file.write(latex_table)
                









