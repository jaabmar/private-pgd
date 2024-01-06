

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
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
marker_styles = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'x', '+']



project_name = "experiment_scale"



script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
k = 32
name = "full"

epsilons = [0.2,1.0, 2.5]
#epsilons = [2.5]
print("HI")
#methods = [ "advanced_extended+KWay_0", "pgm_euclid+AIM_2","pgm_euclid+AIM_3", "pgm_euclid+MST", "private_gsd+KWay", "gem+KWay" ]
methods = ["advanced_extended+KWay_0", "pgm_euclid+AIM_2", "pgm_euclid+MST", "private_gsd+KWay", "gem+KWay", "rap+KWay"]


for epsilon in epsilons:
    print(f"start with {epsilon}")

    with open(os.path.join(script_folder, f"save_stats_{epsilon}_2"), "rb") as file:
        filtered_data = pickle.load(file)

    def custom_sort_key(key):
        epsilon, name = key.split('+')
        epsilon = float(epsilon)
        
        # Assign a weight to the names based on your desired order
        name_weight = {
            "advanced_extended": 1,
            "private_gsd":3,
            "pgm_euclid": 2,
            "gem":4
        }
        
        # If the name is not in name_weight, assign a default weight
        weight = name_weight.get(name, float('inf'))
        
        # Sort first by epsilon, then by the weight of the name
        return (epsilon, weight)





    statistics = {
        #"wdist_2_2Way_avg": "$SW_2^2$ distance", 
                "synth_gradboost_test": "test error",
                                            "cov_fixed_frobenius_norm": "cov. mat. error", 
                "rand_coun_new_mean_dist": "count. queries",
                                            "rand_thrs_query_mean_dist": "thresh. queries",
                            "wdist_1_2Way_avg": "$SW_1$ dist." , 
                            "newl1_2Way_avg" : "TV dist.",
                            #"newl1_3Way_avg" : "TV distance 3Way",
                            #"wdist_1_3Way_avg": "$SW_1$ distance 3Way" , 
                }


    # axis = {
    #     "elapsed_time": 1.0, 
    #     # "wdist_2_2Way_avg": "$SW_2^2$ distance", 
    #             "rand_thrs_query_mean_dist": 1.3, 
    #             "synth_gradboost_test": 0.25,
    #             "rand_coun_new_mean_dist":1.4,
    #                         "cov_fixed_frobenius_norm": 1.7, 

    #             }




    inference_types = {"pgm_euclid":"PGM", "advanced_extended":"PrivPGD", "private_gsd": "PrivGSD", "gem":"GEM", "rap":"RAP "}






    # Create a folder to save the legend plot
    script_file = os.path.abspath(__file__)  # Gets the absolute path of the currently executed file
    script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
    legend_plot_folder = os.path.join(script_folder, 'legend_plots')
    os.makedirs(legend_plot_folder, exist_ok=True)



    # Create a dictionary to store legend information
    legend_info = {}
    plt.rcParams.update({'font.size': 24})
    plt.rcParams['lines.linewidth'] = 4.0
    plt.rcParams['lines.markersize'] = 16.0  # Adjust the value as needed
    #plt.rcParams['text.usetex'] = True


    corr = "dataset_n"

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

    # Now, create the LaTeX table
    # Construct the LaTeX table
    # latex_table = "\\begin{tabular}[t!]{|c|" + "c|" * len(statistics.values()) + "}\\n"
    # latex_table += "\\hline\\n"
    # latex_table += "Method & " + " & ".join(statistics.values()) + " \\\\\\n"
    # latex_table += "\\hline\\n"

    # for method in methods:
    #     inference_type,mechanism = method.split('+')

    #     if "AIM" in mechanism:
    #         mechanism = "AIM-"+mechanism.split('_')[1]
    #     elif "MST" not in mechanism:
    #         mechanism  = mechanism+"-2"
            
    #     if "advanced_extended" in inference_type:
    #         mechanism = mechanism.split('_')[0]


    #     row = [" "+inference_types[inference_type]+" "+mechanism]
    #     for plot_stat in statistics.keys():
    #         win_count = winners[method][plot_stat]
    #         second_count = second_winners[method][plot_stat]
    #         row.append(f"{win_count} ({second_count})")
    #     latex_table += " & ".join(row) + " \\\\\\n"
    #     latex_table += "\\hline\\n"

    # latex_table += "\\end{tabular}"

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


    # Update the file with the new tabl
    print(latex_table)
    # Saving to a .tex file
    with open(os.path.join(script_folder, f"counts_{epsilon}.tex"), 'w') as file:
        file.write(latex_table)
                









