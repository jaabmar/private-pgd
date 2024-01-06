

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

#epsilons = [0.2, 1.0, 2.5]
epsilons = [2.5]
print("HI")
methods = ["advanced_extended+KWay_0" , "pgm_euclid+AIM_2", "pgm_euclid+AIM_3", "pgm_euclid+MST", "private_gsd+KWay", "gem+KWay" ]
#methods = ["advanced_extended+KWay_1" ]


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




    plot_stats = ["wdist_2_workload_Extended_45_avg", "synth_gradboost_test", "cov_diff_spectral_norm", "rand_coun_query_mean_dist", "rand_thrs_query_mean_dist"]

    statistics = {
                "rand_thrs_query_mean_dist": "thresholding queries",
                "synth_gradboost_test": "class/reg test error",
                "rand_coun_new_mean_dist": "counting queries",
                            "cov_fixed_frobenius_norm": "covariance matrix", 
                        "wdist_1_2Way_avg": "$SW_1$ distance" , 
                            "wdist_2_2Way_avg": "$SW_1$ distance" , 
                        "newl1_2Way_avg" : "TV distance",
                        "newl1_3Way_avg" : "TV distance 3Way",
                        "newl1_2Way_max" : "TV distance",
                        "newl1_3Way_max" : "TV distance 3Way",
                        "wdist_1_3Way_avg": "$SW_1$ distance 3Way" , 
                        "wdist_1_3Way_max": "$SW_1$ distance 3Way" , 
                        "wdist_1_2Way_max": "$SW_1$ distance 3Way" , 
                        "wdist_2_3Way_max": "$SW_1$ distance 3Way" , 
                        "wdist_2_2Way_max": "$SW_1$ distance 3Way" , 
                        "w_mean": "$SW_1$ distance 3Way" , 
                }


    # axis = {
    #     "elapsed_time": 1.0, 
    #     # "wdist_2_2Way_avg": "$SW_2^2$ distance", 
    #             "rand_thrs_query_mean_dist": 1.3, 
    #             "synth_gradboost_test": 0.25,
    #             "rand_coun_new_mean_dist":1.4,
    #                         "cov_fixed_frobenius_norm": 1.7, 

    #             }




    inference_types = {"pgm_euclid":"PGM", "advanced_extended":"PrivPGD", "private_gsd": "PrivGSD", "gem":"GEM"}






    # Create a folder to save the legend plot
    script_file = os.path.abspath(__file__)  # Gets the absolute path of the currently executed file
    script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
    legend_plot_folder = os.path.join(script_folder, 'legend_plots')
    os.makedirs(legend_plot_folder, exist_ok=True)



    # Create a dictionary to store legend information
    legend_info = {}
    plt.rcParams.update({'font.size': 28})
    plt.rcParams['lines.linewidth'] = 4.0
    plt.rcParams['lines.markersize'] = 18.0  # Adjust the value as needed
    #plt.rcParams['text.usetex'] = True


    corr = "dataset_n"



    print("HI")
    for plot_stat in statistics.keys():
        # Prepare data here... 
        xdata = {}
        ybase = {}
        ycompare = {}
        plt.figure(figsize=(10, 8))
        plt.title("")


        for dataset in filtered_data.keys():
            if "acs" in dataset:
                continue
            if epsilon >2.5:
                epsilon = int(epsilon)


        for (j,method) in enumerate(methods):
            ycompare[method] = {}
            yfull = []
            xvalues = []
            for dataset in filtered_data.keys():
                if epsilon >2.5:
                    epsilon = int(epsilon)

                name_run= f"{epsilon}+{method}"

                if name_run not in filtered_data[dataset] or plot_stat not in filtered_data[dataset][name_run]  or corr not in filtered_data[dataset][name_run]:
                    continue
                if  plot_stat == "synth_gradboost_test" and "regression" not in dataset:
                    ycompare[method][dataset] = 1- np.mean(filtered_data[dataset][name_run][plot_stat])
                else:
                    ycompare[method][dataset] = np.mean(filtered_data[dataset][name_run][plot_stat])

                yfull.append( ycompare[method][dataset])
                xvalues.append(np.mean(filtered_data[dataset][name_run][corr]))


            if yfull is not None:
                plt.scatter(np.array(xvalues), np.array(yfull), color=colors[j], marker=marker_styles[j])




        # Set axis labels
        #plt.ylabel('$', fontsize=32)

        # Manually set y-axis ticks and labels
        #yticks = np.array([0.5, 1.0, 1.5, 2.0, 10.0, 100.0])
        #ytick_labels = ["0.5", "1.0", "1.5", "2.0", "5.0", "10.0", "50.0"]
        #plt.yticks(np.log2(yticks), ytick_labels)
        plt.grid(True, which='both')

        plt.xlabel('# data points in 100k', fontsize=28)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        #plt.axhline(0.0, linestyle="--", color="black")
        # Save the current statistic plot
        os.makedirs(os.path.join(script_folder, f"plots_full/{epsilon}/"), exist_ok=True)
        filename = os.path.join(script_folder, f'plots_full/{epsilon}/{plot_stat}.pdf')
        plt.savefig(filename)
        filename = os.path.join(script_folder, f'plots_full/{epsilon}/{plot_stat}.png')
        plt.savefig(filename)  
        plt.close()






def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #fig.savefig(filename, bbox_inches=bbox)


plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 10.0  # Adjust the value as needed
# Create a legend plot
lines = []
for (j,method ) in enumerate(methods):
    inference_type, mechanism = method.split('+')
    if "AIM" in mechanism:
        mechanism = "AIM-"+mechanism.split('_')[1]
    elif "MST" not in mechanism:
        mechanism  = mechanism+"-2"
        
    if "advanced_extended" in inference_type:
        mechanism = mechanism.split('_')[0]

    inference_type = inference_types[inference_type]

    line, = plt.plot([], [], label=f"{inference_type}+{mechanism}", linestyle='', marker=marker_styles[j],  color=colors[j])

    lines.append(line)



# Place the legend inside the legend axes, adjusted automatically
legend = plt.legend(handles=lines, ncol=5,framealpha=1, frameon=False)

plt.axis('off')
plt.savefig("legend_2.png")
fig  = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#fig.savefig(filename, bbox_inches=bbox)
# Save the legend plot
os.makedirs(os.path.join(script_folder, f"legend_plots_full/"), exist_ok=True)

legend_filename = os.path.join(os.path.join(script_folder,f"legend_plots_full"), f'exp12legend_plot.pdf')
plt.savefig(legend_filename, bbox_inches=bbox)
legend_filename = os.path.join(os.path.join(script_folder,f"legend_plots_full"), f'exp12legend_plot.png')
plt.savefig(legend_filename, bbox_inches=bbox)
plt.close()





