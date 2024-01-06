

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
colors = ['#377eb8', '#ff7f00', '#4daf4a',
'#f781bf', '#a65628', '#984ea3',
'#999999', '#e41a1c', '#dede00']


project_name = "experiment_scale"



script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
k = 32
name = "full"

#epsilons = [0.2, 1.0, 2.5]
epsilons = [0.2, 1.0 , 2.5]
print("HI")
method_base = "advanced_extended+KWay" 

method = "localPGM+KWay"

with open(os.path.join(script_folder, f"save_stats"), "rb") as file:
    filtered_data = pickle.load(file)

def custom_sort_key(key):
    epsilon, name = key.split('+')
    epsilon = float(epsilon)
    
    # Assign a weight to the names based on your desired order
    name_weight = {
        "advanced_extended": 1,
        "private_gsd":3,
        "pgm_euclid": 2,
        "gem":4, 
        "rap":5, 
        "appgm":6
    }
    
    # If the name is not in name_weight, assign a default weight
    weight = name_weight.get(name, float('inf'))
    
    # Sort first by epsilon, then by the weight of the name
    return (epsilon, weight)


plot_stats = ["wdist_2_workload_Extended_45_avg", "synth_gradboost_test", "cov_diff_spectral_norm", "rand_coun_query_mean_dist", "rand_thrs_query_mean_dist"]

statistics = {
                    "wdist_1_2Way_avg": "$\mathrm{SW}_1$ distance" , 
                    "newl1_2Way_avg" : "TV distance",

            }




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




inference_types = {"pgm_euclid":"PGM", "advanced_extended":"PrivPGD", "private_gsd": "Private GSD", "gem":"GEM", "rap":"RAP", "localPGM":"AP-PGM"}


corr = "dataset_n"

for epsilon in epsilons:

    xdata = {}
    ybase = {}
    ycompare = {}
    plt.figure(figsize=(10, 8))
    plt.title("")



    j = 0
    for plot_stat in statistics.keys():

        for dataset in filtered_data.keys():
            if "acs" in dataset:
                continue
            if epsilon >2.5:
                epsilon = int(epsilon)

            name_run_KWay= f"{epsilon}+{method_base}"
    
            if name_run_KWay not in filtered_data[dataset] or plot_stat not in filtered_data[dataset][name_run_KWay] or corr not in filtered_data[dataset][name_run_KWay]:
                continue

            ybase[dataset] = np.mean(filtered_data[dataset][name_run_KWay][plot_stat])
            xdata[dataset] = np.mean(filtered_data[dataset][name_run_KWay][corr])




        ycompare = {}
        yratio = []
        xvalues = []
        for dataset in filtered_data.keys():
            if epsilon >2.5:
                epsilon = int(epsilon)
            name_run= f"{epsilon}+{method}"

            if plot_stat == "newl1_2Way_avg":
                plot_stat_new = "l1_mean"
            elif plot_stat =="wdist_1_2Way_avg":
                plot_stat_new = "w_mean"
            else:
                plot_stat_new = plot_stat

            if name_run not in filtered_data[dataset] or plot_stat_new not in filtered_data[dataset][name_run]:
                continue
            if  plot_stat == "synth_gradboost_test" and "regression" not in dataset:
                ycompare[dataset] = 1- np.mean(filtered_data[dataset][name_run][plot_stat_new])
            else:
                ycompare[dataset] = np.mean(filtered_data[dataset][name_run][plot_stat_new])

            if dataset in ybase:
                yratio.append( ycompare[dataset]/ybase[dataset])
                xvalues.append(xdata[dataset])


        if yratio is not None:
            plt.scatter(np.array(xvalues)/100000,np.log10(np.array(yratio)),   color=colors[j], marker=marker_styles[j])
            j = j+1

    plt.ylabel('$\log_{10}$ ratio', fontsize=32)
    plt.grid(True, which='both')

    plt.xlabel('# data points in 100k', fontsize=28)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
    plt.axhline(0.0, linestyle="--", color="black")
    # Save the current statistic plot
    os.makedirs(os.path.join(script_folder, f"plots_appgm/"), exist_ok=True)
    filename = os.path.join(script_folder, f'plots_appgm/{epsilon}.pdf')
    plt.savefig(filename)
    filename = os.path.join(script_folder, f'plots_appgm/{epsilon}.png')
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
for (j,stat ) in enumerate(statistics):

    #if "advanced_extended" in inference_type:
    #mechanism = mechanism.split('_')[0]

    

    line, = plt.plot([], [], label=statistics[stat], linestyle='', marker=marker_styles[j],  color=colors[j])

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
os.makedirs(os.path.join(script_folder, f"legend_plots_appgm/"), exist_ok=True)

legend_filename = os.path.join(os.path.join(script_folder,f"legend_plots_appgm"), f'exp12legend_plot.pdf')
plt.savefig(legend_filename, bbox_inches=bbox)
legend_filename = os.path.join(os.path.join(script_folder,f"legend_plots_appgm"), f'exp12legend_plot.png')
plt.savefig(legend_filename, bbox_inches=bbox)
plt.close()





