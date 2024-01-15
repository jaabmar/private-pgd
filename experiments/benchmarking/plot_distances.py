

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
from collections import defaultdict

import numpy as np
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
marker_styles = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'x', '+']
colors = ['#377eb8', '#ff7f00', '#4daf4a',
'#f781bf', '#a65628', '#984ea3',
'#999999', '#e41a1c', '#dede00']


project_name = "experiment_scale"

maps_dataset={
    "ans_employment_CA_2018":"Emp. ", 
    "ans_income_CA_2018": "Inc. ", 
        "ans_traveltime_CA_2018": "Tra. ", 
    "ans_publiccoverage_CA_2018":"Pub. ", 
    "ans_mobility_CA_2018": "Mob. ", 
        "nyc-taxi-green-dec-2016_regression":"Taxi " , 
        "black_friday_regression":"Fri. ",
    "medical_charges_regression": "Med. ",
     "Diabetes130US":"Diab. " 
    }

# maps_dataset={
#     "Diabetes130US":"Diab.",
#     "Higgs":"Higgs",
#     "SGEMM_GPU_kernel_performance_regression":"GPU", 
#     "ans_employment_CA_2018":"ACS Emp.", 
#     "ans_income_CA_2018": "ACS Inc.", 
#     "ans_mobility_CA_2018": "ACS Mob.", 
#     "ans_publiccoverage_CA_2018":"ACS Pub.", 
#     "ans_traveltime_CA_2018": "ACS Tra.", 
#     "black_friday_regression":"Fri.", 
#     "covertype":"Cov.", 
#     "diamonds_regression":"Diam.", 
#     "electricity": "Ele.", 
#     "medical_charges_regression": "Med.", 
#     "nyc-taxi-green-dec-2016_regression":"Taxi", 
#     "particulate-matter-ukair-2017_regression" : "PMU", 
# }

indices_datasets = {key : i+1 for i, key in enumerate(maps_dataset.keys())}





script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
k = 32
name = "full"
ylim = 4

epsilons = [0.2, 1.0 , 2.5]
print("HI")

methods = ["privpgd+KWay", "pgm_euclid+AIM", "pgm_euclid+MST", "private_gsd+KWay", "gem+KWay"]

with open(os.path.join(script_folder, f"save_stats"), "rb") as file:
    filtered_data = pickle.load(file)


def custom_sort_key(key):
    epsilon, name = key.split('+')
    epsilon = float(epsilon)
    
    # Assign a weight to the names based on your desired order
    name_weight = {
        "privpgd": 1,
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




statistics = {
                    "wdist_1_2Way_avg": "$SW_1$ distance"  ,
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




inference_types = {"pgm_euclid":"PGM", "privpgd":"PrivPGD", "private_gsd": "Private GSD", "gem":"GEM", "rap":"RAP"}



for epsilon in epsilons:

    for plot_stat in statistics.keys():
        ymean = {}
        ystd = {}
        plt.figure(figsize=(28, 9))
        plt.title("")
        maxexceed  = 0



        for dataset in maps_dataset.keys():
            if  dataset not in maps_dataset:
                continue
            ymean[dataset] = {}
            ystd[dataset]={}
            for (j,method) in enumerate(methods):            
                name_run= f"{epsilon}+{method}"

                if name_run not in filtered_data[dataset] or plot_stat not in filtered_data[dataset][name_run]:
                    continue
                if  plot_stat == "synth_gradboost_test" and "regression" not in dataset:
                    ymean[dataset][method] = 1- np.mean(filtered_data[dataset][name_run][plot_stat])
                    ystd[dataset][method] = np.std(filtered_data[dataset][name_run][plot_stat])

                else:
                    ymean[dataset][method] = np.mean(filtered_data[dataset][name_run][plot_stat])
                    ystd[dataset][method] = np.std(filtered_data[dataset][name_run][plot_stat])


        # normalize by minimum mean of all methods:
        for dataset in ymean.keys():
            privpgd_mean = ymean[dataset]["privpgd+KWay"]
            for method in ymean[dataset].keys():
                ymean[dataset][method] = ymean[dataset][method]/privpgd_mean
                ystd[dataset][method] = ystd[dataset][method]/privpgd_mean

        # Now order the datasets by the gap between privpgd and the best method (except privpgd)
        # First compute the gap for each dataset    
        gap = {}    
        for dataset in ymean.keys():
            assert("privpgd+KWay" in ymean[dataset])
            gap[dataset] =  min([ymean[dataset][method] for method in ymean[dataset].keys() if method != "privpgd+KWay"])

        # Now sort the datasets by the gap
        #datasets = sorted(gap.keys(), key=lambda x: gap[x], reverse=True)
        #find the dataset with the largest gap such that gap < 1
        datasets = gap.keys()


            
        # now get a map from dataset to index
        indices_datasets = {dataset: i for i, dataset in enumerate(datasets)}

        # Now plot the data in the following way. For each dataset in datasets, set yticks = inidices_datasets[dataset] (we keep the ordering)
        # Then plot the data for each method in methods, with x = indices_datasets[dataset] and y = ymean[dataset][method] including the standard deviation

        # First set the xticks
        plt.xticks([indices_datasets[dataset] for dataset in datasets], [maps_dataset[dataset] for dataset in datasets],  fontsize=20)
        #show all yticks and rotate them vertically 
        plt.tick_params(axis='x', which='both', labelsize=24, labelrotation=90)

        # Now plot the data
        #handle the case where the method is not in ymean[method]

        nexceed = {}
        for (j,method) in enumerate(methods):
            x, y, yerru, yerrl =[], [], [],[]
            for dataset in maps_dataset.keys():
                if not dataset in nexceed:
                    nexceed[dataset] = 0
                if method in ymean[dataset]:
                    if ymean[dataset][method] <= ylim:
                        x.append(indices_datasets[dataset])

                        yval = ymean[dataset][method]

                        y.append(yval)
                        yerru.append(min(ystd[dataset][method], ylim-yval))
                        yerrl.append(ystd[dataset][method])

                    else:
                        nexceed[dataset] += 1
                        x.append(indices_datasets[dataset])
                        y.append(ylim+ nexceed[dataset]*0.4)
                        yerru.append(0)
                        yerrl.append(0)
                        # yerru.append(ylim+ nexceed[plot_stat]*0.3)
                        # yerrl.append(ylim+ nexceed[plot_stat]*0.3)


            # Combine into a 2xN array
            yerr_adjusted = np.vstack([yerrl, yerru])
            plt.errorbar(x, y, yerr=yerr_adjusted, label=f"{inference_types[method.split('+')[0]]}+{method.split('+')[1]}", linestyle='', marker=marker_styles[j],  color=colors[j])
            maxexceed = max(max(nexceed.values()), maxexceed)



        # for (j,method) in enumerate(methods):
        #     x, y, yerr =[], [], []
        #     for dataset in datasets:
        #         if method in ymean[dataset]:
        #             x.append(indices_datasets[dataset])
        #             y.append(ymean[dataset][method])
        #             yerr.append(ystd[dataset][method])


        #     plt.errorbar(x, y, yerr=yerr, label=f"{inference_types[method.split('+')[0]]}+{method.split('+')[1]}", linestyle='', marker=marker_styles[j],  color=colors[j])
        

        # for the first index where gap <1, draw a horizontal line
        # threshold =0
        # for dataset in datasets:
        #     if gap[dataset] < 1:
        #         continue
        #     threshold += 1
        # if threshold >0 and threshold < len(datasets):
        #     #let the line be dashed and grey
        #     plt.axvline(x=threshold+1/2-1, color='grey', linestyle='--')
        


        # set the ylims to be upper bounded by 2 and lower bounded by 0 
        plt.axhline(y=ylim, color='gray', linestyle='--')
            #plt.text(10, ylim+0.2, f'{ylim}+', horizontalalignment='right')
        plt.ylim(bottom=-0.2, top=ylim+maxexceed*0.4+0.25)  


        yticks = np.arange(0, ylim+1,  1.0).tolist()  # Standard ticks from 0 to ylim
        
        # Adding a special tick for "ylim+"
        yticks_labels = [str(tick) for tick in yticks]
        if maxexceed% 2 ==0:
            yticks.append(ylim+(maxexceed//2 + 0.5)*0.4)
            yticks_labels.append(f">{ylim}")
        else:
            yticks.append(ylim+(maxexceed//2 + 1)*0.4)
            yticks_labels.append(f">{ylim}")
            
        # plt.xlabel('Dataset', fontsize=28)
        #remove the grid around the plot
        plt.yticks(yticks, yticks_labels)

        plt.grid(False)
        #remove the raster around the plot
        plt.box(False) 


        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.2)
        #plt.ylabel("")
        #plt.legend()
        # Save the current statistic plot

        # Save the current statistic plot
        os.makedirs(os.path.join(script_folder, f"plots_distances/{epsilon}/"), exist_ok=True)
        filename = os.path.join(script_folder, f'plots_distances/{epsilon}/{plot_stat}.pdf')
        plt.savefig(filename)
        filename = os.path.join(script_folder, f'plots_distances/{epsilon}/{plot_stat}.png')
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
        mechanism = "+AIM"#+mechanism.split('_')[1]
    elif "MST" not in mechanism:
        mechanism  = ""
    else:
        mechanism = "+MST"
        
    #if "advanced_extended" in inference_type:
    #mechanism = mechanism.split('_')[0]

    
    inference_type = inference_types[inference_type]

    line, = plt.plot([], [], label=f"{inference_type}{mechanism}", linestyle='', marker=marker_styles[j],  color=colors[j])

    lines.append(line)



# Place the legend inside the legend axes, adjusted automatically
legend = plt.legend(handles=lines, ncol=6,framealpha=1, frameon=False)

plt.axis('off')
plt.savefig("legend_2.png")
fig  = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#fig.savefig(filename, bbox_inches=bbox)
# Save the legend plot
os.makedirs(os.path.join(script_folder, f"legend_plots/"), exist_ok=True)

legend_filename = os.path.join(os.path.join(script_folder,f"legend_plots"), f'exp12legend_plot.pdf')
plt.savefig(legend_filename, bbox_inches=bbox)
legend_filename = os.path.join(os.path.join(script_folder,f"legend_plots"), f'exp12legend_plot.png')
plt.savefig(legend_filename, bbox_inches=bbox)
plt.close()





