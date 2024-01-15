

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
list_maps_dataset=[{
    "ans_employment_CA_2018":"Emp. ", 
    "ans_income_CA_2018": "Inc. ", 
        "ans_traveltime_CA_2018": "Tra. ", 
    "ans_publiccoverage_CA_2018":"Pub. ", 
    "ans_mobility_CA_2018": "Mob. ", 

}, {      "nyc-taxi-green-dec-2016_regression":"Taxi " , 
        "black_friday_regression":"Fri. ",
    "medical_charges_regression": "Med. ",
     "Diabetes130US":"Diab. " 
    }]






script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
k = 32
name = "full"

epsilons = [0.2, 1.0 , 2.5]
print("HI")


def transform(x):
    return x
ylim = 4
ylim_neg = 0

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
                            "synth_gradboost_test": "down.",
            "cov_fixed_frobenius_norm": "cov.",
            "rand_thresholding_query_mean_dist": "thrs.",
            "rand_counting_query_mean_dist": "count."

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

shift_factor = 6

inference_types = {"pgm_euclid":"PGM", "privpgd":"PrivPGD", "private_gsd": "Private GSD", "gem":"GEM", "rap":"RAP"}

total_records = {}
dimensions = {}

for (ndata,maps_dataset) in enumerate(list_maps_dataset):
    for epsilon in epsilons:
        plt.figure(figsize=(28, 9))
        plt.title("")
        xticks = []
        xticksmetric = []
        maxexceed = 0
        counter =0
        for dataset in maps_dataset:

            ymean = {}
            ystd = {}


            for plot_stat in statistics.keys():


                ymean[plot_stat] = {}
                ystd[plot_stat]={}
                for (j,method) in enumerate(methods):            
                    name_run= f"{epsilon}+{method}"

                    if name_run not in filtered_data[dataset] or plot_stat not in filtered_data[dataset][name_run]:
                        continue
                    if "dataset_n" in filtered_data[dataset][name_run]:
                        total_records[dataset] = int(np.mean(filtered_data[dataset][name_run]["dataset_n"]))
                        dimensions[dataset] = int(np.mean(filtered_data[dataset][name_run]["dataset_d"]))
                    if  plot_stat == "synth_gradboost_test" and "regression" not in dataset:
                        ymean[plot_stat][method] = 1- np.mean(filtered_data[dataset][name_run][plot_stat])
                        ystd[plot_stat][method] = np.std(filtered_data[dataset][name_run][plot_stat])

                    else:
                        ymean[plot_stat][method] = np.mean(filtered_data[dataset][name_run][plot_stat])
                        ystd[plot_stat][method] = np.std(filtered_data[dataset][name_run][plot_stat])


            # normalize by minimum mean of all methods:
            for plot_stat in ymean.keys():
                privpgd_mean = ymean[plot_stat]["privpgd+KWay"]
                for method in ymean[plot_stat].keys():
                    ymean[plot_stat][method] = ymean[plot_stat][method]/privpgd_mean
                    ystd[plot_stat][method] = ystd[plot_stat][method]/privpgd_mean

            # Now order the datasets by the gap between privpgd and the best method (except privpgd)
            # First compute the gap for each dataset    
            # gap = {}    
            # for dataset in ymean.keys():
            #     assert("privpgd+KWay" in ymean[dataset])
            #     gap[dataset] =  min([ymean[dataset][method] for method in ymean[dataset].keys() if method != "privpgd+KWay"])

            # # Now sort the datasets by the gap
            # datasets = sorted(gap.keys(), key=lambda x: gap[x], reverse=True)
            # #find the dataset with the largest gap such that gap < 1


                
            # now get a map from dataset to index
            indices_datasets = {plot_stat: i for i, plot_stat in enumerate(statistics.keys())}
            # Now plot the data in the following way. For each dataset in datasets, set yticks = inidices_datasets[dataset] (we keep the ordering)
            # Then plot the data for each method in methods, with x = indices_datasets[dataset] and y = ymean[dataset][method] including the standard deviation

            # First set the xticks

            # plt.xticks([indices_datasets[plot_stat] for plot_stat in statistics.keys()], [statistics[plot_stat] for plot_stat in statistics.keys()],  fontsize=20)
            # #show all yticks and rotate them vertically 
            # plt.tick_params(axis='x', which='both', labelsize=24, labelrotation=0)

            # Now plot the data
            #handle the case where the method is not in ymean[method]
            shift = counter *shift_factor
            xticks.append([shift +(shift_factor -1/2)/2, maps_dataset[dataset]+ "(n="+ str(total_records[dataset])+", d="+str(dimensions[dataset])+")"])

            nexceed = {}
            for (j,method) in enumerate(methods):
                x, y, yerru, yerrl =[], [], [],[]
                for plot_stat in statistics.keys():
                    if not plot_stat in nexceed:
                        nexceed[plot_stat] = 0
                    if method in ymean[plot_stat]:
                        if transform(ymean[plot_stat][method]) <= ylim:
                            x.append(indices_datasets[plot_stat]+shift+1)
                            xticksmetric.append((indices_datasets[plot_stat]+shift+1, statistics[plot_stat]))

                            yval = transform(ymean[plot_stat][method])

                            y.append(yval)

                            
                            yerru.append(min(transform(ymean[plot_stat][method] +ystd[plot_stat][method]) - yval, ylim-yval))
                            yerrl.append(transform(ystd[plot_stat][method]+ymean[plot_stat][method]) - yval)

                        else:
                            nexceed[plot_stat] += 1
                            x.append(indices_datasets[plot_stat] + shift+1)
                            y.append(ylim+ nexceed[plot_stat]*0.4)
                            yerru.append(0)
                            yerrl.append(0)
                            # yerru.append(ylim+ nexceed[plot_stat]*0.3)
                            # yerrl.append(ylim+ nexceed[plot_stat]*0.3)


                # Combine into a 2xN array
                yerr_adjusted = np.vstack([yerrl, yerru])
                plt.errorbar(x, y, yerr=yerr_adjusted, label=f"{inference_types[method.split('+')[0]]}+{method.split('+')[1]}", linestyle='', marker=marker_styles[j],  color=colors[j])
            
            #set the y ticks as follows. Put them in the interval 0 to ylim. For any value above ylim, then add a dashed line and a tick ylim+
            maxexceed = max(max(nexceed.values()), maxexceed)


           
            counter += 1



        #plt.subplots_adjust(bottom=2.0) # Increase the value to increase space

        #yticks.append(ylim)  # Adding ylim to the list of ticks

        for j in range(len(maps_dataset.keys())-1):
            #plot a vertical line
            plt.axvline(x=(j+1)*shift_factor -1/2, color='gray', linestyle=':')

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


        plt.yticks(yticks, yticks_labels)

        plt.xticks([tick[0] for tick in xticks], [tick[1] for tick in xticks],  fontsize=24)
        # #show all yticks and rotate them vertically 
        # plt.tick_params(axis='x', which='both', labelsize=24, labelrotation=0)

        # set the ylims to be upper bounded by 2 and lower bounded by 0 
        #plt.ylim(bottom=0, top=ylim)
            
        # plt.xlabel('Dataset', fontsize=28)
        #remove the grid around the plot


        # Creating a second x-axis
        ax1 = plt.gca()
        xlims_range = ax1.get_xlim()

        ax2 = ax1.twiny()
        ax2.set_xlim(xlims_range)

        xticksmetric = list(set(xticksmetric))
        xticksmetric.sort(key=lambda x: x[0])
        ax2.set_xticks([tick[0] for tick in xticksmetric], [tick[1] for tick in xticksmetric],  fontsize=24, rotation=90)
        plt.grid(False)
        #remove the raster around the plot
        plt.box(False) 

        plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.05)
        plt.ylabel("")
        #plt.legend()
        # Save the current statistic plot

        # Save the current statistic plot
        os.makedirs(os.path.join(script_folder, f"plots_dataset_full/{epsilon}/"), exist_ok=True)
        filename = os.path.join(script_folder, f'plots_dataset_full/{epsilon}/{ndata}.pdf')
        plt.savefig(filename)
        filename = os.path.join(script_folder, f'plots_dataset_full/{epsilon}/{ndata}.png')
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





