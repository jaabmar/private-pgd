

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
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'x', '+']

statistics = {
    "wdist_2_2Way_avg": "$SW_2^2$ distance", 
            "rand_thrs_query_mean_dist": "thresholding queries",
            "synth_gradboost_test": "class/reg test error",
            "rand_coun_new_mean_dist": "counting queries",
                        "cov_fixed_frobenius_norm": "covariance matrix", 
                        "w_mean": "w_mean" , 
                        "l1_2Way_avg" : "TV distance"

            }

inference_types = {"pgm_euclid":"PGM", "advanced_extended":"PrivPGD", "private_gsd": "PrivGSD", "gem":"GEM"}



project_name = "experiment_scale"



script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
k = 32
name = "full"



#epsilons = [2.5]


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







epsilons = [2.5]
otheralgos =["gem", "private_gsd"]

for epsilon in epsilons:
    print(f"start with {epsilon}")

    with open(os.path.join(script_folder, f"save_stats_{epsilon}_2"), "rb") as file:
        filtered_data = pickle.load(file)

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
    methods = ["pgm_euclid+AIM", "pgm_euclid+MST", "gem+KWay", "private_gsd+KWay" ]
    methods_base = ["KWay"]
    for plot_stat in statistics.keys():
        for method_base in methods_base:
            plt.figure(figsize=(10, 6))
            plt.title("")

            for method in methods:
                y = []
                xmst = []
                xkway =[]
                y_reg = []
                xmst_reg = []
                xkway_reg =[]
                for dataset in filtered_data.keys():
                    if epsilon >2.5:
                        epsilon = int(epsilon)

                    name_run_KWay= f"{epsilon}+advanced_extended+{method_base}"
                    name_run_MST = f"{epsilon}+{otheralgo}+{method}"

                    if name_run_MST not in filtered_data[dataset] or name_run_KWay not in filtered_data[dataset]:
                        continue
                    if plot_stat not in filtered_data[dataset][name_run_MST] or plot_stat not in filtered_data[dataset][name_run_KWay] or corr not in filtered_data[dataset][name_run_KWay]:
                        continue
                    if "regression" not in dataset:
                        if plot_stat == "synth_gradboost_test":
                            xmst.append(1- np.mean(filtered_data[dataset][name_run_MST][plot_stat]))
                            xkway.append(1- np.mean(filtered_data[dataset][name_run_KWay][plot_stat]))
                        else:
                            xmst.append(np.mean(filtered_data[dataset][name_run_MST][plot_stat]))
                            xkway.append(np.mean(filtered_data[dataset][name_run_KWay][plot_stat]))
                        y.append(np.mean(filtered_data[dataset][name_run_KWay][corr]))

                    else:
                        xmst.append(np.mean(filtered_data[dataset][name_run_MST][plot_stat]))
                        xkway.append(np.mean(filtered_data[dataset][name_run_KWay][plot_stat]))
                        y.append(np.mean(filtered_data[dataset][name_run_KWay][corr]))


                if len(y) ==0:
                    plt.close()

                    continue
                else:
                    print("not close")
                    # Plot the shaded area for standard deviation
                xmst = np.array(xmst)
                xkway = np.array(xkway)
                y = np.array(y) /100000
                ratio = xmst/xkway
                ratio_transformed = np.log10(ratio)

                mask_xl = ratio_transformed < 0  # Values in x that are less than 0
                mask_xu = ratio_transformed >= 0  # Values in x that are greater than or equal to 0

                # Use the masks to create the four new vectors
                xl = ratio_transformed[mask_xl]
                yl = y[mask_xl]
                xu = ratio_transformed[mask_xu]
                yu = y[mask_xu]



                thrs = 2

                #np.where(ratio > thrs, thrs+np.log2(ratio-thrs),ratio)
                #ratio_reg_transformed = np.where(ratio_reg > 2, 2+np.log10(ratio_reg-2), ratio_reg)
                
                plt.scatter(yl, xl, color='red', marker='s')
                plt.scatter(yu, xu, color='blue', marker='d')


                slope, intercept = np.polyfit(y,ratio_transformed ,1)

                # Create a linear trendline function

                plt.plot( [0, max(max(y),1)],[0.0,0.0],  color='black', marker='', linestyle="-", linewidth = 3)
                #plt.plot([y,y], [xmst, xkway], color='blue', linestyle='-', marker=False, label=False)

                # Plot the linear trendline as a grey dashed line
                trendline_x = np.array([min(y), max(y)])  # Generate a range of x values
                trendline = slope * trendline_x + intercept

                plt.plot(trendline_x,  trendline, color='gray', linestyle='--', linewidth = 3)

                # if epsilon == 2.5:
                #     plt.ylim(-axis[plot_stat], axis[plot_stat])




                # Set axis labels
                plt.ylabel('$\log_{10}$ ratio', fontsize=32)

                # Manually set y-axis ticks and labels
                #yticks = np.array([0.5, 1.0, 1.5, 2.0, 10.0, 100.0])
                ytick_labels = ["0.5", "1.0", "1.5", "2.0", "5.0", "10.0", "50.0"]
                #plt.yticks(np.log2(yticks), ytick_labels)
                plt.grid(True, which='both')

                plt.xlabel('# data points in 100k', fontsize=28)
                plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)

                # Save the current statistic plot
                os.makedirs(os.path.join(script_folder, f"plots_cor_{otheralgo}/{epsilon}_{method}+{method_base}/"), exist_ok=True)
                filename = os.path.join(script_folder, f'plots_cor_{otheralgo}/{epsilon}_{method}+{method_base}/{corr}_{plot_stat}.pdf')
                plt.savefig(filename)
                filename = os.path.join(script_folder, f'plots_cor_{otheralgo}/{epsilon}_{method}+{method_base}/{corr}_{plot_stat}.png')
                plt.savefig(filename)  
                plt.close()
                print(f"save {method_base}")


                import os




    def export_legend(legend, filename="legend.png"):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig(filename, bbox_inches=bbox)


    plt.rcParams.update({'font.size': 14})
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8.0  # Adjust the value as needed
    # Create a legend plot
    lines = []
    line, = plt.plot([], [], label="PrivPGD performing better", linestyle='', marker='d', color='blue')
    lines.append(line)

    line, = plt.plot([], [], label="PGM performing better     ", linestyle='', marker='s', color='red')

    lines.append(line)



    # Place the legend inside the legend axes, adjusted automatically
    legend = plt.legend(handles=lines, ncol=2,framealpha=1, frameon=False)

    plt.axis('off')
    plt.savefig("legend_2.png")
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #fig.savefig(filename, bbox_inches=bbox)
    # Save the legend plot
    os.makedirs(os.path.join(script_folder, f"legend_plots_{otheralgo}/"), exist_ok=True)

    legend_filename = os.path.join(os.path.join(script_folder,f"legend_plots_{otheralgo}"), f'exp12legend_plot.pdf')
    plt.savefig(legend_filename, bbox_inches=bbox)
    plt.close()





