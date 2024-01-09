

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





script_file = os.path.abspath(__file__) 
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script
k = 32


epsilons = [0.2, 1.0 , 2.5]

methods_base = []

methods = {"advanced_extended+KWay" :"PrivPGD" , "pgm_euclid+AIM" :"PGM+AIM", "pgm_euclid+MST":"PGM+MST", "private_gsd+KWay":"Private GSD", "gem+KWay":"GEM", "rap+KWay":"RAP", "localPGM+KWay":"AP-PGM"} 
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
        "localPGM":6
    }
    
    # If the name is not in name_weight, assign a default weight
    weight = name_weight.get(name, float('inf'))
    
    # Sort first by epsilon, then by the weight of the name
    return (epsilon, weight)


inference_types = {"pgm_euclid":"PGM", "advanced_extended":"PrivPGD", "private_gsd": "Private GSD", "gem":"GEM", "rap":"RAP", "localPGM":"AP-PGM"}

statistics = {
            "wdist_1_2Way_avg": "$SW_1$ distance" , 
                "newl1_2Way_avg" : "TV distance",
            }


maps_dataset={
    "Diabetes130US":"Diabetes",
    "Higgs":"Higgs",
    "SGEMM_GPU_kernel_performance_regression":"SGEMM GPU", 
    "ans_employment_CA_2018":"ACS Employment", 
    "ans_income_CA_2018": "ACS Income", 
    "ans_mobility_CA_2018": "ACS Mobility", 
    "ans_publiccoverage_CA_2018":"ACS Public Coverage", 
    "ans_traveltime_CA_2018": "ACS Traveltime", 
    "black_friday_regression":"Black Firday", 
    "covertype":"Covertype", 
    "diamonds_regression":"Diamonds", 
    "electricity": "Electricity", 
    "medical_charges_regression": "Medical Charges", 
    "nyc-taxi-green-dec-2016_regression":"NYC Taxi", 
    "particulate-matter-ukair-2017_regression" : "Particulate Matter Ukair", 
}



# def process_data(entry, column):
#     if column == "dataset":
#         return maps_dataset[entry]

#     else:
#         return entry




plt.rcParams.update({'font.size': 28})
plt.rcParams['lines.linewidth'] = 4.0
plt.rcParams['lines.markersize'] = 18.0  # Adjust the value as needed
#plt.rcParams['text.usetex'] = True



# Create a folder to save the legend plot
script_file = os.path.abspath(__file__)  # Gets the absolute path of the currently executed file
script_folder = os.path.dirname(script_file)  # Extracts the folder (directory) containing the script


with open(os.path.join(script_folder, f"save_stats"), "rb") as file:
    filtered_data = pickle.load(file)

for epsilon in epsilons:


    new_data = {}
    header = ["dataset", "algorithm"] + [statistics[key] for key in statistics.keys()]

    header_keys = ["dataset", "inference"] + list(statistics.keys())



    data = new_data
    data["header"] = header
    data["data"] =[]
    for dataset in filtered_data:
        if dataset not in maps_dataset:
            continue

        for key in filtered_data[dataset]:
            epsilon_val, inference = key.split('+',1)
            if float(epsilon_val) != float(epsilon):
                continue
            if inference not in methods:
                continue
            data_temp = {"dataset":dataset, "inference":methods[inference]}
            if "localPGM" not in inference:
                for key_run in statistics:
                    if key_run in filtered_data[dataset][key]:
                        tmp =  filtered_data[dataset][key][key_run]
                        mean = np.mean(tmp)
                        std_dev = np.std(tmp)

                        # Formatting the output
                        data_temp[key_run] = "{:.2g} (±{:.2g})".format(mean, std_dev)

                    else:
                        data_temp[key_run] = "-"
            else:
                for key_run in statistics:
                    if key_run ==  "wdist_1_2Way_avg":
                        tmp =  filtered_data[dataset][key]["w_mean"]
                        mean = np.mean(tmp)
                        std_dev = np.std(tmp)
                        # Formatting the output
                        data_temp[key_run] = "{:.2g} (±{:.2g})".format(mean, std_dev)
                    elif key_run ==  "newl1_2Way_avg":
                        tmp =  filtered_data[dataset][key]["l1_mean"]
                        mean = np.mean(tmp)
                        std_dev = np.std(tmp)
                        # Formatting the output
                        data_temp[key_run] = "{:.2g} (±{:.2g})".format(mean, std_dev)
                    else:
                        data_temp[key_run] = "-"
            data["data"].append(data_temp)







    min_values = {stat: {} for stat in statistics.keys()}
    for dataset in filtered_data:
        for stat in statistics.keys():
            min_mean = float('inf')
            for key in filtered_data[dataset]:
                epsilon_val, inference = key.split('+', 1)
                if float(epsilon_val) != float(epsilon) or inference not in methods:
                    continue
                if stat in filtered_data[dataset][key]:
                    mean = np.mean(filtered_data[dataset][key][stat])
                    if mean < min_mean:
                        min_mean = mean
            if min_mean < float('inf'):
                min_values[stat][dataset] = min_mean

    # Function to process data and highlight minimum values in bold
    def process_data(value, stat, dataset):
        try:
            mean = float(value.split(" (")[0])
            if float(mean) == min_values[stat][dataset]:
                return f"\\textbf{{{value}}}"
        except Exception as e:
            pass
        return value

    # Step 2: Generate LaTeX table
    header = ["dataset", "inference"] + list(statistics.keys())
    keys_inf = {value: j for (j, value) in enumerate(methods.values())}
    maps_keys = {value: j for (j, value) in enumerate(maps_dataset.values())}

    temp = data['data']
    sorted_temp = sorted(temp, key=lambda x: (maps_keys[maps_dataset[x['dataset']]], keys_inf[x['inference']]))

    latex_table = "\\begin{table*}[ht]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{" + "c" * len(header) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += " & ".join(header) + " \\\\\n"
    latex_table += "\\midrule\n"

    previous_dataset = None
    row_span_count = 0

    for i, row in enumerate(sorted_temp):
        current_dataset = maps_dataset[row.get("dataset", "")]

        if current_dataset != previous_dataset and previous_dataset is not None:
            latex_table = latex_table.replace(f"{{placeholder_{previous_dataset}}}", f"\\multirow{{{row_span_count}}}{{*}}{{{previous_dataset}}}")
            row_span_count = 1
        else:
            row_span_count += 1

        if current_dataset != previous_dataset:
            latex_table += f"{{placeholder_{current_dataset}}} & "
            previous_dataset = current_dataset
        else:
            latex_table += " & "

        latex_table += " & ".join(process_data(row[stat], stat, row['dataset']) for stat in statistics.keys()) + " \\\\\n"

        if i == len(sorted_temp) - 1:
            latex_table = latex_table.replace(f"{{placeholder_{current_dataset}}}", f"\\multirow{{{row_span_count}}}{{*}}{{{current_dataset}}}")

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Your Caption Here}\n"
    latex_table += "\\end{table*}\n"









    # keys_inf = {value:j for (j, value) in enumerate(methods.values())}
    # maps_keys = {value:j for (j, value) in enumerate(maps_dataset.values())}

    # temp = new_data['data']

    # sorted_tmp = sorted_data = sorted(temp, key=lambda x: (maps_keys[maps_dataset[x['dataset']]], keys_inf[x['inference']]))
    # new_data['data'] = sorted_tmp


    # json_data = new_data
    # # Parse the JSON data
    # header = json_data["header"]
    # data = json_data["data"]

    # # Generate LaTeX code for the table with dynamic column count using booktabs
    # latex_table = "\\begin{table*}[ht]\n"
    # latex_table += "\\centering\n"
    # latex_table += "\\begin{tabular}{" + "c" * len(header) + "}\n"
    # latex_table += "\\toprule\n"
    # latex_table += " & ".join(header) + " \\\\ \n"
    # latex_table += "\\midrule\n"

    # previous_dataset = None
    # row_span_count = 0

    # for i, row in enumerate(data):
    #     current_dataset = maps_dataset[row.get("dataset", "")]

    #     if current_dataset != previous_dataset and previous_dataset is not None:
    #         # Insert the multirow command for the previous dataset
    #         latex_table = latex_table.replace(f"{{placeholder_{previous_dataset}}}", f"\\multirow{{{row_span_count}}}{{*}}{{{previous_dataset}}}")
    #         row_span_count = 1
    #     else:
    #         row_span_count += 1

    #     if current_dataset != previous_dataset:
    #         latex_table += f"{{placeholder_{current_dataset}}} & "
    #         previous_dataset = current_dataset
    #     else:
    #         latex_table += " & "

    #     latex_table += " & ".join(str(process_data(row[column], column)) for column in header_keys[1:]) + " \\\\ \n"

    #     # Handle the last dataset entry
    #     if i == len(data) - 1:
    #         latex_table = latex_table.replace(f"{{placeholder_{current_dataset}}}", f"\\multirow{{{row_span_count}}}{{*}}{{{current_dataset}}}")

    # latex_table += "\\bottomrule\n"
    # latex_table += "\\end{tabular}\n"
    # latex_table += "\\caption{Regression tasks for $\\epsilon " + str(epsilon) + "$ }\n"
    # latex_table += "\\end{table*}\n"


    # json_data = new_data
    # # Parse the JSON data
    # header = json_data["header"]
    # data = json_data["data"]

    # # Determine the number of columns
    # num_columns = len(header)

    # # Generate LaTeX code for the table with dynamic column count using booktabs
    # latex_table = "\\begin{table*}[ht]\n"
    # latex_table += "\\centering\n"
    # latex_table += "\\begin{tabular}{" + "c" * num_columns + "}\n"
    # latex_table += "\\toprule\n"
    # latex_table += " & ".join(header) + " \\\\ \n"
    # latex_table += "\\midrule\n"
    # for row in data:
    #     if "regression" in row["dataset"]:
    #         latex_table += " & ".join(str(process_data(row[column],column)) for column in header_keys) + " \\\\ \n"
    # latex_table += "\\bottomrule\n"
    # latex_table += "\\end{tabular}\n"
    # latex_table += "\\caption{Regression tasks for $\\epsilon " + str(epsilon) + "$ }\n"
    # latex_table += "\\end{table*}\n"



    # Save to a .tex file
    script_folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_folder, f"table_regression_{epsilon}.tex"), "w") as tex_file:
        tex_file.write(latex_table)
        



    
    # json_data = new_data
    # # Parse the JSON data
    # header = json_data["header"]
    # data = json_data["data"]

    # # Determine the number of columns
    # num_columns = len(header)

    # # Generate LaTeX code for the table with dynamic column count
    # latex_table = "\\begin{table*}[ht]\n"
    # latex_table += "\\centering\n"
    # latex_table += "\\begin{tabular}{|" + "c|" * num_columns + "}\n"
    # latex_table += "\\hline\n"
    # latex_table += " & ".join(header) + " \\\\ \n"
    # latex_table += "\\hline\n"
    # for row in data:
    #     if "regression" not in row["dataset"]:
    #         latex_table += " & ".join(str(process_data(row[column],column)) for column in header_keys) + " \\\\ \n"
    # latex_table += "\\hline\n"
    # latex_table += "\\end{tabular}\n"
    # latex_table += "\\caption{Classification tasks for $\\epsilon "+str(epsilon)+"$ }\n"

    # latex_table += "\\end{table*}\n"

    # # Save to a .tex file
    # script_folder = os.path.dirname(os.path.abspath(__file__))
    # with open(os.path.join(script_folder, f"table_classification_{epsilon}.tex"), "w") as tex_file:
    #     tex_file.write(latex_table)










    # Create a dictionary to store legend information

