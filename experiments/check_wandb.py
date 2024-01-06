import pdb
import wandb
import sys
import os

def custom_equality(str1, str2):
    try:
        return float(str1) == float(str2)
    except ValueError:
        return str1 == str2


def get_last_entry_name(path):
    base_name = os.path.basename(path)  # Get the last entry name
    name_without_extension = os.path.splitext(base_name)[0]  # Remove the .csv extension
    return name_without_extension
def check_existing_run(runs, parameters, full_dataset_name):
    dataset_name = get_last_entry_name(full_dataset_name)
    for trun in runs:
        try:
            matches = [custom_equality(trun.config.get(k), v) for k, v in parameters.items()]
            if all(matches) and trun.state in ["running", "finished"] and trun.config.get('dataset',"") == dataset_name:
                return True
        except:
            pass
        
    return False


import csv

def read_args_from_csv(csv_path):
    args = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            args.extend(row)
    return args


def process_combinations(project_name, all_combinations):
    combinations_list = all_combinations #all_combinations.split(',')
    results = []
    api = wandb.Api()
    runs = api.runs(path="eth-sml-privacy-project/{}".format(project_name))
    for combo in combinations_list:

        dataset_name, params = combo.split(':')
        params = params.replace('\n', '')

        parameters = dict(arg.split('=') for arg in params.split(' ') if arg)
        if check_existing_run(runs, parameters, dataset_name):
            results.append("1")
        else:
            results.append("0")
    return ','.join(results)



if __name__ == "__main__":
        # Check if there's an argument for CSV, if so, load from CSV

    # if len(sys.argv) > 2 and sys.argv[2].endswith('.csv'):
    #     args = read_args_from_csv(sys.argv[2])
    # else:
    #     args = sys.argv[2:]

    # The rest of your script

    project_name = sys.argv[1]
    input_file = sys.argv[2]
    
    with open(input_file, 'r') as file:
        lines = file.readlines()

    print(process_combinations(project_name, lines))
