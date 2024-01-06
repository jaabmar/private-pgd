import argparse
import os
import yaml
import wandb
from examples.experiment import experiment
# Constants
BASE_FILE = "experiments/config/base.yaml"

def determine_value_type(value):
    """ Determine the type of a value. """
    if isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, str):
        return str
    return None


def parse_dataset_config(dataset_config):
    parts = dataset_config.split('_')
    
    if len(parts) >= 3:  # Ensure there are at least three parts
        # The last part is the k-value
        disc_k = parts[-1]
        
        # The second to last part is the type
        disc_type = parts[-2]
        
        # All preceding parts make up the dataset name
        dataset_name = '_'.join(parts[:-2])
        
        return dataset_name, disc_type, disc_k
    else:
        return None, None, None
    


def convert_str_to_bool(value):
    """ Convert string to boolean if applicable. """
    value_lower = value.lower()
    if value_lower in ('true', 't', '1'):
        return True
    elif value_lower in ('false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def load_base_yaml():
    """ Load base YAML configuration file. """
    with open(BASE_FILE, encoding="utf-8") as file:
        return yaml.safe_load(file)

def setup_arg_parser(base_yaml):
    """ Setup argument parser with base YAML configuration. """
    parser = argparse.ArgumentParser(description='GAN Hyperparameters')
    parser.add_argument("--base_path", type=str, help="Path to the dataset csv file")
    parser.add_argument("--regularized", type=str, default="false")

    # Add hyperparameters from YAML to argparse
    for param_name, default_value in base_yaml.items():
        parser.add_argument(f"--{param_name}", 
                            type=determine_value_type(default_value), 
                            default=None, 
                            help=f"Default value from YAML: {default_value}")

    return parser

def main():
    print("Starting main function")
    base_yaml = load_base_yaml()

    parser = setup_arg_parser(base_yaml)
    args = parser.parse_args()

    # Set debug mode if required
    if getattr(args, "debug", None) == 1:
        os.environ["WANDB_MODE"] = "dryrun"

    # Update base_yaml with arguments
    for arg, value in vars(args).items():
        if value is not None:
            base_yaml[arg] = value

    base_yaml["dataset"] = os.path.basename(os.path.normpath(base_yaml["base_path"]))


    # Start the run
    base_path = base_yaml["base_path"]
    run_name = (
        base_yaml["dataset"]
        + "_"
        + base_yaml["inference_type"]
        + "_"
        + base_yaml["mechanism"]
    )
    run = wandb.init(dir=os.environ.get("WANDB_DIR", None), project=base_yaml["project_name"], config=base_yaml, name=run_name)
    # Get the file name without extension
    wandb.config["train_dataset"] = os.path.join(base_path, "data_disc.csv")
    wandb.config["domain"] =     os.path.join(base_path, "domain.json")
    wandb.config["savedir"] = base_path
    #wandb.config["save"] = os.path.join(base_path, "synth_" + run.id)
    wandb.config["run_id"] = run.id

    results = experiment(dict(wandb.config))
    for key in results:
        run.summary[key] = results[key]

    dataset_name, disc_type, disc_k = parse_dataset_config(wandb.config['dataset'])
    
    run.summary['dataset_name'] = dataset_name
    run.summary['dataset_disc_type'] = disc_type
    run.summary['dataset_disc_k'] = int(disc_k)  # convert k to integer


    wandb.finish()

if __name__ == "__main__":
    main()
