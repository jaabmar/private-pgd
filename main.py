import csv
import os
import time
from typing import Any, Dict, Tuple

import click

from inference.dataset import Dataset
from inference.evaluation import Evaluator
from mechanisms.utils_mechanisms import generate_all_kway_workload


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_data_and_prepare_workload(hp: Dict[str, Any]) -> Tuple[Dataset, Any]:
    """Load dataset and prepare the workload."""
    data = Dataset.load(
        hp["train_dataset"],
        hp["domain"],
    )
    workload = generate_all_kway_workload(
        data=data,
        degree=hp["degree"],
        num_marginals=hp["num_marginals"],
    )
    return data, workload


def initialize_mechanism_and_inference(hp: Dict[str, Any]) -> Tuple[Any, Any]:
    """Initialize the mechanism and inference method."""
    mechanisms = {
        "KWay": ("mechanisms.kway", "KWay"),
        "MWEM": ("mechanisms.mwem", "MWEM"),
        "MST": ("mechanisms.mst", "MST"),
    }
    inference_methods = {
        "pgm_euclid": ("inference.pgm.inference", "FactoredInference"),
        "privpgd": (
            "inference.privpgd.inference",
            "AdvancedSlicedInference",
        ),
    }

    mechanism_module, mechanism_class = mechanisms[hp["mechanism"]]
    inference_module, inference_class = inference_methods[hp["inference_type"]]

    Mechanism = getattr(
        __import__(mechanism_module, fromlist=[mechanism_class]),
        mechanism_class,
    )
    InferenceMethod = getattr(
        __import__(inference_module, fromlist=[inference_class]),
        inference_class,
    )
    return Mechanism, InferenceMethod


@click.command()
@click.option(
    "--savedir",
    default="src/data/datasets/acs_income_CA_2018_default_32/",
    help="Directory to save the generated synthetic dataset.",
)
@click.option(
    "--train_dataset",
    default="src/data/datasets/acs_income_CA_2018_default_32/data_disc.csv",
    help="File path for the training dataset (CSV format).",
)
@click.option(
    "--test_dataset",
    default="src/data/datasets/acs_income_CA_2018_default_32/testdata_disc.csv",
    help="File path for the test dataset (CSV format).",
)
@click.option(
    "--domain",
    default="src/data/datasets/acs_income_CA_2018_default_32/domain.json",
    help="File path for the domain description (JSON format).",
)
@click.option(
    "--epsilon",
    default=2.5,
    type=float,
    help="Privacy budget (epsilon) for differential privacy.",
)
@click.option(
    "--delta",
    default=0.00001,
    type=float,
    help="Delta parameter for differential privacy.",
)
@click.option(
    "--degree",
    default=2,
    type=int,
    help="Degree of the marginals used in the workload.",
)
@click.option(
    "--num_marginals",
    default=None,
    type=int,
    help="Number of marginals to consider. 'None' for all marginals..",
)
@click.option(
    "--max_model_size",
    default=1000,
    type=float,
    help="Maximum allowable size of the model in MegaBytes.",
)
@click.option(
    "--iters",
    default=1000,
    type=int,
    help="Number of iterations for PGM mirror descent or PrivPGD particle gradient descent.",
)
@click.option(
    "--n_particles",
    default=100000,
    type=int,
    help="Number of particles for PrivPGD.",
)
@click.option(
    "--data_init",
    default=None,
    help="Initialization method for PrivPGD particles. 'None' for random initialization.",
)
@click.option(
    "--inference_type",
    default="privpgd",
    type=click.Choice(["privpgd", "pgm_euclid"]),
    help="Type of inference method to use.",
)
@click.option(
    "--warm_start",
    default=True,
    type=bool,
    help="Whether to use warm start in the algorithm.",
)
@click.option(
    "--mechanism",
    default="KWay",
    type=click.Choice(["KWay", "MST", "MWEM"]),
    help="Type of mechanism to use in the privacy-preserving algorithm.",
)
@click.option(
    "--lr", default=10.0, type=float, help="Learning rate for PGM or PrivPGD."
)
@click.option(
    "--descent_type",
    default="MD",
    type=click.Choice(["GD", "MD"]),
    help="Descent algorithm for PGM, either Gradient Descent (GD) or Mirror Descent (MD).",
)
@click.option(
    "--optimizer_pgm",
    default="Adam",
    type=click.Choice(["Adam", "SGD", "RMSProp"]),
    help="Optimizer type for PGM.",
)
@click.option(
    "--scheduler_step",
    default=50,
    type=float,
    help="Scheduler step size for PGM or PrivPGD.",
)
@click.option(
    "--scheduler_gamma",
    default=0.75,
    type=float,
    help="Scheduler gamma (i.e., multiplicative factor) for PGM or PrivPGD.",
)
@click.option(
    "--num_projections",
    default=10,
    type=int,
    help="Number of projections to compute SW2 for PrivPGD.",
)
@click.option(
    "--scale_reg",
    default=0.0,
    type=float,
    help="Regularization parameter for constraints in PrivPGD.",
)
@click.option(
    "--p_mask",
    default=0.8,
    type=float,
    help="Percentage of randomly masked gradients in PrivPGD.",
)
@click.option(
    "--batch_size",
    default=5,
    type=int,
    help="Batch size in PrivPGD.",
)
@click.option(
    "--rounds",
    default=100,
    type=int,
    help="Number of rounds for the MWEM mechanism.",
)
@click.option(
    "--records",
    default=None,
    type=int,
    help="Number of records to generate for PGM. 'None' for same size as original dataset.",
)
def experiment(
    savedir,
    train_dataset,
    test_dataset,
    domain,
    epsilon,
    delta,
    degree,
    num_marginals,
    max_model_size,
    iters,
    n_particles,
    data_init,
    inference_type,
    warm_start,
    mechanism,
    lr,
    descent_type,
    optimizer_pgm,
    scheduler_step,
    scheduler_gamma,
    num_projections,
    scale_reg,
    p_mask,
    batch_size,
    rounds,
    records,
):
    """
    Run an experiment with specified parameters for privacy-preserving data synthesis.
    This tool supports various mechanisms and settings, allowing for extensive configurability.
    """

    params = locals()

    data, workload = load_data_and_prepare_workload(params)
    Mechanism, InferenceMethod = initialize_mechanism_and_inference(params)

    generation_engine = InferenceMethod(
        domain=data.domain,
        N=data.df.shape[0],
        hp=params,
    )

    mechanism = Mechanism(hp=params)  # here we define bounded

    start_time = time.time()
    synth, loss = mechanism.run(
        data=data,
        workload=workload,
        engine=generation_engine,
        records=params["records"],
    )
    end_time = time.time()

    print(f"Total loss: {loss}, Elapsed time: {end_time-start_time}")
    if params["savedir"]:
        synth.df.to_csv(
            os.path.join(params["savedir"], "synth_data.csv"), index=False
        )

    print("Starting to evaluate...")
    evaluator = Evaluator(data=data, synth=data, workload=workload)
    evaluator.set_compression()
    evaluator.update_synth(synth)
    results, _ = evaluator.evaluate()

    dataset_name = os.path.basename(os.path.dirname(train_dataset))
    experiment_results = {
        "dataset_name": dataset_name,
        "time": time.time() - start_time,
        "loss": loss,
        **flatten_dict(
            {
                k: v
                for k, v in params.items()
                if k
                not in ["savedir", "train_dataset", "test_dataset", "domain"]
            }
        ),
        **flatten_dict(results),
    }

    # File to save results
    results_file = os.path.join(savedir, "experiment_results.csv")

    # Check if the file exists and whether a header is needed
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_results)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    experiment()
