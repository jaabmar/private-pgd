import csv
import logging
import os
import time
from typing import Any, Dict, Tuple

import click
from utils_examples import flatten_dict

from inference.dataset import Dataset
from inference.evaluation import Evaluator
from mechanisms.utils_mechanisms import generate_all_kway_workload

logging.basicConfig(level=logging.INFO)  # Configure logging level


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
        "AIM": ("mechanisms.aim", "AIM"),
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

    if (
        inference_class == "AdvancedSlicedInference"
        and mechanism_class != "KWay"
    ):
        raise ValueError(
            "AdvancedSlicedInference must be used with KWay mechanism only."
        )

    Mechanism = getattr(
        __import__(mechanism_module, fromlist=[mechanism_class]),
        mechanism_class,
    )
    InferenceMethod = getattr(
        __import__(inference_module, fromlist=[inference_class]),
        inference_class,
    )

    # Create an instance of the mechanism with the appropriate parameters
    if hp["mechanism"] == "KWay":
        Mechanism = Mechanism(
            epsilon=hp["epsilon"],
            delta=hp["delta"],
            degree=hp["degree"],
            bounded=True,
        )
    elif hp["mechanism"] == "MWEM":
        Mechanism = Mechanism(
            epsilon=hp["epsilon"],
            delta=hp["delta"],
            rounds=hp["rounds"],
            max_model_size=hp["max_model_size"],
            bounded=True,
        )
    elif hp["mechanism"] == "AIM":
        Mechanism = Mechanism(
            epsilon=hp["epsilon"],
            delta=hp["delta"],
            max_model_size=hp["max_model_size"],
            bounded=True,
        )
    else:  # MST
        # For other mechanisms, modify as needed
        Mechanism = Mechanism(
            epsilon=hp["epsilon"], delta=hp["delta"], bounded=True
        )

    return Mechanism, InferenceMethod


@click.command()
@click.option(
    "--savedir",
    default="../data/datasets/acs_income_CA_2018_default_32/",
    help="Directory to save the generated synthetic dataset.",
)
@click.option(
    "--train_dataset",
    default="../data/datasets/acs_income_CA_2018_default_32/data_disc.csv",
    help="File path for the training dataset (CSV format).",
)
@click.option(
    "--domain",
    default="../data/datasets/acs_income_CA_2018_default_32/domain.json",
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
    help="Maximum allowable size of the PGM model in MegaBytes.",
)
@click.option(
    "--iters",
    default=1000,
    type=int,
    help="Number of iterations for PrivPGD particle gradient descent or PGM mirror descent.",
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
    default=False,
    type=bool,
    help="Whether to use warm start in PGM.",
)
@click.option(
    "--mechanism",
    default="KWay",
    type=click.Choice(["KWay", "MST", "MWEM", "AIM"]),
    help="Type of mechanism to use in the privacy-preserving algorithm.",
)
@click.option(
    "--lr", default=0.1, type=float, help="Learning rate for PGM or PrivPGD."
)
@click.option(
    "--scheduler_step",
    default=50,
    type=float,
    help="Scheduler step size for PrivPGD.",
)
@click.option(
    "--scheduler_gamma",
    default=0.75,
    type=float,
    help="Scheduler gamma (i.e., multiplicative factor) for PrivPGD.",
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
    default=80,
    type=int,
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
    mechanism, inference_method = initialize_mechanism_and_inference(params)

    generation_engine = inference_method(
        domain=data.domain,
        hp=params,
    )

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
                if k not in ["savedir", "train_dataset", "domain"]
            }
        ),
        **flatten_dict(results),
    }

    # File to save results
    results_file = os.path.join(savedir, "experiment_results.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_results)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    experiment()
