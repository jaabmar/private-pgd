import csv
import logging
import os
import time

import click
from utils_examples import flatten_dict

from inference.dataset import Dataset
from inference.evaluation import Evaluator
from inference.pgm.inference import FactoredInference
from mechanisms.aim import AIM
from mechanisms.utils_mechanisms import generate_all_kway_workload

logging.basicConfig(level=logging.INFO)  # Configure logging level


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
    "--test_dataset",
    default="../data/datasets/acs_income_CA_2018_default_32/testdata_disc.csv",
    help="File path for the test dataset (CSV format).",
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
@click.option("--lr", default=10.0, type=float, help="Learning rate.")
@click.option("--iters", default=1000, type=int, help="Number of iterations.")
@click.option(
    "--max_model_size",
    default=1000,
    type=float,
    help="Maximum allowable size of the model.",
)
@click.option(
    "--warm_start",
    default=True,
    type=bool,
    help="Whether to use warm start.",
)
@click.option(
    "--records",
    default=None,
    type=int,
    help="Number of records to generate. 'None' for same size as original dataset.",
)
def run_aim_pgm(
    savedir,
    train_dataset,
    test_dataset,
    domain,
    epsilon,
    delta,
    lr,
    iters,
    max_model_size,
    warm_start,
    records,
):
    """
    Run AIM+PGM with specified parameters for privacy-preserving data synthesis.
    """

    params = locals()

    data = Dataset.load(
        params["train_dataset"],
        params["domain"],
    )

    generation_engine = FactoredInference(
        domain=data.domain,
        hp=params,
    )

    mechanism = AIM(
        epsilon=params["epsilon"],
        delta=params["delta"],
        max_model_size=params["max_model_size"],
        bounded=True,
    )

    workload = generate_all_kway_workload(data=data, degree=2, num_marginals=5)

    start_time = time.time()
    synth, loss = mechanism.run(
        data=data, engine=generation_engine, workload=workload
    )
    end_time = time.time()

    print(f"Total loss: {loss}, Elapsed time: {end_time-start_time}")
    if params["savedir"]:
        synth.df.to_csv(
            os.path.join(params["savedir"], "aim_pgm_synth_data.csv"),
            index=False,
        )

    print("Starting to evaluate...")
    evaluator = Evaluator(
        data=data,
        synth=data,
        workload=workload,
    )
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
    results_file = os.path.join(savedir, "aim_pgm_results.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_results)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    run_aim_pgm()
