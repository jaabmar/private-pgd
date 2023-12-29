import csv
import logging
import os
import time
from functools import partial

import click
import torch
from constraints import RegularizedGradientDescent, approx_thres_func_constraint
from utils_examples import flatten_dict

from inference.dataset import Dataset
from inference.evaluation import Evaluator
from inference.privpgd.inference import AdvancedSlicedInference
from mechanisms.kway import KWay
from mechanisms.utils_mechanisms import generate_all_kway_workload

logging.basicConfig(level=logging.INFO)  # Configure logging level


@click.command()
@click.option(
    "--savedir",
    default="../src/data/datasets/acs_income_CA_2018_default_32/",
    help="Directory to save the generated synthetic dataset.",
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
    "--iters",
    default=1000,
    type=int,
    help="Number of iterations for PrivPGD particle gradient descent.",
)
@click.option(
    "--n_particles",
    default=100000,
    type=int,
    help="Number of particles for PrivPGD.",
)
@click.option("--lr", default=10.0, type=float, help="Learning rate.")
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
def run_constraint_privpgd(
    savedir,
    epsilon,
    delta,
    iters,
    n_particles,
    lr,
    scheduler_step,
    scheduler_gamma,
    num_projections,
    scale_reg,
    p_mask,
    batch_size,
):
    """
    Run PrivPGD with an approximate thresholding function as additional domain-specific constraint.

    This method applies PrivPGD using an approximate thresholding function as an additional constraint, specifically
    tailored for the ACS Income dataset with privacy parameters for the constraint epsilon=0.5 and delta=0.000002.
    The total privacy budget for the operation is calculated as the sum of these values and the ones provided through
    the '--epsilon' and '--delta' command-line parameters.
    """

    params = locals()

    train_dataset = (
        "../src/data/datasets/acs_income_CA_2018_default_32/data_disc.csv"
    )
    domain = "../src/data/datasets/acs_income_CA_2018_default_32/domain.json"

    data = Dataset.load(
        train_dataset,
        domain,
    )

    all_2_way_workload = generate_all_kway_workload(data=data, degree=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    taus = torch.tensor([1.5767367], dtype=torch.float32).to(device)
    thetas = torch.tensor(
        [
            [
                0.0,
                0.0,
                0.8843252,
                0.0,
                0.661784,
                0.0,
                0.0,
                0.0,
                0.557365,
                0.49245986,
                0.9226635,
            ]
        ],
        dtype=torch.float32,
    ).to(device)
    noisy_estimate = torch.tensor([0.6325920820236206], dtype=torch.float32).to(
        device
    )
    sigma = 5.0

    constraint_function_partial = partial(
        approx_thres_func_constraint, thetas=thetas, taus=taus, sigma=sigma
    )

    regularizer = RegularizedGradientDescent(
        constraint_function=constraint_function_partial,
        noisy_estimate=noisy_estimate,
    )

    generation_engine = AdvancedSlicedInference(
        domain=data.domain, hp=params, constraint_regularizer=regularizer
    )

    mechanism = KWay(
        epsilon=params["epsilon"],
        delta=params["delta"],
        degree=2,
        bounded=True,
    )

    start_time = time.time()
    synth, loss = mechanism.run(
        data=data,
        workload=all_2_way_workload,
        engine=generation_engine,
    )
    end_time = time.time()

    print(f"Total loss: {loss}, Elapsed time: {end_time-start_time}")
    if params["savedir"]:
        synth.df.to_csv(
            os.path.join(
                params["savedir"], "constraint_privpgd_synth_data.csv"
            ),
            index=False,
        )

    print("Starting to evaluate...")
    evaluator = Evaluator(data=data, synth=data, workload=all_2_way_workload)
    evaluator.set_compression()
    evaluator.update_synth(synth)
    results, _ = evaluator.evaluate()

    dataset_name = os.path.basename(os.path.dirname(train_dataset))
    experiment_results = {
        "dataset_name": dataset_name,
        "time": time.time() - start_time,
        "loss": loss,
        **flatten_dict(
            {k: v for k, v in params.items() if k not in ["savedir"]}
        ),
        **flatten_dict(results),
    }

    # File to save results
    results_file = os.path.join(savedir, "constraint_privpdg_results.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_results)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    run_constraint_privpgd()
