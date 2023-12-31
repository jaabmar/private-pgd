import csv
import logging
import os
import time

import click
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
@click.option("--lr", default=0.1, type=float, help="Learning rate.")
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
def run_privpgd(
    savedir,
    train_dataset,
    domain,
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
    Run PrivPGD with specified parameters and all 2-way marginals as workload for privacy-preserving data synthesis.
    """

    params = locals()

    data = Dataset.load(
        params["train_dataset"],
        params["domain"],
    )

    all_2_way_workload = generate_all_kway_workload(data=data, degree=2)

    generation_engine = AdvancedSlicedInference(
        domain=data.domain,
        hp=params,
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
            os.path.join(params["savedir"], "privpgd_synth_data.csv"),
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
            {
                k: v
                for k, v in params.items()
                if k not in ["savedir", "train_dataset", "domain"]
            }
        ),
        **flatten_dict(results),
    }

    # File to save results
    results_file = os.path.join(savedir, "privpdg_results.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=experiment_results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_results)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    run_privpgd()
