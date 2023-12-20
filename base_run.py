import itertools
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wandb

from inference.dataset import Dataset
from inference.evaluation import Evaluator


def generate_all_kway_workload(
    data: "Dataset",
    degree: int = 2,
    num_marginals: Optional[int] = None,
) -> List[tuple]:
    """
    Generate all k-way marginals workload.

    Args:
        data (Dataset): The dataset to generate workload for.
        degree (int): The degree of combinations.
        num_marginals (Optional[int]): Number of marginals to generate.

    Returns:
        List[tuple]: A list of attribute combinations.
    """
    workload = list(itertools.combinations(data.domain.attrs, degree))
    if num_marginals is not None:
        workload = [
            workload[i]
            for i in np.random.choice(
                len(workload), num_marginals, replace=False
            )
        ]
    return workload


def get_run_name(params: Dict[str, str]) -> str:
    """Generate a run name based on dataset, inference type, and mechanism."""
    return (
        f"{params['dataset']}_"
        f"{params['inference_type']}_"
        f"{params['mechanism']}"
    )


def initialize_wandb(params: Dict[str, Any], run_name: str) -> None:
    """Initialize the Weights & Biases (wandb) run."""
    wandb.init(
        dir="wandb/",
        project=params["project_name"],
        config=params,
        name=run_name,
    )
    wandb.config["save"] = os.path.join(
        params["base_path"], f"synth_{wandb.run.id}"
    )


def load_data_and_prepare_workload(
    base_path: str, hp: Dict[str, Any]
) -> Tuple[Dataset, Any]:
    """Load dataset and prepare the workload."""
    data = Dataset.load(
        os.path.join(base_path, "data_disc.csv"),
        os.path.join(base_path, "domain.json"),
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
        "MWEM": ("mechanisms.mwem_pgm", "MWEM"),
        "MST": ("mechanisms.mst", "MST"),
    }
    inference_methods = {
        "pgm_euclid": ("inference.pgm.inference", "FactoredInference"),
        "priv_pgd": (
            "inference.privpgd.inference",
            "AdvancedSlicedInference",
        ),
    }

    mechanism_module, mechanism_class = mechanisms[hp["mechanism"]]
    inference_module, inference_class = inference_methods[hp["inference_type"]]

    Mechanism = getattr(
        __import__(mechanism_module, fromlist=[mechanism_class]),
        mechanism_class,
    )(hp=hp)
    InferenceMethod = getattr(
        __import__(inference_module, fromlist=[inference_class]),
        inference_class,
    )
    return Mechanism, InferenceMethod


def train(params: Dict[str, Any]) -> None:
    """
    Train a model using specified parameters.

    Args:
        params (Dict[str, Any]): A dictionary of training parameters.

    Returns:
        None
    """
    if not params:
        raise ValueError("No training parameters provided.")

    run_name = get_run_name(params)
    initialize_wandb(params, run_name)

    data, workload = load_data_and_prepare_workload(
        params["base_path"], dict(wandb.config)
    )
    Mechanism, InferenceMethod = initialize_mechanism_and_inference(
        dict(wandb.config)
    )

    evaluator = Evaluator(data=data, synth=data, workload=workload)
    generation_engine = InferenceMethod(
        domain=data.domain,
        N=data.df.shape[0],
        evaluator=evaluator,
        hp=dict(wandb.config),
    )

    start_time = time.time()
    synth, loss = Mechanism.run(
        data=data, workload=workload, engine=generation_engine
    )
    end_time = time.time()

    wandb.log({"syntdataloss": loss, "elapsed_time": end_time - start_time})

    if dict(wandb.config)["save"]:
        synth.df.to_csv(dict(wandb.config)["save"], index=False)

    print("Starting to evaluate...")
    evaluator.set_compression()
    evaluator.update_synth(synth)
    evaluator.evaluate()

    wandb.finish()
