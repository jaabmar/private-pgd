import json
import os
import pickle
from io import StringIO

import numpy as np
import openml
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from data.data_handler import DataHandler

base_path = "/cluster/work/yang/donhausk/privacy-ot/data"
unprocessed_data = "/cluster/work/yang/donhausk/privacy-ot/up_data"


def fetch_and_preprocess_data(
    dataset_name, dataset_id, disc_k, url=None, handler_type="default"
):
    if handler_type == "default":
        Handler = DataHandler
    else:
        raise ValueError(f"Unknown handler_type: {handler_type}")
    path = os.path.join(base_path, f"{dataset_name}_{handler_type}_{disc_k}")

    # Check and create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print("already exists")
        return

    # Directory for unprocessed data
    unprocessed_dataset_path = os.path.join(unprocessed_data, dataset_name)
    if not os.path.exists(unprocessed_dataset_path):
        os.makedirs(unprocessed_dataset_path)

    unprocessed_file_path = os.path.join(
        unprocessed_dataset_path, f"{dataset_name}.csv"
    )

    # Fetching dataset
    if os.path.exists(unprocessed_file_path):
        data_df = pd.read_csv(unprocessed_file_path)
    elif url:  # If a URL is provided, we fetch data from it
        try:
            response = requests.get(url, timeout=1000)
        except requests.Timeout:
            print("The request timed out")
        except requests.RequestException as e:
            print("An error occurred:", e)
        data_str = response.content.decode("utf-8")
        data_df = pd.read_csv(StringIO(data_str))
        # Save the data for future use
        data_df.to_csv(unprocessed_file_path, index=False)
    else:  # We assume it's an OpenML dataset
        datasett = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = datasett.get_data(target=datasett.default_target_attribute)
        data_df = pd.concat([X, y], axis=1)
        # Save the data for future use
        data_df.to_csv(unprocessed_file_path, index=False)

    data_df.columns = data_df.columns.str.replace(".", "")

    # preprocess out the first column
    data_df = data_df.dropna()
    data_df = data_df.iloc[:, 1:]

    # Process the entire dataset using DataHandler
    datahandler = Handler(data_df)
    processed_data, _, _ = datahandler.forward(disc_k)
    # Split the indices to align with the train and test splits
    train_indices, test_indices = train_test_split(
        processed_data.index, test_size=0.2, random_state=42
    )

    # Use the indices to slice both the original and processed dataframes
    train_df_processed = processed_data.iloc[train_indices]
    test_df_processed = processed_data.iloc[test_indices]
    train_df_original = data_df.iloc[train_indices]
    test_df_original = data_df.iloc[test_indices]
    # Save processed and original data
    save_process_data(
        train_df_processed, train_df_original, path, datahandler, disc_k
    )
    save_process_data(
        test_df_processed,
        test_df_original,
        path,
        datahandler,
        test=True,
    )


def save_process_data(df_processed, df_original, path, datahandler, test=False):
    # Determine the name based on the test flag
    name_file = "testdata" if test else "data"

    # Save the processed dataframe to CSV
    df_processed.to_csv(
        os.path.join(path, f"{name_file}_disc.csv"), index=False
    )

    # Save the original dataframe to CSV
    df_original.to_csv(
        os.path.join(path, f"{name_file}_original.csv"), index=False
    )

    if not test:
        # Save processed_domain as JSON
        with open(
            os.path.join(path, "domain.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(preprocess_for_json(datahandler.domain), f)

        # Save inverse_mapping using pickle
        with open(os.path.join(path, "inverse_mapping.pkl"), "wb") as f:
            pickle.dump(datahandler.inverse_mapping, f)


def preprocess_for_json(data):
    if isinstance(data, dict):
        return {k: preprocess_for_json(v) for k, v in data.items()}
    elif isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.float64):
        return float(data)
    else:
        return data


# The dataset links and other details
datasets = [
    ("electricity", 44120, 7, None),
    ("covertype", 44121, 10, None),
    ("pol", 44122, 26, None),
    ("house_16H", 44123, 16, None),
    ("MagicTelescope", 44125, 10, None),
    ("bank-marketing", 44126, 7, None),
    ("default-of-credit-card-clients", 45020, 20, None),
    ("Higgs", 44129, 24, None),
    ("eye_movements", 44130, 20, None),
    ("Diabetes130US", 45022, 7, None),
]


datasets_regression = [
    ("cpu_act_regression", 44132, 21, None),
    ("pol_regression", 44133, 26, None),
    ("elevators_regression", 44134, 16, None),
    ("wine_quality_regression", 44136, 11, None),
    ("houses_regression", 44138, 8, None),
    ("house_16H_regression", 44139, 16, None),
    ("diamonds_regression", 44140, 6, None),
    ("Brazilian_houses_regression", 44141, 8, None),
    ("Bike_Sharing_Demand_regression", 44142, 6, None),
    ("nyc-taxi-green-dec-2016_regression", 44143, 9, None),
    ("house_sales_regression", 44144, 15, None),
    ("sulfur_regression", 44145, 6, None),
    ("medical_charges_regression", 44146, 5, None),
    ("MiamiHousing2016_regression", 44147, 14, None),
]


# Input data
dataset_names = [
    "yprop_4_1",
    "analcatdata_supreme",
    "visualizing_soil",
    "black_friday",
    "diamonds",
    "Mercedes_Benz_Greener_Manufacturing",
    "Brazilian_houses",
    "Bike_Sharing_Demand",
    "OnlineNewsPopularity",
    "nyc-taxi-green-dec-2016",
    "house_sales",
    "particulate-matter-ukair-2017",
    "SGEMM_GPU_kernel_performance",
]

n_features = [62, 7, 4, 9, 9, 359, 11, 11, 59, 16, 17, 6, 9]
n_samples = [
    8885,
    4052,
    8641,
    166821,
    53940,
    4209,
    10692,
    17379,
    39644,
    581835,
    21613,
    394299,
    241600,
]

# New links provided
new_links = [
    "https://www.openml.org/d/44054",
    "https://www.openml.org/d/44055",
    "https://www.openml.org/d/44056",
    "https://www.openml.org/d/44057",
    "https://www.openml.org/d/44059",
    "https://www.openml.org/d/44061",
    "https://www.openml.org/d/44062",
    "https://www.openml.org/d/44063",
    "https://www.openml.org/d/44064",
    "https://www.openml.org/d/44065",
    "https://www.openml.org/d/44066",
    "https://www.openml.org/d/44068",
    "https://www.openml.org/d/44069",
]

# Process data and filter out datasets
datasets_regression_2 = []

for i in range(len(dataset_names)):
    # Check the constraints
    if (
        n_features[i] <= 25
        and n_samples[i] >= 30000
        and "diamonds" not in dataset_names[i]
        and "nyc-taxi" not in dataset_names[i]
    ):
        # Construct the entry
        print("entry!")
        entry = (
            dataset_names[i] + "_regression",
            int(new_links[i].split("/")[-1]),
            n_features[i],
            None,  # No additional link information was provided
        )
        datasets_regression_2.append(entry)

for name, id_data, _, url_dir in datasets_regression_2:
    for k in [8, 16, 64, 96]:
        fetch_and_preprocess_data(
            name, id_data, k, url_dir, handler_type="default"
        )

for name, id_data, _, url_dir in datasets_regression:
    for k in [8, 16, 64, 96]:
        fetch_and_preprocess_data(
            name, id_data, k, url_dir, handler_type="default"
        )

for name, id_data, _, url_dir in datasets:
    for k in [8, 16, 64, 96]:
        fetch_and_preprocess_data(
            name, id_data, k, url_dir, handler_type="default"
        )
