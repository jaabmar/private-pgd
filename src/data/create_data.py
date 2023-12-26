import json
import os
import pickle

import pandas as pd
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)
from sklearn.model_selection import train_test_split

from data.data_handler import DataHandler

base_path = "datasets"


def split_training_testing_data(dataset_path):
    df = pd.read_csv(dataset_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv(os.path.dirname(dataset_path) + "train_data.csv")
    test.to_csv(os.path.dirname(dataset_path) + "test_data.csv")


def create_all_ACS_data(disc_k=35, year="2018"):
    state = "CA"
    data_tasks = {
        "employment": ACSEmployment,
        "publiccoverage": ACSPublicCoverage,
        "mobility": ACSMobility,
        "income": ACSIncome,
        "traveltime": ACSTravelTime,
    }
    data_source = ACSDataSource(
        survey_year=year, horizon="5-Year", survey="person"
    )
    acs_data = data_source.get_data(states=[state], download=True)
    for task, datatask in data_tasks.items():
        # get all paths
        path = os.path.join(
            base_path,
            "acs_{}_{}_{}_default_{}/".format(task, state, year, disc_k),
        )

        # If the directory doesn't exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created.")
        else:
            continue
        # get all acs data and searate label column
        features, label, _ = datatask.df_to_numpy(acs_data)
        data_raw = pd.DataFrame(features)
        data_raw["label"] = label
        data_raw.columns = [str(col) for col in data_raw.columns]
        # Process the entire dataset using DataHandler
        datahandler = DataHandler(data_raw)
        processed_data, _, _ = datahandler.forward(disc_k)
        # Split the indices to align with the train and test splits
        train_indices, test_indices = train_test_split(
            processed_data.index, test_size=0.2, random_state=42
        )
        # Use the indices to slice both the original and processed dataframes
        train_df_processed = processed_data.iloc[train_indices]
        test_df_processed = processed_data.iloc[test_indices]
        train_df_original = data_raw.iloc[train_indices]
        test_df_original = data_raw.iloc[test_indices]

        # Save processed and original data
        save_process_data(
            train_df_processed,
            train_df_original,
            path,
            datahandler,
            testdata=False,
        )
        save_process_data(
            test_df_processed,
            test_df_original,
            path,
            datahandler,
            testdata=True,
        )


def save_process_data(
    df_processed, df_original, path, datahandler, testdata=False
):
    # Determine the name based on the test flag
    name = "testdata" if testdata else "data"

    # Save the processed dataframe to CSV
    df_processed.to_csv(os.path.join(path, f"{name}_disc.csv"), index=False)

    # Save the original dataframe to CSV
    df_original.to_csv(os.path.join(path, f"{name}_original.csv"), index=False)
    if not testdata:
        # Save processed_domain as JSON
        with open(
            os.path.join(path, "domain.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(datahandler.domain, f)

        # Save inverse_mapping using pickle
        with open(os.path.join(path, "inverse_mapping.pkl"), "wb") as f:
            pickle.dump(datahandler.inverse_mapping, f)


# Function to convert a string column to numerical values and create mapping
def convert_to_number(column, string_to_number_mapping):
    unique_values = column.unique()
    mapping = {value: index for index, value in enumerate(unique_values)}
    string_to_number_mapping[column.name] = mapping
    return column.map(mapping)


for k in [32]:
    create_all_ACS_data(disc_k=k)
