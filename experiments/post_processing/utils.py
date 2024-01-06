import os

def get_filtered_runs(api, project_name, entity=None, filters={}, relational_filters={}):
    """
    Fetches and filters runs from the project.

    Args:
        api: The wandb API instance.
        project_name: The name of the project.
        entity: The entity name (optional).
        filters: A dictionary of filters where keys are the field names and values are the expected values.
        relational_filters: A dictionary of filters where keys are the field names and values are tuples of (operator, value).

    Returns:
        A list of filtered runs.
    """
    # Fetch all finished runs from the project
    all_runs = api.runs(path=f"{entity}/{project_name}" if entity else project_name, filters={"state": "finished"})

    # Filter runs
    filtered_runs = [run for run in all_runs if match_filters(run, filters, relational_filters)]

    return filtered_runs

def match_filters(run, filters, relational_filters):
    """
    Checks if a run matches the provided filters.

    Args:
        run: The run to check.
        filters: A dictionary of filters where keys are the field names and values are the expected values.
        relational_filters: A dictionary of filters where keys are the field names and values are tuples of (operator, value).

    Returns:
        True if the run matches all filters, False otherwise.
    """
    # Create a unified dictionary
    unified_dict = dict(run.config)
    unified_dict.update(run.summary)

    # Check if all keys in the filter are present in the unified dictionary and their values match the filter
    match = all(key in unified_dict and unified_dict[key] in values for key, values in filters.items())

    # Check relational filters
    for key, (op, val) in relational_filters.items():
        if key in unified_dict:
            if op == ">" and unified_dict[key] <= val:
                match = False
            elif op == "<" and unified_dict[key] >= val:
                match = False
            elif op == "=" and unified_dict[key] != val:
                match = False
            # ... Add more relational operators as needed

    return match


def get_paths():
    """
    Returns the paths to the data directory and the entity name.

    Returns:
        A dictionary with the keys "base_path" and "entity_name".
    """
    # Get the base path
    main_path = os.environ.get("EXPERIMENT_BASE_PATH", None)
    if main_path is None:
        raise ValueError("Environment variable BASE_PATH not set.")

    # Get the entity name
    entity_name = os.environ.get("WANDB_ENTITY_NAME", None)
    if entity_name is None:
        raise ValueError("Environment variable WANDB_ENTITY not set.")

    return {"main_path": main_path, "entity_name": entity_name}