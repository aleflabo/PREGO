import json
from typing import Any, Dict, List

import numpy as np


def eliminate_consecutive_duplicates(arr: np.ndarray) -> np.ndarray:
    """
    Eliminate consecutive duplicate values in a numpy array.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Array with consecutive duplicates removed.
    """
    # Initialize result with the first element
    result = [arr[0]]
    # Iterate through the array and add elements that are not duplicates
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            result.append(arr[i])
    return np.array(result)


def find_changes(arr: np.ndarray) -> List[int]:
    """
    Find indices where changes occur in the array.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        List[int]: List of indices where changes occur.
    """
    # Initialize result list
    result = []
    # Iterate through the array and record indices where changes occur
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            result.append(i)
    result.append(len(arr))
    return result


def aggregate(data: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Aggregate predictions and ground truth data, and save the results to a JSON file.

    Args:
        data (Dict[str, Dict[str, Any]]): Input data containing predictions and ground truth.
        output_path (str): Path to save the aggregated data as a JSON file.
    """
    aggregated_data = {}
    window_size = 200
    # Iterate through each key-value pair in the data
    for key, value in data.items():
        predictions = value["pred"]
        ground_truth = value["gt"]
        new_predictions = np.zeros_like(predictions)
        start_indices = []
        end_indices = []

        # Process predictions in windows
        for start in range(0, len(predictions), window_size):
            end = start + window_size
            if end > len(predictions):
                end = len(predictions)
            counts = np.bincount(predictions[start:end])
            new_predictions[start:end] = np.argmax(counts)
            start_indices.append(start)
            end_indices.append(end)

        # Find changes and eliminate consecutive duplicates
        changes_predictions = find_changes(new_predictions)
        changes_ground_truth = find_changes(ground_truth)
        new_predictions = eliminate_consecutive_duplicates(new_predictions)
        ground_truth = eliminate_consecutive_duplicates(ground_truth)

        # Store the results in the aggregated data dictionary
        aggregated_data[key] = {
            "pred": new_predictions.tolist(),
            "gt": ground_truth.tolist(),
            "changes_pred": changes_predictions,
            "changes_gt": changes_ground_truth,
        }

    # Save the aggregated data as a JSON file
    with open(output_path, "w") as fp:
        json.dump(aggregated_data, fp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate predictions and ground truth data."
    )
    parser.add_argument("input_path", type=str, help="Path to the input JSON file.")
    parser.add_argument(
        "output_path", type=str, help="Path to save the aggregated JSON file."
    )
    args = parser.parse_args()

    # Load the input data from a JSON file
    with open(args.input_path, "r") as fp:
        data = json.load(fp)
    # Call the aggregate function with the loaded data and output path
    aggregate(data, args.output_path)
