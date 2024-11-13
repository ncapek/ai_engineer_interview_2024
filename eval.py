"""
Evaluation script for a trained machine learning model.

This module loads a trained model, evaluates its performance on a given evaluation dataset,
and outputs various evaluation metrics. It supports command-line arguments for specifying
the paths to the model and evaluation data. The evaluation results, including accuracy,
classification report, and confusion matrix, are logged and saved as a JSON file.
"""
import argparse
import json
import os
from logging import Logger
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import load_config
from train import load_dataset, prepare_dataset
from utils import setup_logger

# Set up logging
eval_logger = setup_logger('eval_logger')


def load_model(model_path: str, logger: Logger):
    """
    Loads a serialized model from the specified path.
    """
    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)
    return model


def evaluate_model(model, df: pd.DataFrame, config_dict: dict, logger: Logger):
    """
    Evaluates the model on the evaluation dataset.
    """
    # Extract the target column from config
    target_column = config_dict['datasets']['target_column']

    # Separate features and target
    x_eval, y_eval = df['text'], df[target_column]

    # Make predictions
    predictions = model.predict(x_eval)

    # Calculate metrics
    accuracy = accuracy_score(y_eval, predictions)
    report = classification_report(y_eval, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_eval, predictions)

    # Log the results
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Classification report:\n%s", report)
    logger.info("Confusion matrix:\n%s", conf_matrix)

    # Store results in a JSON file (configured in output section of config)
    eval_results_path = config_dict['output']['eval_results_path']
    eval_results = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()  # Convert to list if needed
    }

    # Ensure the directory exists before saving the file
    Path(eval_results_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the results to the JSON file
    with open(eval_results_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4)

    logger.info(f"Evaluation results saved to {eval_results_path}")


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a dataset.")
    parser.add_argument(
        '--model', type=str, required=True, help="Path to the serialized model file.")
    parser.add_argument(
        '--eval_data', type=str, required=True, help="Path to the evaluation data file.")
    return parser.parse_args()


if __name__ == "__main__":
    # Load configuration
    config = load_config('config.yml')

    # Parse arguments
    args = parse_args()

    # Load and prepare evaluation data
    eval_data = load_dataset(args.eval_data, config, eval_logger)
    eval_data = prepare_dataset(eval_data, config, eval_logger)

    # Load the trained model
    best_model = load_model(args.model, eval_logger)

    # Evaluate the model
    evaluate_model(best_model, eval_data, config, eval_logger)
