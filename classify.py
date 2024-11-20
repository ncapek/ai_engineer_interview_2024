"""
Classify.py

This script is designed to classify input data using a pre-trained model and save the classification
results.
"""

import argparse
from logging import Logger
from pathlib import Path

import pandas as pd

from config import load_config
from eval import load_model
from train import prepare_dataset
from utils import setup_logger

# Set up logging
classify_logger = setup_logger('classify_logger')


def classify_data(model, df: pd.DataFrame, logger: Logger):
    """
    Classifies the input data using the provided model and adds the predicted categories.
    """
    # Extract features from the input data (assuming 'text' column contains the input data)
    x_input = df['text']

    # Make predictions
    predictions = model.predict(x_input)

    # Add the predictions as a new column in the DataFrame
    df['category'] = predictions

    logger.info("Classification completed. Categories added.")

    return df


def save_classified_data(df: pd.DataFrame, config_dict: dict, logger: Logger):
    """
    Saves the classified data to a JSONL file.
    """
    # Ensure the directory exists before saving the file
    output_path = config_dict['output']['classify_results_path']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to JSONL format
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    logger.info(f"Classified data saved to {output_path}")


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Classify data using a trained model.")
    parser.add_argument(
        '--model', type=str, required=True, help="Path to the serialized model file.")
    parser.add_argument(
        '--input_data', type=str, required=True, help="Path to the input data file (JSONL format).")
    return parser.parse_args()


if __name__ == "__main__":
    # Load configuration
    config = load_config('config.yml')

    # Parse arguments
    args = parse_args()

    # Load input data
    input_data = pd.read_json(args.input_data, orient='records', lines=True)

    # Preprocess input data
    input_data = prepare_dataset(input_data, config, classify_logger)

    # Load the trained model
    classification_model = load_model(args.model, classify_logger)

    # Classify the data
    classified_data = classify_data(classification_model, input_data, classify_logger)

    # Save the classified data
    save_classified_data(classified_data, config, classify_logger)
