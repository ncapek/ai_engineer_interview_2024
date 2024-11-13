"""
This module handles the training of a machine learning model using datasets
and performs grid search for hyperparameter tuning.
"""

import argparse
import os
from logging import Logger
from typing import Optional, Dict, Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import load_config
from utils import setup_logger

# Set up logging
train_logger = setup_logger('train_logger')


def prepare_dataset(input_df: pd.DataFrame, config_params: dict, logger: Logger) -> pd.DataFrame:
    """
    Prepares the dataset for text classification by combining 'headline' and 'short_description'
    into a single 'text' column and selecting only 'text' and the target column.
    """
    required_columns = config_params['datasets'].get('required_columns', [])
    target_column = config_params['datasets'].get('target_column', 'category')

    missing_columns = [col for col in required_columns if col not in input_df.columns]
    if missing_columns:
        logger.error("Missing columns in DataFrame: %s", missing_columns)
        raise ValueError(f"The following required columns are missing: {missing_columns}")

    input_df['headline'] = input_df['headline'].fillna('')
    input_df['short_description'] = input_df['short_description'].fillna('')

    input_df['text'] = input_df['headline'] + " " + input_df['short_description']
    logger.info("Combined 'headline' and 'short_description' into 'text' column.")

    input_df = input_df[['text', target_column]]
    logger.info("Selected 'text' and '%s' columns. Dataset preparation complete.", target_column)

    return input_df


def load_dataset(file_path: str, config_dict: Dict[str, Any], logger: Logger)\
        -> Optional[pd.DataFrame]:
    """
    Loads a JSON lines dataset from the specified file path with error handling and logging.
    """
    encoding = config_dict['datasets'].get('encoding', 'utf-8')
    lines = config_dict['datasets'].get('lines', True)

    try:
        # Load dataset
        df = pd.read_json(file_path, lines=lines, encoding=encoding)
        if df.empty:
            logger.warning("Loaded dataset from %s is empty.", file_path)
            return None
        logger.info("Successfully loaded dataset from %s. Shape: %s", file_path, df.shape)
        return df

    except ValueError as ve:
        logger.error("ValueError while loading JSON from %s: %s", file_path, ve)
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)

    return None


def get_scorer(metric_name: str):
    """
    Retrieves a scoring function based on the specified metric name.
    """
    if metric_name == "f1_weighted":
        return make_scorer(f1_score, average="weighted")
    if metric_name == "f1_macro":
        return make_scorer(f1_score, average="macro")
    if metric_name == "accuracy":
        return make_scorer(accuracy_score)
    raise ValueError(f"Unknown metric: {metric_name}")


def extract_config_values(config_dict: dict):
    """
    Extract configuration values for cleaner code and easier access.
    """
    return {
        'target_column': config_dict['datasets']['target_column'],
        'vectorizer_params': config_dict['hyperparameters']['TfidfVectorizer'],
        'classifier_params': config_dict['hyperparameters']['LogisticRegression'],
        'model_params': config_dict['model'],
        'scoring_metric': get_scorer(config_dict['model']['scoring']),
    }


def create_pipeline(vectorizer_params, classifier_params):
    """Creates and returns the model pipeline."""
    return Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_df=vectorizer_params['max_df'],
            min_df=vectorizer_params['min_df'],
            stop_words=vectorizer_params['stop_words']
        )),
        ('classifier', LogisticRegression(
            max_iter=classifier_params['max_iter'],
            random_state=classifier_params['random_seed'],
            C=classifier_params['C'],
            solver=classifier_params['solver']
        ))
    ])


def create_param_grid(vectorizer_params, classifier_params):
    """Create a parameter grid for GridSearchCV."""
    return {
        'vectorizer__max_df': vectorizer_params['max_df'],
        'vectorizer__min_df': vectorizer_params['min_df'],
        'vectorizer__stop_words': vectorizer_params['stop_words'],
        'classifier__C': classifier_params['C'],
        'classifier__solver': classifier_params['solver'],
        'classifier__max_iter': classifier_params['max_iter']
    }


def train_model(train_data_df: pd.DataFrame, dev_data_df: pd.DataFrame, config_params: dict,
                logger: Logger):
    """Train model with grid search."""

    logger.info("Starting model training with hyperparameter tuning...")

    # Extract configuration values
    config_values = extract_config_values(config_params)

    # Separate features and target from the datasets
    x_train, y_train = train_data_df['text'], train_data_df[config_values['target_column']]
    x_dev, y_dev = dev_data_df['text'], dev_data_df[config_values['target_column']]

    # Define the training pipeline and parameter grid
    pipeline = create_pipeline(config_values['vectorizer_params'],
                               config_values['classifier_params'])
    param_grid = create_param_grid(config_values['vectorizer_params'],
                                   config_values['classifier_params'])

    # Perform grid search with the dev set as validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=[(train_data_df.index, dev_data_df.index)],
        scoring=config_values['model_params']['scoring'],
        verbose=config_values['model_params']['verbose'],
        n_jobs=config_values['model_params']['n_jobs']
    )

    # Fit the model
    logger.info("Fitting the model with GridSearchCV...")
    grid_search.fit(x_train, y_train)

    # Log best parameters and score
    logger.info("Best parameters found: %s", grid_search.best_params_)
    logger.info("Best cross-validated accuracy: %.4f", grid_search.best_score_)

    # Evaluate on dev data and log results
    best_model = grid_search.best_estimator_
    dev_predictions = best_model.predict(x_dev)
    report = classification_report(y_dev, dev_predictions)
    logger.info("Classification report on dev set:\n%s", report)

    # Save the best model
    save_path = config_values['model_params']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    joblib.dump(best_model, save_path)
    logger.info("Model saved to %s", save_path)

    return best_model, grid_search.best_score_


def parse_args():
    """
    Parse command-line arguments for training and validation dataset file paths.
    """
    parser = argparse.ArgumentParser(description="Load train and validation datasets for training.")
    parser.add_argument('--train', type=str, required=True, help="Path to the training dataset.")
    parser.add_argument('--dev', type=str, required=True, help="Path to the development dataset.")
    parser.add_argument('--test', type=str, required=True, help="Path to the testing dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    # Load config
    config = load_config('config.yml')

    # Parse command-line arguments
    args = parse_args()

    # Load datasets
    train_data = load_dataset(args.train, config, train_logger)
    dev_data = load_dataset(args.dev, config, train_logger)

    # Apply the function to train and dev datasets
    train_data = prepare_dataset(train_data, config, train_logger)
    dev_data = prepare_dataset(dev_data, config, train_logger)

    _, _ = train_model(train_data, dev_data, config, train_logger)
