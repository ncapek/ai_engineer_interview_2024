# ai_engineer_interview_2024

This repository contains the solution for an AI engineer case study in which a text classification model is built for news categorization. The project implements scripts for training, evaluating, and classifying news data based on predefined categories.

## Approach

The solution follows the requirements of the case study by implementing a production-like pipeline for model training, evaluation, and classification using Python. The process is structured around three key scripts (`train.py`, `eval.py`, and `classify.py`), as well as a configuration file (`config.yml`) to manage paths and other parameters.

Key Highlights:
- **Data preparation**: The headline and short_description columns were combined into a single text column, which is then used to derive features
- **Preprocessing**: A minimalistic modelling approach was selected, with a TFIDF vectorization combined with english stopword removal
- **Model**: A simple logistic regression model was used
- **Hyperparameter tuning**: The config yaml files specifies hyperparameter space to search for TFIDF and LogisticRegrssion. Weighted F1 score was used to handle class imbalance better than accuracy.
- **Evaluation**: Evaluation based on accuracy, confusion matrix, and classification report for deeper insights.
- **Classification**: A separate script to classify the input data and add predicted categories to the output.
- **CI on GithubActions**: A simple linting check was setup on GitHub actions.
- **Logging**: Logging is dumped into app.log as well as in console.

## Files and Structure

- **`train.py`**: 
  - **Inputs**: Training data, validation (dev) data.
  - **Outputs**: Serialized model file (specified in config).
  
- **`eval.py`**: 
  - **Inputs**: Serialized model file, evaluation data in JSONL format.
  - **Outputs**: Model accuracy, confusion matrix, classification report.

- **`classify.py`**: 
  - **Inputs**: Serialized model file, data to classify (JSONL with `headline` and `short_description` columns, may or may not contain other columns).
  - **Outputs**: Input JSONL with a new `category` field set to model’s predicted values.

- **`requirements.txt`**: 
  - List of dependencies for the project in Pip format.

- **`config.yml`**: 
  - Configuration file for paths, model settings, and other parameters.

## Improvements and Next Steps

- **Removing hard coded values**: Despite efforts for maximal configurability, current code still contains some hard coded values here and there.
- **Target Class Imbalance**: The target class has high imbalance and cardinality, potentially requiring grouping or resampling techniques for improved performance.
- **Preprocessing Enhancements**: Future improvements may include exploring stemming, lemmatization, or different vectorization techniques like word embeddings or contextual models (e.g., BERT).
- **Software engineering improvements**: The project can be better structured, i.e. packaging functionality, etc
- **Better models**: The chosen model (pipeline) was selected for its simplicity. Efforts were made to explore a LLM based approach, but abandoned due to high computational requirements. Other options might be more suitable, i.e. word embeddings, etc.
- **Better customizability**: Code could be altered to allow for setting a variety of learning algorithms and/or preprocessing techniques.
- **Add unit tests**: Unit tests might be added depending on time options and integrated into CI pipeline.
- **More data**: Since the cardinality of the target is high and the size of the validation and test datasets is quite low, some categories likely don't have enough instances for very good evaluation.

## Instructions

### Install Dependencies
1. Get the repo: `git clone https://github.com/ncapek/ai_engineer_interview_2024.git`
2. Switch to branch: `git switch feature/case_study`
3. Create virtual environment: `python -m venv venv`
4. Activate virtual environment (windows machine): `venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Move datasets into a data directory in the repo
7. Run train script: `python train.py --train data/train.jsonl --dev data/dev.jsonl`
8. Run eval script: `python eval.py --model models/best_model.pkl --eval_data data/test.jsonl`
9. Run classify script: `python classify.py --model models/best_model.pkl --input_data data/test.jsonl`

## Final comments
- The solution is not perfect, a larger emphasis was placed on technical requirements than finding the best model
- With more time a more accurate approach might be identified and developed
- Hyperparameter tuning on my desktop machine takes approximately 3 minutes.
- Final accuracy on test set: 0.4685
- The project was fun