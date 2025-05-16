# Purchase Prediction Model for Push Notifications

## Overview

This module implements a machine learning pipeline to predict which users are likely to purchase specific products. These predictions are used to send targeted push notifications, optimizing marketing efforts while minimizing user disruption.

## Business Context

- Push notifications in our app have only a 5% open rate
- Irrelevant notifications cause user fatigue and may lead to app uninstalls
- We need to balance precision (relevance) with reach (recall)

## Implementation

### Data Preparation
- Filters orders to include only those with 5+ items (business requirement)
- Processes date features from timestamp data
- Encodes categorical variables using different techniques
- Splits data into train/validation/test sets preserving order integrity

### Model Selection
The module compares different pipeline configurations:
1. Frequency Encoding + LogisticRegression
2. Target Encoding + LogisticRegression

(More in the notebook)

For each model:
- Finds the optimal threshold that maximizes F0.3 score (precision-focused metric)
- Evaluates performance on validation data
- Selects the best performing model based on F0.3 score

### Model Versioning
The system implements a robust model versioning system that:
- Creates timestamped versions of each trained model
- Stores metadata including performance metrics
- Maintains a model registry for tracking model history
- Uses symbolic links to point to the active model

## Usage

1. **Train and select the best model**:
```
poetry run python -m src.module_3.main
```
2. **Make predictions with the active model**:
```
poetry run python -m src.module_3.predict src/module_3/to_predict_sample.csv
```
## Performance Considerations

The model prioritizes precision over recall to avoid sending irrelevant notifications. This approach ensures:
- Higher quality user experience
- Less notification fatigue
- More targeted marketing campaigns

## Directory Structure

- `config.py`: Configuration settings and model versioning functionality
- `preprocessing.py`: Feature transformation and encoding classes
- `train.py`: Model training and evaluation functions
- `predict.py`: Inference functionality for production use
- `main.py`: End-to-end pipeline execution