# Automated ML Pipeline with GitHub Actions
## IE7374 GitHub Actions Lab - Saurabh Suresh Kothari

## Overview

This project demonstrates an automated machine learning pipeline that trains a text classification model using GitHub Actions. The pipeline automatically retrains when the training dataset is updated, stores models as GitHub artifacts, and tracks metrics in the repository.

## Project Purpose

The system classifies customer support tickets into four categories: connectivity, billing, hardware, and software. It uses a Random Forest classifier with TF-IDF vectorization to process and classify ticket descriptions.

## Key Features

- Automatic model retraining when dataset changes
- Manual training workflow option
- Model versioning with timestamps
- Metrics tracking in Git
- Model storage as GitHub artifacts (90-day retention)
- Scheduled model evaluation

## Technology Stack

- Python 3.10
- scikit-learn for machine learning
- NLTK for text preprocessing
- GitHub Actions for automation
- pandas for data manipulation

## Project Structure

```
automated-ml-pipeline/
├── .github/workflows/
│   ├── train_on_data_change.yml    # Automatic training on data changes
│   ├── manual_training.yml         # Manual training trigger
│   └── scheduled_evaluation.yml    # Periodic model evaluation
├── data/
│   └── support_tickets.csv         # Training dataset
├── src/
│   ├── prepare_data.py            # Data preprocessing
│   ├── train_classifier.py        # Model training
│   └── evaluate_model.py          # Model evaluation
├── models/                         # Model files (stored as artifacts)
├── metrics/                        # Training metrics (tracked in Git)
└── requirements.txt
```

## Prerequisites

- Python 3.9 or higher
- Git
- GitHub account
- pip package manager

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/automated-ml-pipeline.git
cd automated-ml-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure GitHub Repository

Go to your repository settings on GitHub:

1. Navigate to Settings > Actions > General
2. Under "Workflow permissions", select "Read and write permissions"
3. Click Save

This allows workflows to commit metrics back to the repository.

## Running Locally

### Preprocess Data

```bash
python3 src/prepare_data.py
```

This script:
- Loads the raw dataset
- Cleans and normalizes text
- Removes stopwords and performs lemmatization
- Saves processed data to data/support_tickets_processed.csv

### Train Model

```bash
python3 src/train_classifier.py
```

This script:
- Loads preprocessed data
- Creates TF-IDF features
- Trains a Random Forest classifier
- Saves model files to models/ directory
- Generates training metrics in metrics/ directory

### Evaluate Model

```bash
timestamp=$(ls models/model_*.joblib | head -1 | sed 's/.*model_\(.*\)\.joblib/\1/')
python src/evaluate_model.py --timestamp $timestamp
```

This script:
- Loads a trained model by timestamp
- Evaluates performance on test data
- Generates classification report
- Saves evaluation results

## Automated Workflows

### Automatic Training

The workflow triggers automatically when data/support_tickets.csv is modified and pushed to the main branch.

**To trigger:**

```bash
echo "52,connectivity,Network connection dropped,high" >> data/support_tickets.csv
git add data/support_tickets.csv
git commit -m "Add new support ticket"
git push origin main
```

The workflow will:
1. Detect the dataset change
2. Preprocess the updated data
3. Train a new model
4. Store the model as a GitHub artifact
5. Commit updated metrics to the repository

### Manual Training

To manually trigger training:

1. Go to the Actions tab in your GitHub repository
2. Select "Manual Training" workflow
3. Click "Run workflow"
4. Optionally enter a reason for training
5. Click the green "Run workflow" button

### Scheduled Evaluation

The evaluation workflow runs automatically every day at midnight UTC. It can also be triggered manually from the Actions tab.

## Workflow Details

### train_on_data_change.yml

**Trigger:** Push to main branch with changes to data/support_tickets.csv

**Steps:**
1. Checkout code
2. Set up Python environment
3. Cache pip dependencies
4. Install requirements
5. Verify data file exists
6. Run data preprocessing
7. Train classification model
8. Upload model as artifact (90-day retention)
9. Commit metrics to repository
10. Create workflow summary

### manual_training.yml

**Trigger:** Manual workflow dispatch

**Steps:** Same as automatic training, but can be triggered on demand with an optional reason parameter.

### scheduled_evaluation.yml

**Trigger:** Daily at midnight UTC or manual dispatch

**Steps:**
1. Download latest model artifact
2. Run evaluation on test dataset
3. Generate performance report
4. Upload evaluation results as artifact
5. Display summary in workflow

## Dataset Format

The training dataset (data/support_tickets.csv) has the following structure:

```csv
ticket_id,category,description,priority
1,connectivity,My internet connection keeps dropping,high
2,billing,I was charged twice this month,medium
...
```

**Columns:**
- ticket_id: Unique identifier
- category: One of connectivity, billing, hardware, software
- description: Text description of the issue
- priority: low, medium, or high

## Model Details

**Algorithm:** Random Forest Classifier
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2

**Feature Extraction:** TF-IDF Vectorizer
- max_features: 500
- min_df: 2
- max_df: 0.8
- ngram_range: (1, 2)

**Text Preprocessing:**
- Lowercase conversion
- Special character removal
- Stopword removal
- Lemmatization
