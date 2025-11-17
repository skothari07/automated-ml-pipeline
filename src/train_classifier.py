import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def load_processed_data():
    data_path = os.path.join('data', 'support_tickets_processed.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        print("Please run prepare_data.py first")
        return None
    
    print(f"Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)
    return df

def create_features(df, max_features=500):
    print("\nCreating TF-IDF features...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(df['description_processed'])
    y = df['category']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X, y, vectorizer

def train_model(X_train, y_train, n_estimators=100):
    print("\nTraining Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training complete")
    
    return model

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_score_weighted': float(f1_score(y_test, y_pred, average='weighted')),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted')),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted'))
    }
    
    print("\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score (weighted): {metrics['f1_score_weighted']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics

def save_model_locally(model, vectorizer, timestamp):
    os.makedirs('models', exist_ok=True)
    
    model_path = os.path.join('models', f'model_{timestamp}.joblib')
    vectorizer_path = os.path.join('models', f'vectorizer_{timestamp}.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\nModels saved locally:")
    print(f"  Classifier: {model_path}")
    print(f"  Vectorizer: {vectorizer_path}")
    
    return model_path, vectorizer_path

def save_metrics(metrics, train_size, test_size, timestamp):
    os.makedirs('metrics', exist_ok=True)
    
    metrics_data = {
        'timestamp': timestamp,
        'training_samples': train_size,
        'test_samples': test_size,
        **metrics
    }
    
    metrics_path = os.path.join('metrics', f'training_metrics_{timestamp}.json')
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    return metrics_path

def main():
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTraining session: {timestamp}")
    
    df = load_processed_data()
    if df is None:
        return False
    
    print(f"\nDataset size: {len(df)} records")
    
    X, y, vectorizer = create_features(df)
    
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    model = train_model(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    model_path, vectorizer_path = save_model_locally(model, vectorizer, timestamp)
    
    metrics_path = save_metrics(metrics, X_train.shape[0], X_test.shape[0], timestamp)
    
    metadata = {
        'timestamp': timestamp,
        'model_path': model_path,
        'vectorizer_path': vectorizer_path,
        'metrics_path': metrics_path,
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'feature_extractor': 'TfidfVectorizer',
        'max_features': 500
    }
    
    metadata_path = os.path.join('models', f'metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)