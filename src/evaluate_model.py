import pandas as pd
import numpy as np
import os
import json
import joblib
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def load_model_and_vectorizer(timestamp):
    model_path = os.path.join('models', f'model_{timestamp}.joblib')
    vectorizer_path = os.path.join('models', f'vectorizer_{timestamp}.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"Error: Model files not found for timestamp {timestamp}")
        return None, None
    
    print(f"Loading model: {model_path}")
    print(f"Loading vectorizer: {vectorizer_path}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

def load_test_data():
    data_path = os.path.join('data', 'support_tickets_processed.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Test data not found at {data_path}")
        return None
    
    print(f"Loading test data from: {data_path}")
    df = pd.read_csv(data_path)
    
    return df

def evaluate_model(model, vectorizer, df):
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    print(f"\nTest set size: {len(df)} records")
    
    X = vectorizer.transform(df['description_processed'])
    y_true = df['category']
    
    print("\nGenerating predictions...")
    y_pred = model.predict(X)
    
    metrics = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'test_samples': len(df),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted'))
    }
    
    print("\nEvaluation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score (weighted): {metrics['f1_score_weighted']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    per_class_metrics = {}
    for label in np.unique(y_true):
        mask = y_true == label
        per_class_metrics[str(label)] = {
            'precision': float(precision_score(y_true[mask], y_pred[mask], average='binary', pos_label=label)),
            'recall': float(recall_score(y_true[mask], y_pred[mask], average='binary', pos_label=label)),
            'f1_score': float(f1_score(y_true[mask], y_pred[mask], average='binary', pos_label=label)),
            'support': int(mask.sum())
        }
    
    metrics['per_class'] = per_class_metrics
    
    return metrics

def save_evaluation_results(metrics, model_timestamp):
    os.makedirs('metrics', exist_ok=True)
    
    eval_path = os.path.join('metrics', f'evaluation_{metrics["timestamp"]}.json')
    
    evaluation_data = {
        'model_timestamp': model_timestamp,
        'evaluation_timestamp': metrics['timestamp'],
        'metrics': metrics
    }
    
    with open(eval_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"\nEvaluation results saved to: {eval_path}")
    
    return eval_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--timestamp', required=True, help='Model timestamp to evaluate')
    
    args = parser.parse_args()
    
    model, vectorizer = load_model_and_vectorizer(args.timestamp)
    
    if model is None:
        return 1
    
    df = load_test_data()
    
    if df is None:
        return 1
    
    metrics = evaluate_model(model, vectorizer, df)
    
    eval_path = save_evaluation_results(metrics, args.timestamp)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())