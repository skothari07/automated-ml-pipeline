import pandas as pd
import re
import os
import json
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def prepare_data():
    print("\n" + "="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    data_path = os.path.join('data', 'support_tickets.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return False
    
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"\nDataset Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Features: {list(df.columns)}")
    print(f"  Categories: {df['category'].nunique()}")
    
    print("\nCategory Distribution:")
    for category, count in df['category'].value_counts().items():
        print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\nProcessing text data...")
    df['description_clean'] = df['description'].apply(clean_text)
    df['description_processed'] = df['description'].apply(preprocess_text)
    
    output_path = os.path.join('data', 'support_tickets_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    stats = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'total_records': len(df),
        'num_categories': df['category'].nunique(),
        'avg_words_original': float(df['description'].str.split().str.len().mean()),
        'avg_words_processed': float(df['description_processed'].str.split().str.len().mean())
    }
    
    stats_path = os.path.join('metrics', 'preprocessing_stats.json')
    os.makedirs('metrics', exist_ok=True)
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nPreprocessing Statistics:")
    print(f"  Average words (original): {stats['avg_words_original']:.1f}")
    print(f"  Average words (processed): {stats['avg_words_processed']:.1f}")
    print(f"  Reduction: {((stats['avg_words_original'] - stats['avg_words_processed']) / stats['avg_words_original'] * 100):.1f}%")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = prepare_data()
    exit(0 if success else 1)