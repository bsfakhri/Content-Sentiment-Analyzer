import pandas as pd
import aivm_client as aic
import torch
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

@dataclass
class Config:
    """Application configuration settings"""
    MODEL_NAME: str = "BertTinySentimentTwitter"
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4

def analyze_text(text: str) -> Dict:
    """Analyze text sentiment using AIVM model."""
    try:
        tokenized_inputs = aic.tokenize(text)
        encrypted_inputs = aic.BertTinyCryptensor(tokenized_inputs[0], tokenized_inputs[1])
        result = aic.get_prediction(encrypted_inputs, Config.MODEL_NAME)
        
        logits = result[0]
        probabilities = torch.nn.functional.softmax(logits, dim=0)
        predicted_class = torch.argmax(logits).item()
        prob_list = probabilities.tolist()
        
        sorted_probs, _ = torch.sort(probabilities, descending=True)
        confidence = (sorted_probs[0] / (sorted_probs[0] + sorted_probs[1])).item()
        
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        predicted_sentiment = sentiment_map[predicted_class]
        
        return {
            'sentiment': predicted_sentiment,
            'confidence': confidence,
            'probabilities': prob_list
        }
    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        return None

def process_batch(texts: List[str]) -> List[Dict]:
    """Process a batch of texts in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_to_text = {executor.submit(analyze_text, text): text for text in texts}
        for future in as_completed(future_to_text):
            result = future.result()
            results.append(result if result else {
                'sentiment': "Error",
                'confidence': 0.0,
                'probabilities': [0, 0, 0]
            })
    return results

def main():
    start_time = time.time()
    
    # Read the dataset
    print("Loading dataset...")
    df = pd.read_csv('flight_data_cleaned.csv')
    df = df[:300]
    
    # Print column names to debug
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Assuming the column names might be different, let's try to identify them
    text_column = next((col for col in df.columns if 'text' in col.lower()), None)
    sentiment_column = next((col for col in df.columns if 'sentiment' in col.lower()), None)
    platform_column = next((col for col in df.columns if 'platform' in col.lower()), None)
    
    if not all([text_column, sentiment_column]):
        raise ValueError("Required columns not found in the dataset. Please check column names.")
    
    # Keep only required columns and preprocess
    columns_to_keep = [col for col in [text_column, sentiment_column, platform_column] if col is not None]
    analysis_df = df[columns_to_keep].copy()
    
    # Rename columns to standard names
    column_mapping = {
        text_column: 'Text',
        sentiment_column: 'Sentiment'
    }
    if platform_column:
        column_mapping[platform_column] = 'Platform'
    
    analysis_df = analysis_df.rename(columns=column_mapping)
    
    # Ensure Sentiment is properly capitalized
    if 'Sentiment' in analysis_df.columns:
        analysis_df['Sentiment'] = analysis_df['Sentiment'].str.capitalize()
    
    # Initialize lists for results
    all_results = []
    texts = analysis_df['Text'].tolist()
    
    # Process in batches with progress bar
    print("\nStarting sentiment analysis...")
    for i in tqdm(range(0, len(texts), Config.BATCH_SIZE)):
        batch_texts = texts[i:i + Config.BATCH_SIZE]
        batch_results = process_batch(batch_texts)
        all_results.extend(batch_results)
    
    # Add results to dataframe
    analysis_df['Predicted_Sentiment'] = [r['sentiment'] for r in all_results]
    analysis_df['Confidence'] = [r['confidence'] for r in all_results]
    
    # Add comparison column
    analysis_df['Matches_Original'] = (analysis_df['Sentiment'].str.strip() == 
                                     analysis_df['Predicted_Sentiment'])
    
    # Calculate accuracy metrics
    matching = analysis_df['Matches_Original'].sum()
    total = len(analysis_df)
    accuracy = matching / total * 100
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nAnalysis complete!")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Overall accuracy: {accuracy:.2f}%")
    
    # Save results
    output_file = 'sentiment_analysis_results_after_training.csv'
    analysis_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Display detailed statistics
    if 'Platform' in analysis_df.columns:
        print("\nPlatform Statistics:")
        platform_stats = analysis_df.groupby('Platform').agg({
            'Sentiment': 'count',
            'Confidence': ['mean', 'std'],
            'Matches_Original': 'mean'
        }).round(3)
        platform_stats.columns = ['Count', 'Avg Confidence', 'Std Confidence', 'Accuracy']
        print(platform_stats)
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    confusion_matrix = pd.crosstab(
        analysis_df['Sentiment'],
        analysis_df['Predicted_Sentiment'],
        normalize='index'
    ).round(3) * 100
    print(confusion_matrix)

if __name__ == "__main__":
    main()