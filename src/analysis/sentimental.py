import pandas as pd
import os
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Note: Make sure to install required libraries:
# pip install pandas nltk
# and download NLTK data:
# python -m nltk.downloader vader_lexicon punkt stopwords

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
POSTS_DATA_PATH = BASE_DIR / "data" / "processed"
COMMENTS_DATA_PATH = BASE_DIR / "data" / "processed" / "cmts"
POSTS_RESULTS_PATH = BASE_DIR / "data" / "results" 
COMMENTS_RESULTS_PATH = BASE_DIR / "data" / "results" / "cmts"

# Sentiment Analyzer Class
class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        """Clean and preprocess text for sentiment analysis."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtag symbols but keep words
        text = re.sub(r'#', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment_scores(self, text):
        """Get VADER sentiment scores."""
        if not text or pd.isna(text):
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        return self.sia.polarity_scores(text)
    
    def classify_sentiment(self, compound_score):
        """Classify sentiment based on compound score."""
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_dataframe(self, df, text_column='text', processed_col_name='processed_text'):
        """Analyze sentiment for a DataFrame."""
        df = df.copy()
        
        print(f"Cleaning text for {len(df)} records...")
        df[processed_col_name] = df[text_column].apply(self.clean_text)
        
        print(f"Analyzing sentiment...")
        
        # Get sentiment scores
        sentiment_scores = df[processed_col_name].apply(self.get_sentiment_scores)
        
        df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
        df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
        df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        
        # Classify sentiment
        df['sentiment_label'] = df['sentiment_compound'].apply(self.classify_sentiment)
        
        return df
    
    def print_summary(self, df, data_type='records', original_col='text'):
        """Print sentiment analysis summary."""
        print("\n" + "=" * 50)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("=" * 50)
        
        total = len(df)
        print(f"\nTotal {data_type} analyzed: {total}")
        
        # Sentiment distribution
        print("\nSentiment Distribution:")
        sentiment_counts = df['sentiment_label'].value_counts()
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            count = sentiment_counts.get(sentiment, 0)
            pct = (count / total) * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {sentiment:10} {count:6} ({pct:5.1f}%) {bar}")
        
        # Average scores
        print("\nAverage Scores:")
        print(f"  Positive:  {df['sentiment_positive'].mean():.4f}")
        print(f"  Neutral:   {df['sentiment_neutral'].mean():.4f}")
        print(f"  Negative:  {df['sentiment_negative'].mean():.4f}")
        print(f"  Compound:  {df['sentiment_compound'].mean():.4f}")
        
        # Most positive and negative
        if len(df) > 0 and df['sentiment_compound'].notna().any():
            print("\nExtreme Examples:")
            most_positive = df.loc[df['sentiment_compound'].idxmax()]
            most_negative = df.loc[df['sentiment_compound'].idxmin()]
            
            print(f"  Most positive ({most_positive['sentiment_compound']:.3f}):")
            text_content = str(most_positive[original_col])[:80]
            print(f"    \"{text_content}...\"")
            print(f"  Most negative ({most_negative['sentiment_compound']:.3f}):")
            text_content = str(most_negative[original_col])[:80]
            print(f"    \"{text_content}...\"")
        
        print("=" * 50)

def process_posts(analyzer):
    """Process LinkedIn posts sentiment analysis."""
    print("\n" + "=" * 60)
    print("PROCESSING LINKEDIN POSTS")
    print("=" * 60)
    
    # Check if posts data directory exists
    if not POSTS_DATA_PATH.exists():
        print(f"Error: Directory not found: {POSTS_DATA_PATH}")
        return 0, 0, 0
    
    # Find all CSV files (exclude cmts subdirectory)
    csv_files = [f for f in POSTS_DATA_PATH.glob("*.csv") if f.is_file()]
    
    if not csv_files:
        print(f"No CSV files found in: {POSTS_DATA_PATH}")
        return 0, 0, 0
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, file in enumerate(csv_files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"  {i}. {file.name} ({size_kb:.1f} KB)")
    
    # Create results directory
    POSTS_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    successful_files = 0
    failed_files = 0
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {csv_file.name}")
        print('=' * 60)
        
        # Read CSV
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"Loaded {len(df)} rows")
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            failed_files += 1
            continue
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            print(f"'text' column not found in {csv_file.name}")
            print(f"Available columns: {df.columns.tolist()}")
            failed_files += 1
            continue
        
        # Analyze sentiment
        try:
            df = analyzer.analyze_dataframe(df, text_column='text', processed_col_name='processed_text')
            
            # Generate output filename
            output_filename = f"results_{csv_file.stem}.csv"
            output_path = POSTS_RESULTS_PATH / output_filename
            
            # Save results
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nResults saved to: {output_path}")
            
            # Print summary
            analyzer.print_summary(df, data_type='posts', original_col='text')
            
            # Show sample results
            print("\n=== Sample Results ===")
            sample_cols = ['text', 'processed_text', 'sentiment_label', 'sentiment_compound']
            sample_df = df[sample_cols].head(3)
            
            for idx, row in sample_df.iterrows():
                print(f"\nPost {idx + 1}:")
                print(f"Original: {str(row['text'])[:80]}...")
                print(f"Processed: {str(row['processed_text'])[:80]}...")
                print(f"Sentiment: {row['sentiment_label']} (Score: {row['sentiment_compound']:.4f})")
            
            total_processed += len(df)
            successful_files += 1
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            failed_files += 1
            continue
    
    return total_processed, successful_files, failed_files

def process_comments(analyzer):
    """Process LinkedIn comments sentiment analysis."""
    print("\n" + "=" * 60)
    print("PROCESSING LINKEDIN COMMENTS")
    print("=" * 60)
    
    # Check if comments data directory exists
    if not COMMENTS_DATA_PATH.exists():
        print(f"Info: Comments directory not found: {COMMENTS_DATA_PATH}")
        print(f"Skipping comments processing...")
        return 0, 0, 0
    
    # Find all CSV files
    csv_files = list(COMMENTS_DATA_PATH.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in: {COMMENTS_DATA_PATH}")
        return 0, 0, 0
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:")
    total_size = 0
    for i, file in enumerate(csv_files, 1):
        size_kb = file.stat().st_size / 1024
        total_size += size_kb
        print(f"  {i}. {file.name} ({size_kb:.1f} KB)")
    print(f"\nTotal size: {total_size:.1f} KB")
    
    # Create results directory
    COMMENTS_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {COMMENTS_RESULTS_PATH}")
    
    total_processed = 0
    successful_files = 0
    failed_files = 0
    
    # Process each CSV file
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing [{idx}/{len(csv_files)}]: {csv_file.name}")
        print('=' * 60)
        
        # Read CSV
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"Loaded {len(df)} rows")
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            failed_files += 1
            continue
        
        # Check if 'Comment' column exists
        if 'Comment' not in df.columns:
            print(f"'Comment' column not found in {csv_file.name}")
            print(f"Available columns: {df.columns.tolist()}")
            failed_files += 1
            continue
        
        # Analyze sentiment
        try:
            df = analyzer.analyze_dataframe(df, text_column='Comment', processed_col_name='processed_comment')
            
            # Generate output filename
            output_filename = f"results_{csv_file.stem}.csv"
            output_path = COMMENTS_RESULTS_PATH / output_filename
            
            # Save results
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nResults saved to: {output_path}")
            
            # Print summary
            analyzer.print_summary(df, data_type='comments', original_col='Comment')
            
            # Show sample results
            print("\n=== Sample Results ===")
            if len(df) > 0:
                sample_cols = ['Comment', 'processed_comment', 'sentiment_label', 'sentiment_compound']
                sample_df = df[sample_cols].head(3)
                
                for row_idx, row in sample_df.iterrows():
                    print(f"\nComment {row_idx + 1}:")
                    print(f"Original: {str(row['Comment'])[:80]}...")
                    print(f"Processed: {str(row['processed_comment'])[:80]}...")
                    print(f"Sentiment: {row['sentiment_label']} (Score: {row['sentiment_compound']:.4f})")
            
            total_processed += len(df)
            successful_files += 1
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            failed_files += 1
            continue
    
    return total_processed, successful_files, failed_files

def main():
    print("=" * 60)
    print("LinkedIn Sentiment Analyzer")
    print("Posts & Comments Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    print("\nInitializing sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    
    # Process posts
    posts_processed, posts_success, posts_failed = process_posts(analyzer)
    
    # Process comments
    comments_processed, comments_success, comments_failed = process_comments(analyzer)
    
    # Final overall summary
    print("\n" + "=" * 60)
    print("OVERALL PROCESSING SUMMARY")
    print("=" * 60)
    
    print("\nPOSTS:")
    print(f"  Successfully processed: {posts_success}")
    print(f"  Failed: {posts_failed}")
    print(f"  Total posts analyzed: {posts_processed}")
    if posts_success > 0:
        print(f"  Results saved to: {POSTS_RESULTS_PATH}")
    
    print("\nCOMMENTS:")
    print(f"  Successfully processed: {comments_success}")
    print(f"  Failed: {comments_failed}")
    print(f"  Total comments analyzed: {comments_processed}")
    if comments_success > 0:
        print(f"  Results saved to: {COMMENTS_RESULTS_PATH}")
    
    print("\nGRAND TOTAL:")
    print(f"  Files processed: {posts_success + comments_success}")
    print(f"  Files failed: {posts_failed + comments_failed}")
    print(f"  Records analyzed: {posts_processed + comments_processed}")
    
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()