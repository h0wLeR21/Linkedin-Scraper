import pandas as pd
import json
import re
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
POSTS_RESULTS_PATH = BASE_DIR / "data" / "results"
COMMENTS_RESULTS_PATH = BASE_DIR / "data" / "results" / "cmts"
OUTPUT_CSV_PATH = BASE_DIR / "data" / 'results' /"combined_posts_comments_long.csv"
OUTPUT_JSON_PATH = BASE_DIR / "data" / 'results' /"combined_posts_comments_nested.json"

def extract_urn_from_filename(filename):
    """Extract URN ID from comment filename."""
    match = re.search(r'linkedin-(\d+)', filename)
    if match:
        return match.group(1)
    return None

def load_posts_data():
    """Load all posts data with sentiment analysis."""
    print("=" * 60)
    print("Loading Posts Data")
    print("=" * 60)
    
    posts_files = list(POSTS_RESULTS_PATH.glob("results_*.csv"))
    
    if not posts_files:
        print(f"No results files found in: {POSTS_RESULTS_PATH}")
        return None
    
    print(f"Found {len(posts_files)} post result file(s)")
    
    all_posts = []
    for file in posts_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            all_posts.append(df)
            print(f"  ✓ Loaded {file.name}: {len(df)} posts")
        except Exception as e:
            print(f"  ✗ Error loading {file.name}: {e}")
    
    if all_posts:
        combined_posts = pd.concat(all_posts, ignore_index=True)
        print(f"\nTotal posts loaded: {len(combined_posts)}")
        return combined_posts
    
    return None

def extract_urn_id(urn_string):
    """Extract numeric ID from URN string."""
    if pd.isna(urn_string):
        return None
    
    match = re.search(r'(\d+)$', str(urn_string))
    if match:
        return match.group(1)
    return None

def load_comments_data():
    """Load all comments data with sentiment analysis."""
    print("\n" + "=" * 60)
    print("Loading Comments Data")
    print("=" * 60)
    
    if not COMMENTS_RESULTS_PATH.exists():
        print(f"Comments directory not found: {COMMENTS_RESULTS_PATH}")
        return None
    
    comment_files = list(COMMENTS_RESULTS_PATH.glob("results_*.csv"))
    
    if not comment_files:
        print(f"No comment files found in: {COMMENTS_RESULTS_PATH}")
        return None
    
    print(f"Found {len(comment_files)} comment result file(s)")
    
    all_comments = []
    for file in comment_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            
            # Extract URN ID from filename
            urn_id = extract_urn_from_filename(file.name)
            if urn_id:
                df['post_urn_id'] = urn_id
                all_comments.append(df)
                print(f"  ✓ Loaded {file.name}: {len(df)} comments (URN: {urn_id})")
            else:
                print(f"  ⚠ Could not extract URN from {file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {file.name}: {e}")
    
    if all_comments:
        combined_comments = pd.concat(all_comments, ignore_index=True)
        print(f"\nTotal comments loaded: {len(combined_comments)}")
        return combined_comments
    
    return None

def create_long_format_csv(posts_df, comments_df):
    """Create long format CSV (Option 2)."""
    print("\n" + "=" * 60)
    print("Creating Long Format CSV")
    print("=" * 60)
    
    # Extract URN ID from posts
    if 'full_urn' in posts_df.columns:
        posts_df['post_urn_id'] = posts_df['full_urn'].apply(extract_urn_id)
        print(f"Extracted URN IDs from {posts_df['post_urn_id'].notna().sum()} posts")
    else:
        print("Warning: 'full_urn' column not found in posts data")
        return None
    
    # Keep ALL columns from posts
    posts_subset = posts_df.copy()
    
    # Rename all post columns to be clear (add 'post_' prefix except post_urn_id)
    posts_subset.columns = ['post_' + col if col != 'post_urn_id' else col 
                            for col in posts_subset.columns]
    
    if comments_df is None or len(comments_df) == 0:
        print("No comments data available. Creating posts-only CSV.")
        posts_subset['has_comment'] = False
        return posts_subset
    
    # Keep ALL columns from comments
    comments_subset = comments_df.copy()
    
    # Rename all comment columns to be clear (add 'comment_' prefix except post_urn_id)
    comments_subset.columns = ['comment_' + col if col != 'post_urn_id' else col 
                               for col in comments_subset.columns]
    
    # Merge posts and comments
    print(f"\nMerging {len(posts_subset)} posts with {len(comments_subset)} comments...")
    combined = pd.merge(
        posts_subset,
        comments_subset,
        on='post_urn_id',
        how='left'
    )
    
    print(f"Combined dataset: {len(combined)} rows")
    
    # Add flags for easier filtering
    combined['has_comment'] = combined['comment_Comment'].notna() if 'comment_Comment' in combined.columns else False
    
    # Check if sentiment columns exist before creating sentiment_match
    if 'post_sentiment_label' in combined.columns and 'comment_sentiment_label' in combined.columns:
        combined['sentiment_match'] = (
            combined['post_sentiment_label'] == combined['comment_sentiment_label']
        )
    
    return combined

def create_nested_json(posts_df, comments_df):
    """Create nested JSON structure (Option 3)."""
    print("\n" + "=" * 60)
    print("Creating Nested JSON Structure")
    print("=" * 60)
    
    # Extract URN ID from posts
    if 'full_urn' in posts_df.columns:
        posts_df['post_urn_id'] = posts_df['full_urn'].apply(extract_urn_id)
        print(f"Extracted URN IDs from {posts_df['post_urn_id'].notna().sum()} posts")
    else:
        print("Warning: 'full_urn' column not found in posts data")
        return None
    
    # Group comments by post URN ID
    comments_grouped = {}
    if comments_df is not None and len(comments_df) > 0:
        print(f"\nGrouping {len(comments_df)} comments by post...")
        for post_urn in comments_df['post_urn_id'].unique():
            post_comments = comments_df[comments_df['post_urn_id'] == post_urn]
            
            # Convert comments to list of dictionaries (include ALL columns)
            comments_list = []
            for _, comment_row in post_comments.iterrows():
                comment_dict = comment_row.to_dict()
                # Remove post_urn_id from comment data (it's in parent)
                comment_dict.pop('post_urn_id', None)
                comments_list.append(comment_dict)
            
            comments_grouped[post_urn] = comments_list
        
        print(f"Grouped comments for {len(comments_grouped)} posts")
    
    # Create nested structure
    nested_data = []
    print(f"\nCreating nested records for {len(posts_df)} posts...")
    
    for _, post_row in posts_df.iterrows():
        post_urn_id = post_row.get('post_urn_id')
        
        # Get comments for this post
        post_comments = comments_grouped.get(post_urn_id, [])
        
        # Calculate comment statistics
        comment_stats = {
            'total_comments': len(post_comments),
            'positive_comments': sum(1 for c in post_comments if c.get('sentiment_label') == 'Positive'),
            'negative_comments': sum(1 for c in post_comments if c.get('sentiment_label') == 'Negative'),
            'neutral_comments': sum(1 for c in post_comments if c.get('sentiment_label') == 'Neutral'),
            'avg_sentiment_compound': (
                sum(c.get('sentiment_compound', 0) for c in post_comments) / len(post_comments)
                if post_comments else 0
            )
        }
        
        # Create nested record with ALL post data
        record = post_row.to_dict()
        record['comment_stats'] = comment_stats
        record['comments'] = post_comments
        
        nested_data.append(record)
    
    print(f"Created {len(nested_data)} nested records")
    return nested_data

def generate_summary(csv_df, json_data):
    """Generate comprehensive summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    # CSV stats
    total_rows = len(csv_df)
    total_posts = csv_df['post_urn_id'].nunique()
    
    if 'has_comment' in csv_df.columns:
        total_comments = csv_df['has_comment'].sum()
        posts_with_comments = csv_df[csv_df['has_comment']]['post_urn_id'].nunique()
    else:
        total_comments = 0
        posts_with_comments = 0
    
    posts_without_comments = total_posts - posts_with_comments
    
    print(f"\nTotal unique posts: {total_posts}")
    print(f"Posts with comments: {posts_with_comments}")
    print(f"Posts without comments: {posts_without_comments}")
    print(f"Total comments: {total_comments}")
    print(f"Total rows in CSV: {total_rows}")
    
    if total_comments > 0:
        avg_comments = total_comments / posts_with_comments
        print(f"Average comments per post (with comments): {avg_comments:.2f}")
        
        # Post sentiment distribution
        print("\nPost Sentiment Distribution:")
        post_sentiment = csv_df.drop_duplicates('post_urn_id')['post_sentiment_label'].value_counts()
        for sentiment, count in post_sentiment.items():
            pct = (count / total_posts) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")
        
        # Comment sentiment distribution
        if 'comment_sentiment_label' in csv_df.columns:
            print("\nComment Sentiment Distribution:")
            comment_sentiment = csv_df[csv_df['has_comment']]['comment_sentiment_label'].value_counts()
            for sentiment, count in comment_sentiment.items():
                pct = (count / total_comments) * 100
                print(f"  {sentiment}: {count} ({pct:.1f}%)")
        
        # Sentiment match analysis
        if 'sentiment_match' in csv_df.columns:
            matching = csv_df[csv_df['has_comment']]['sentiment_match'].sum()
            match_pct = (matching / total_comments) * 100
            print(f"\nComments matching post sentiment: {matching} ({match_pct:.1f}%)")
        
        # Top commented posts from JSON
        print("\nTop 5 Most Commented Posts:")
        sorted_posts = sorted(json_data, 
                            key=lambda x: x['comment_stats']['total_comments'], 
                            reverse=True)[:5]
        
        for idx, record in enumerate(sorted_posts, 1):
            if record['comment_stats']['total_comments'] > 0:
                print(f"  {idx}. URN {record['post_urn_id']}: {record['comment_stats']['total_comments']} comments")
                post_text = record.get('text', '')
                print(f"     Post: {str(post_text)[:60]}...")
                sentiment_label = record.get('sentiment_label', 'N/A')
                print(f"     Sentiment: Post={sentiment_label}, "
                      f"Avg Comment={record['comment_stats']['avg_sentiment_compound']:.3f}")

def main():
    print("=" * 60)
    print("Hybrid Format: Long CSV + Nested JSON")
    print("=" * 60)
    
    # Load posts data
    posts_df = load_posts_data()
    if posts_df is None or len(posts_df) == 0:
        print("\nError: No posts data found. Please run sentiment analysis first.")
        return
    
    # Load comments data
    comments_df = load_comments_data()
    
    # Create Long Format CSV (Option 2)
    print("\n" + "=" * 60)
    print("CREATING LONG FORMAT CSV")
    print("=" * 60)
    
    csv_df = create_long_format_csv(posts_df, comments_df)
    if csv_df is None:
        print("\nError: Failed to create CSV format.")
        return
    
    # Save CSV
    csv_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print(f"\n✓ Long Format CSV saved to: {OUTPUT_CSV_PATH}")
    print(f"  Total rows: {len(csv_df)}")
    print(f"  Total columns: {len(csv_df.columns)}")
    
    # Create Nested JSON (Option 3)
    print("\n" + "=" * 60)
    print("CREATING NESTED JSON")
    print("=" * 60)
    
    json_data = create_nested_json(posts_df, comments_df)
    if json_data is None:
        print("\nError: Failed to create JSON format.")
        return
    
    # Save JSON
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Nested JSON saved to: {OUTPUT_JSON_PATH}")
    print(f"  Total records: {len(json_data)}")
    
    # Generate summary
    generate_summary(csv_df, json_data)
    
    # Show CSV sample
    print("\n" + "=" * 60)
    print("Sample Long Format CSV (first 3 rows)")
    print("=" * 60)
    sample_cols = [col for col in ['post_text', 'post_sentiment_label', 
                                     'comment_Comment', 'comment_sentiment_label', 'has_comment'] 
                   if col in csv_df.columns]
    if sample_cols:
        print(csv_df[sample_cols].head(3).to_string(max_colwidth=40))
    
    # Show JSON sample
    print("\n" + "=" * 60)
    print("Sample Nested JSON (first post)")
    print("=" * 60)
    if json_data:
        sample = json_data[0].copy()
        # Truncate comments for display
        if len(sample.get('comments', [])) > 2:
            sample['comments'] = sample['comments'][:2] + [{'...': f"{len(sample['comments'])-2} more comments"}]
        sample_str = json.dumps(sample, indent=2, ensure_ascii=False, default=str)
        print(sample_str[:1500] + "\n...")
    
    print("\n" + "=" * 60)
    print("Complete! Files Created:")
    print("=" * 60)
    print(f"\n1. LONG FORMAT CSV: {OUTPUT_CSV_PATH}")
    print(f"   ✓ One row per comment (post data repeated)")
    print(f"   ✓ Easy to filter and analyze in Excel/Pandas")
    print(f"   ✓ Great for: detailed analysis, filtering, ML")
    print(f"\n2. NESTED JSON: {OUTPUT_JSON_PATH}")
    print(f"   ✓ One object per post with nested comments")
    print(f"   ✓ Maintains hierarchy")
    print(f"   ✓ Great for: programmatic access, APIs, apps")
    print("=" * 60)

if __name__ == "__main__":
    main()